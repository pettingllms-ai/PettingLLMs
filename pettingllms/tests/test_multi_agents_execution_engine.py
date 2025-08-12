import os
import asyncio
from types import SimpleNamespace

import pytest

from transformers import AutoTokenizer
AutoTokenizer = None

from pettingllms.trainer.multi_agents_execution_engine import MultiAgentsExecutionEngine
from pettingllms.router.router import Router
from pettingllms.parser.chat_template.parser import ChatTemplateParser


class _AttrDict(SimpleNamespace):
    def get(self, key, default=None):
        return getattr(self, key, default)


def _build_minimal_config(model_path: str) -> _AttrDict:
    # rollout and model sub-configs required by Router
    rollout = _AttrDict(
        n=1,
        response_length=128,
        temperature=0.7,
        top_p=0.95,
        val_kwargs=_AttrDict(top_p=0.95, temperature=0.0),
    )
    model = _AttrDict(path=model_path)
    actor_rollout_ref = _AttrDict(rollout=rollout, model=model)

    # training/data/env configs used by engine
    data = _AttrDict(
        gen_batch_size=1,
        gen_n_samples=1,
        sample_temperature=0.7,
        max_prompt_length=1024,
        max_response_length=256,
    )
    env = _AttrDict(
        name="code_env",
        benchmark="deepmind/code_contests",
        max_turns=2,
        resolve=False,
        multi_modal=False,
        batched_init=False,  # 避免测试时实际加载数据集
    )
    multi_agent_interaction = _AttrDict(
        turn_order=["code_generator", "test_generator"],
        num_interacting_agents=2,
        shared_observation=True,
    )

    # top-level config with attribute access
    cfg = _AttrDict(
        data=data,
        env=env,
        multi_agent_interaction=multi_agent_interaction,
        actor_rollout_ref=actor_rollout_ref,
    )
    return cfg


# -------------------- Dummy 实现，避免依赖真实 env/agent --------------------
class DummyEnv:
    def __init__(self, env_idx: int, rollout_idx: int, max_turns: int, **kwargs):
        self.env_idx = env_idx
        self.rollout_idx = rollout_idx
        self.max_turns = max_turns
        self.turn = 0
        self.state = SimpleNamespace(
            problem="Add numbers",
            current_code=None,
            current_test_input=None,
            current_code_output=None,
            current_test_output=None,
            golden_code="print(input())",
            golden_test_input=["1"],
            golden_test_output=["1"],
        )

    def step(self, role: str, action):
        self.turn += 1
        # no-op; keep minimal compatibility
        return None


class DummyEnvBatch:
    def __init__(self, env_idx_list, rollout_idx_list, max_turns: int, config=None):
        self.env_list = [DummyEnv(i, j, max_turns) for i, j in zip(env_idx_list, rollout_idx_list)]


class _BaseDummyAgent:
    def __init__(self, rollout_idx=None, **kwargs):
        self.rollout_idx = rollout_idx
        self._current_prompt = {"text": "", "image": None}
        self._action = None
        self._reward = 0.0

    @property
    def current_prompt(self):
        return self._current_prompt

    @property
    def action(self):
        return self._action

    @property
    def reward(self):
        return self._reward

    def calculate_reward(self, env, mode="sum"):
        # 简单固定奖励，验证数据流
        self._reward = 1.0
        return self._reward


class DummyCodeAgent(_BaseDummyAgent):
    def update_from_env(self, env):
        self._current_prompt = {"text": "You are a helper that generates code. Please output code.", "image": None}

    def update_from_model(self, response: str):
        # 直接将响应设为 action
        self._action = response
        return self._action


class DummyTestAgent(_BaseDummyAgent):
    def update_from_env(self, env):
        self._current_prompt = {"text": "You are a helper that generates tests. Please output tests.", "image": None}

    def update_from_model(self, response: str):
        self._action = response
        return self._action


class DummyRouter(Router):
    async def generate_sequences(self, batch, application_id: str, **sampling_params):
        # 为每个 prompt 生成一个响应 token 序列
        responses_token_ids = []
        for formatted_prompt in batch.non_tensor_batch["formatted_prompts"]:
            if "generates code" in formatted_prompt or "output code" in formatted_prompt:
                text = "**Code:**\n```python\nprint(1)\n```\n\n**Explanation:**\nOK"
            else:
                text = "**Test Input:**\n```\n1\n```\n\n**Test Output:**\n```\n1\n```\n\n**Explanation:**\nOK"
            token_ids = self.tokenizer(text, add_special_tokens=False, return_tensors=None)["input_ids"]
            # postprocess_batch 期望每个样本是长度为 n 的 list[list[int]]
            responses_token_ids.append([token_ids])
        return await self.postprocess_batch(batch, responses_token_ids, n=1)


def test_generate_single_rollout_minimal():
    model_path = os.environ.get("TEST_QWEN_MODEL", "Qwen/Qwen2.5-0.5B")

    if AutoTokenizer is None:
        pytest.skip("skip: transformers 未安装")

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    except Exception as e:
        pytest.skip(f"skip: cannot load tokenizer for {model_path}: {e}")

    cfg = _build_minimal_config(model_path)

    # 替换注册表为 Dummy 实现，避免导入真实 env/agent 失败
    import pettingllms.trainer.multiagentssys_register as reg

    reg.ENV_CLASS_MAPPING["code_env"] = DummyEnv
    reg.ENV_BATCH_CLASSES = getattr(reg, "ENV_BATCH_CLASSES", {})
    reg.ENV_BATCH_CLASSES["code_env"] = DummyEnvBatch
    reg.AGENT_CLASS_MAPPING["code_generator"] = DummyCodeAgent
    reg.AGENT_CLASS_MAPPING["test_generator"] = DummyTestAgent

    # 路由器：若提供 VLLM_ADDR 则用真路由器，否则 DummyRouter
    addr = os.environ.get("VLLM_ADDR")
    if addr:
        router = Router(config=cfg, tokenizer=tokenizer, addresses=[addr])
    else:
        router = DummyRouter(config=cfg, tokenizer=tokenizer, addresses=[])

    model_name = "code_generator_model"
    engine = MultiAgentsExecutionEngine(
        config=cfg,
        tokenizer_dict={model_name: tokenizer},
        processor_dict={model_name: None},  # key 必须存在
        router_dict={model_name: router},
        agent_policy_mapping={
            "code_generator": model_name,
            "test_generator": model_name,
        },
        env_args={},
        max_workers=2,
    )

    # 引擎内部未初始化 chat_parser_dict，这里补齐
    engine.chat_parser_dict = {model_name: ChatTemplateParser.get_parser(tokenizer, disable_thinking=False)}

    async def _run():
        return await engine.generate_single_rollout(rollout_idx=0, timing_raw={}, meta_info={})

    result = asyncio.run(_run())

    assert isinstance(result, dict) and model_name in result
    dpr = result[model_name]
    assert hasattr(dpr, "non_tensor_batch")
    assert "reward" in dpr.non_tensor_batch
    assert isinstance(dpr.non_tensor_batch["reward"], list)
    assert len(dpr.non_tensor_batch["reward"]) >= 1


if __name__ == "__main__":
    asyncio.run(test_generate_single_rollout_minimal())


