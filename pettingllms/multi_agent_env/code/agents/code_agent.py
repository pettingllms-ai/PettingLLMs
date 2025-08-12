import copy
import logging
from typing import Any
from pettingllms.multi_agent_env.base.agent import Agent, AgentData
from pettingllms.multi_agent_env.base.env import Env

logger = logging.getLogger(__name__)


def truncatefn(s, length=300):
    if isinstance(s, str):
        pass
    else:
        s = str(s)
    if len(s) <= length:
        return s

    return s[: length // 2] + "...(truncated) ..." + s[-length // 2 :]


class CodeGenerationAgent(Agent):
    """
    Agent specialized for generating code to solve programming problems.
    """

    def __init__(self, rollout_idx: int | None = None, **kwargs):
        """
        Initialize the Code Generation Agent's data.
        """
        super().__init__()
        self.rollout_idx = rollout_idx
        # Accept other unrelated keyword arguments for compatibility
        for key, value in (kwargs or {}).items():
            setattr(self, key, value)

    def update_from_env(self, env_data: Env):
        # Save environment data
        self.env_data = env_data

        # Support passing either the raw environment (with state) or a wrapped Env
        state = getattr(env_data, "state", None)
        agent_obs = getattr(env_data, "agent_observations", None)

        def as_text(value: Any) -> str:
            if value is None:
                return ""
            if isinstance(value, list):
                return "\n".join([str(v) for v in value])
            return str(value)

        if state is not None:
            question = as_text(getattr(state, "problem", ""))
            current_code = getattr(state, "current_code", None)
            current_test_input = getattr(state, "current_test_input", None)
            current_code_output = getattr(state, "current_code_output", None)
            current_test_output = getattr(state, "current_test_output", None)
        elif agent_obs is not None:
            question = as_text(agent_obs.get("question", ""))
            current_code = agent_obs.get("current_code", None)
            current_test_input = agent_obs.get("current_test_input", None)
            current_code_output = agent_obs.get("current_code_output", None)
            current_test_output = agent_obs.get("current_test_output", None)
        else:
            question = ""
            current_code = None
            current_test_input = None
            current_code_output = None
            current_test_output = None

        need_generate = current_code in (None, "") or current_test_input in (None, "")

        if need_generate:
            # Generation mode
            formatted_prompt = (
                f"You are a helpful assistant that generates code to solve programming problems.\n\n"
                f"Problem:\n{question}\n\n"
                f"Please generate correct, efficient, and readable code that solves the problem and can pass comprehensive tests.\n\n"
                f"Respond in the format:\n\n"
                f"**Code:**\n```python\n# your code here\n```\n\n"
                f"**Explanation:**\nreasoning here."
            )
        else:
            # Refinement mode
            formatted_prompt = (
                f"You are a helpful assistant that refines code to pass tests.\n\n"
                f"Problem:\n{question}\n\n"
                f"Current tests (inputs):\n{as_text(current_test_input)}\n\n"
                f"Expected outputs:\n{as_text(current_test_output)}\n\n"
                f"Current code output:\n{as_text(current_code_output)}\n\n"
                f"Please refine the code to pass all tests.\n\n"
                f"Respond in the format:\n\n"
                f"**Code:**\n```python\n# corrected code here\n```\n\n"
                f"**Explanation:**\nexplanation here."
            )

        self.agent_data.current_prompt = {"text": formatted_prompt, "image": None}
        
    def update_from_model(self, response: str):
        # Parse the response and update agent_data
        import re
        
        # Parse code
        code = ""
        
        # Try to match the code block in our prompt format
        pattern_code = r'\*\*Code:\*\*\s*```(?:[a-zA-Z]+)?\s*([\s\S]*?)```'
        code_matches = re.findall(pattern_code, response, re.DOTALL)

        # Fallback: no backticks, still extract using our anchors
        if not code_matches:
            pattern_code_plain = r'\*\*Code:\*\*\s*([\s\S]*?)(?=\*\*Explanation:\*\*|$)'
            code_matches = re.findall(pattern_code_plain, response, re.DOTALL)

        # Final fallback: capture any triple-backtick code block (no title)
        if not code_matches:
            any_block = re.findall(r'```(?:[a-zA-Z]+)?\s*([\s\S]*?)```', response, re.DOTALL)
            if any_block:
                code_matches = any_block

        if code_matches:
            code = code_matches[0].strip()

        # Update the agent's current action (environment expects a raw code string)
        self.agent_data.current_action = code
        return self.agent_data.current_action

    def calculate_reward(self, env_data: Env, mode: str = "sum") -> float:
        """
        Compute reward based on environment state, supporting three modes:
        - generated: use generated_pass_ratio (prefer generated_test_vs_generated_code_match_ratio, fallback to generated_test_vs_golden_code_match_ratio)
        - golden: use golden_pass_ratio (golden_test_vs_generated_code_match_ratio)
        - sum/both/others: sum of both
        """
        state = getattr(env_data, "state", None)
        generated_pass_ratio = 0.0
        golden_pass_ratio = 0.0

        if state is not None:
            # Generated tests vs generated code
            gen_vs_gen = getattr(state, "generated_test_vs_generated_code_match_ratio", None)
            # Generated tests vs golden code (as fallback)
            gen_vs_gold = getattr(state, "generated_test_vs_golden_code_match_ratio", None)
            # Golden tests vs generated code
            gold_vs_gen = getattr(state, "golden_test_vs_generated_code_match_ratio", None)

            if isinstance(gen_vs_gen, (int, float)):
                generated_pass_ratio = float(gen_vs_gen)
            elif isinstance(gen_vs_gold, (int, float)):
                generated_pass_ratio = float(gen_vs_gold)

            if isinstance(gold_vs_gen, (int, float)):
                golden_pass_ratio = float(gold_vs_gen)

        m = (mode or "sum").lower()
        if m in ("generated", "gen"):
            reward = generated_pass_ratio
        elif m in ("golden", "gold"):
            reward = golden_pass_ratio
        else:
            reward = generated_pass_ratio + golden_pass_ratio

        # Record and return
        self.agent_data.agent_reward = reward
        if self.agent_data.info is None:
            self.agent_data.info = {}
        self.agent_data.info.update({
            "generated_pass_ratio": generated_pass_ratio,
            "golden_pass_ratio": golden_pass_ratio,
            "reward_mode": m,
        })
        return reward

    
    
    
    
    
    def reset(self):
        """
        Reset the agent's internal state for a new episode.
        """
        self.agent_data = AgentData()

 