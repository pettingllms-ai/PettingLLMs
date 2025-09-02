import logging
from typing import Any, Dict, Optional

from pettingllms.multi_agent_env.base.agent import Agent
from pettingllms.multi_agent_env.base.env import Env
from pettingllms.utils.logger_config import get_multi_logger

# 我们将在 utils.py 里实现下面这些
from .utils import (
    build_prompt_from_obs,
    extract_action_from_text,
    choose_executable_action,
)

logger = logging.getLogger(__name__)


class AlfWorldAgent(Agent):

    def __init__(self, rollout_idx: Optional[int] = None, **kwargs):
        super().__init__()
        self.rollout_idx = rollout_idx
        for k, v in (kwargs or {}).items():
            setattr(self, k, v)
        self.multi_logger = get_multi_logger()
        self.current_prompt = None
        self.current_action = None

    def update_from_env(self, env):
        task_description = env.task_description
        current_observation = env.observation
        admissible_actions = ", ".join(env.admissible_actions)
        action_history = ", ".join(env.action_history)
        

        format_prompt = f"""
    You are an expert agent operating in the ALFRED Embodied Environment. Your task is to: {task_description}
    Below are the corresponding actions you took:
    {action_history}
    Your current observation is:
    {current_observation}
    Your admissible actions for the current situation are: [{admissible_actions}].
    Now it's your turn to take an action.
    You should first reason step-by-step about the current situation. 
    Once you've finished your reasoning, you should choose an admissible action for the current step and present it within <action> </action> tags.
    """.strip()

        self.current_prompt = {"text": format_prompt, "image": None}


    def update_from_model(self, response: str):
        """从模型文本中抽取动作（字符串），并尝试对齐 admissible actions。"""
        obs = getattr(self.env_data, "agent_observations", None) or {}
        admissible = obs.get("admissible_actions", []) or []
        intent = extract_action_from_text(response)
        action = choose_executable_action(intent, admissible)
        self.current_action = action
        return action

    async def step(self, env_worker: Any = None):
        action = self.current_action
        obs, reward, done, info = await env_worker.step.remote(action)
        if len(self.reward_history) > 0:
            self.agent_reward = float(reward) - float(self.reward_history[-1])
        else:
            self.agent_reward = float(reward)
        self.reward_history.append(float(reward))
        self.done = bool(done)
        if self.done:
            self.is_pass = True


    def reset(self):
        self.current_action = None
        self.current_prompt = None
        self.current_response = None
        self.current_reward = None
        self.current_info = None
        self.done = False
        self.reward_history.clear()
