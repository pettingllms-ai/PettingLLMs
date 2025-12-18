import re
import json
import copy
import logging
import ast
from typing import Any, List, Tuple, Dict, Optional

from pettingllms.multi_agent_env.base.agent import Agent, AgentData
from pettingllms.multi_agent_env.base.env import Env

from pettingllms.multi_agent_env.stateful.prompt import build_plan_prompt
from pettingllms.multi_agent_env.stateful.utils import  extract_final_action
logger = logging.getLogger(__name__)

def truncatefn(s, length=300):
    if not isinstance(s, str):
        s = str(s)
    return s if len(s) <= length else s[: length // 2] + "...(truncated)..." + s[-length // 2 :]





class PlanAgent(Agent):
    """
    Unified PlanWalker:
    - benchmark: plan_path | eight_queens | blocksworld | suduku
    - Only prompt changes with benchmark; evaluation and write-back pipeline remains consistent.
    """

    def __init__(self, rollout_idx: int | None = None, benchmark: str = "plan_path", **kwargs):
        super().__init__()
        self.rollout_idx = rollout_idx
        self.benchmark = benchmark
        for k, v in (kwargs or {}).items():
            setattr(self, k, v)
        self.action_list = []
        self.state_list = []
        self.success = False
        self.agent_reward = 0.0

    def reset(self):
        self.action_list = []
        self.state_list = []
        self.success = False
        self.agent_reward = 0.0

    # ===================== Prompt Construction (Externalized) =====================
    def update_from_env(self, turn_idx: int, env_data: Env):
        self.env_data = env_data
        state = getattr(env_data, "state", None)
        formatted_prompt = f"You are a planning and reasoning agent. You will receive: The original task description, The Code Agent's code, The code execution output. Your job is to reason carefully, decide the final action, and format your response exactly as specified. Instructions: Read the task, inspect the code, and verify the execution output against the task requirements. If the code/output is correct and sufficient, adopt it; otherwise, improve or override it with your own reasoning. Keep your reasoning concise but explicit: justify why the final action is correct. Formatting is mandatory. Output the action list after a ####. "
        if self.benchmark in ("plan_path", "sokoban"):
            formatted_prompt+= "Format: output a single JSON list of moves after #### (e.g., a list of 'U','D','L','R'). Do not output placeholders. Please think step by step. Do not directly give the final action list. \n"

        formatted_prompt+= build_plan_prompt(self.benchmark,turn_idx, state)
        formatted_prompt+= f"Here is code agent's code: {state.tool_code}.\n"
        formatted_prompt+= f"Here is code agent's execution output: {state.tool_execution_output}. "
        formatted_prompt+= f"Here is code agent's action: {state.tool_action}.\n"

        # Get image from environment state
        image_data = None
        if hasattr(state, 'observation') and state.observation is not None:
            from PIL import Image
            if isinstance(state.observation, Image.Image):
                image_data = state.observation

        self.current_prompt = {"text": formatted_prompt, "image": image_data}


    
    def update_from_model(self, response: str):
        if response is None:
            self.current_action = []
            return self.current_action
            
        self.current_action = extract_final_action(response, self.benchmark)
        if self.current_action is None:
            self.current_action = []
        return self.current_action

    async def step(self, env_data: Env, env_worker: Any = None):
        env_data.state.plan_action = self.current_action
        state = env_data.state
        self.state_list.append(state)
        state.step(self.current_action)
        self.action_list.append(self.current_action)
        if self.current_action is None or self.current_action == []:
            self.agent_reward = -2
        else:
            self.agent_reward = state.reward
        if hasattr(state, 'done') and state.done:
            env_data.done = True
            self.success = True
            env_data.success = True
        
    
    def calculate_reward(self, env_data: Env):
        self.agent_reward = self.agent_reward+env_data.state.reward
        self.reward_history.append(self.agent_reward)