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
        self.current_response = None  # Store the full model response for history

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
            formatted_prompt+= "Format: output a single JSON list of moves after #### (e.g., a list of 'U','D','L','R'). Do not output placeholders. Please think step by step. Do not directly give the final action list. If the code produces numerical results, convert them using: action_map = {0:'U', 1:'D', 2:'L', 3:'R'}\n6. Ensure your code is executable and produces clear output\n\n\n"
        if self.benchmark == "suduku":
            formatted_prompt+= "Example: #### [[1,2,3,4],[3,4,1,2],[2,1,4,3],[4,3,2,1]] or #### [[0,1,2],[0,2,3],[1,0,4]].\n"


        formatted_prompt+= build_plan_prompt(self.benchmark,turn_idx, state)

        # Add history information when turn > 0
        if turn_idx > 0 and hasattr(state, 'code_history'):
            history_formatted_prompt = ""
            history_formatted_prompt += "\n\n=== HISTORY FROM PREVIOUS TURNS ===\n"
            for i in range(len(state.code_history)):
                history_formatted_prompt += f"\n--- Turn {i+1} ---\n"
                if i < len(state.code_history):
                    history_formatted_prompt += f"Tool Agent's Code:\n{truncatefn(state.code_history[i], 500)}\n"
                if i < len(state.execution_history):
                    history_formatted_prompt += f"Execution Output:\n{truncatefn(state.execution_history[i], 300)}\n"
                if i < len(state.tool_action_history):
                    history_formatted_prompt += f"Tool Agent's Action: {state.tool_action_history[i]}\n"
                if i < len(state.plan_action_history):
                    history_formatted_prompt += f"Plan Agent's Action: {state.plan_action_history[i]}\n"
            history_formatted_prompt += "\n=== END OF HISTORY ===\n\n"
            #formatted_prompt += history_formatted_prompt

        formatted_prompt+= f"Here is code agent's code: {state.tool_code}.\n"
        formatted_prompt+= f"Here is code agent's execution output: {state.tool_execution_output}. "
        formatted_prompt+= f"Here is code agent's action: {state.tool_action}.\n"



        self.current_prompt = {"text": formatted_prompt, "image": None}


    def update_from_model(self, response: str):
        self.current_response = response  # Store full response for history
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

        # Save to history before stepping
        # Note: We save the truncated reasoning (response) before ####, and the action after ####
        if hasattr(state, 'plan_action_history'):
            state.plan_action_history.append(copy.deepcopy(self.current_action))

        if hasattr(state, 'plan_reasoning_history') and self.current_response:
            # Extract reasoning part (before ####)
            reasoning_part = self.current_response
            if "####" in self.current_response:
                reasoning_part = self.current_response.split("####")[0].strip()
            state.plan_reasoning_history.append(truncatefn(reasoning_part, 300))

        state.step(self.current_action)
        self.action_list.append(self.current_action)
        if self.current_action is None or self.current_action == []:
            self.agent_reward = 0.0  # No action - failure
        else:
            self.agent_reward = state.reward  # Binary reward from state (0 or 1)
        if hasattr(state, 'done') and state.done:
            env_data.done = True
            self.success = True
            env_data.state.success = True
        
    
    def calculate_reward(self, env_data: Env):
        self.agent_reward = self.agent_reward+env_data.state.reward
        self.reward_history.append(self.agent_reward)