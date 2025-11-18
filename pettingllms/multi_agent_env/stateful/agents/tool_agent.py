import logging
import copy
from typing import Any, List, Optional

from pettingllms.multi_agent_env.base.agent import Agent
from pettingllms.multi_agent_env.base.env import Env
from pettingllms.multi_agent_env.math.math_worker import get_code_execution_output
from pettingllms.multi_agent_env.stateful.utils import (
    extract_code_from_response,
    extract_actions_from_code_output
)
from pettingllms.multi_agent_env.stateful.prompt import build_tool_prompt

logger = logging.getLogger(__name__)


class ToolAgent(Agent):
    """
    Code generation style tool agent
    - Determines initial/subsequent prompts via benchmark parameter
    - Other logic (execution, parsing, scoring, write-back, completion) remains unchanged
    """

    def __init__(self, rollout_idx: int | None = None, benchmark: str = "plan_path", **kwargs):
        super().__init__()
        self.rollout_idx = rollout_idx
        self.benchmark = benchmark
        self.agent_reward_history = []
        self.success = False
        self.agent_reward = 0.0
        for k, v in (kwargs or {}).items():
            setattr(self, k, v)

    def update_from_env(self, turn_idx: int, env_data: Env):
        """Update agent prompt based on environment state"""
        self.env_data = env_data
        state = getattr(env_data, "state", None)

        formatted_prompt = (
            "You are an AI assistant specialized in solving planning problems through code generation. Your task is to analyze the given scenario and generate Python code that produces a sequence of actions to solve the problem.\n\nInstructions:\n1. Write Python code enclosed in ```python and ``` tags\n2. Your code should output an action sequence using print() function\n"

        )

        # Add history information when turn > 0
        if turn_idx > 0 and hasattr(state, 'code_history'):
            history_formatted_prompt = ""
            history_formatted_prompt += "\n\n=== HISTORY FROM PREVIOUS TURNS ===\n"
            for i in range(len(state.code_history)):
                history_formatted_prompt += f"\n--- Turn {i+1} ---\n"
                if i < len(state.code_history):
                    # Truncate code to 500 characters
                    code_preview = state.code_history[i]
                    if len(code_preview) > 500:
                        code_preview = code_preview[:250] + "\n...(truncated)...\n" + code_preview[-250:]
                    history_formatted_prompt += f"Previous Code:\n{code_preview}\n"
                if i < len(state.execution_history):
                    # Truncate execution output to 300 characters
                    exec_preview = state.execution_history[i]
                    if len(exec_preview) > 300:
                        exec_preview = exec_preview[:150] + "...(truncated)..." + exec_preview[-150:]
                    history_formatted_prompt += f"Previous Execution Output:\n{exec_preview}\n"
                if i < len(state.tool_action_history):
                    history_formatted_prompt += f"Previous Tool Action: {state.tool_action_history[i]}\n"
                if i < len(state.plan_action_history):
                    history_formatted_prompt += f"Previous Plan Action: {state.plan_action_history[i]}\n"
            history_formatted_prompt += "\n=== END OF HISTORY ===\n\n"
            #formatted_prompt += history_formatted_prompt
        formatted_prompt += build_tool_prompt(self.benchmark, turn_idx, state)
        
        if self.benchmark in ("plan_path", "sokoban"):
            formatted_prompt += (
                "3. Actions should be represented as a list of strings: ['U', 'D', 'L', 'R'] (Up, Down, Left, Right)\n"
                "4. You may return either the complete action sequence to reach the goal, or a partial sequence if you're uncertain\n"
                "5. Remember, please do not return the result directly, you need to print the final result. If your algorithm produces numerical results, convert them using: action_map = {0:'U', 1:'D', 2:'L', 3:'R'}\n6. Ensure your code is executable and produces clear output\n\n\n5. Remember, please do not return the result directly, you need to print the final result. \n6. Please use algorithm like BFS or A* to solve the problem. Very important, print the final result. \n7. Important: Your code must output the final action sequence in this exact format:\n**Actions List**: [\"U\", \"R\", \"D\", \"L\"] (example for path planning)\n\nNote: If your algorithm produces numerical results, convert them using action_map = {{0:'U', 1:'D', 2:'L', 3:'R'}} before outputting.\n\n"
                
            )
        elif self.benchmark == "suduku":
            formatted_prompt += (
                "3. For Sudoku, return the complete grid solution.\n"
                "4. Ensure your code is executable and produces clear output\n\n"
            )
        else:
            formatted_prompt += (
                "3. Actions should be represented as a list of strings: ['U', 'D', 'L', 'R'] (Up, Down, Left, Right)\n"
                "4. You may return either the complete action sequence to reach the goal, or a partial sequence if you're uncertain\n"
                "5. Ensure your code is executable and produces clear output\n\n"
            )
        
        if self.benchmark in ("plan_path", "sokoban"):
            formatted_prompt += ""
        elif self.benchmark == "suduku":
            formatted_prompt += (
                "Important: Your code must output the final action in this exact format:\n"
                "**Actions List**: [[1,2,3,4],[3,4,1,2],[2,1,4,3],[4,3,2,1]] (complete grid)\n"
            )
        else:
            formatted_prompt += (
                "Important: Your code must output the final action sequence in this exact format:\n"
                "**Actions List**: [\"U\", \"R\", \"D\", \"L\"] (example). If solved/no moves needed, output an empty list [].\n"
            )
        
        self.current_prompt = {"text": formatted_prompt, "image": None}

    def update_from_model(self, response: str):
        """Extract code from model response"""
        if response is None:
            self.current_code = ""
            return self.current_code
            
        self.current_code = extract_code_from_response(response)
        return self.current_code

    async def step(self, env_data: Env, env_worker: Any = None):
        """Execute code, parse actions, score and update environment"""
        generated_code = self.current_code or ""
        if self.current_code is None:
            self.agent_reward = 0.0  # No code - failure
        env_data.state.code_generated_action = generated_code

        code_execution_output = None
        try:
            code_execution_output = await get_code_execution_output(
                generated_code,
                timeout=20.0,
                ray_actor=env_worker,
            )
            env_data.state.code_execution_output = code_execution_output
        except Exception as e:
            code_execution_output = f"error: {e}"
            env_data.state.code_execution_output = code_execution_output

        if code_execution_output is None:
            self.agent_reward = 0.0  # No output - failure

        env_data.state.tool_execution_output = code_execution_output
        env_data.state.tool_code = generated_code

        self.current_action = extract_actions_from_code_output(code_execution_output or "", self.benchmark)

        env_data.state.tool_action = self.current_action

        # Save to history after executing code and getting results
        if hasattr(env_data.state, 'code_history'):
            env_data.state.code_history.append(generated_code if generated_code else "")

        if hasattr(env_data.state, 'execution_history'):
            env_data.state.execution_history.append(code_execution_output if code_execution_output else "")

        if hasattr(env_data.state, 'tool_action_history'):
            env_data.state.tool_action_history.append(copy.deepcopy(self.current_action) if self.current_action else [])

        state = copy.deepcopy(env_data.state)
        state.step(self.current_action)

        if self.benchmark in ("plan_path", "sokoban") and self.current_action is None:
            self.agent_reward = 0.0  # No action - failure
        else:
            self.agent_reward = state.reward  # Binary reward from state (0 or 1)


        if hasattr(state, 'done') and state.done:
            self.success = True
    
    def calculate_reward(self, env_data: Env):
        self.agent_reward = self.agent_reward+env_data.state.reward
        self.reward_history.append(self.agent_reward)
    

    def reset(self):
        """Reset agent state"""
        self.current_action = None
        self.current_prompt = None
        self.current_response = None
        self.current_reward = None
        self.current_info = None
        self.done = False
        self.is_pass = False
        self.success = False
        self.agent_reward = 0.0
