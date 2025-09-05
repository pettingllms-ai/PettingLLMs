import copy
import logging
from typing import Any
from pettingllms.multi_agent_env.base.agent import Agent, AgentData
from pettingllms.multi_agent_env.base.env import Env
from pettingllms.utils.logger_config import get_multi_logger
from typing import List
from pettingllms.multi_agent_env.code.code_utils import (
        evaluate_code_against_tests,
    )
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

        self.multi_logger = get_multi_logger()

    def reset(self):
        super().reset()

    def update_from_env(self, turn_idx: int, env_data: Env):
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
        

        question = getattr(state, "problem", None)
        current_code = getattr(state, "generated_code", None)
        current_test_input = getattr(state, "generated_test_input", None)
        current_test_output = getattr(state, "generated_test_output", None)
        current_code_output = getattr(state, "exe_code_generated_test_output", None)
        
        mismatch_cases = getattr(state, "generated_test_vs_generated_code_match_cases", None)
       
        formatted_prompt_for_mismatch_cases = "The previous history of mismatch cases between the current generated test cases and the current code execution result:\n"
        for idx, code in enumerate(state.generated_code_history):
            if state.generated_test_vs_generated_code_mismatch_cases_history[idx] is not None:
                formatted_prompt_for_mismatch_cases += f"Code {idx+1}:\n{code}\n"
                for mismatch_case in state.generated_test_vs_generated_code_mismatch_cases_history[idx]:
                    formatted_prompt_for_mismatch_cases += f"Input: {mismatch_case['test_input']}\n"
                    formatted_prompt_for_mismatch_cases += f"Expected output: {mismatch_case['generated_test_output']}\n"
                    formatted_prompt_for_mismatch_cases += f"Actual mismatch output: {mismatch_case['code_execution_output']}\n"
        
        if turn_idx == 0:
            # Generation mode
            formatted_prompt = (
                f"You are a helpful assistant that generates code to solve programming problems.\n\n"
                f"You need to think first then write python script."
                f"You should use input() to input and print() to output in your script.\n```"
                f"Problem:\n{question}\n\n"
                f"Please generate correct, efficient, and readable code that solves the problem and can pass comprehensive tests.\n\n"
                f"Respond in the format:\n\n"
                f"**Code:**\n```python\n# your code here\n```\n\n"
               
            )
        else:
            # Refinement mode
            formatted_prompt = (
                f"You are a helpful assistant that refines code to pass tests. You need to think first then refine and generate new python script.\n\n"
                f" Given a coding task, you need to check the current generated test cases and the current code execution result, if the mismatch is caused by the current generated test cases, please refine the code to pass all tests.\n"
                f"This is the problem:\n{question}\n\n"
                f"You need to think first then write python script.")
            formatted_prompt += formatted_prompt_for_mismatch_cases + (
                f"Please first judge the mismatch between the current generated test cases history and the current code execution result history, if the mismatch is caused by the current code, please refine the code to pass all tests. If the mismatch is not caused by the current code, please answer the original code.\n"
                f"Refine the code to pass all tests.\n\n"
                f"Respond in the format:\n\n"
                f"**Code:**\n```python\n# corrected code here\n```\n\n"
            )

        self.current_prompt = {"text": formatted_prompt, "image": None}
        
    
    def update_from_model(self, response: str):
        # Parse the response and update agent_data
        import re
        
        # Parse code
        code = ""
        
        # Try to match the code block in our prompt format
        matches = re.findall(r"```python(.*?)```", response, re.DOTALL)
        if matches:
            code = matches[-1].strip()
        else:
            code = "We can not extract the code in the output. "

            # Update the agent's current action (environment expects a raw code string)
        self.current_action = code

        return self.current_action

    async def step(self, env_data: Env, env_worker:Any=None):
        """
        the action is the generated code, you should execute the code and get the output, and then update the state
        """
        # 1) Parse and update generated code
        gen_code = self.current_action
        env_data.state.generated_code = gen_code
        
        # 2) Evaluate generated test vs generated code (if exists)
        #    Allow reading from state.current_test_input/current_test_output
        ground_truth_test_input = env_data.state.ground_truth_test_input or []
        ground_truth_test_output = env_data.state.ground_truth_test_output or []
        passed_ratio = 0.0
        #print(f"env_worker_idx: {env_worker.get_idx.remote()}")
        if isinstance(ground_truth_test_input, list) and isinstance(ground_truth_test_output, list) and ground_truth_test_input and ground_truth_test_output:
            try:
                passed_ratio, passed_cases, failed_cases = await evaluate_code_against_tests(
                    gen_code, ground_truth_test_input, ground_truth_test_output, timeout=15.0, ray_actor=env_worker,rollout_idx=self.rollout_idx
                )
                env_data.state.ground_truth_test_vs_generated_code_match_cases = passed_cases
                env_data.state.ground_truth_test_vs_generated_code_mismatch_cases = failed_cases
                env_data.state.ground_truth_test_vs_generated_code_match_ratio = passed_ratio
                if passed_ratio < 1.0:
                    passed_ratio = 0.0
                if passed_ratio >= 1.0 and len(ground_truth_test_input) > 0:
                    self.done = True
                    self.is_pass = True
                
        
            except Exception as e:
                print(f"Warning: Failed to evaluate code against tests: {e}")
                passed_ratio, passed_cases, failed_cases = 0.0, [], []
        self.agent_reward = passed_ratio
        self.reward_history.append(passed_ratio)
        self.value=passed_ratio



    
    
    def calculate_reward(self, env_data: List[Env]) -> float:
        """
        Compute reward based on environment state.
        Uses generated_test_vs_generated_code_match_ratio for reward calculation.
        """
        state = getattr(env_data[0], "state", None)
        pass_ratio = 0.0

        if state is not None:
            # Generated tests vs generated code
            ground_truth_vs_generated = getattr(state, "ground_truth_test_vs_generated_code_match_ratio", None)
            if isinstance(ground_truth_vs_generated, (int, float)):
                pass_ratio = float(ground_truth_vs_generated)
            elif ground_truth_vs_generated is not None:
                try:
                    pass_ratio = float(ground_truth_vs_generated)
                except Exception:
                    pass

        # Record and return
        self.agent_reward = pass_ratio
        self.reward_history.append(self.agent_reward)

        
        return self.agent_reward

    
    
    
    
    
    def reset(self):
        """
        Reset the agent's internal state for a new episode.
        """
        self.current_action = None
        self.current_prompt = None
        self.current_response = None
        self.current_reward = None
        self.current_info = None
        self.current_action = None
        self.current_prompt = None
        self.current_response = None

 