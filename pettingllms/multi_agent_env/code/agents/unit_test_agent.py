import copy
import logging
from typing import Any

from pettingllms.multi_agent_env.base.agent import Agent, AgentData
from pettingllms.multi_agent_env.base.env import Env
from pettingllms.utils.logger_config import get_multi_logger
from pettingllms.multi_agent_env.code.code_utils import extract_test_cases
from typing import List
logger = logging.getLogger(__name__)
from pettingllms.multi_agent_env.code.code_utils import (
        evaluate_code_against_tests,
    )

def truncatefn(s, length=300):
    if isinstance(s, str):
        pass
    else:
        s = str(s)
    if len(s) <= length:
        return s

    return s[: length // 2] + "...(truncated) ..." + s[-length // 2 :]


class UnitTestGenerationAgent(Agent):
    """
    Agent specialized for generating unit test cases.
    """

    def __init__(self, rollout_idx: int | None = None, **kwargs):
        """
        Initialize the Unit Test Generation Agent.
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
        """
        Update the agent's internal prompt after an environment step.
        Rules:
        - If either state.current_code or state.current_test_input is None/empty, prompt to generate test cases.
        - Otherwise, refine or correct tests based on existing code and test cases.
        """
        # Save environment data
        self.env_data = env_data

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
        mismatch_cases = getattr(state, "generated_test_vs_generated_code_mismatch_cases", None)
        formatted_prompt_for_mismatch_cases = ""
        for idx, code in enumerate(state.generated_code_history):
            if state.generated_test_vs_generated_code_mismatch_cases_history[idx] is not None:
                formatted_prompt_for_mismatch_cases += f"Code {idx+1}:\n{code}\n"
                for mismatch_case in state.generated_test_vs_generated_code_mismatch_cases_history[idx]:
                    formatted_prompt_for_mismatch_cases += f"Input: {mismatch_case['test_input']}\n"
                    formatted_prompt_for_mismatch_cases += f"LLM generated test case output: {mismatch_case['generated_test_output']}\n"
                    formatted_prompt_for_mismatch_cases += f"LLM code execution output: {mismatch_case['code_execution_output']}\n"
                    formatted_prompt_for_mismatch_cases += f"Two outputs are not the same.\n"
        
        if turn_idx == 0:
            # Test-case generation mode
            formatted_prompt = (
                f" You are a helpful assistant that generates test examples for coding tasks.  \n"
                f" User: Given a coding task, instead of providing the final script, your task is to generate some new test exampless (both input, output and explanation).\n"
                f"This is the problem:\n{question}\n\n"
                f"You need to provide some new test examples as much as possible. For coverage, you should provide at least 3 test examples. A good test example should be completely accurate and conform to the problem's format requirements, while also possessing enough discriminative power to distinguish correct code from incorrect code.\n"
                f"Before providing a test example, you must think carefully and reason step by step to derive an input and output you are very confident are correct. For example, start by designing an input you can reliably handle, then compute the output step by step. If you're unsure about the output, revise or re-design the input to ensure accuracy. Directly providing input/output pairs without this process is discouraged, as it often results in low accuracy.\n"
                f"Finally, after completing these previous thinking and derivation steps (you should not write the final test example unless you have gone through these steps very thoroughly), you MUST put your final test example in the following format:\n\n"
                f"**Test Cases:**\n"
                f"**Test Input:**\n```input here```\n\n**Test Output:**\n```output here```\n\n"
                f"**Format Example:**\n"
                f"**Test Cases:**\n"
                f"**Test Input:**\n```3\n0\n9\n1\n-1\n```\n\n**Test Output:**\n```1\n```\n\n"
                
            )
        else:
           
            formatted_prompt =""
            formatted_prompt += (
                f" You are a helpful assistant that refines or corrects test examples for coding tasks.  \n"
                f" User: Given a coding task, instead of providing the final script, your task is to refine or correct test examples.\n"
                f"This is the problem:\n{question}\n\n")
            formatted_prompt +=formatted_prompt_for_mismatch_cases + (
                f"First, you need to judge the mismatch history between the current generated test cases and the current code execution result, if the mismatch is caused by the current generated test cases, please refine the test cases to pass all tests.\n"
                f"Then, you need to refine the code to pass all tests.\n"
                f"Finally, you MUST put your final test example in the following format:\n\n"
                f"**Test Cases:**\n"
                f"**Test Input:**\n```input here```\n\n**Test Output:**\n```output here```\n\n"
                f"**Format Example:**\n"
                f"**Test Cases:**\n"
                f"**Test Input:**\n```3\n0\n9\n1\n-1\n```\n\n**Test Output:**\n```1\n```\n\n"
                
            )

        self.current_prompt = {"text": formatted_prompt, "image": None}
            
    def update_from_model(self, response: str):
        # Parse the response and update agent_data
        import re
        test_action = extract_test_cases(response)
        
        # Parse test cases
        self.current_action = test_action
   
        
        return self.current_action

    async def step(self, env_data: Env, env_worker:Any=None):
        """
        the action is the generated test cases, you should execute the test cases with the generated code and get the output, and then update the state
        """
        # 1) Parse and update generated test cases
        gen_inputs = self.current_action["input"]
        gen_outputs = self.current_action["output"]
        passed_ratio = 0.0
        env_data.state.generated_test_input = gen_inputs
        env_data.state.generated_test_output = gen_outputs
        golden_code = getattr(env_data.state, "golden_code", None)
        # 2) Evaluate generated test vs generated code (if generated code exists)
        if gen_inputs and gen_outputs and getattr(env_data.state, "generated_code", None):
            try:
                env_passed_ratio, env_passed_cases, env_failed_cases = await evaluate_code_against_tests(
                    env_data.state.generated_code, gen_inputs, gen_outputs, timeout=20.0,ray_actor=env_worker,rollout_idx=self.rollout_idx
                )
               

                env_data.state.generated_test_vs_generated_code_match_cases = env_passed_cases
                env_data.state.generated_test_vs_generated_code_mismatch_cases = env_failed_cases
                env_data.state.generated_test_vs_generated_code_match_ratio = env_passed_ratio
                if env_passed_ratio >= 1.0 and len(gen_inputs) > 0:
                    self.done = True
                else:
                    env_data.state.generated_code_history.append(env_data.state.generated_code)
                    env_data.state.generated_test_vs_generated_code_mismatch_cases_history.append(env_failed_cases)
                   
            except Exception as e:
                print(f"Warning: Failed to evaluate generated test against code: {e}")
                env_passed_ratio, env_passed_cases, env_failed_cases = 0.0, [], []

        if gen_inputs and gen_outputs and golden_code:
            try:
                passed_ratio, passed_cases, failed_cases = await evaluate_code_against_tests(
                    env_data.state.golden_code, gen_inputs, gen_outputs, timeout=20.0,ray_actor=env_worker,rollout_idx=self.rollout_idx
                )
                env_data.state.generated_test_input = gen_inputs
                env_data.state.generated_test_output = gen_outputs

                env_data.state.generated_test_vs_golden_code_match_cases = passed_cases
                env_data.state.generated_test_vs_golden_code_mismatch_cases = failed_cases
                env_data.state.generated_test_vs_golden_code_match_ratio = passed_ratio
                if passed_ratio >= 1.0 and len(gen_inputs) > 0:
                    self.is_pass = True
        
        
                    
            except Exception as e:
                print(f"Warning: Failed to evaluate generated test against code: {e}")
                passed_ratio, passed_cases, failed_cases = 0.0, [], []
        
        elif gen_inputs and gen_outputs and golden_code is None:
            code_is_correct=(env_data.state.ground_truth_test_vs_generated_code_match_ratio==1.0)
            if code_is_correct:
                if env_passed_ratio>=1.0:
                    passed_ratio=1.0
                    self.is_pass = True
                else:
                    passed_ratio=0.0
            else:
                if env_passed_ratio>=1.0:
                    passed_ratio=0.0
                else:
                    passed_ratio=1.0
                    self.is_pass = True
      
        
        self.agent_reward = passed_ratio
        self.reward_history.append(passed_ratio)
        self.value=passed_ratio
                

    
    def calculate_reward(self, env_data: List[Env]) -> float:
        """Compute reward for test agent based on generated tests vs generated code match ratio."""
        state = getattr(env_data[0], "state", None)
        pass_ratio = 0.0
        if state is not None:
            gen_tests_vs_golden_code = getattr(state, "generated_test_vs_golden_code_match_ratio", None)
            if isinstance(gen_tests_vs_golden_code, (int, float)):
                pass_ratio = float(gen_tests_vs_golden_code)

        self.agent_reward = pass_ratio
        self.reward_history.append(pass_ratio)
        return self.agent_reward

        
    def select_env(self, env_data: List[Env]) -> List[int]:
        if all(env.done for env in env_data):
            return -1
        
        
        found_match_env_class=False
        
        
        for env in env_data:
            if env.done:
                continue
                
            state = getattr(env, "state", None)
            if state is None:
                continue
            
            if not found_match_env_class:
                golden_test_vs_generated_code_match_ratio = getattr(state, "golden_test_vs_generated_code_match_ratio", None)
                generated_test_vs_golden_code_match_ratio = getattr(state, "generated_test_vs_golden_code_match_ratio", None)
                if golden_test_vs_generated_code_match_ratio == 1 and generated_test_vs_golden_code_match_ratio == 0:
                    found_match_env_class = True
                    return env.env_idx
            
            # 检查 golden_test_vs_generated_code_match_ratio 为 0
            if not found_match_env_class:
                generated_test_vs_golden_code_match_ratio = getattr(state, "generated_test_vs_golden_code_match_ratio", None)
                if generated_test_vs_golden_code_match_ratio == 0:
                    found_match_env_class = True
                    return env.env_idx
        
        
        if not found_match_env_class:
            for env in env_data:
                if not env.done:
                    return env.env_idx
        
        
        return -1

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