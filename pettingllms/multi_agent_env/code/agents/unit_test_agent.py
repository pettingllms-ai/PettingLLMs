import copy
import logging
from typing import Any

from pettingllms.multi_agent_env.base.agent import Agent, AgentData
from pettingllms.multi_agent_env.base.env import Env
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

        text_format=(
            f"**Test Cases:**\n"
                f"1. **Test Input:**\n```input here```\n\n **Test Output:**\n```output here```\n\n"
                f"2. **Test Input:**\n```input here```\n\n **Test Output:**\n```output here```\n\n"
                f"3. **Test Input:**\n```input here```\n\n **Test Output:**\n```output here```\n\n"
                f"**Format Example:**\n"
                f"**Test Cases:**\n"
                f"1. **Test Input:**\n```3\n0\n9\n1\n-1\n```\n\n**Test Output:**\n```1\n```\n\n"
                f"2. **Test Input:**\n```3\n0\n9\n1\n-3\n```\n\n**Test Output:**\n```2\n```\n\n"
                f"3. **Test Input:**\n```3\n0\n6\n1\n-2\n```\n\n**Test Output:**\n```3\n```\n\n"
            )
        text_format_single=f"**Test Input:**\n```input here```\n\n **Test Output:**\n```output here```\n\n"
        question = getattr(state, "problem", None)
        current_code = getattr(state, "generated_code", None)
        mismatch_cases = getattr(state, "generated_test_vs_generated_code_mismatch_cases", None)
        formatted_prompt_for_mismatch_cases = "The previous history of mismatch cases between your previous generated test case and another LLM generated code execution result:\n"
        for idx, code in enumerate(state.generated_code_history):
            if state.generated_test_vs_generated_code_mismatch_cases_history[idx] is not None:
                formatted_prompt_for_mismatch_cases += f"Code {idx+1}:\n{code}\n"
                for mismatch_case in state.generated_test_vs_generated_code_mismatch_cases_history[idx]:
                    formatted_prompt_for_mismatch_cases += f"Input: {mismatch_case['test_input']}\n"
                    formatted_prompt_for_mismatch_cases += f"You previous generated test case output: {mismatch_case['generated_test_output']}\n"
                    formatted_prompt_for_mismatch_cases += f"Another LLM code execution output: {mismatch_case['code_execution_output']}\n"
                    formatted_prompt_for_mismatch_cases += f"Two outputs are not the same.\n"
        
        if turn_idx == 0:
            # Test-case generation mode
            formatted_prompt = (
                f" You are a helpful assistant that the task is to generate unit test cases for coding tasks.  \n"
                f" User: Given a coding task, instead of providing the final script, your task is to generate some new test cases (both input, output).\n"
                f"This is the problem:\n{question}\n\n"
                f"You need to provide a new test case.\n"
                f"Before providing a test example, you must think carefully and reason step by step to derive an input and output you are very confident are correct. "
                f"Leverage your mathematical reasoning skills - if the problem involves mathematical concepts, formulas, algorithms, or numerical computations, "
                f"use rigorous mathematical analysis to verify your test case. Apply mathematical reasoning methods such as:\n"
                f"- Algebraic manipulation and equation solving\n"
                f"- Logical deduction and proof techniques\n"
                f"- Pattern recognition and mathematical induction\n"
                f"- Computational verification of mathematical properties\n"
                f"- Edge case analysis using mathematical bounds and constraints\n"
                f"Then reason through the expected output for your chosen input using these mathematical principles.\n"
                f"You MUST put your final test case in the following format:\n\n"
                +text_format_single
               
            )
        else:
           
            formatted_prompt =""
            formatted_prompt += (
                f" You are a helpful assistant that checks and refines test cases for coding tasks.  \n"
                f" User: Given a coding task, you need to generate test case align with the problem description.\n"
                f"This is the problem:\n{question}\n\n")
            formatted_prompt +=formatted_prompt_for_mismatch_cases + (
                f"First, according to the problem and the code generated by another LLM, you need to think if the previous test case is correct, "
                f"if you misunderstood the task or had wrong reasoning before, then give a test case which is correct.\n"
                f"Use your mathematical reasoning skills to thoroughly analyze the problem. If it involves mathematical concepts, "
                f"apply rigorous mathematical analysis including:\n"
                f"- Mathematical verification of the expected relationships\n"
                f"- Step-by-step mathematical derivation of correct outputs\n"
                f"- Validation using mathematical properties and constraints\n"
                f"- Error analysis to identify where previous reasoning went wrong\n"
                f"Ensure your corrected test case is mathematically sound and aligns with the problem requirements.\n"
                +text_format_single
                
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
        #passed_ratio = 0.0
        #self.agent_reward = passed_ratio
        env_data.state.generated_test_input = gen_inputs
        env_data.state.generated_test_output = gen_outputs
        golden_code = getattr(env_data.state, "golden_code", None)
        # 2) Evaluate generated test vs generated code (if generated code exists)
        if gen_inputs and gen_outputs and getattr(env_data.state, "generated_code", None):
            try:
                env_passed_ratio, env_passed_cases, env_failed_cases = await evaluate_code_against_tests(
                    env_data.state.generated_code, gen_inputs, gen_outputs, timeout=30.0,ray_actor=env_worker,rollout_idx=self.rollout_idx
                )
            except Exception as e:
                print(f"Warning: Failed to evaluate generated test against code: {e}")
                env_passed_ratio, env_passed_cases, env_failed_cases = 0.0, [], ["error: {e}"]
               

            env_data.state.generated_test_vs_generated_code_match_cases = env_passed_cases
            env_data.state.generated_test_vs_generated_code_mismatch_cases = env_failed_cases
            env_data.state.generated_test_vs_generated_code_match_ratio = env_passed_ratio
            env_data.state.generated_code_history.append(env_data.state.generated_code)
            env_data.state.generated_test_vs_generated_code_mismatch_cases_history.append(env_failed_cases)
            
            if env_passed_ratio >= 1.0 and len(gen_inputs) > 0:
                env_data.done = True
                
                   
            

        if gen_inputs and gen_outputs and golden_code:
            try:
                passed_ratio, passed_cases, failed_cases = await evaluate_code_against_tests(
                    env_data.state.golden_code, gen_inputs, gen_outputs, timeout=20.0,ray_actor=env_worker,rollout_idx=self.rollout_idx
                )
            except Exception as e:
                print(f"Warning: Failed to evaluate generated test against code: {e}")
                passed_ratio, passed_cases, failed_cases = 0.0, [], ["error: {e}"]
            env_data.state.generated_test_input = gen_inputs
            env_data.state.generated_test_output = gen_outputs

            env_data.state.generated_test_vs_golden_code_match_cases = passed_cases
            env_data.state.generated_test_vs_golden_code_mismatch_cases = failed_cases
            env_data.state.generated_test_vs_golden_code_match_ratio = passed_ratio
            #self.agent_reward = passed_ratio
            if passed_ratio >= 1.0 and len(generated_test_input) > 0:
                self.success = True
                
            else:
                self.success = False
               
    
    def calculate_reward(self, env_data: Env):
        self.agent_reward = env_data.state.generated_test_vs_golden_code_match_ratio+env_data.state.ground_truth_test_vs_generated_code_match_ratio
                

        
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