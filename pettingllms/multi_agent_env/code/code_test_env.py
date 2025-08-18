import logging
import copy
import io
import sys
import time
import typing
import multiprocessing as mp
from typing import Any, Dict, Optional, Tuple, List

from pettingllms.multi_agent_env.code.agents.code_agent import CodeGenerationAgent
from pettingllms.multi_agent_env.code.agents.unit_test_agent import UnitTestGenerationAgent
from pettingllms.multi_agent_env.base.env import MultiAgentsEnvironment
from pettingllms.multi_agent_env.code.code_utils import (
        load_problem_batch,
        extract_code_from_response,
        evaluate_code_against_tests,
    )
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class CodeTestEnvState:
    problem: str=None
    generated_code: str=None
    generated_test_input: List[str]=None
    generated_test_output: List[str]=None
    ground_truth_test_input: List[str]=None
    ground_truth_test_output: List[str]=None
    exe_code_generated_test_output: List[str]=None
    exe_code_ground_truth_test_output: List[str]=None
    # Evaluation results: generated test vs generated code
    ground_truth_test_vs_generated_code_mismatch_cases: List[Dict]=None
    ground_truth_test_vs_generated_code_match_cases: List[Dict]=None
    ground_truth_test_vs_generated_code_match_ratio: float=None
    generated_test_vs_generated_code_match_cases: List[Dict]=None
    generated_test_vs_generated_code_mismatch_cases: List[Dict]=None
    generated_test_vs_generated_code_match_ratio: float=None


class CodeTestEnv(MultiAgentsEnvironment):
    """
    Environment for code generation and testing tasks with dual-agent interaction.
    
    This environment coordinates between code generation and unit test generation agents,
    similar to how WebEnv coordinates between code and visual agents.
    """

    def __init__(
        self, 
        env_idx: int,
        rollout_idx: int,
        max_turns: int,
        config: dict | None = None,
      
    ):
        """
        Initialize the code test environment.

    
        """
        super().__init__(env_idx=env_idx, rollout_idx=rollout_idx, max_turns=max_turns, config=config)
        self.state=CodeTestEnvState()


   
    async def step(self, role: str, action: str):
        if role == "code_generator":
            await self._code_step(action)
        elif role == "test_generator":
            await self._test_step(action)
        else:
            raise ValueError(f"Invalid role: {role}")

        pass

    async def _code_step(self, action: str):
        """
        the action is the generated code, you should execute the code with the generated test cases, and get the output, and then update the state
        """
        # 1) Update generated code
        generated_code = action
        self.state.generated_code = generated_code

        # 2) Evaluate generated test vs generated code (if exists)
        #    Allow reading from state.current_test_input/current_test_output
        ground_truth_test_input = self.state.ground_truth_test_input or []
        ground_truth_test_output = self.state.ground_truth_test_output or []
        if isinstance(ground_truth_test_input, list) and isinstance(ground_truth_test_output, list) and ground_truth_test_input and ground_truth_test_output:
            try:
                passed_ratio, passed_cases, failed_cases = await evaluate_code_against_tests(
                    generated_code, ground_truth_test_input, ground_truth_test_output, timeout=5.0
                )
            except Exception as e:
                print(f"Warning: Failed to evaluate code against tests: {e}")
                passed_ratio, passed_cases, failed_cases = 0.0, [], []

           

           
            self.state.ground_truth_test_vs_generated_code_match_cases = passed_cases
            self.state.ground_truth_test_vs_generated_code_mismatch_cases = failed_cases
            self.state.ground_truth_test_vs_generated_code_match_ratio = passed_ratio

    async def _test_step(self, action: dict):
        """
        the action is the generated test cases, you should execute the test cases with the generated code and get the output, and then update the state
        """
        # 1) Parse and update generated test cases
        gen_inputs = action["input"]
        gen_outputs = action["output"]
            
        self.state.generated_test_input = gen_inputs
        self.state.generated_test_output = gen_outputs

        # 2) Evaluate generated test vs generated code (if generated code exists)
        if gen_inputs and gen_outputs and getattr(self.state, "generated_code", None):
            try:
                passed_ratio, passed_cases, failed_cases = await evaluate_code_against_tests(
                    self.state.generated_code, gen_inputs, gen_outputs, timeout=5.0
                )

                self.state.generated_test_vs_generated_code_match_cases = passed_cases
                self.state.generated_test_vs_generated_code_mismatch_cases = failed_cases
                self.state.generated_test_vs_generated_code_match_ratio = passed_ratio
                
                # 3) Check if all generated tests pass with generated code
                # If match ratio is 1.0 (100%), set environment to done state
                if passed_ratio >= 1.0 and len(gen_inputs) > 0:
                    self.done = True
                    self.is_pass = True
                    self.termination_reason = "all_tests_passed"
                    logger.info(f"All {len(gen_inputs)} generated test cases passed! Setting environment to done state.")
                
            except Exception as e:
                print(f"Warning: Failed to evaluate generated test against code: {e}")
                self.state.generated_test_vs_generated_code_match_cases = []
                self.state.generated_test_vs_generated_code_mismatch_cases = []
                self.state.generated_test_vs_generated_code_match_ratio = 0.0



class CodeTestEnvBatch:
    def __init__(self, env_idx_list: List[int], rollout_idx_list: List[int], samples: int, max_turns: int, config: dict, mode="train"):
        
        self.problem_list=load_problem_batch(config.env.benchmark, len(env_idx_list),mode=mode)
        self.env_list=[]
        if mode=="validation":
            rollout_idx_list=range(len(self.problem_list)*samples)
   
        if not self.problem_list:
            raise ValueError(f"Failed to load problems from benchmark: {config.env.benchmark}. Please check if the dataset is available and accessible.")
    
        
        for i,problem in enumerate(self.problem_list):
            state=CodeTestEnvState(problem=problem["question"],ground_truth_test_input=problem["example_input"],ground_truth_test_output=problem["example_output"])
            for s in range(samples):
                env=CodeTestEnv(env_idx=i, rollout_idx=rollout_idx_list[i*samples+s], max_turns=max_turns, config=None)
                env.state=copy.deepcopy(state)
                self.env_list.append(env)
        if len(self.env_list)!=len(rollout_idx_list):
            raise ValueError(f"len(self.env_list)!=len(rollout_idx_list), {len(self.env_list)}!={len(rollout_idx_list)}")