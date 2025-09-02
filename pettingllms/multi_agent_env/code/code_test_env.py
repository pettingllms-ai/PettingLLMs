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
    golden_code: str=None
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
    generated_test_vs_golden_code_match_cases: List[Dict]=None
    generated_test_vs_golden_code_mismatch_cases: List[Dict]=None
    generated_test_vs_golden_code_match_ratio: float=None

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
        
        self.backend = "ray_docker" 


   
    async def step(self, role: str, action: str,env_worker:Any=None):
        if role == "code_generator":
            await self._code_step(action,env_worker)
        elif role == "test_generator":
            await self._test_step(action,env_worker)
        else:
            raise ValueError(f"Invalid role: {role}")

        pass

    def reset(self):
        """
        Reset the environment to the initial state.
        class CodeTestEnvState:
    problem: str=None
    golden_code: str=None
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
    generated_test_vs_golden_code_match_cases: List[Dict]=None
    generated_test_vs_golden_code_mismatch_cases: List[Dict]=None
    generated_test_vs_golden_code_match_ratio: float=None
        """

        self.state.generated_code=None
        self.state.generated_test_input=None
        self.state.generated_test_output=None
        self.state.exe_code_generated_test_output=None
        self.state.exe_code_ground_truth_test_output=None
        self.state.ground_truth_test_vs_generated_code_mismatch_cases=None
        self.state.ground_truth_test_vs_generated_code_match_cases=None
        self.state.ground_truth_test_vs_generated_code_match_ratio=None
        self.state.generated_test_vs_generated_code_match_cases=None
        self.state.generated_test_vs_generated_code_mismatch_cases=None
        self.state.generated_test_vs_generated_code_match_ratio=None
        self.state.generated_test_vs_golden_code_match_cases=None
        self.state.generated_test_vs_golden_code_mismatch_cases=None
        self.state.generated_test_vs_golden_code_match_ratio=None


class CodeTestEnvBatch:
    def __init__(self, env_idx_list: List[int], env_indices: List[int], rollout_idx_list: List[int], samples: int, max_turns: int, config: dict, mode="train", *, env_workers: List=None):
        if mode=="train":
            self.problem_list=load_problem_batch(env_indices)
        else:
            self.problem_list=load_problem_batch(env_indices,mode=mode)
            samples=1
        self.env_list=[]
        if mode=="validate":
            rollout_idx_list=range(len(self.problem_list)*samples)
   
        if not self.problem_list:
            raise ValueError(f"Failed to load problems from benchmark: {config.env.benchmark}. Please check if the dataset is available and accessible.")

        for i,problem in enumerate(self.problem_list):
            ground_truth_test_input=problem["test_input"]
            ground_truth_test_output=problem["test_output"]
            state=CodeTestEnvState(problem=problem["question"],ground_truth_test_input=ground_truth_test_input,ground_truth_test_output=ground_truth_test_output)
            for s in range(samples):
                env=CodeTestEnv(env_idx=i, rollout_idx=rollout_idx_list[i*samples+s], max_turns=max_turns, config=None)
                env.state=copy.deepcopy(state)
                self.env_list.append(env)
        if len(self.env_list)!=len(rollout_idx_list):
            raise ValueError(f"len(self.env_list)!=len(rollout_idx_list), {len(self.env_list)}!={len(rollout_idx_list)}")