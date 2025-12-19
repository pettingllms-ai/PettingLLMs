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
from pettingllms.multi_agent_env.base.env import Env
from pettingllms.multi_agent_env.code.code_utils import (
        load_problem_batch,
        extract_code_from_response,
        evaluate_code_against_tests,
    )
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

@dataclass
class CodeEnvState:
    problem: str=None
    golden_code: str=None
    generated_code_history: List[str]=field(default_factory=list)
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
    ground_truth_test_vs_generated_code_match_ratio: float=0
    generated_test_vs_generated_code_match_cases: List[Dict]=None
    generated_test_vs_generated_code_mismatch_cases: List[Dict]=None
    generated_test_vs_generated_code_mismatch_cases_history: List[Dict]=field(default_factory=list)
    generated_test_vs_generated_code_match_ratio: float=0
    generated_test_vs_golden_code_match_cases: List[Dict]=None
    generated_test_vs_golden_code_mismatch_cases: List[Dict]=None
    generated_test_vs_golden_code_match_ratio: float=0

class CodeEnv(Env):
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
        self.state=CodeEnvState()
        
        self.backend = "ray_docker" 


   

    def reset(self):

        self.state.generated_code=None
        self.state.generated_code_history=[]
        self.state.generated_test_input=None
        self.state.generated_test_output=None
        self.state.exe_code_generated_test_output=None
        self.state.exe_code_ground_truth_test_output=None
        self.state.ground_truth_test_vs_generated_code_mismatch_cases=None
        self.state.ground_truth_test_vs_generated_code_match_cases=None
        self.state.ground_truth_test_vs_generated_code_match_ratio=0
        self.state.generated_test_vs_generated_code_match_cases=None
        self.state.generated_test_vs_generated_code_mismatch_cases=None
        self.state.generated_test_vs_generated_code_mismatch_cases_history=[]
        self.state.generated_test_vs_generated_code_match_ratio=0
        self.state.generated_test_vs_golden_code_match_cases=None
        self.state.generated_test_vs_golden_code_mismatch_cases=None
        self.state.generated_test_vs_golden_code_match_ratio=0


class CodeEnvBatch:
    def __init__(self, env_idx_list: List[int], env_indices: List[int], rollout_idx_list: List[int], samples: int, max_turns: int, config: dict, mode="train", *, env_workers: List=None):
        # Convert env_indices to list for safety
        safe_env_indices = list(env_indices) if not isinstance(env_indices, list) else env_indices
        
        if mode=="train":
            dataset_name = getattr(config.env, "dataset", "apps") if hasattr(config, "env") and hasattr(config.env, "dataset") else "apps"
            self.problem_list=load_problem_batch(safe_env_indices, dataset_name=dataset_name, mode="train")
        else:
            benchmark_name=getattr(config.env,"benchmark") if hasattr(config,"env") and hasattr(config.env,"benchmark") else "test"
            #difficulty=getattr(config,"difficulty") if hasattr(config,"difficulty") else "difficult"
            self.problem_list=load_problem_batch(safe_env_indices,mode=mode,benchmark_name=benchmark_name)
            samples=1
        self.env_list=[]
        if mode=="validate":
            rollout_idx_list=range(len(self.problem_list)*samples)
   

        for i,problem in enumerate(self.problem_list):
            ground_truth_test_input=problem["test_input"]
            ground_truth_test_output=problem["test_output"]
            state=CodeEnvState(problem=problem["question"],ground_truth_test_input=ground_truth_test_input,ground_truth_test_output=ground_truth_test_output)
            for s in range(samples):
                env=CodeEnv(env_idx=i, rollout_idx=rollout_idx_list[i*samples+s], max_turns=max_turns, config=None)
                env.state=copy.deepcopy(state)
                self.env_list.append(env)
        if len(self.env_list)!=len(rollout_idx_list):
            raise ValueError(f"len(self.env_list)!=len(rollout_idx_list), {len(self.env_list)}!={len(rollout_idx_list)}")