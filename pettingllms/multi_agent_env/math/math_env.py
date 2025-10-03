import logging
import copy
from typing import Any, Dict, Optional, List
from dataclasses import dataclass

from pettingllms.multi_agent_env.base.env import Env
from pettingllms.multi_agent_env.math.math_utils import (
    load_math_problem_batch,
)

logger = logging.getLogger(__name__)

@dataclass
class MathEnvState:
    problem: str = None
    ground_truth_answer: str = None
    reasoning_generated_solution: str = None
    code_generated_solution: str = None
    reasoning_extracted_answer: str = None
    code_extracted_answer: str = None
    reasoning_is_correct: bool = False
    code_is_correct: bool = False
    code_reasoning_aligned: bool = False
class MathEnv(Env):
    """
    Environment for mathematical problem solving tasks with single-agent interaction.
    
    This environment handles mathematical problem solving where an agent generates
    step-by-step solutions and receives feedback based on answer correctness.
    """

    def __init__(
        self, 
        env_idx: int,
        rollout_idx: int,
        max_turns: int,
        config: dict | None = None,
    ):
        """
        Initialize the math test environment.
        """
        super().__init__(env_idx=env_idx, rollout_idx=rollout_idx, max_turns=max_turns, config=config)
        self.state = MathEnvState()

    def reset(self):
        self.state.reasoning_generated_solution = None
        self.state.code_generated_solution = None
        self.state.reasoning_extracted_answer = None
        self.state.code_extracted_answer = None
        self.state.reasoning_is_correct = None


class MathEnvBatch:
    def __init__(self, env_idx_list: List[int],env_indices: List[int], rollout_idx_list: List[int], samples: int, max_turns: int, config: dict, mode="train", *, env_workers: List = None):
        benchmark_name=getattr(config,"benchmark") if hasattr(config,"benchmark") else "MATH500"
        self.problem_list = load_math_problem_batch(env_indices, mode=mode, config=config,benchmark_name=benchmark_name)
        self.env_list = []
        if mode == "validate":
            rollout_idx_list = range(len(self.problem_list) * samples)
   
        if not self.problem_list:
            raise ValueError(f"Failed to load problems from math dataset. Please check if the dataset is available and accessible.")

        for i, problem in enumerate(self.problem_list):
            state = MathEnvState(
                problem=problem["question"],
                ground_truth_answer=problem["solution"],
            )
            for s in range(samples):
                env = MathEnv(env_idx=i, rollout_idx=rollout_idx_list[i*samples+s], max_turns=max_turns, config=None)
                env.state = copy.deepcopy(state)
                self.env_list.append(env)
                
        if len(self.env_list) != len(rollout_idx_list):
            raise ValueError(f"len(self.env_list)!=len(rollout_idx_list), {len(self.env_list)}!={len(rollout_idx_list)}")
