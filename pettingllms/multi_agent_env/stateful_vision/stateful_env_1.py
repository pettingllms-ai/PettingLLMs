import logging
import copy
from dataclasses import dataclass
from typing import Any, Dict, Optional, List, Tuple

from pettingllms.multi_agent_env.base.env import Env
from pettingllms.multi_agent_env.stateful.utils import (
    load_plan_path_problem_batch,
)
from pettingllms.multi_agent_env.stateful.env_state import (
    PlanPathGridEnvState, 
    EightQueensEnvState, 
    BlocksworldEnvState, 
    SudukuEnvState,
    get_state_class_by_benchmark
)

logger = logging.getLogger(__name__)


class StatefulEnv(Env):

    def __init__(
        self,
        env_idx: int,
        rollout_idx: int,
        max_turns: int,
        config: dict | None = None,
    ):
        super().__init__(env_idx=env_idx, rollout_idx=rollout_idx, max_turns=max_turns, config=config)
        self.state = None
        self.done = False  
    def reset(self):
        if self.state is not None:
            
            self.state.tool_action = []  # List[str]    
            self.state.tool_code = ""    # str 
            self.state.tool_execution_output = ""  # str 
            self.state.plan_action = []  # List[str] type
            
            
            if hasattr(self.state, 'reasoning_generated_plan'):
                self.state.reasoning_generated_plan = None
            if hasattr(self.state, 'code_generated_plan'):
                self.state.code_generated_plan = None
            if hasattr(self.state, 'reasoning_extracted_path'):
                self.state.reasoning_extracted_path = None
            if hasattr(self.state, 'code_extracted_path'):
                self.state.code_extracted_path = None
            if hasattr(self.state, 'reasoning_is_feasible'):
                self.state.reasoning_is_feasible = None
            if hasattr(self.state, 'code_is_feasible'):
                self.state.code_is_feasible = None
            if hasattr(self.state, 'reasoning_is_optimal'):
                self.state.reasoning_is_optimal = None
            if hasattr(self.state, 'code_is_optimal'):
                self.state.code_is_optimal = None
            if hasattr(self.state, 'code_reasoning_aligned'):
                self.state.code_reasoning_aligned = None



class StatefulEnvBatch:
  

    def __init__(
        self,
        env_idx_list: List[int],
        env_indices: List[int],
        rollout_idx_list: List[int],
        samples: int,
        max_turns: int,
        config: dict,
        mode: str = "train",
        *,
        env_workers: None = None,  # Optional: external worker
    ):
        
        
        self.env_list: List[StatefulEnv] = []

        if mode == "validate":
            env_idx_list = range(100)
            samples = samples
            rollout_idx_list = range(100*samples)
            

        benchmark_name = getattr(config.env, "benchmark") if hasattr(config, "env") and hasattr(config.env, "benchmark") else "plan_path"
        dataset_name = getattr(config.env, "dataset") if hasattr(config, "env") and hasattr(config.env, "dataset") else "default"
        
        # Load problem batch based on benchmark and dataset
        problem_list = load_plan_path_problem_batch(
            env_indices=env_indices,
            dataset_name=dataset_name,
            split=mode,
            mode=mode,
            config=config,
            benchmark_name=benchmark_name
        )
        
        # Seed configuration: train and test use different seed ranges
        # Train: 0-99999, Test/Validate: 100000-199999
        # This ensures train and test use different environments
        TRAIN_SEED_OFFSET = 0
        TEST_SEED_OFFSET = 100000
        
        for i in range(len(env_idx_list)):
            prob = problem_list[i] if i < len(problem_list) else problem_list[i % len(problem_list)]
            
            state_class = get_state_class_by_benchmark(benchmark_name)
            
            
            if benchmark_name == "plan_path":
                # Use env_idx (not env_indices) to ensure same seed for same environment across different training steps
                base_seed = env_indices[i] if i < len(env_indices) else i
                if mode == "train":
                    seed = TRAIN_SEED_OFFSET + base_seed
                else:  # test or validate
                    seed = TEST_SEED_OFFSET + base_seed
                state = state_class(seed=seed, config=config)
            elif benchmark_name == "sokoban":
                base_seed = env_indices[i] if i < len(env_indices) else i
                if mode == "train":
                    seed = TRAIN_SEED_OFFSET + base_seed
                else:  # test or validate
                    seed = TEST_SEED_OFFSET + base_seed
                state = state_class(seed=seed, config=config)
            elif benchmark_name == "EightQueens":
                state = state_class(N=prob["N"])
            elif benchmark_name == "Blocksworld":
                state = state_class(
                    init_stacks=prob["init_stacks"],
                    goal_stacks=prob["goal_stacks"]
                )
            elif benchmark_name == "suduku":
                base_seed = env_indices[i] if i < len(env_indices) else i
                if mode == "train":
                    seed = TRAIN_SEED_OFFSET + base_seed
                else:  # test or validate
                    seed = TEST_SEED_OFFSET + base_seed
                state = state_class(seed=seed, config=config)
            else:
                raise ValueError(f"Unsupported benchmark: {benchmark_name}")
            
            for s in range(samples):
                env = StatefulEnv(env_idx=i, rollout_idx=rollout_idx_list[i * samples + s], config=config, max_turns=max_turns)
                
                env.state = copy.deepcopy(state)
                self.env_list.append(env)

        if len(self.env_list) != len(rollout_idx_list):
            raise ValueError(
                f"len(self.env_list)!=len(rollout_idx_list), {len(self.env_list)}!={len(rollout_idx_list)}"
            )
