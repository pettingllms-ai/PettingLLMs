"""
Mixed environment batch that combines math and code environments.

Loads both math and code datasets, creates a mixed list of MathEnv and CodeEnv
instances, each tagged with a `problem_type` attribute ("math" or "code") so
that the autoevol engine can select the correct reward function per env.
"""

import copy
import random
import logging
from typing import List

from pettingllms.multi_agent_env.math.math_env import MathEnv, MathEnvState
from pettingllms.multi_agent_env.math.math_utils import load_math_problem_batch
from pettingllms.multi_agent_env.code.code_env import CodeEnv, CodeEnvState
from pettingllms.multi_agent_env.code.code_utils import load_problem_batch

logger = logging.getLogger(__name__)


class MixedEnvBatch:
    """
    Environment batch that mixes math and code problems.

    Config fields used (under ``config.env``):
        - dataset_math: math training dataset name (default "polaris")
        - dataset_code: code training dataset name (default "code_contests")
        - benchmark_math: math validation benchmark (default "AIME24")
        - benchmark_code: code validation benchmark (default "code_contests")
        - math_ratio: fraction of batch devoted to math (default 0.5)
    """

    def __init__(
        self,
        env_idx_list: List[int],
        env_indices: List[int],
        rollout_idx_list: List[int],
        samples: int,
        max_turns: int,
        config,
        mode: str = "train",
        *,
        env_workers: List = None,
    ):
        safe_env_indices = list(env_indices) if not isinstance(env_indices, list) else env_indices

        # Read per-type config with sensible defaults
        env_cfg = config.env if hasattr(config, "env") else config
        dataset_math = getattr(env_cfg, "dataset_math", "polaris")
        dataset_code = getattr(env_cfg, "dataset_code", "code_contests")
        benchmark_math = getattr(env_cfg, "benchmark_math", "AIME24")
        benchmark_code = getattr(env_cfg, "benchmark_code", "code_contests")
        math_ratio = float(getattr(env_cfg, "math_ratio", 0.5))

        self.env_list: List = []

        if mode == "train":
            n_total = len(safe_env_indices)
            n_math = max(1, int(n_total * math_ratio))
            n_code = max(1, n_total - n_math)

            math_indices = safe_env_indices[:n_math]
            code_indices = safe_env_indices[n_math : n_math + n_code]

            math_problems = load_math_problem_batch(
                math_indices, mode="train", dataset_name=dataset_math, config=config,
            )
            code_problems = load_problem_batch(
                code_indices, dataset_name=dataset_code, mode="train",
            )

            # Build envs: math first, then code
            rollout_cursor = 0
            for i, problem in enumerate(math_problems):
                state = MathEnvState(
                    problem=problem["question"],
                    ground_truth_answer=problem["solution"],
                )
                for s in range(samples):
                    env = MathEnv(
                        env_idx=i,
                        rollout_idx=rollout_idx_list[rollout_cursor],
                        max_turns=max_turns,
                        config=None,
                    )
                    env.state = copy.deepcopy(state)
                    env.problem_type = "math"
                    self.env_list.append(env)
                    rollout_cursor += 1

            math_env_count = len(math_problems)
            for i, problem in enumerate(code_problems):
                state = CodeEnvState(
                    problem=problem["question"],
                    ground_truth_test_input=problem["test_input"],
                    ground_truth_test_output=problem["test_output"],
                )
                for s in range(samples):
                    env = CodeEnv(
                        env_idx=math_env_count + i,
                        rollout_idx=rollout_idx_list[rollout_cursor],
                        max_turns=max_turns,
                        config=None,
                    )
                    env.state = copy.deepcopy(state)
                    env.problem_type = "code"
                    self.env_list.append(env)
                    rollout_cursor += 1

            # Trim rollout_idx_list if we loaded fewer problems than requested
            if len(self.env_list) != len(rollout_idx_list):
                logger.warning(
                    f"MixedEnvBatch: env_list size {len(self.env_list)} != "
                    f"rollout_idx_list size {len(rollout_idx_list)}, adjusting."
                )

        else:  # validate
            math_problems = load_math_problem_batch(
                [], mode="validate", config=config, benchmark_name=benchmark_math,
            )
            code_problems = load_problem_batch(
                [], mode="validate", benchmark_name=benchmark_code,
            )

            all_problems = []
            for p in math_problems:
                all_problems.append(("math", p))
            for p in code_problems:
                all_problems.append(("code", p))

            total_envs = len(all_problems) * samples
            rollout_idx_list = list(range(total_envs))

            rollout_cursor = 0
            for i, (ptype, problem) in enumerate(all_problems):
                if ptype == "math":
                    state = MathEnvState(
                        problem=problem["question"],
                        ground_truth_answer=problem["solution"],
                    )
                    for s in range(samples):
                        env = MathEnv(
                            env_idx=i,
                            rollout_idx=rollout_idx_list[rollout_cursor],
                            max_turns=max_turns,
                            config=None,
                        )
                        env.state = copy.deepcopy(state)
                        env.problem_type = "math"
                        self.env_list.append(env)
                        rollout_cursor += 1
                else:
                    state = CodeEnvState(
                        problem=problem["question"],
                        ground_truth_test_input=problem["test_input"],
                        ground_truth_test_output=problem["test_output"],
                    )
                    for s in range(samples):
                        env = CodeEnv(
                            env_idx=i,
                            rollout_idx=rollout_idx_list[rollout_cursor],
                            max_turns=max_turns,
                            config=None,
                        )
                        env.state = copy.deepcopy(state)
                        env.problem_type = "code"
                        self.env_list.append(env)
                        rollout_cursor += 1

        if not self.env_list:
            raise ValueError(
                "MixedEnvBatch: no environments created. Check dataset paths."
            )

        logger.info(
            f"MixedEnvBatch created: {len(self.env_list)} envs "
            f"(math={sum(1 for e in self.env_list if e.problem_type == 'math')}, "
            f"code={sum(1 for e in self.env_list if e.problem_type == 'code')})"
        )
