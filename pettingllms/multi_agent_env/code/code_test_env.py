import logging
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
        parse_test_cases_from_response,
        evaluate_code_against_tests,
        evaluate_tests_against_golden_code,
    )
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class CodeTestEnvState:
    problem: str=None
    current_code: str=None
    current_test_input: str=None
    current_code_output: str=None
    current_test_output: str=None
    golden_code: str=None
    golden_test_input: Any=None
    golden_test_output: Any=None
    # Evaluation results: consider three combinations
    # 1) generated test vs generated code
    generated_test_vs_generated_code_mismatch_cases: List[Dict]=None
    generated_test_vs_generated_code_match_cases: List[Dict]=None
    generated_test_vs_generated_code_mismatch_ratio: float=None
    generated_test_vs_generated_code_match_ratio: float=None

    # 2) golden test vs generated code
    golden_test_vs_generated_code_mismatch_cases: List[Dict]=None
    golden_test_vs_generated_code_match_cases: List[Dict]=None
    golden_test_vs_generated_code_mismatch_ratio: float=None
    golden_test_vs_generated_code_match_ratio: float=None

    # 3) generated test vs golden code
    generated_test_vs_golden_code_mismatch_cases: List[Dict]=None
    generated_test_vs_golden_code_match_cases: List[Dict]=None
    generated_test_vs_golden_code_mismatch_ratio: float=None
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


   
    def step(self, role: str, action: str):
        if role == "code_agent":
            self._code_step(action)
        elif role == "test_agent":
            self._test_step(action)
        else:
            raise ValueError(f"Invalid role: {role}")

        pass

    def _code_step(self, action: str):
        """
        the action is the generated code, you should execute the code with the golden test cases and the generated test cases, and get the output, and then update the state
        """
        # 1) Update generated code
        generated_code = extract_code_from_response(action) if isinstance(action, str) else str(action)
        self.state.current_code = generated_code

        # 2) Evaluate with golden test cases: golden test vs generated code
        golden_inputs = self.state.golden_test_input or []
        golden_outputs = self.state.golden_test_output or []

        if golden_inputs and golden_outputs:
            _, detailed_info = evaluate_code_against_tests(
                generated_code, golden_inputs, golden_outputs, timeout=1.0
            )

            passed_cases = detailed_info.get("passed_cases", [])
            failed_cases = detailed_info.get("failed_cases", [])

            # Normalize to a list of dictionaries
            def _to_dict_list(cases):
                result = []
                for c in cases:
                    if hasattr(c, "to_dict"):
                        result.append(c.to_dict())
                    elif isinstance(c, dict):
                        result.append(c)
                    else:
                        # Best-effort serialization
                        result.append({"repr": repr(c)})
                return result

            passed_cases_dicts = _to_dict_list(passed_cases)
            failed_cases_dicts = _to_dict_list(failed_cases)

            total = max(1, len(passed_cases_dicts) + len(failed_cases_dicts))

            self.state.golden_test_vs_generated_code_match_cases = passed_cases_dicts
            self.state.golden_test_vs_generated_code_mismatch_cases = failed_cases_dicts
            self.state.golden_test_vs_generated_code_match_ratio = len(passed_cases_dicts) / total
            self.state.golden_test_vs_generated_code_mismatch_ratio = len(failed_cases_dicts) / total

        # 3) Evaluate generated test vs generated code (if exists)
        #    Allow reading from state.current_test_input/current_test_output
        gen_inputs = self.state.current_test_input or []
        gen_outputs = self.state.current_test_output or []
        if isinstance(gen_inputs, list) and isinstance(gen_outputs, list) and gen_inputs and gen_outputs:
            _, detailed_info_gen = evaluate_code_against_tests(
                generated_code, gen_inputs, gen_outputs, timeout=1.0
            )

            passed_cases = detailed_info_gen.get("passed_cases", [])
            failed_cases = detailed_info_gen.get("failed_cases", [])

            def _to_dict_list2(cases):
                result = []
                for c in cases:
                    if hasattr(c, "to_dict"):
                        result.append(c.to_dict())
                    elif isinstance(c, dict):
                        result.append(c)
                    else:
                        result.append({"repr": repr(c)})
                return result

            passed_cases_dicts = _to_dict_list2(passed_cases)
            failed_cases_dicts = _to_dict_list2(failed_cases)
            total = max(1, len(passed_cases_dicts) + len(failed_cases_dicts))

            self.state.generated_test_vs_generated_code_match_cases = passed_cases_dicts
            self.state.generated_test_vs_generated_code_mismatch_cases = failed_cases_dicts
            self.state.generated_test_vs_generated_code_match_ratio = len(passed_cases_dicts) / total
            self.state.generated_test_vs_generated_code_mismatch_ratio = len(failed_cases_dicts) / total

    def _test_step(self, action: str):
        """
        the action is the generated test cases, you should execute the test cases with the generated code  and the golden code and get the output, and then update the state
        """
        # 1) Parse and update generated test cases
        test_cases = parse_test_cases_from_response(action) if isinstance(action, str) else []
        # Split parsed test cases into inputs/outputs and store in current state fields
        gen_inputs = [tc.get("input", "") for tc in test_cases]
        gen_outputs = [tc.get("output", "") for tc in test_cases]
        self.state.current_test_input = gen_inputs
        self.state.current_test_output = gen_outputs

        # 2) Evaluate generated test vs golden code
        if test_cases and self.state.golden_code:
            _, detailed_info_golden = evaluate_tests_against_golden_code(
                test_cases, self.state.golden_code, timeout=1.0
            )

            passed_cases = detailed_info_golden.get("passed_cases", [])
            failed_cases = detailed_info_golden.get("failed_cases", [])

            def _to_dict_list(cases):
                result = []
                for c in cases:
                    if hasattr(c, "to_dict"):
                        result.append(c.to_dict())
                    elif isinstance(c, dict):
                        result.append(c)
                    else:
                        result.append({"repr": repr(c)})
                return result

            passed_cases_dicts = _to_dict_list(passed_cases)
            failed_cases_dicts = _to_dict_list(failed_cases)
            total = max(1, len(passed_cases_dicts) + len(failed_cases_dicts))

            self.state.generated_test_vs_golden_code_match_cases = passed_cases_dicts
            self.state.generated_test_vs_golden_code_mismatch_cases = failed_cases_dicts
            self.state.generated_test_vs_golden_code_match_ratio = len(passed_cases_dicts) / total
            self.state.generated_test_vs_golden_code_mismatch_ratio = len(failed_cases_dicts) / total

        # 3) Evaluate generated test vs generated code (if generated code exists)
        if test_cases and getattr(self.state, "current_code", None):
            _, detailed_info_codegen = evaluate_code_against_tests(
                self.state.current_code, gen_inputs, gen_outputs, timeout=1.0
            )

            passed_cases = detailed_info_codegen.get("passed_cases", [])
            failed_cases = detailed_info_codegen.get("failed_cases", [])

            def _to_dict_list2(cases):
                result = []
                for c in cases:
                    if hasattr(c, "to_dict"):
                        result.append(c.to_dict())
                    elif isinstance(c, dict):
                        result.append(c)
                    else:
                        result.append({"repr": repr(c)})
                return result

            passed_cases_dicts = _to_dict_list2(passed_cases)
            failed_cases_dicts = _to_dict_list2(failed_cases)
            total = max(1, len(passed_cases_dicts) + len(failed_cases_dicts))

            self.state.generated_test_vs_generated_code_match_cases = passed_cases_dicts
            self.state.generated_test_vs_generated_code_mismatch_cases = failed_cases_dicts
            self.state.generated_test_vs_generated_code_match_ratio = len(passed_cases_dicts) / total
            self.state.generated_test_vs_generated_code_mismatch_ratio = len(failed_cases_dicts) / total

class CodeTestEnvBatch:
    def __init__(self, env_idx_list: List[int], rollout_idx_list: List[int], max_turns: int, config: dict):
        self.env_list=[]
        self.problem_list=load_problem_batch(config.env.benchmark, len(env_idx_list))
        for i,problem in enumerate(self.problem_list):
            state=CodeTestEnvState(problem=problem["problem"], golden_code=problem["golden_code"], golden_test_input=problem["golden_test_input"], golden_test_output=problem["golden_test_output"])
            env=CodeTestEnv(env_idx=env_idx_list[i], rollout_idx=rollout_idx_list[i], max_turns=max_turns, config=None)
            env.state=state
            self.env_list.append(env)