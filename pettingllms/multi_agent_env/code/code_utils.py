"""
Utility functions for code generation and testing.

This module contains utilities for code execution, validation, data loading,
and metric computation. It references the eval part of the CURE-main project
and supports streaming data loading.
"""

import os
import sys
import multiprocessing as mp
import re
import random
import asyncio
import concurrent.futures
import subprocess
import tempfile
import shutil
import contextlib
import textwrap
import traceback as _traceback
import errno
import signal
from typing import Any
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List, Union
from tqdm import tqdm
import numpy as np

# Import worker utilities
from .code_worker import (
    _worker_docker,
    _await_ray_object_ref,
    test_if_eq,
    get_ray_docker_worker_cls,
)

try:
    from datasets import load_dataset as hf_load_dataset
    DATASETS_AVAILABLE = True
except ImportError:
    print("Warning: The 'datasets' library is unavailable; some features are limited")
    DATASETS_AVAILABLE = False

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    print("Warning: The 'pandas' library is unavailable; some features are limited")
    PANDAS_AVAILABLE = False



def load_problem_batch( 
    indices: List[int],
    dataset_name: str="apps",
    benchmark_name: str="livecodebench",
    mode: str = "train"
) -> List[Dict[str, Any]]:
    """
    Load a batch of programming problems.
    
    Args:
        indices: List of indices to load
        dataset_name: Dataset name for training (e.g., "code_contests", "apps")
        benchmark_name: Benchmark name for validation (e.g., "livecodebench", "code_contests", "apps")
        mode: "train" or "validate"
        
    Returns:
        A list of dicts with keys question/test_input/test_output/solution
    """
    
    current_dir = Path(__file__).parent.parent.parent.parent
    local_datasets_dir = current_dir / "data" / "code"
    
    if mode == "train":
        parquet_file = local_datasets_dir / "train" / f"{dataset_name}.parquet"
    else:
        parquet_file = local_datasets_dir / "test" / f"{benchmark_name}.parquet"
    
    if not parquet_file.exists():
        raise FileNotFoundError(
            f"Parquet file not found: {parquet_file}. "
            f"Please run scripts/dataprocess/load_code.py to generate data."
        )
    
    print(f"Loading dataset from: {parquet_file}")
    try:
        ds = hf_load_dataset("parquet", data_files=str(parquet_file), split="train")
        print(f"Successfully loaded dataset with {len(ds)} samples")
    except Exception as e:
        raise Exception(f"Failed to load dataset: {e}")
    
    batch_results = []
    
    if mode == "train":
        # Ensure indices don't exceed dataset length using modulo operation
        safe_indices = [idx % len(ds) for idx in indices]
        for idx in safe_indices:
            example = ds[idx]
            problem_dict = _format_competition_problem(example, idx, mode="train")
            if problem_dict:
                batch_results.append(problem_dict)
    else:
        for i, example in enumerate(ds):
            problem_dict = _format_competition_problem(example, i, mode="validate")
            if problem_dict:
                batch_results.append(problem_dict)
                if i % 100 == 0:
                    print(f"Loaded validation problem {i+1}/{len(ds)}")
    
    print(f"Successfully loaded {len(batch_results)} problems")
    return batch_results




def _format_competition_problem(example: Dict, index: int, mode: str = "train") -> Optional[Dict]:
    """
    Format a competition problem example into a standardized dictionary.
    
    Args:
        example: Raw example from dataset
        index: Index of the example
        mode: "train" or "validate"
        
    Returns:
        Formatted problem dictionary or None if invalid
    """
    try:
        question = example.get("question", "")
        test_input = example.get("test_input", "")
        test_output = example.get("test_output", "")
        if mode == "train":
            if len(test_input) > 4:
                test_input = test_input[:4]
            if len(test_output) > 4:
                test_output = test_output[:4]
        if mode == "train":
            solution = example.get("solution", "")
        else:  # validation mode
            solution = example.get("solution", "")
        
        if not question or not test_input or not test_output:
            print(f"Warning: Skipping example {index}: missing required fields")
            return None
        
        return {
            "question": question,
            "test_input": test_input,
            "test_output": test_output,
            "solution": solution
        }
        
    except Exception as e:
        print(f"Warning: Error formatting example {index}: {e}")
        return None

# =================== Code execution and validation ===================


async def get_code_execution_output(
    code: str,
    input_val: str = "",
    timeout: float = 40.0,
    ray_actor: Any | None = None,
) -> str:
    """
    Execute Python code with input and return the output.
    Uses Ray worker for execution with proper timeout handling for concurrent rollouts.
    
    Args:
        code: Python code to execute
        input_val: Input data for the code
        timeout: Execution timeout
        ray_actor: Ray actor for code execution
        
    Returns:
        Code execution output as string
    """
    try:
        if ray_actor is None:
            raise ValueError("ray_actor is required")
        
    
        timeout_buffer = max(timeout * 2.0, 30.0)  
        total_timeout = timeout + timeout_buffer
        obj_ref = ray_actor.run.remote(code, input_val, "", timeout)  
        result_dict = await _await_ray_object_ref(obj_ref, total_timeout)
        
        if isinstance(result_dict, dict):
            execution_output = result_dict.get("code_execution_output", "")
        else:
            execution_output = str(result_dict)
            
        if isinstance(execution_output, str) and execution_output.startswith("error:"):
            print(f"Warning: Ray execution returned error: {execution_output}")
        else:
            print(f"Success: Ray execution successful, output length: {len(str(execution_output))} characters")
            
        return execution_output
        
    except asyncio.TimeoutError as e:
        error_msg = f"Ray execution timed out after {total_timeout}s"
        print(f"Error: {error_msg}")
        return f"error: {error_msg}"
    except Exception as e:
        error_msg = f"Ray execution failed: {e}"
        print(f"Error: {error_msg}")
        return f"error: {error_msg}"





async def evaluate_code_against_tests(
    code: str, 
    test_inputs: List[str], 
    test_outputs: List[str],
    timeout: float = 40.0,
    *,
    image: str = "python:3.11-slim",
    ray_actor: Any | None = None,
    rollout_idx: int | None = None,
) -> Tuple[float, List, List]:
    """
    Evaluate code against test cases and return detailed results.
    Uses async execution for improved performance.
    
    Args:
        code: Code to evaluate
        test_inputs: List of test inputs
        test_outputs: List of expected outputs
        timeout: Execution timeout
        
    Returns:
        (passed_ratio, passed_cases, failed_cases)
    """
    if not test_inputs or not test_outputs:
        return 0.0, [], []
    
    
    total_tests = len(test_inputs)
    results: List[Dict[str, Any]] = []

    actors = [ray_actor]

    obj_refs = []


    try:
        for i in range(total_tests):
            safe_rollout_idx = rollout_idx if rollout_idx is not None else 0
            actor = actors[safe_rollout_idx % len(actors)]
            obj_refs.append(
                actor.run.remote(code, test_inputs[i], test_outputs[i], timeout, image)
            )
            
        # Significantly increase timeout buffer to handle large-scale concurrency
        timeout_buffer = min(timeout * 1.5, 120.0)  # Maximum 120 seconds buffer
        total_timeout = timeout + timeout_buffer
        # print(f"Starting {total_tests} code test tasks, timeout: {total_timeout}s")
        
        async_tasks = [
            _await_ray_object_ref(obj_ref, total_timeout)
            for obj_ref in obj_refs
        ]        
        results_or_exc = await asyncio.gather(*async_tasks, return_exceptions=True)


        processed_results: List[Dict[str, Any]] = []
        for i, item in enumerate(results_or_exc):
            if isinstance(item, Exception):
                processed_results.append({
                    "test_input": test_inputs[i],
                    "code_execution_output": f"error: {item}",
                    "test_output": test_outputs[i],
                    "passed": False,
                })
                print(f"item code_execution_output: {item}")
            else:
                #print(f"item code_execution_output: {item.get('code_execution_output')}")
                processed_results.append(item)
        results = processed_results
        
        success_count = sum(1 for r in results if not str(r.get("code_execution_output", "")).startswith("error:"))
        error_count = len(results) - success_count
        
    except asyncio.TimeoutError as e:
        print(f"Ray execution timed out: {e}")
      
        results = [{
            "test_input": test_inputs[i] if i < len(test_inputs) else "",
            "code_execution_output": f"error: timeout - {e}",
            "test_output": test_outputs[i] if i < len(test_outputs) else "",
            "passed": False,
        } for i in range(len(test_inputs))]
    except Exception as e:
        print(f"Ray execution failed: {e}")
        # Other error cases: return error results
        results = [{
            "test_input": test_inputs[i] if i < len(test_inputs) else "",
            "code_execution_output": f"error: {e}",
            "test_output": test_outputs[i] if i < len(test_outputs) else "",
            "passed": False,
        } for i in range(len(test_inputs))]

  
    passed_tests = 0
    passed_cases: List[Dict[str, Any]] = []
    failed_cases: List[Dict[str, Any]] = []

    for i, result in enumerate(results):
        actual_output = result.get("code_execution_output")
        expected_output = result.get("test_output")
        if_passed = result.get("passed", False)
        test_case_info = {
            "test_input": test_inputs[i],
            "code_execution_output": actual_output,
            "generated_test_output": expected_output,
            "passed": if_passed,
        }

        if actual_output is None:
            if_passed = False
        elif isinstance(actual_output, str) and actual_output.startswith("error:"):
            if_passed = False
        else:
            if_passed = await test_if_eq(actual_output, str(expected_output))

        if if_passed:
            passed_tests += 1
            passed_cases.append(test_case_info)
        else:
            failed_cases.append(test_case_info)

    passed_ratio = passed_tests / total_tests if total_tests > 0 else 0.0
    return passed_ratio, passed_cases, failed_cases


# =================== Test case parsing ===================

def extract_test_cases(text: str):
    # unify line endings
    s = text.replace("\r\n", "\n").replace("\r", "\n")

    # support ``` or ```txt / ```python etc.
    input_blocks = re.findall(
        r"\*\*Test Input:\*\*\s*```(?:[a-zA-Z0-9_+\-]*\n)?(.*?)```",
        s, flags=re.DOTALL
    )
    output_blocks = re.findall(
        r"\*\*Test Output:\*\*\s*```(?:[a-zA-Z0-9_+\-]*\n)?(.*?)```",
        s, flags=re.DOTALL
    )

    # remove leading and trailing whitespace, but keep line endings in content
    test_input = [blk.strip() for blk in input_blocks]
    test_output = [blk.strip() for blk in output_blocks]

    # align length (prevent unequal length)
    n = min(len(test_input), len(test_output))
    test_input = test_input[:n]
    test_output = test_output[:n]

    test_action = {"input": test_input, "output": test_output}
    return test_action




def extract_code_from_response(response: str) -> str:
    """
    Extract code from agent response.
    
    Args:
        response: Agent response string
        
    Returns:
        Extracted code string
    """
    # Look for Python code block
    python_pattern = r'```python\s*(.*?)```'
    matches = re.findall(python_pattern, response, re.DOTALL)
    
    if matches:
        return matches[-1].strip()  # Return the last code block
    
    # Look for generic code block
    code_pattern = r'```\s*(.*?)```'
    matches = re.findall(code_pattern, response, re.DOTALL)
    
    if matches:
        return matches[-1].strip()
    
    # If no code block found, return entire response
    return response.strip()