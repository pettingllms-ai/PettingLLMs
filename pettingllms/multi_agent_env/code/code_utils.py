"""
Utility functions for code generation and testing.

This module contains utilities for code execution, validation, data loading,
and metric computation. It references the eval part of the CURE-main project
and supports streaming data loading.
"""

import os
import sys
import json
import io
import time
import typing
import multiprocessing
import multiprocessing as mp
import re
import random
import asyncio
import concurrent.futures
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List, Union
from tqdm import tqdm
import numpy as np
import itertools
from dataclasses import dataclass
from huggingface_hub import hf_hub_download

@dataclass
class evaluate_result:
    """
    Dataclass for test results
    """
    test_case_id: int
    input: str
    expected_output: str
    actual_output: str
    passed: bool
    error_type: Optional[str] = None
    
    def to_dict(self) -> Dict:
        """Convert to dict format to keep backward compatibility"""
        return {
            "test_case_id": self.test_case_id,
            "input": self.input,
            "expected_output": self.expected_output,
            "actual_output": self.actual_output,
            "passed": self.passed,
            "error_type": self.error_type
        }

try:
    from datasets import load_dataset as hf_load_dataset
    DATASETS_AVAILABLE = True
except ImportError:
    print("⚠️ The 'datasets' library is unavailable; some features are limited")
    DATASETS_AVAILABLE = False

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    print("⚠️ The 'pandas' library is unavailable; some features are limited")
    PANDAS_AVAILABLE = False


def load_problem_batch(
    dataset_name: str,
    batch_size: int,
    split: str = "train",
    mode: str = "train"
) -> List[Dict[str, Any]]:
    """
    Load a batch of programming problems.
    
    Args:
        dataset_name: Dataset name (e.g., "deepmind/code_contests", "Gen-Verse/CodeContests")
        batch_size: Batch size
        split: Dataset split ("train", "test", etc.)
        
    Returns:
        A list of dicts of length batch_size with keys problem/test_input/test_output
    """
    if not DATASETS_AVAILABLE:
        print("❌ datasets library unavailable")
        return []
    
    
    if dataset_name == "CodeContests_train":
        default_split = "train"
    else:
        default_split = "test"
    
  
    if split == "train":
        split = default_split
    
    if mode == "validation":
        print(f"🔄 Loading all problems from dataset {dataset_name} (split={split})...")
    else:
        print(f"🔄 Loading {batch_size} problems from dataset {dataset_name}...")
    
    # 首先尝试从本地 datasets 文件夹加载
    local_dataset_path = None
    try:
        # 检查是否存在本地数据集
        current_dir = Path(__file__).parent.parent.parent.parent  # 回到 pettingllms 根目录
        local_datasets_dir = current_dir / "data" / "datasets" / dataset_name.lower().replace("/", "_")
        
        # 检查是否存在 parquet 文件
        parquet_file = local_datasets_dir / f"{split}.parquet"
        if parquet_file.exists():
            print(f"📁 Found local dataset at: {local_datasets_dir}")
            local_dataset_path = str(local_datasets_dir)
        else:
            print(f"📁 Local dataset not found at: {local_datasets_dir}")
    except Exception as e:
        print(f"⚠️ Error checking local dataset: {e}")
    
    # 如果本地存在数据集，优先使用本地数据
    if local_dataset_path:
        try:
            print(f"🔄 Loading from local dataset: {local_dataset_path}")
            ds = hf_load_dataset("parquet", data_files=f"{local_dataset_path}/{split}.parquet", split=split)
            print(f"✅ Successfully loaded local dataset with {len(ds)} samples")
        except Exception as e:
            print(f"❌ Error loading local dataset: {e}")
            local_dataset_path = None
    
    # 如果本地加载失败或不存在，则从 Hugging Face 加载
    if not local_dataset_path:
        hf_dataset_name = f"Gen-Verse/{dataset_name}"
        print(f"🌐 Loading from Hugging Face: {hf_dataset_name}")
        
        try:
            # Use streaming=True to avoid downloading the entire dataset
            ds = hf_load_dataset(hf_dataset_name, split=split, streaming=True)
        except Exception as e:
            print(f"❌ Error loading dataset with streaming: {e}")
            print("💡 Trying fallback method without streaming...")
            try:
                # Fallback: load traditional way
                ds = hf_load_dataset(hf_dataset_name, split=split)
            except Exception as e2:
                print(f"❌ Failed to load dataset: {e2}")
                return []
    
    batch_results = []
    
    if mode == "validation":
        iterator = ds
       
    else:
        try:
          
            iterator = ds.take(batch_size * 2)
        except AttributeError:
          
            iterator = itertools.islice(ds, batch_size * 2)
    
    for i, example in enumerate(iterator):
        if mode != "validation":
            if len(batch_results) >= batch_size:
                break
            
        problem_dict = _format_competition_problem(example, i)
        if problem_dict:
            batch_results.append(problem_dict)
            print(f"✅ Loaded problem {len(batch_results)} (index={i})")
    
    if not batch_results:
        raise Exception("No valid problems found in streaming mode")
        
    print(f"✅ Successfully loaded {len(batch_results)} problems")
    return batch_results



def _format_competition_problem(example: Dict, index: int) -> Optional[Dict]:
    """
    Convert raw dataset format and extract problem data in the required format.
    
    Args:
        example: Raw dataset sample
        index: Sample index
        
    Returns:
        Dict with keys: question, example_input, example_output, test_input, test_output
        or None if extraction fails
    """
    # Check if example is a dictionary
    if not isinstance(example, dict):
        raise TypeError(f"Expected dict but got {type(example)} for problem {index}")
    
    # First, format the problem
    question = _extract_problem(example)
    if not question:
        raise ValueError(f"Failed to extract problem from example {index}")
    
    # Extract example inputs/outputs
    example_input = example.get("example_input", [])
    example_output = example.get("example_output", [])
    
    # Extract test inputs/outputs
    test_input = example.get("test_input", [])
    if not test_input:
        print(f"Warning: No test_input found in example {index}, using empty list")
        test_input = []
    test_output = example.get("test_output", [])
    if not test_output:
        print(f"Warning: No test_output found in example {index}, using empty list")
        test_output = []
    
    # Return in the required format
    return {
        "question": question,
        "example_input": example_input,
        "example_output": example_output,
        "test_input": test_input,
        "test_output": test_output
    }

def _extract_problem(example: Dict) -> Optional[str]:
    """
    Extract problem from raw dataset example.
    
    Args:
        example: Raw dataset sample
        
    Returns:
        problem string or None
    """
    # Try various field names
    code_fields = ["problem", "description", "question", "prompt"]
    
    for field in code_fields:
        if field in example and example[field]:
            code = example[field]
            if isinstance(code, str):
                return code
            elif isinstance(code, list) and len(code) > 0:
                return code[0] if isinstance(code[0], str) else str(code[0])
    
    # If not found, return None
    return None


# =================== Code execution and validation ===================

async def worker(script, input_val, timeout: float = 10.0):
    """
    Worker function for executing code in a separate process.
    Based on the reference worker function provided.
    """
    # Create an iterator over the input lines.
    input_lines = iter(input_val.splitlines())

    # Override the input() function in the exec context.
    def fake_input(prompt=""):
        try:
            return next(input_lines)
        except StopIteration:
            raise EOFError("No more input")

    # Redirect sys.stdout to capture printed output.
    stdout_capture = io.StringIO()
    original_stdout = sys.stdout
    original_stdin = sys.stdin  # Save original stdin
    sys.stdout = stdout_capture
    sys.stdin = io.StringIO(input_val)  # Simulate stdin with input_val

    context = {
        "__name__": "__main__",   # Ensures that `if __name__ == "__main__": ...` will fire
        "input": fake_input,
        "List": typing.List,
        "Tuple": typing.Tuple,
        "Optional": typing.Optional,
    }

    try:
        # Use asyncio.wait_for to implement timeout
        await asyncio.wait_for(
            asyncio.to_thread(exec, script, context),
            timeout=timeout
        )
        printed_output = stdout_capture.getvalue()

    except asyncio.TimeoutError:
        printed_output = None  # Return None for timeout
        
    except SystemExit:
        printed_output = stdout_capture.getvalue()
       
    except Exception as e:
        printed_output = f"error: {e}"

    finally:
        sys.stdout = original_stdout
        sys.stdin = original_stdin
        
    return printed_output




async def test_if_eq(x, y):
    """
    Test equality of two outputs ignoring whitespace differences.
    Based on the reference test_if_eq function provided.
    """
    return " ".join(x.split()) == " ".join(y.split())





async def evaluate_code_against_tests(
    code: str, 
    test_inputs: List[str], 
    test_outputs: List[str],
    timeout: float = 10.0,
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
    
    if len(test_inputs) != len(test_outputs):
        return 0.0, [], []
    
    total_tests = len(test_inputs)
    
    # Execute all test cases asynchronously
    tasks = [worker(code, test_input, timeout) for test_input in test_inputs]
    actual_outputs = await asyncio.gather(*tasks)
    
    # Process results
    passed_tests = 0
    passed_cases = []
    failed_cases = []
    
    for i, (test_input, expected_output, actual_output) in enumerate(zip(test_inputs, test_outputs, actual_outputs)):
        test_case_info = {"input": test_input,"generated_test_case_output":actual_output,"generated_code_execution_output":expected_output}
        
        if actual_output is None: # Check for timeout
            if_passed = False
            error_type = "timeout"
        elif actual_output.startswith("error:"): # Check for execution errors
            if_passed = False
            error_type = actual_output.replace("error: ", "")
        else:
            if_passed = await test_if_eq(actual_output, str(expected_output))
            error_type = None
        
        if if_passed:
            passed_tests += 1
            passed_cases.append(test_case_info)
        else:
            failed_cases.append(test_case_info)
        
        passed_ratio = passed_tests / total_tests
    
    return passed_ratio, passed_cases, failed_cases

def modify(c):
    c = c.replace("plaintext\n", "")
    c = c.replace("\\n", "\n")
    if not c.endswith("\n"):
        c += "\n"
    return c
# =================== Test case parsing ===================
def extract_test_cases(full_output):
    # First, try extracting with the updated triple-backtick pattern
    pattern_input_backticks = r'\*\*Test Input:\*\*\s*```(.*?)```'
    pattern_output_backticks = r'\*\*Test Output:\*\*\s*```(.*?)```'
    matches_input = re.findall(pattern_input_backticks, full_output, re.DOTALL)
    matches_output = re.findall(pattern_output_backticks, full_output, re.DOTALL)

    # For Test Input: either use the updated triple-backtick version or fallback to plain text
    if matches_input:
        test_input = [modify(matches_input[-1].lstrip('\n'))]
    else:
        # Fallback pattern without backticks: capture until **Test Output:**
        pattern_input_plain = r'\*\*Test Input:\*\*\s*([\s\S]*?)(?=\*\*Test Output:\*\*)'
        matches_input_plain = re.findall(pattern_input_plain, full_output, re.DOTALL)
        if matches_input_plain:
            test_input = [modify(matches_input_plain[-1].strip())]
        else:
            test_input = []
    
    # For Test Output: either use the updated triple-backtick version or fallback to plain text
    if matches_output:
        test_output = [modify(matches_output[-1].lstrip('\n'))]
    else:
        # Fallback: capture until the **Explanation:** marker or end-of-string
        pattern_output_plain = r'\*\*Test Output:\*\*\s*([\s\S]*?)(?=\*\*Explanation:|\*\*Test Input:|$)'
        matches_output_plain = re.findall(pattern_output_plain, full_output, re.DOTALL)
        if matches_output_plain:
            test_output = [modify(matches_output_plain[-1].strip())]
        else:
            test_output = []
    
    test_action= {"input": test_input, "output": test_output}
    
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


# =================== Metric computation ===================

def compute_pass_at_k_metrics(
    results: List[Dict], 
    k_values: List[int] = [1, 5, 10]
) -> Dict[str, float]:
    """
    Compute Pass@K metrics.
    
    Args:
        results: Evaluation results list
        k_values: List of K values
        
    Returns:
        Dict of Pass@K metrics
    """
    if not results:
        return {f"pass@{k}": 0.0 for k in k_values}
    
    # Compute pass status for each problem
    problem_results = {}
    for result in results:
        problem_id = result.get("task_id", result.get("problem_id", "unknown"))
        if problem_id not in problem_results:
            problem_results[problem_id] = []
        
        passed = result.get("success", False) or result.get("all_passed", False)
        problem_results[problem_id].append(passed)
    
    metrics = {}
    total_problems = len(problem_results)
    
    for k in k_values:
        passed_problems = 0
        for problem_id, passes in problem_results.items():
            # Take top-k results
            k_results = passes[:k]
            if any(k_results):  # At least one pass
                passed_problems += 1
        
        pass_rate = passed_problems / total_problems if total_problems > 0 else 0.0
        metrics[f"pass@{k}"] = pass_rate
    
    return metrics


def compute_basic_metrics(results: List[Dict]) -> Dict[str, Any]:
    """
    Compute basic evaluation metrics.
    
    Args:
        results: Evaluation results list
        
    Returns:
        Dict of basic metrics
    """
    if not results:
        return {
            "total_tasks": 0,
            "success_rate": 0.0,
            "average_iterations": 0.0,
            "average_test_pass_rate": 0.0
        }
    
    total_tasks = len(results)
    successful_tasks = sum(1 for r in results if r.get("success", False))
    
    # Compute average iterations
    iterations = [r.get("total_iterations", 0) for r in results]
    avg_iterations = sum(iterations) / len(iterations) if iterations else 0.0
    
    # Compute average test pass rate
    test_pass_rates = []
    for r in results:
        if "final_test_results" in r and "pass_rate" in r["final_test_results"]:
            test_pass_rates.append(r["final_test_results"]["pass_rate"])
        elif "code_evaluation" in r and "pass_rate" in r["code_evaluation"]:
            test_pass_rates.append(r["code_evaluation"]["pass_rate"])
    
    avg_test_pass_rate = sum(test_pass_rates) / len(test_pass_rates) if test_pass_rates else 0.0
    
    return {
        "total_tasks": total_tasks,
        "successful_tasks": successful_tasks,
        "success_rate": successful_tasks / total_tasks,
        "average_iterations": avg_iterations,
        "average_test_pass_rate": avg_test_pass_rate,
        "total_errors": total_tasks - successful_tasks
    }


def compute_error_analysis(results: List[Dict]) -> Dict[str, Any]:
    """
    Compute error analysis metrics.
    
    Args:
        results: Evaluation results list
        
    Returns:
        Error analysis dict
    """
    error_types = {
        "timeout_errors": 0,
        "execution_errors": 0,
        "output_mismatches": 0,
        "exceptions": 0,
        "no_solution": 0
    }
    
    termination_reasons = {}
    
    for result in results:
        # Analyze termination reasons
        reason = result.get("termination_reason", "unknown")
        termination_reasons[reason] = termination_reasons.get(reason, 0) + 1
        
        # Analyze error types
        if "iterations" in result:
            for iteration in result["iterations"]:
                if "code_execution_result" in iteration:
                    exec_result = iteration["code_execution_result"]
                    stats = exec_result.get("statistics", {})
                    
                    for error_type in error_types:
                        error_types[error_type] += stats.get(error_type, 0)
    
    return {
        "error_statistics": error_types,
        "termination_reasons": termination_reasons
    }


def compute_comprehensive_metrics(
    results: List[Dict], 
    k_values: List[int] = [1, 5, 10]
) -> Dict[str, Any]:
    """
    Compute comprehensive evaluation metrics.
    
    Args:
        results: Evaluation results list
        k_values: List of K values for Pass@K
        
    Returns:
        Comprehensive metrics dict
    """
    basic_metrics = compute_basic_metrics(results)
    pass_at_k_metrics = compute_pass_at_k_metrics(results, k_values)
    error_analysis = compute_error_analysis(results)
    
    return {
        **basic_metrics,
        **pass_at_k_metrics,
        **error_analysis,
        "evaluation_timestamp": time.time(),
        "num_evaluated_tasks": len(results)
    }


# =================== Helper functions ===================

def save_evaluation_results(
    results: Dict[str, Any], 
    output_path: str,
    pretty_print: bool = True
) -> None:
    """
    Save evaluation results to file.
    
    Args:
        results: Evaluation results dict
        output_path: Output file path
        pretty_print: Whether to pretty print JSON
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        if pretty_print:
            json.dump(results, f, indent=2, ensure_ascii=False)
        else:
            json.dump(results, f, ensure_ascii=False)
            
    print(f"💾 Evaluation results saved to: {output_path}")


def print_evaluation_summary(metrics: Dict[str, Any]) -> None:
    """
    Print evaluation summary.
    
    Args:
        metrics: Evaluation metrics dict
    """
    print(f"\n🎯 Evaluation Summary:")
    print(f"  📊 Total tasks: {metrics.get('total_tasks', 0)}")
    print(f"  ✅ Successful: {metrics.get('successful_tasks', 0)}")
    print(f"  📈 Success rate: {metrics.get('success_rate', 0):.2%}")
    print(f"  🔄 Avg iterations: {metrics.get('average_iterations', 0):.1f}")
    print(f"  🧪 Avg test pass rate: {metrics.get('average_test_pass_rate', 0):.2%}")
    
    # Print Pass@K metrics
    for k in [1, 5, 10]:
        if f"pass@{k}" in metrics:
            print(f"  📊 Pass@{k}: {metrics[f'pass@{k}']:.2%}")
    
    # Print error statistics
    if "error_statistics" in metrics:
        print(f"\n❌ Error statistics:")
        for error_type, count in metrics["error_statistics"].items():
            if count > 0:
                print(f"  {error_type}: {count}")


# =================== Main Evaluation Functions ===================

def evaluate_code_generation_task(
    code: str,
    problem: Dict,
    timeout: float = 1.0
) -> Dict[str, Any]:
    """
    Evaluate single code generation task
    
    Args:
        code: Generated code
        problem: Problem dictionary with keys: question, example_input, example_output, test_input, test_output
        timeout: Execution timeout
        
    Returns:
        Evaluation result dictionary
    """
    # Get test cases
    test_inputs = problem.get("test_input", [])
    test_outputs = problem.get("test_output", [])
    
    # If no test cases, use example test cases
    if not test_inputs or not test_outputs:
        test_inputs = problem.get("example_input", [])
        test_outputs = problem.get("example_output", [])
    
    if not test_inputs or not test_outputs:
        return {
            "success": False,
            "error": "No available test cases",
            "pass_rate": 0.0
        }
    
    # Execute evaluation
    reward, detailed_info = evaluate_code_against_tests(
        code, test_inputs, test_outputs, timeout
    )
    
    overall_result = detailed_info.get("overall_result", {})
    
    return {
        "success": overall_result.get("all_passed", False),
        "pass_rate": overall_result.get("pass_rate", 0.0),
        "passed_tests": overall_result.get("passed_tests", 0),
        "total_tests": overall_result.get("total_tests", 0),
        "reward": reward,
        "detailed_results": detailed_info,
        "execution_statistics": detailed_info.get("statistics", {})
    }


def test_load_problem(benchmark: str, batch_size: int):
    # Get problems
    results= load_problem_batch(
        dataset_name=benchmark,
        batch_size=batch_size
    )
       

if __name__ == "__main__":
    for benchmark in ["CodeContests_train"]:
        print(f"test load {benchmark}")
        test_load_problem(f"{benchmark}", 2)