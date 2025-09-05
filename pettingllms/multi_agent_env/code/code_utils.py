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
import subprocess
import tempfile
import shutil
import contextlib
import textwrap
import traceback as _traceback
import errno
import signal
from typing import Any
def _stdin_from_input_val_like_inproc(input_val: Any) -> str:
    """
    生成与 `_worker_inproc` 近似语义的 stdin：
    - 若输入为 list/tuple，则按行拼接
    - 统一换行符为 \n
    - 确保最后一行以换行结尾，避免 input() 读到 EOF
    不额外移除代码块/标记（与 inproc 保持一致）。
    """
    if isinstance(input_val, (list, tuple)):
        input_text = "\n".join([str(x).rstrip("\n") for x in input_val])
    else:
        input_text = str(input_val)
    input_text = input_text.replace("\r\n", "\n").replace("\r", "\n")
    if not input_text.endswith("\n"):
        input_text = input_text + "\n"
    return input_text

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
    indices: List[int],
    dataset_name: str="train",
    split: str = "train",
    mode: str = "train"
) -> List[Dict[str, Any]]:
    """
    Load a batch of programming problems.
    
    Args:
        batch_size: Batch size
        dataset_name: Dataset name (e.g., "deepmind/code_contests", "Gen-Verse/CodeContests")
        split: Dataset split ("train", "test", etc.)
        mode: "train" or "validate"
        
    Returns:
        A list of dicts with keys question/test_input/test_output/solution
    """
    if not DATASETS_AVAILABLE:
        print("❌ datasets library unavailable")
        return []
    
    if mode == "validate":
        print(f"🔄 Loading all problems from dataset {dataset_name} (split={split})...")
    else:
        print(f"🔄 Loading {len(indices)} problems from dataset {dataset_name}...")
    
    # 获取本地数据集路径
    current_dir = Path(__file__).parent.parent.parent.parent  # 回到 pettingllms 根目录
    local_datasets_dir = current_dir / "datasets" / "code" / dataset_name.lower().replace("/", "_")
    split_name = "train" if mode == "train" else "test"
    parquet_file = local_datasets_dir / f"{split_name}.parquet"
    if mode == "train":
        if not parquet_file.exists():
            raise FileNotFoundError(f"❌ Train mode requires local dataset at {parquet_file}, but file not found!")
        
        print(f"📁 Loading from local dataset: {local_datasets_dir}")
        try:
            ds = hf_load_dataset("parquet", data_files=str(parquet_file), split=split)
            print(f"✅ Successfully loaded local dataset with {len(ds)} samples")
        except Exception as e:
            raise Exception(f"❌ Failed to load local dataset: {e}")
        batch_results = []
        
        for i, idx in enumerate(indices):
            example = ds[idx]
            problem_dict = _format_competition_problem(example, idx, mode="train")
            if problem_dict:
                batch_results.append(problem_dict)
                
        return batch_results
    
    # validation mode: try local first, if not found, then download
    else:
        if not parquet_file.exists():
            raise FileNotFoundError(
                f"❌ Validation mode requires local test set {parquet_file}, not found! Please run scripts/dataprocess/load_train_code.py to generate data."
            )
        print(f"📁 Loading test set from local: {local_datasets_dir}")
        try:
            # parquet single file default split name is "train"
            ds = hf_load_dataset("parquet", data_files=str(parquet_file), split="train")
            print(f"✅ Test set loaded successfully, {len(ds)} samples")
        except Exception as e:
            raise Exception(f"❌ Failed to load local dataset: {e}")
        

        batch_results = []
        for i, example in enumerate(ds):
            problem_dict = _format_competition_problem(example, i, mode="validate")
            if problem_dict:
                batch_results.append(problem_dict)
                if i % 100 == 0:  # print progress every 100 samples
                    print(f"🔄 Loaded validation problem {i+1}/{len(ds)}")
        
        print(f"✅ Successfully returned {len(batch_results)} validation samples")
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
        if len(test_input)>4:
            test_input=test_input[:4]
        test_output = example.get("test_output", "")
        if len(test_output)>4:
            test_output=test_output[:4]
        if mode == "train":
            solution = example.get("solution", "")
        else:  # validation mode
            solution = ""  
        
        if not question or not test_input or not test_output:
            print(f"⚠️ Skipping example {index}: missing required fields")
            return None
        
        return {
            "question": question,
            "test_input": test_input,
            "test_output": test_output,
            "solution": solution
        }
        
    except Exception as e:
        print(f"⚠️ Error formatting example {index}: {e}")
        return None

# =================== Code execution and validation ===================


async def _worker_docker(
    script: str,
    input_val: str,
    expected_output: str,
    timeout: float = 40.0,
    image: str = "python:3.11-slim"
):
    tmpdir = tempfile.mkdtemp(prefix="pllm_exec_",dir="tmp")
    script_path = os.path.join(tmpdir, "script.py")
    def cleanup_tmpdir():
        if not os.path.exists(tmpdir):
            return
        try:
            shutil.rmtree(tmpdir, ignore_errors=False)
        except Exception:
            print(f"failed to remove tmpdir: {tmpdir}")
            return
    stdin_file = None
    stdout_file = None
    stderr_file = None
    printed_output = None
    
    try:
        with open(script_path, "w", encoding="utf-8") as f:
            f.write(script)

        stdin_text = _stdin_from_input_val_like_inproc(input_val)
        stdin_path = os.path.join(tmpdir, "stdin.txt")
        stdout_path = os.path.join(tmpdir, "stdout.txt")
        stderr_path = os.path.join(tmpdir, "stderr.txt")

        # pre-write stdin content, and redirect stdout/stderr to temporary files to avoid communication through pipes
        with open(stdin_path, "w", encoding="utf-8") as f_in:
            f_in.write(stdin_text)

        stdin_file = open(stdin_path, "rb")
        stdout_file = open(stdout_path, "wb")
        stderr_file = open(stderr_path, "wb")

        try:
            env = dict(os.environ)
            env.update({
                "PYTHONFAULTHANDLER": "1",
                "PYTHONUNBUFFERED": "1",
                "PYTHONWARNINGS": "default",
                "PYTHONTRACEMALLOC": "5",
                "PYTHONIOENCODING": "utf-8",
            })

            proc = await asyncio.create_subprocess_exec(
                sys.executable, "-X", "dev", "-W", "default", "-u", script_path,
                stdin=stdin_file,
                stdout=stdout_file,
                stderr=stderr_file,
                cwd=tmpdir,
                env=env,
                start_new_session=True,
            )

            try:
                await asyncio.wait_for(proc.wait(), timeout=timeout-2)
                rc = proc.returncode
            except asyncio.TimeoutError:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except Exception:
                    try:
                        proc.kill()
                    except Exception:
                        pass
                # 在超时时向 stderr 追加父进程侧的说明，便于诊断
                try:
                    with open(stderr_path, "ab") as f_err_append:
                        msg = f"[parent] Timeout after {timeout}s; process killed.\n".encode()
                        f_err_append.write(msg)
                except Exception:
                    pass
                try:
                    await proc.wait()
                except Exception:
                    pass
                rc = None
                printed_output = None
                print("printed_output: None (timeout)")
                try:
                    if proc.returncode is None:
                        os.killpg(proc.pid, signal.SIGKILL)
                except Exception:
                    try:
                        proc.kill()
                    except Exception:
                        pass
                try:
                    await proc.wait()
                except Exception:
                    pass
                rc = proc.returncode
            if printed_output is None and rc is None:
                pass
            elif rc is not None:
                try:
                    with open(stdout_path, "rb") as f_out:
                        out_bytes = f_out.read()
                except Exception:
                    out_bytes = b""
                try:
                    with open(stderr_path, "rb") as f_err:
                        err_bytes = f_err.read()
                except Exception:
                    err_bytes = b""

                if rc == 0:
                    printed_output = out_bytes.decode(errors="replace")
                else:
                    err_text = (err_bytes or b"").decode(errors="replace").strip()
                    out_text = (out_bytes or b"").decode(errors="replace").strip()
                    combined = err_text or out_text
                    if "Traceback (most recent call last):" in combined:
                        last_line = combined.strip().splitlines()[-1]
                        combined = last_line
                    printed_output = f"error: exit {rc}: {combined}"
        finally:
            for file_handle, file_name in [(stdin_file, "stdin"), (stdout_file, "stdout"), (stderr_file, "stderr")]:
                if file_handle is not None:
                    try:
                        if not file_handle.closed:
                            file_handle.close()
                    except Exception as e:
                        print(f"关闭 {file_name} 文件句柄失败: {e}")
                        
    except Exception as e:
        # top level fallback, keep the same behavior as the original implementation: convert exception to readable string
        printed_output = f"error: {e}"

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
        cleanup_tmpdir()

    if_passed = await test_if_eq(printed_output, str(expected_output)) if printed_output is not None else False
    
    result = {
        "test_input": input_val,
        "code_execution_output": printed_output,
        "test_output": expected_output,
        "passed": if_passed,
    }
    return result


_RAY_TASK_HANDLE = None  # 缓存 Ray 远程函数句柄


async def _await_ray_object_ref(obj_ref, timeout_seconds: float = 10.0):
    import ray
    import time
    
    start_time = time.time()
    while True:
        ready, _ = ray.wait([obj_ref], timeout=0.1)
        if ready:
            return ray.get(obj_ref)
        
        elapsed = time.time() - start_time
        if elapsed > timeout_seconds:
            raise asyncio.TimeoutError(f"Ray task timed out after {timeout_seconds}s")
        

        await asyncio.sleep(0.01)


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
            
        async_tasks = [
            _await_ray_object_ref(obj_ref, (timeout - 3.0)/total_tests)
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
    except Exception as e:
        print(f"Ray execution failed, falling back to docker: {e}")
        try:
            # 确保所有参数都是有效的
            if not isinstance(code, str):
                print(f"Warning: code parameter is not string: {type(code)}")
                code = str(code) if code is not None else ""
            
            if not isinstance(test_inputs, list):
                print(f"Warning: test_inputs parameter is not list: {type(test_inputs)}")
                test_inputs = [test_inputs] if test_inputs is not None else []
            
            if not isinstance(test_outputs, list):
                print(f"Warning: test_outputs parameter is not list: {type(test_outputs)}")
                test_outputs = [test_outputs] if test_outputs is not None else []
            
            # 确保列表长度一致
            total_tests = max(len(test_inputs), len(test_outputs))
            if len(test_inputs) < total_tests:
                test_inputs.extend([""] * (total_tests - len(test_inputs)))
            if len(test_outputs) < total_tests:
                test_outputs.extend([""] * (total_tests - len(test_outputs)))
            
            tasks = [
                asyncio.create_task(
                    _worker_docker(code, test_inputs[i], test_outputs[i], timeout, image)
                ) for i in range(total_tests)
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            processed_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    print(f"Docker worker {i} failed: {result}")
                    processed_results.append({
                        "test_input": test_inputs[i] if i < len(test_inputs) else "",
                        "code_execution_output": f"error: {result}",
                        "test_output": test_outputs[i] if i < len(test_outputs) else "",
                        "passed": False,
                    })
                else:
                    processed_results.append(result)
            
            results = processed_results
            
        except Exception as fallback_error:
            print(f"Fallback to docker also failed: {fallback_error}")
            # 最后的fallback：返回错误结果
            results = [{
                "test_input": test_inputs[i] if i < len(test_inputs) else "",
                "code_execution_output": f"error: fallback failed - {fallback_error}",
                "test_output": test_outputs[i] if i < len(test_outputs) else "",
                "passed": False,
            } for i in range(max(len(test_inputs), len(test_outputs), 1))]

  
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




def get_ray_docker_worker_cls():
    try:
        import ray  # type: ignore
    except Exception as e:
        print(f"Failed to import ray: {e}")
        return None

    # Check if we already have a cached class
    if hasattr(get_ray_docker_worker_cls, "_cls"):
        return getattr(get_ray_docker_worker_cls, "_cls")

    try:
        _max_conc = 800

        @ray.remote(num_cpus=0.1, max_concurrency=_max_conc)
        class _RayDockerWorker:
            def __init__(self, idx):
                if not isinstance(idx, (int, float)):
                    print(f"Warning: idx parameter is not numeric: {type(idx)}, converting to int")
                    try:
                        self.idx = int(idx) if idx is not None else 0
                    except (ValueError, TypeError):
                        self.idx = 0
                else:
                    self.idx = int(idx)

            def get_idx(self):
                return self.idx

            async def run(
                self,
                script: str,
                input_val: str,
                expected_output: str,
                timeout: float = 10.0,
                image: str = "python:3.11-slim",
            ) -> Dict[str, Any]:
                
                
                try:
                    return await _worker_docker(
                        script=script,
                        input_val=input_val,
                        expected_output=expected_output,
                        timeout=timeout,
                        image=image,
                    )
                except Exception as e:
                    print(f"RayDockerWorker.run failed: {e}")
                    return {
                        "code_execution_output": f"error: {e}",
                        "passed": False,
                        "error": str(e)
                    }

        RayDockerWorker = _RayDockerWorker
        setattr(get_ray_docker_worker_cls, "_cls", RayDockerWorker)
        return RayDockerWorker
        
    except Exception as e:
        print(f"Failed to create RayDockerWorker class: {e}")
        return None




# ============ RayDockerWorker 池管理 ============
_RAY_DOCKER_ACTOR_POOL: List[Any] | None = None




def modify(c):
    c = c.replace("plaintext\n", "")
    c = c.replace("\\n", "\n")
    if not c.endswith("\n"):
        c += "\n"
    return c
# ===================TODO: Test case parsing ===================
def extract_test_cases(text: str):
    """
    从包含多组 **Test Input:** / **Test Output:** 代码块的字符串中提取内容。
    返回形如 {"input": [..], "output": [..]} 的字典。
    """
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


# =================== Metric computation ===================



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


def test_load_problem(batch_size: int):
    # Get problems
    results= load_problem_batch(
        batch_size=batch_size,

    )
    for result in results:
        print("--------------------------------Here is the solution--------------------------------")
        print(result["solution"])
       

if __name__ == "__main__":
    for benchmark in ["CodeContests"]:
        print(f"test load {benchmark}")
        test_load_problem(5)