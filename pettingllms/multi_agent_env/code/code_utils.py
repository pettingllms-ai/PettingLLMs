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
    local_datasets_dir = current_dir / "datasets" / "code"
    
    if mode == "train":
        parquet_file = local_datasets_dir / "train" / f"{dataset_name}_train.parquet"
    else:
        parquet_file = local_datasets_dir / "test" / f"{benchmark_name}_test.parquet"
    
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
        for idx in indices:
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
        if len(test_input)>4:
            test_input=test_input[:4]
        test_output = example.get("test_output", "")
        if len(test_output)>4:
            test_output=test_output[:4]
        if mode == "train":
            solution = example.get("solution", "")
        else:  # validation mode
            solution = example.get("solution", "")
        
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
    # Ensure base tmp directory exists
    try:
        os.makedirs("tmp", exist_ok=True)
    except Exception:
        pass
    tmpdir = tempfile.mkdtemp(prefix="pllm_exec_",dir="tmp")
    script_path = os.path.join(tmpdir, "script.py")
    def cleanup_tmpdir():
        if not os.path.exists(tmpdir):
            return
        try:
            shutil.rmtree(tmpdir, ignore_errors=False)
        except Exception:
            try:
                subprocess.run(["rm", "-rf", tmpdir], timeout=5, capture_output=True)
            except Exception:
                pass
    stdin_file = None
    stdout_file = None
    stderr_file = None
    printed_output = None
    
    try:
        with open(script_path, "w", encoding="utf-8") as f:
            f.write(script)

        runner_path = os.path.join(tmpdir, "runner.py")
        runner_code = textwrap.dedent(
            """
            import sys, io, typing

            def _main():
                input_data = sys.stdin.read()
                input_lines = iter(input_data.splitlines())

                def fake_input(prompt: str = "") -> str:
                    try:
                        return next(input_lines)
                    except StopIteration:
                        raise EOFError("No more input")

                original_stdin = sys.stdin
                sys.stdin = io.StringIO(input_data)

                context = {
                    "__name__": "__main__",
                    "input": fake_input,
                    "List": typing.List,
                    "Tuple": typing.Tuple,
                    "Optional": typing.Optional,
                }

                try:
                    with open("script.py", "r", encoding="utf-8") as sf:
                        code_text = sf.read()
                    try:
                        exec(code_text, context)
                    except SystemExit:
                        # 与参考实现一致：捕获 SystemExit，仍然保留已打印输出
                        pass
                    except Exception as e:
                        # 统一错误格式
                        print(f"error: {e}")
                finally:
                    sys.stdin = original_stdin

            if __name__ == "__main__":
                _main()
            """
        )
        with open(runner_path, "w", encoding="utf-8") as f:
            f.write(runner_code)

        stdin_text = input_val
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
                sys.executable, "-X", "dev", "-W", "default", "-u", runner_path,
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
                        print(f"close {file_name} file handle failed: {e}")
                        
    except Exception as e:
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
            print(f"⚠️ Ray execution returned error: {execution_output}")
        else:
            print(f"✅ Ray execution successful, output length: {len(str(execution_output))} characters")
            
        return execution_output
        
    except asyncio.TimeoutError as e:
        error_msg = f"Ray execution timed out after {total_timeout}s"
        print(f"❌ {error_msg}")
        return f"error: {error_msg}"
    except Exception as e:
        error_msg = f"Ray execution failed: {e}"
        print(f"❌ {error_msg}")
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
            
        # 大幅增加超时缓冲时间以处理大规模并发
        timeout_buffer = min(timeout * 1.5, 120.0)  # 最多120秒缓冲
        total_timeout = timeout + timeout_buffer
        #print(f"🚀 开始执行 {total_tests} 个代码测试任务，超时时间: {total_timeout}s")
        
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
        #print(f"✅ Ray代码测试任务完成: {success_count} 成功, {error_count} 失败")
    except asyncio.TimeoutError as e:
        print(f"❌ Ray execution timed out: {e}")
        # 超时情况：返回超时错误结果
        results = [{
            "test_input": test_inputs[i] if i < len(test_inputs) else "",
            "code_execution_output": f"error: timeout - {e}",
            "test_output": test_outputs[i] if i < len(test_outputs) else "",
            "passed": False,
        } for i in range(len(test_inputs))]
    except Exception as e:
        print(f"❌ Ray execution failed: {e}")
        # 其他错误情况：返回错误结果
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
        _max_conc = 20

        # 优化配置：支持500个rollout，每个rollout可能有多个测试用例
        # 使用极少的CPU资源但支持大量并发
        @ray.remote(num_cpus=0.001, max_concurrency=10000)
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
                timeout: float = 40.0,  # 与外层函数保持一致
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


# =================== Test case parsing ===================

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