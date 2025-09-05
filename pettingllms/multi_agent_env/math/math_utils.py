"""
Utility functions for mathematical problem solving and evaluation.

This module contains utilities for loading math datasets, evaluating solutions,
and computing metrics for mathematical problem solving tasks.
"""

import os
import json
import random
import asyncio
import re
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List, Union
from dataclasses import dataclass
import ray
import shutil
import tempfile
import time

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



def extract_code(response: str) -> str:
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
    tasks = [
                asyncio.create_task(
                    _worker_docker(code, test_inputs[i], test_outputs[i], timeout, image)
                ) for i in range(total_tests)
            ]
    results = await asyncio.gather(*tasks)
  
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



def _ensure_ray_initialized() -> bool:
    from pettingllms.utils.logger_config import get_multi_logger
    multi_logger = get_multi_logger()
    import ray  

    if not ray.is_initialized():
        multi_logger.log_ray_status(mode="train", context="test_ray_log_function ")
       
        
        try:
            num_cpus_env = os.getenv("RAY_NUM_CPUS")
            multi_logger.log_ray_status(mode="train", context="before_code_utils_ray_init")
            init_kwargs = dict(
                ignore_reinit_error=True,
                include_dashboard=False,
                logging_level="ERROR",
            )
            if num_cpus_env:
                try:
                    num_cpus = float(num_cpus_env)
                    if num_cpus > 0:
                        init_kwargs["num_cpus"] = num_cpus
                    else:
                        print(f"Warning: RAY_NUM_CPUS must be positive, got {num_cpus_env}")
                except (ValueError, TypeError):
                    print(f"Warning: invalid RAY_NUM_CPUS value: {num_cpus_env}, using default")

            ray.init(**init_kwargs)

            try:
                cluster = ray.cluster_resources()
                avail = ray.available_resources()
                multi_logger.log_ray_status(
                    mode="train", context="after_code_utils_ray_init"
                )
            except Exception as e:
                print(f"Warning: failed to get ray cluster info: {e}")
                pass
        except Exception as e:
            print(f"Failed to initialize ray: {e}")
            multi_logger.log_ray_status(mode="train", context="code_utils_ray_init_failed")
            return False
    else:
        try:
            import ray  
            from pettingllms.utils.logger_config import get_multi_logger
            multi_logger = get_multi_logger()
            cluster = ray.cluster_resources()
            avail = ray.available_resources()
            
        except Exception as e:
            print(f"Warning: failed to get ray cluster info: {e}")
            pass

    return True







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





async def evaluate_code(
    code: str, 
    test_inputs: List[str], 
    test_outputs: List[str],
    timeout: float = 40.0,
    *,
    backend: str = "ray_docker",
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
    if backend == "ray_docker" and _ensure_ray_initialized():
        try:
            actors = [ray_actor]

            obj_refs = []
  

            actor_idx = ray.get(ray_actor.get_idx.remote())
            for i in range(total_tests):
                safe_rollout_idx = rollout_idx if rollout_idx is not None else 0
                actor = actors[safe_rollout_idx % len(actors)]
                obj_refs.append(
                    actor.run.remote(code, test_inputs[i], test_outputs[i], timeout, image)
                )
            
            async_tasks = [
                _await_ray_object_ref(obj_ref, timeout + 5.0)
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
                
                # 处理可能的异常结果
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
                
    
        # 非 ray 分支：使用 docker 后端
        tasks = [
            asyncio.create_task(
                _worker_docker(code, timeout, image)
            ) for i in range(total_tests)
        ]
        results = await asyncio.gather(*tasks)

  
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



def _ensure_ray_initialized() -> bool:
    from pettingllms.utils.logger_config import get_multi_logger
    multi_logger = get_multi_logger()
    import ray  

    if not ray.is_initialized():
        multi_logger.log_ray_status(mode="train", context="test_ray_log_function ")
       
        
        try:
            num_cpus_env = os.getenv("RAY_NUM_CPUS")
            multi_logger.log_ray_status(mode="train", context="before_code_utils_ray_init")
            init_kwargs = dict(
                ignore_reinit_error=True,
                include_dashboard=False,
                logging_level="ERROR",
            )
            if num_cpus_env:
                try:
                    num_cpus = float(num_cpus_env)
                    if num_cpus > 0:
                        init_kwargs["num_cpus"] = num_cpus
                    else:
                        print(f"Warning: RAY_NUM_CPUS must be positive, got {num_cpus_env}")
                except (ValueError, TypeError):
                    print(f"Warning: invalid RAY_NUM_CPUS value: {num_cpus_env}, using default")

            ray.init(**init_kwargs)

            try:
                cluster = ray.cluster_resources()
                avail = ray.available_resources()
                multi_logger.log_ray_status(
                    mode="train", context="after_code_utils_ray_init"
                )
            except Exception as e:
                print(f"Warning: failed to get ray cluster info: {e}")
                pass
        except Exception as e:
            print(f"Failed to initialize ray: {e}")
            multi_logger.log_ray_status(mode="train", context="code_utils_ray_init_failed")
            return False
    else:
        try:
            import ray  
            from pettingllms.utils.logger_config import get_multi_logger
            multi_logger = get_multi_logger()
            cluster = ray.cluster_resources()
            avail = ray.available_resources()
            
        except Exception as e:
            print(f"Warning: failed to get ray cluster info: {e}")
            pass

    return True












async def _worker_docker(
    script: str,
    timeout: float = 40.0,
    image: str = "python:3.11-slim"
) -> str:
    tmpdir = tempfile.mkdtemp(prefix="pllm_exec_",dir="tmp")
    script_path = os.path.join(tmpdir, "script.py")
    def cleanup_tmpdir():
        if not os.path.exists(tmpdir):
            return
        
        for attempt in range(3):
            try:
                shutil.rmtree(tmpdir, ignore_errors=False)
                print(f"成功删除临时目录: {tmpdir}")
                return
            except OSError as e:
                print(f"删除临时目录失败 (尝试 {attempt + 1}/3): {e}")
                if attempt < 2:
                    # 如果删除失败，尝试强制删除所有文件
                    try:
                        for root, dirs, files in os.walk(tmpdir):
                            for file in files:
                                file_path = os.path.join(root, file)
                                try:
                                    os.chmod(file_path, 0o777)
                                    os.remove(file_path)
                                except Exception:
                                    pass
                            for dir_name in dirs:
                                dir_path = os.path.join(root, dir_name)
                                try:
                                    os.chmod(dir_path, 0o777)
                                except Exception:
                                    pass
                        # 再次尝试删除目录
                        os.rmdir(tmpdir)
                        print(f"强制删除临时目录成功: {tmpdir}")
                        return
                    except Exception as force_e:
                        print(f"强制删除也失败: {force_e}")
                        time.sleep(0.1)  # 短暂等待后重试
                else:
                    # 最后一次尝试，使用 ignore_errors=True
                    shutil.rmtree(tmpdir, ignore_errors=True)
                    print(f"使用 ignore_errors 删除临时目录: {tmpdir}")
    
    stdout_file = None
    stderr_file = None
    printed_output = None
    
    try:
        with open(script_path, "w", encoding="utf-8") as f:
            f.write(script)


       
        stdout_path = os.path.join(tmpdir, "stdout.txt")
        stderr_path = os.path.join(tmpdir, "stderr.txt")
        stdout_file = open(stdout_path, "wb")
        stderr_file = open(stderr_path, "wb")

        try:
            proc = await asyncio.create_subprocess_exec(
                "python", script_path,
                stdout=stdout_file,
                stderr=stderr_file,
                cwd=tmpdir,
                start_new_session=True,
            )

            try:
                await asyncio.wait_for(proc.wait(), timeout=timeout-10)
                rc = proc.returncode
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
                try:
                    await proc.wait()
                except Exception:
                    pass
              
                rc = None
                printed_output = None
                print("printed_output: None (timeout)")

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
            for file_handle, file_name in [(stdout_file, "stdout"), (stderr_file, "stderr")]:
                if file_handle is not None:
                    try:
                        if not file_handle.closed:
                            file_handle.close()
                    except Exception as e:
                        print(f"关闭 {file_name} 文件句柄失败: {e}")
                        
    except Exception as e:
        # 顶层兜底，保持与原实现一致的行为：将异常转为可读字符串
        printed_output = f"error: {e}"
        print(f"_worker_docker 执行异常: {e}")

    return printed_output


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





async def get_code_execution_output(
    code: str, 
    timeout: float = 40.0,
    ray_actor: Any | None = None,
) -> str:
    """
    Execute Python code and return the output.
    Uses Ray worker for execution.
    
    Args:
        code: Python code to execute
        timeout: Execution timeout
        ray_actor: Ray actor for code execution
        
    Returns:
        Code execution output as string
    """
    try:
        if ray_actor is None:
            raise ValueError("ray_actor is required")
        
        # 使用 Ray actor 执行代码
        result = await ray_actor.run.remote(code, timeout)
        return result
        
    except Exception as e:
        print(f"Ray execution failed: {e}")
        return f"error: {e}"



def _ensure_ray_initialized() -> bool:
    from pettingllms.utils.logger_config import get_multi_logger
    multi_logger = get_multi_logger()
    import ray  

    if not ray.is_initialized():
        multi_logger.log_ray_status(mode="train", context="test_ray_log_function ")
       
        
        try:
            num_cpus_env = os.getenv("RAY_NUM_CPUS")
            multi_logger.log_ray_status(mode="train", context="before_code_utils_ray_init")
            init_kwargs = dict(
                ignore_reinit_error=True,
                include_dashboard=False,
                logging_level="ERROR",
            )
            if num_cpus_env:
                try:
                    num_cpus = float(num_cpus_env)
                    if num_cpus > 0:
                        init_kwargs["num_cpus"] = num_cpus
                    else:
                        print(f"Warning: RAY_NUM_CPUS must be positive, got {num_cpus_env}")
                except (ValueError, TypeError):
                    print(f"Warning: invalid RAY_NUM_CPUS value: {num_cpus_env}, using default")

            ray.init(**init_kwargs)

            try:
                cluster = ray.cluster_resources()
                avail = ray.available_resources()
                multi_logger.log_ray_status(
                    mode="train", context="after_code_utils_ray_init"
                )
            except Exception as e:
                print(f"Warning: failed to get ray cluster info: {e}")
                pass
        except Exception as e:
            print(f"Failed to initialize ray: {e}")
            multi_logger.log_ray_status(mode="train", context="code_utils_ray_init_failed")
            return False
    else:
        try:
            import ray  
            from pettingllms.utils.logger_config import get_multi_logger
            multi_logger = get_multi_logger()
            cluster = ray.cluster_resources()
            avail = ray.available_resources()
            
        except Exception as e:
            print(f"Warning: failed to get ray cluster info: {e}")
            pass

    return True




def get_ray_docker_worker_cls():
    try:
        import ray  # type: ignore
    except Exception as e:
        print(f"Failed to import ray: {e}")
        return None

    try:
        _ensure_ray_initialized()
    except Exception as e:
        print(f"Failed to ensure ray initialized: {e}")
        return None

    if hasattr(get_ray_docker_worker_cls, "_cls"):
        return getattr(get_ray_docker_worker_cls, "_cls")

    try:
        _max_conc_env = os.getenv("RAY_ACTOR_MAX_CONCURRENCY")
        try:
            _max_conc = int(_max_conc_env) if _max_conc_env else 8
        except (ValueError, TypeError):
            print(f"Warning: invalid RAY_ACTOR_MAX_CONCURRENCY value: {_max_conc_env}, using default 8")
            _max_conc = 8

        @ray.remote(num_cpus=0.02, max_concurrency=_max_conc)
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
                """获取 actor 的索引"""
                return self.idx

            async def run(
                self,
                script: str,
                timeout: float = 40.0,
                image: str = "python:3.11-slim",
            ) -> str:
                """
                Execute Python script using Docker and return output.
                
                Args:
                    script: Python script to execute
                    timeout: Execution timeout
                    image: Docker image to use
                    
                Returns:
                    Script execution output as string
                """
                try:
                    return await _worker_docker(
                        script=script,
                        timeout=timeout,
                        image=image,
                    )
                except Exception as e:
                    print(f"RayDockerWorker.run failed: {e}")
                    return f"error: {e}"

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
    # 统一换行
    s = text.replace("\r\n", "\n").replace("\r", "\n")

    # 支持 ``` 或 ```txt / ```python 等形式的代码块
    input_blocks = re.findall(
        r"\*\*Test Input:\*\*\s*```(?:[a-zA-Z0-9_+\-]*\n)?(.*?)```",
        s, flags=re.DOTALL
    )
    output_blocks = re.findall(
        r"\*\*Test Output:\*\*\s*```(?:[a-zA-Z0-9_+\-]*\n)?(.*?)```",
        s, flags=re.DOTALL
    )

    # 去掉首尾空白，但保留内容中的换行
    test_input = [blk.strip() for blk in input_blocks]
    test_output = [blk.strip() for blk in output_blocks]

    # 对齐长度（防止不等长）
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






















def load_math_problem_batch(
    batch_size: int = 10,
    dataset_name: str = "train",
    split: str = "train",
    mode: str = "train",
    config: dict = None
) -> List[Dict[str, Any]]:
    """
    Load a batch of mathematical problems.
    
    Args:
        batch_size: Batch size
        dataset_name: Dataset name (统一使用 "train")
        split: Dataset split (保留兼容性，但实际不使用)
        mode: "train" or "validate"
        config: Configuration dict
        
    Returns:
        A list of dicts with keys question/solution
    """
    if not DATASETS_AVAILABLE:
        print("❌ datasets library unavailable")
        return []
    
    # 期望的目录结构：datasets/math/train/{train.parquet,test.parquet}
    current_dir = Path(__file__).parent.parent.parent.parent  # 回到 pettingllms 根目录
    local_datasets_dir = current_dir / "datasets" / "math" / dataset_name.lower().replace("/", "_")
    split_name = "train" if mode == "train" else "test"
    parquet_file = local_datasets_dir / f"{split_name}.parquet"
    print(f"📄 目标文件: {parquet_file}")
    
    if mode == "train":
        if not parquet_file.exists():
            raise FileNotFoundError(f"❌ Train mode requires local dataset at {parquet_file}, but file not found!")
        
        print(f"📁 从本地加载数学训练集: {local_datasets_dir}")
        try:
            ds = hf_load_dataset("parquet", data_files=str(parquet_file), split="train")
            print(f"✅ 数学训练集加载成功，共 {len(ds)} 条")
        except Exception as e:
            raise Exception(f"❌ Failed to load local dataset: {e}")
        
        if len(ds) < batch_size:
            raise Exception(f"❌ Local dataset only has {len(ds)} samples, but batch_size is {batch_size}")
        
        indices = random.sample(range(len(ds)), batch_size)
        batch_results = []
        
        for i, idx in enumerate(indices):
            example = ds[idx]
            problem_dict = _format_math_problem(example, idx, mode="train")
            if problem_dict:
                batch_results.append(problem_dict)
                print(f"✅ Loaded math train problem {i+1}/{batch_size} (index={idx})")
        
        print(f"✅ 成功返回 {len(batch_results)} 条数学训练样本")
        return batch_results
    
    # validation mode: 加载测试集
    else:
        if not parquet_file.exists():
            raise FileNotFoundError(
                f"❌ 验证模式需要本地数学测试集 {parquet_file}，未找到！请先运行 scripts/dataprocess/load_train_math.py 生成数据。"
            )
        print(f"📁 从本地加载数学测试集: {local_datasets_dir}")
        try:
            # parquet 单文件默认 split 名称为 "train"
            ds = hf_load_dataset("parquet", data_files=str(parquet_file), split="train")
            print(f"✅ 数学测试集加载成功，共 {len(ds)} 条")
        except Exception as e:
            raise Exception(f"❌ Failed to load local dataset: {e}")
        
        # 加载所有验证数据
        batch_results = []
        for i, example in enumerate(ds):
            problem_dict = _format_math_problem(example, i, mode="validate")
            if problem_dict:
                batch_results.append(problem_dict)
                if i % 100 == 0:  # 每100个打印一次进度
                    print(f"🔄 Loaded math validation problem {i+1}/{len(ds)}")
        
        print(f"✅ 成功返回 {len(batch_results)} 条数学验证样本")
        return batch_results



def _format_math_problem(example: Dict, index: int, mode: str = "train") -> Optional[Dict]:
    """
    Format a math problem example into a standardized dictionary.
    
    Args:
        example: Raw example from dataset (期望格式: question/solution)
        index: Index of the example
        mode: "train" or "validate"
        
    Returns:
        Formatted problem dictionary or None if invalid
    """
    try:
        question = example.get("question", "")
        solution = example.get("solution", "")
        
        # 根据mode处理solution字段
        if mode == "train":
            # 训练模式：保留solution作为答案
            answer = solution
        else:  # validation mode
            # 验证模式：solution设为空（因为是测试）
            answer = ""
        
        # 验证必要字段
        if not question:
            print(f"⚠️ Skipping example {index}: missing question field")
            return None
        
        return {
            "question": question,
            "solution": answer  # 统一使用solution字段
        }
        
    except Exception as e:
        print(f"⚠️ Error formatting example {index}: {e}")
        return None



async def evaluate_math_solution(
    generated_solution: str,
    ground_truth_answer: str
) -> Tuple[bool, Optional[str]]:
    """
    Evaluate a mathematical solution against the ground truth answer.
    
    Args:
        solution: Generated solution string
        ground_truth_answer: Ground truth answer
        
    Returns:
        (is_correct, extracted_answer)
    """
   
    if generated_solution is None:
        return False

    is_correct = generated_solution == ground_truth_answer
    return is_correct
     


# Test function
def test_load_math_problems(batch_size: int = 5):
    """Test loading math problems"""
    results = load_math_problem_batch(batch_size=batch_size)
    for i, result in enumerate(results):
        print(f"\n--- Problem {i+1} ---")
        print(f"Problem: {result['question']}")
        print(f"Answer: {result['solution']}")


if __name__ == "__main__":
    print("Testing math problem loading...")
    test_load_math_problems(3)
