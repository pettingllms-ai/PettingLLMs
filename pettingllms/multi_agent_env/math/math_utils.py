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
import subprocess
import signal
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List, Union
from dataclasses import dataclass
import ray
import shutil
import tempfile
import time
import contextlib

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

def extract_answer(solution_str):
    """
    Extract answer from solution string using \\boxed{} format.
    
    Args:
        solution_str: Solution text containing \\boxed{answer}
        
    Returns:
        Extracted answer string or None if not found
    """
    boxed_pattern = r"\\boxed\s*\{([^{}]+(?:\{[^{}]*\}[^{}]*)*)\}"
    matches = re.findall(boxed_pattern, solution_str)
    
    if matches:
        return matches[-1].strip()
    
    solution = re.findall(r"####\s*(.+?)(?:\n|$)", solution_str)
    if solution:
        return solution[-1].strip()
    
    return None

def extract_reasoning_steps(response: str):
    """
    Extract reasoning steps from agent response.
    
    Args:
        response: Agent response string
        
    Returns:
        Extracted reasoning steps
    """
    match = re.search(r"\*\*Reasoning Steps:\*\*\s*```(.*?)```", response, re.DOTALL)
    if not match:
        return []
    
    steps_block = match.group(1).strip()
    
    steps = [line.strip() for line in steps_block.split("\n") if line.strip()]
    return steps

def extract_code(response: str) -> str:
    """
    Extract code from agent response.
    
    Args:
        response: Agent response string
        
    Returns:
        Extracted code string
    """
    python_pattern = r'```python\s*(.*?)```'
    matches = re.findall(python_pattern, response, re.DOTALL)
    
    if matches:
        return matches[-1].strip()
    
    code_pattern = r'```\s*(.*?)```'
    matches = re.findall(code_pattern, response, re.DOTALL)
    
    if matches:
        return matches[-1].strip()
    
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


def _ensure_ray_initialized() -> None:
   
    import ray  

    if ray.is_initialized():
        return
    

    init_kwargs = {
        "ignore_reinit_error": True,
        "include_dashboard": False,
        "logging_level": "ERROR",
    }

    num_cpus_env = os.getenv("RAY_NUM_CPUS")
    if num_cpus_env:
        try:
            num_cpus = float(num_cpus_env)
            if num_cpus > 0:
                init_kwargs["num_cpus"] = num_cpus
            else:
                print(f"Warning: RAY_NUM_CPUS must be positive, got {num_cpus_env}")
        except (ValueError, TypeError):
            print(f"Warning: invalid RAY_NUM_CPUS value: {num_cpus_env}, using default")

    ray_tmp_dir = "/tmp/verl_ray"
    ray_spill_dir = "/tmp/verl_spill"
    os.makedirs(ray_tmp_dir, exist_ok=True)
    os.makedirs(ray_spill_dir, exist_ok=True)
    
    init_kwargs["_temp_dir"] = ray_tmp_dir
    spilling_conf = {"type": "filesystem", "params": {"directory_path": [ray_spill_dir]}}
    init_kwargs["_system_config"] = {
        "object_spilling_config": json.dumps(spilling_conf)
    }
    
    ray.init(**init_kwargs)


async def _worker_docker(
    script: str,
    timeout: float = 40.0,
    image: str = "python:3.11-slim"
) -> str:
    try:
        os.makedirs("tmp", exist_ok=True)
    except Exception:
        pass
    tmpdir = tempfile.mkdtemp(prefix="pllm_exec_", dir="tmp")
    script_path = os.path.join(tmpdir, "script.py")
    stdout_path = os.path.join(tmpdir, "stdout.txt")

    with open(script_path, "w", encoding="utf-8") as f:
        f.write(script)

    stdout_file = open(stdout_path, "wb")
    try:
        proc = await asyncio.create_subprocess_exec(
            "python",
            script_path,
            stdout=stdout_file,
            stderr=asyncio.subprocess.DEVNULL,
            cwd=tmpdir,
            start_new_session=True,
        )

        try:
            await asyncio.wait_for(proc.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            try:
                if proc.pid:
                    try:
                        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                    except (ProcessLookupError, PermissionError, OSError):
                        pass
                    
                    proc.kill()
                    
                    try:
                        await asyncio.wait_for(proc.wait(), timeout=2.0)
                    except asyncio.TimeoutError:
                        try:
                            proc.terminate()
                            await asyncio.wait_for(proc.wait(), timeout=1.0)
                        except:
                            pass
            except Exception:
                pass
            finally:
                try:
                    if not stdout_file.closed:
                        stdout_file.close()
                    if os.path.exists(tmpdir):
                        try:
                            shutil.rmtree(tmpdir)
                        except Exception:
                            try:
                                subprocess.run(['rm', '-rf', tmpdir], timeout=5, capture_output=True)
                            except Exception:
                                pass
                except Exception:
                    pass
                
            return "timeout"
    finally:
        if not stdout_file.closed:
            stdout_file.close()

    try:
        with open(stdout_path, "rb") as f_out:
            out_bytes = f_out.read()
        result = out_bytes.decode(errors="replace")
    finally:
        try:
            if os.path.exists(tmpdir):
                try:
                    shutil.rmtree(tmpdir)
                except Exception:
                    try:
                        subprocess.run(['rm', '-rf', tmpdir], timeout=5, capture_output=True)
                    except Exception:
                        pass
        except Exception:
            pass
    
    return result


_RAY_TASK_HANDLE = None


async def get_code_execution_output(
    code: str, 
    timeout: float = 40.0,
    ray_actor: Any | None = None,
) -> str:
    """
    Execute Python code and return the output.
    Uses Ray worker for execution with proper timeout handling for concurrent rollouts.
    
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
        
        timeout_buffer = max(timeout * 2.0, 30.0)
        total_timeout = timeout + timeout_buffer
        
        obj_ref = ray_actor.run.remote(code, timeout)
        result = await _await_ray_object_ref(obj_ref, total_timeout)
        
        if isinstance(result, str) and result.startswith("error:"):
            print(f"Warning: Ray execution returned error: {result}")
        else:
            print(f"Success: Ray execution completed, output length: {len(str(result))} characters")
            
        return result
        
    except asyncio.TimeoutError as e:
        error_msg = f"Ray execution timed out after {total_timeout}s"
        print(f"Error: {error_msg}")
        return f"error: {error_msg}"
    except Exception as e:
        error_msg = f"Ray execution failed: {e}"
        print(f"Error: {error_msg}")
        return f"error: {error_msg}"


def get_ray_docker_worker_cls():
    try:
        import ray
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
            _max_conc = int(_max_conc_env) if _max_conc_env else 20
        except (ValueError, TypeError):
            print(f"Warning: invalid RAY_ACTOR_MAX_CONCURRENCY value: {_max_conc_env}, using default 20")
            _max_conc = 20

        @ray.remote(num_cpus=0.001, max_concurrency=2000)
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
                """Get the actor's index"""
                return self.idx

            async def run(
                self,
                script: str,
                timeout: float = 40.0,
                image: str = "python:3.11-slim",
            ) -> str:
                """
                Execute Python script and return output.
                
                Args:
                    script: Python script to execute
                    timeout: Execution timeout
                    image: Docker image to use (not used in current implementation)
                    
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


_RAY_DOCKER_ACTOR_POOL: List[Any] | None = None


def load_math_problem_batch(
    env_indices: List[int],
    mode: str = "train",
    dataset_name: str = "train_polaris.parquet",
    config: dict = None,
    benchmark_name: str = "AIME24",
    validate_samples: int = 8
) -> List[Dict[str, Any]]:
    """
    Load a batch of mathematical problems.
    
    Args:
        env_indices: List of environment indices
        mode: "train" or "validate"
        config: Configuration dict (unused, kept for compatibility)
        benchmark_name: Name of benchmark dataset
        validate_samples: Number of samples per problem in validate mode
        
    Returns:
        A list of dicts with keys question/solution
    """
    if not DATASETS_AVAILABLE:
        raise ImportError("datasets library is required")
    

    current_dir = Path(__file__).parent.parent.parent.parent
    local_datasets_dir = current_dir / "datasets" / "math"
    
    if mode == "train":
        parquet_file = local_datasets_dir /"train" /f"{dataset_name}.parquet"
    else:
        parquet_file = local_datasets_dir /"test" / f"{benchmark_name}_test.parquet"
    

    if not parquet_file.exists():
        raise print(f"Dataset file not found: {parquet_file}")
    
    print(f"Loading dataset from: {parquet_file}")
    ds = hf_load_dataset("parquet", data_files=str(parquet_file), split="train")
    print(f"Dataset loaded: {len(ds)} samples")
    
    batch_results = []
    
    if mode == "train":
        if len(ds) < len(env_indices):
            raise ValueError(f"Dataset has {len(ds)} samples, but {len(env_indices)} requested")
        
        indices = random.sample(range(len(ds)), len(env_indices))
        for idx in indices:
            problem_dict = _format_math_problem(ds[idx], idx, mode="train")
            if problem_dict:
                batch_results.append(problem_dict)
    else:
        # 验证模式：遍历所有样本，每个样本重复 validate_samples 次
        for i, example in enumerate(ds):
            problem_dict = _format_math_problem(example, i, mode="validate")
            if problem_dict:
                for _ in range(validate_samples):
                    batch_results.append(problem_dict)
    
    print(f"Returning {len(batch_results)} samples")
    return batch_results


def _format_math_problem(example: Dict, index: int, mode: str = "train") -> Optional[Dict]:
    """
    Format a math problem example into a standardized dictionary.
    
    Args:
        example: Raw example from dataset (expected format: question/solution)
        index: Index of the example
        mode: "train" or "validate"
        
    Returns:
        Formatted problem dictionary or None if invalid
    """
    try:
        question = example.get("question", "")
        solution = example.get("solution", "")
        answer = solution
        
        if not question:
            print(f"Warning: Skipping example {index}: missing question field")
            return None
        
        return {
            "question": question,
            "solution": answer
        }
        
    except Exception as e:
        print(f"Warning: Error formatting example {index}: {e}")
        return None
