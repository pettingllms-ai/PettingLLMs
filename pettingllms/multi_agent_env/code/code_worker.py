"""
Worker classes and utilities for code execution.

This module contains the worker implementations for executing code
in isolated environments, including Ray-based distributed workers.
"""

import os
import sys
import asyncio
import subprocess
import tempfile
import shutil
import textwrap
import signal
from typing import Any, Dict
from pathlib import Path


async def test_if_eq(x, y):
    """
    Test equality of two outputs ignoring whitespace differences.
    Based on the reference test_if_eq function provided.
    """
    return " ".join(x.split()) == " ".join(y.split())


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
    # Make tmpdir absolute to avoid path resolution issues when cwd changes
    tmpdir = os.path.abspath(tempfile.mkdtemp(prefix="pllm_exec_",dir="tmp"))
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
                        pass
                    except Exception as e:
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


def get_ray_docker_worker_cls(num_workers=180):
    try:
        import ray  # type: ignore
    except Exception as e:
        print(f"Failed to import ray: {e}")
        return None

    cache_key = f"_cls_{num_workers}"
    if hasattr(get_ray_docker_worker_cls, cache_key):
        return getattr(get_ray_docker_worker_cls, cache_key)

    try:
        import multiprocessing
        total_cpus = multiprocessing.cpu_count()
        cpus_per_worker = min(4.0, (total_cpus * 0.6) / num_workers)
        print(f"Ray worker resource allocation: total_cpus={total_cpus}, num_workers={num_workers}, "
              f"cpus_per_worker={cpus_per_worker:.3f} (60% of total)")
    except Exception as e:
        print(f"Failed to calculate CPU allocation, using default: {e}")
        cpus_per_worker = 0.001

    try:
        _max_conc = 20

        @ray.remote(num_cpus=cpus_per_worker, max_concurrency=10000)
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
                timeout: float = 40.0,  
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
        cache_key = f"_cls_{num_workers}"
        setattr(get_ray_docker_worker_cls, cache_key, RayDockerWorker)
        return RayDockerWorker
        
    except Exception as e:
        print(f"Failed to create RayDockerWorker class: {e}")
        return None

