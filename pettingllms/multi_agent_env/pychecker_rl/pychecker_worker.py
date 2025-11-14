"""
Worker utilities for PyChecker RL simulation execution.

This module contains Ray worker implementations for parallel Verilog simulation
with proper timeout handling and resource management.
"""

import os
import json
import asyncio
import signal
import subprocess
import shutil
import tempfile
import time
import re
from typing import Any, List, Tuple, Dict
import logging
from pettingllms.multi_agent_env.pychecker_rl.pychecker_utils import check_compile_success, check_simulation_pass

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Configure logging to console with basic configuration
console = logging.StreamHandler()
console.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console.setFormatter(formatter)
logger.addHandler(console)

# Default simulation timeouts (overridable via environment variables)
SEQ_SIM_TIMEOUT = int(os.getenv("PYCHECKER_SEQ_SIM_TIMEOUT", "300"))
CMB_SIM_TIMEOUT = int(os.getenv("PYCHECKER_CMB_SIM_TIMEOUT", "180"))

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Configure logging to console with basic configuration
console = logging.StreamHandler()
console.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console.setFormatter(formatter)
logger.addHandler(console)

try:
    import ray
    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False
    ray = None


_RAY_TASK_HANDLE = None
_RAY_PYCHECKER_ACTOR_POOL: List[Any] | None = None


async def _await_ray_object_ref(obj_ref, timeout_seconds: float = 120.0):
    """
    Await a Ray object reference with timeout.
    
    Args:
        obj_ref: Ray object reference to await
        timeout_seconds: Maximum time to wait
        
    Returns:
        Result from Ray task
        
    Raises:
        asyncio.TimeoutError: If task exceeds timeout
    """
    if not RAY_AVAILABLE:
        raise RuntimeError("Ray is not available")
        
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


def _get_gpu_group_id() -> str:
    """
    Get a unique identifier for the current GPU group based on CUDA_VISIBLE_DEVICES.
    This allows multiple training jobs with different GPUs to run concurrently.

    Returns:
        GPU group ID string (e.g., "gpu_0_1" for CUDA_VISIBLE_DEVICES=0,1)
    """
    cuda_visible = os.getenv("CUDA_VISIBLE_DEVICES", "")
    if cuda_visible:
        # Normalize: remove spaces, sort for consistency
        gpu_ids = sorted([g.strip() for g in cuda_visible.split(",") if g.strip()])
        return f"gpu_{'_'.join(gpu_ids)}"
    return "gpu_default"


def _ensure_ray_initialized() -> None:
    """
    Ensure Ray is initialized with proper configuration.

    Initializes Ray if not already running, with custom temp directories
    and resource configurations from environment variables.
    """
    if not RAY_AVAILABLE:
        raise RuntimeError("Ray is not available")

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
    
    try:
        ray.init(**init_kwargs)
    except ValueError as e:
        if "_system_config must not be provided" in str(e):
            print("Detected existing Ray cluster, reconnecting without _system_config...")
            init_kwargs.pop("_system_config", None)
            init_kwargs.pop("_temp_dir", None)
            ray.init(**init_kwargs)
        else:
            raise

def get_ray_pychecker_worker_cls(num_workers=None):
    """
    Get or create the Ray PyChecker worker class.
    
    Returns a Ray remote actor class that can execute PyChecker simulations
    with timeout handling. Uses caching to avoid recreating the class.
    
    Args:
        num_workers: Number of workers (for compatibility, not used in this implementation)
    
    Returns:
        Ray remote actor class for PyChecker simulation execution
    """
    if not RAY_AVAILABLE:
        return None
        
    _ensure_ray_initialized()

    if hasattr(get_ray_pychecker_worker_cls, "_cls"):
        return getattr(get_ray_pychecker_worker_cls, "_cls")

    # Configure for CPU-intensive Verilog compilation tasks
    # Key insight: Different tasks use different workers (assigned deterministically in execution engine)
    # Workers are now assigned based on rollout_idx % num_workers for task isolation
    #
    # For 200+ CPUs and 400-500 tasks:
    # - Lower num_cpus allows more concurrent workers
    # - Each worker handles ONE task at a time (max_concurrency=1)
    # - Example: 0.4 CPU/worker * 500 workers = 200 CPUs
    num_cpus_per_worker = float(os.getenv("PYCHECKER_WORKER_CPUS", "0.1"))
    max_concurrent_tasks = int(os.getenv("PYCHECKER_WORKER_CONCURRENCY", "1"))

    # Optional: Support GPU affinity for temp file location
    # Set PYCHECKER_USE_GPU_TMPDIR=1 to enable GPU-local temp directories
    use_gpu_tmpdir = os.getenv("PYCHECKER_USE_GPU_TMPDIR", "0") == "1"

    logger.info(f"Creating PyChecker worker class with num_cpus={num_cpus_per_worker}, max_concurrency={max_concurrent_tasks}, use_gpu_tmpdir={use_gpu_tmpdir}")

    # Get GPU group ID for this worker pool
    gpu_group_id = _get_gpu_group_id()
    logger.info(f"Worker pool will use GPU group: {gpu_group_id}")

    @ray.remote(num_cpus=num_cpus_per_worker, max_concurrency=max_concurrent_tasks)
    class _RayPyCheckerWorker:
        def __init__(self, idx=None, worker_id=None, num_jobs=1, timeout=300):
            # Support both idx (legacy) and worker_id (new) parameters
            if worker_id is not None:
                self.idx = int(worker_id) if isinstance(worker_id, (int, float, str)) else 0
            elif idx is not None:
                if isinstance(idx, (int, float)):
                    self.idx = int(idx)
                elif isinstance(idx, str) and re.fullmatch(r"\s*-?\d+\s*", idx):
                    self.idx = int(idx)
                else:
                    self.idx = 0
            else:
                self.idx = 0

            self.timeout = timeout

            # Store GPU group ID for this worker pool
            # This allows different CUDA_VISIBLE_DEVICES to have isolated worker pools
            self.gpu_group_id = gpu_group_id

            # Optional: Detect GPU affinity for this worker
            # If PYCHECKER_USE_GPU_TMPDIR is enabled, workers can use GPU-local storage
            self.gpu_id = None
            if use_gpu_tmpdir:
                try:
                    import torch
                    if torch.cuda.is_available():
                        # Simple heuristic: assign GPU based on worker index
                        num_gpus = torch.cuda.device_count()
                        if num_gpus > 0:
                            self.gpu_id = self.idx % num_gpus
                            logger.info(f"Worker {self.idx} (group: {self.gpu_group_id}) assigned to GPU {self.gpu_id}")
                except Exception as e:
                    logger.warning(f"Failed to detect GPU for worker {self.idx}: {e}")

        def get_idx(self):
            """Get the actor's index"""
            return self.idx

        def get_gpu_id(self):
            """Get the assigned GPU ID (if any)"""
            return self.gpu_id

        def get_gpu_group_id(self):
            """Get the GPU group ID for this worker pool"""
            return self.gpu_group_id

        async def run_python_file(
            self,
            python_file_path: str,
            working_directory: str,
            timeout: float = 60.0
        ) -> Tuple[bool, str]:
            try:
                # If no directory in path, use working_directory
                file_dir = os.path.dirname(python_file_path) or working_directory
                file_name = os.path.basename(python_file_path)

                # Execute: cd to file directory, then run python filename
                result = subprocess.run(
                    ["python", file_name],
                    cwd=file_dir,
                    capture_output=True,
                    text=True,
                    timeout=timeout
                )

                if result.returncode != 0:
                    return False, f"Python execution failed: {result.stderr}"
                
                return True, ""
            except subprocess.TimeoutExpired:
                return False, f"Python execution timeout after {timeout} seconds"
            except Exception as e:
                return False, f"Python execution error: {str(e)}"

        

        async def simulate_dut_seq_ray(self, output_dir: str) -> Tuple[int, str, str, float, bool, bool]:
            """
            Ray remote function: Execute sequential circuit simulation in separate process

            Args:
                output_dir: Output directory path

            Returns:
                (returncode, stdout, stderr, reward, compile_success, all_tests_passed) tuple
            """
            
            current_dir = os.path.dirname(os.path.abspath(__file__))
            sim_template_dir = "/home/lah003/workspace/verl_efficient/pettingllms/multi_agent_env/pychecker_rl/sim_seq"
            
            work_dir = os.path.join(output_dir, f"sim_seq")
            
            # Try to find the DUT file - could be top_module.v or top.v
            dut_path = os.path.join(output_dir, "top_module.v")
            if not os.path.exists(dut_path):
                dut_path = os.path.join(output_dir, "top.v")
            
            test_path = os.path.join(output_dir, "testbench.json")
            
            # Check if source files exist
            if not os.path.exists(dut_path):
                logger.error(f"DUT file does not exist: {dut_path}")
                return 1, "", f"DUT file not found: {dut_path}", 0.0, False, False
            if not os.path.exists(test_path):
                logger.error(f"Test file does not exist: {test_path}")
                return 1, "", f"Test file not found: {test_path}", 0.0, False, False
            
            os.makedirs(work_dir, exist_ok=True)

            # Copy simulation framework files
            framework_files = [
                "Makefile", "input.vc", 
                "sim-main.cpp", "rfuzz-harness.h",
                "harness-generator.py"
            ]
            for file_name in framework_files:
                src = os.path.join(sim_template_dir, file_name)
                dest = os.path.join(work_dir, file_name)
                if os.path.exists(src):
                    shutil.copy(src, dest)
                else:
                    logger.warning(f"Missing framework file: {src}")
            
            # Copy task-specific files
            shutil.copy(dut_path, os.path.join(work_dir, "top_module.v"))
            shutil.copy(test_path, os.path.join(work_dir, "testbench.json"))
            
            # Execute compilation and simulation
            make_jobs = os.getenv("PYCHECKER_MAKE_JOBS", "1")
            cmd = f"cd {work_dir} && python harness-generator.py && make -j{make_jobs}"
            process = None
            try:
                # Use Popen with process group to ensure child processes can be killed
                process = subprocess.Popen(
                    cmd,
                    shell=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    preexec_fn=os.setpgrp  # Create new process group
                )

                # Wait with timeout
                stdout, stderr = process.communicate(timeout=SEQ_SIM_TIMEOUT)
                result = subprocess.CompletedProcess(
                    args=cmd,
                    returncode=process.returncode,
                    stdout=stdout,
                    stderr=stderr
                )
            except subprocess.TimeoutExpired:
                logger.error(f"SEQ simulation timeout after {SEQ_SIM_TIMEOUT}s for {output_dir}")

                # Kill entire process group to clean up cc1plus children
                if process:
                    try:
                        pgid = os.getpgid(process.pid)
                        os.killpg(pgid, signal.SIGTERM)
                        time.sleep(0.5)
                        # Force kill if still alive
                        try:
                            os.killpg(pgid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                    except (ProcessLookupError, PermissionError) as e:
                        logger.warning(f"Failed to kill process group: {e}")

                # Get partial output
                try:
                    stdout, stderr = process.communicate(timeout=1) if process else ("", "")
                except:
                    stdout, stderr = "", ""

                result = subprocess.CompletedProcess(
                    args=cmd,
                    returncode=1,
                    stdout=stdout,
                    stderr=f"Simulation timeout after {SEQ_SIM_TIMEOUT} seconds\n{stderr}"
                )
            
            # Save log
            log_file = os.path.join(output_dir, f"simulate_seq.log")
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            with open(log_file, "w") as f:
                f.write(f"Command: {cmd}\n")
                f.write(f"Work directory: {work_dir}\n")
                f.write(f"Return code: {result.returncode}\n")
                f.write("\n=== STDOUT ===\n")
                f.write(result.stdout)
                f.write("\n=== STDERR ===\n")
                f.write(result.stderr)
            
            # Calculate reward based on log content
            log_content = f"Command: {cmd}\nWork directory: {work_dir}\nReturn code: {result.returncode}\n=== STDOUT ===\n{result.stdout}\n=== STDERR ===\n{result.stderr}"
            reward = 0.0
            compile_success = check_compile_success(log_content)
            all_tests_passed = check_simulation_pass(log_content)

            if all_tests_passed:
                reward = 1.0

            return result.returncode, result.stdout, result.stderr, reward, compile_success, all_tests_passed


        async def simulate_dut_cmb_ray(self, output_dir: str) -> Tuple[int, str, str, float, bool, bool]:
            """
            Ray remote function: Execute combinational circuit simulation in separate process

            Args:
                output_dir: Output directory path

            Returns:
                (returncode, stdout, stderr, reward, compile_success, all_tests_passed) tuple
            """
            current_dir = os.path.dirname(os.path.abspath(__file__))
            sim_template_dir = "/home/lah003/workspace/verl_efficient/pettingllms/multi_agent_env/pychecker_rl/sim_cmb"
            
            #task_id = os.path.basename(os.path.normpath(output_dir))
            work_dir = os.path.join(output_dir, f"sim_cmb")
            
            # Try to find the DUT file - could be top_module.v or top.v
            dut_path = os.path.join(output_dir, "top_module.v")
            if not os.path.exists(dut_path):
                dut_path = os.path.join(output_dir, "top.v")
            
            test_path = os.path.join(output_dir, "testbench.json")
            
            # Check if source files exist
            if not os.path.exists(dut_path):
                logger.error(f"DUT file does not exist: {dut_path}")
                return 1, "", f"DUT file not found: {dut_path}", 0.0, False, False
            if not os.path.exists(test_path):
                logger.error(f"Test file does not exist: {test_path}")
                return 1, "", f"Test file not found: {test_path}", 0.0, False, False
            
            os.makedirs(work_dir, exist_ok=True)
            
            # Copy simulation framework files
            framework_files = [
                "Makefile", "input.vc", 
                "sim-main.cpp", "rfuzz-harness.h",
                "harness-generator.py"
            ]
            for file_name in framework_files:
                src = os.path.join(sim_template_dir, file_name)
                dest = os.path.join(work_dir, file_name)
                if os.path.exists(src):
                    shutil.copy(src, dest)
                else:
                    logger.warning(f"Missing framework file: {src}")
            
            # Copy task-specific files
            shutil.copy(dut_path, os.path.join(work_dir, "top_module.v"))
            shutil.copy(test_path, os.path.join(work_dir, "testbench.json"))
            
            # Execute compilation and simulation
            make_jobs = os.getenv("PYCHECKER_MAKE_JOBS", "1")
            cmd = f"cd {work_dir} && python harness-generator.py && make -j{make_jobs}"
            process = None
            try:
                # Use Popen with process group to ensure child processes can be killed
                process = subprocess.Popen(
                    cmd,
                    shell=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    preexec_fn=os.setpgrp  # Create new process group
                )

                # Wait with timeout
                stdout, stderr = process.communicate(timeout=CMB_SIM_TIMEOUT)
                result = subprocess.CompletedProcess(
                    args=cmd,
                    returncode=process.returncode,
                    stdout=stdout,
                    stderr=stderr
                )
            except subprocess.TimeoutExpired:
                logger.error(f"CMB simulation timeout after {CMB_SIM_TIMEOUT}s for {output_dir}")

                # Kill entire process group to clean up cc1plus children
                if process:
                    try:
                        pgid = os.getpgid(process.pid)
                        os.killpg(pgid, signal.SIGTERM)
                        time.sleep(0.5)
                        # Force kill if still alive
                        try:
                            os.killpg(pgid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                    except (ProcessLookupError, PermissionError) as e:
                        logger.warning(f"Failed to kill process group: {e}")

                # Get partial output
                try:
                    stdout, stderr = process.communicate(timeout=1) if process else ("", "")
                except:
                    stdout, stderr = "", ""

                result = subprocess.CompletedProcess(
                    args=cmd,
                    returncode=1,
                    stdout=stdout,
                    stderr=f"Simulation timeout after {CMB_SIM_TIMEOUT} seconds\n{stderr}"
                )

            # Save log
            log_file = os.path.join(output_dir, f"simulate_cmb.log")
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            with open(log_file, "w") as f:
                f.write(f"Command: {cmd}\n")
                f.write(f"Work directory: {work_dir}\n")
                f.write(f"Return code: {result.returncode}\n")
                f.write("\n=== STDOUT ===\n")
                f.write(result.stdout)
                f.write("\n=== STDERR ===\n")
                f.write(result.stderr)

            # Calculate reward based on log content
            log_content = f"Command: {cmd}\nWork directory: {work_dir}\nReturn code: {result.returncode}\n=== STDOUT ===\n{result.stdout}\n=== STDERR ===\n{result.stderr}"
            reward = 0.0
            compile_success = check_compile_success(log_content)
            all_tests_passed = check_simulation_pass(log_content)

            if all_tests_passed:
                reward = 1.0

            return result.returncode, result.stdout, result.stderr, reward, compile_success, all_tests_passed

        async def run_verilog_simulation(
            self,
            task_folder: str,
            circuit_type: str,
            timeout: float = 30.0
        ) -> Tuple[bool, str, Dict[str, Any]]:
            """
            Execute Verilog simulation and return result with reward.
            
            Args:
                task_folder: Task folder containing all artifacts
                circuit_type: "CMB" or "SEQ"
                timeout: Execution timeout
                
            Returns:
                (success, error_message, results_dict) tuple
            """
            try:
                # Normalize circuit type for robust comparison
                normalized_type = (circuit_type or "").strip().upper()

                # Call appropriate simulation function based on circuit type
                if normalized_type == "SEQ":
                    returncode, stdout, stderr, reward, compile_success, all_tests_passed = await self.simulate_dut_seq_ray(task_folder)
                elif normalized_type == "CMB":
                    returncode, stdout, stderr, reward, compile_success, all_tests_passed = await self.simulate_dut_cmb_ray(task_folder)
                else:
                    raise ValueError(f"Unsupported circuit type: {circuit_type}")

                # Prepare results dictionary
                results_dict = {
                    'returncode': returncode,
                    'stdout': stdout,
                    'stderr': stderr,
                    'reward': reward,
                    'compile_success': compile_success,
                    'all_tests_passed': all_tests_passed
                }
                
                if returncode == 0:
                    return True, "", results_dict
                else:
                    error_msg = f"Simulation failed with return code {returncode}: {stderr}"
                    return False, error_msg, results_dict
                    
            except Exception as e:
                error_msg = f"Verilog simulation error: {str(e)}"
                results_dict = {
                    'returncode': 1,
                    'stdout': "",
                    'stderr': str(e),
                    'reward': 0.0,
                    'compile_success': False,
                    'all_tests_passed': False
                }
                return False, error_msg, results_dict

        async def run_stimulus_generation(
            self,
            stimulus_py_path: str,
            task_folder: str,
            timeout: float = 20.0
        ) -> Tuple[bool, str]:
            try:
                # Ensure task_folder exists before using it as cwd
                os.makedirs(task_folder, exist_ok=True)
                
                # Normalize task_folder to absolute path
                task_folder = os.path.abspath(task_folder)
                
                # Resolve stimulus_py_path
                basename = os.path.basename(stimulus_py_path)
                
                if os.path.isabs(stimulus_py_path):
                    # Already absolute, normalize it
                    stimulus_py_path_abs = os.path.abspath(stimulus_py_path)
                    # If absolute path doesn't exist, try to find file in task_folder
                    if not os.path.exists(stimulus_py_path_abs):
                        potential_path_in_task = os.path.join(task_folder, basename)
                        if os.path.exists(potential_path_in_task):
                            stimulus_py_path_abs = os.path.abspath(potential_path_in_task)
                else:
                    # Relative path - try to resolve it relative to task_folder
                    # First try just the basename in task_folder
                    potential_path_in_task = os.path.join(task_folder, basename)
                    if os.path.exists(potential_path_in_task):
                        stimulus_py_path_abs = os.path.abspath(potential_path_in_task)
                    else:
                        # Join with task_folder and normalize
                        stimulus_py_path_abs = os.path.abspath(os.path.join(task_folder, stimulus_py_path))
                
                # Check if file exists
                if not os.path.exists(stimulus_py_path_abs):
                    return False, f"Stimulus file not found: {stimulus_py_path_abs} (original path: {stimulus_py_path}, task_folder: {task_folder})"
                
                # If stimulus_py_path is in task_folder, use relative path
                # Otherwise use absolute path
                try:
                    rel_path = os.path.relpath(stimulus_py_path_abs, task_folder)
                    if not rel_path.startswith('..'):
                        script_path = rel_path
                    else:
                        script_path = stimulus_py_path_abs
                except ValueError:
                    # If paths are on different drives (Windows), use absolute path
                    script_path = stimulus_py_path_abs
                
                # Execute stimulus generation script
                result = subprocess.run(
                    ["python", script_path],
                    cwd=task_folder,
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                )

                if result.returncode != 0:
                    return False, f"Stimulus execution failed: {result.stderr}"

                return True, ""
            except subprocess.TimeoutExpired:
                return False, f"Stimulus execution timeout after {timeout} seconds"
            except Exception as e:
                return False, f"Stimulus execution error: {str(e)}"

    RayPyCheckerWorker = _RayPyCheckerWorker
    setattr(get_ray_pychecker_worker_cls, "_cls", RayPyCheckerWorker)
    return RayPyCheckerWorker


# Alias for compatibility with the registration system
def get_ray_docker_worker_cls():
    """Alias for get_ray_pychecker_worker_cls to match the expected interface"""
    return get_ray_pychecker_worker_cls()


async def get_verilog_simulation_result(
    task_folder: str,
    circuit_type: str,
    verilog_dut: str,
    testbench_json_path: str,
    timeout: float = 60.0,
    ray_actor: Any | None = None,
) -> Tuple[bool, str, Dict[str, Any]]:
    """
    Execute Verilog simulation and return the result.
    
    Uses Ray worker for execution with proper timeout handling for concurrent rollouts.
    
    Args:
        task_folder: Task folder containing all artifacts
        circuit_type: "CMB" or "SEQ"
        verilog_dut: Verilog DUT code
        testbench_json_path: Path to testbench.json
        timeout: Execution timeout in seconds
        ray_actor: Ray actor for execution
        
    Returns:
        (success, error_message, results_dict) tuple
        
    Raises:
        ValueError: If ray_actor is None
    """
    try:
        if ray_actor is None:
            raise ValueError("ray_actor is required")
        
        timeout_buffer = max(timeout * 1.5, 60.0)
        total_timeout = timeout + timeout_buffer
        
        # Align call signature with worker's run_verilog_simulation(task_folder, circuit_type, timeout)
        obj_ref = ray_actor.run_verilog_simulation.remote(
            task_folder, circuit_type, timeout
        )
        result = await _await_ray_object_ref(obj_ref, total_timeout)
        
        return result
        
    except asyncio.TimeoutError as e:
        error_msg = f"Verilog simulation timed out after {total_timeout}s"
        print(f"Error: {error_msg}")
        return False, error_msg, {}
    except Exception as e:
        error_msg = f"Verilog simulation failed: {e}"
        print(f"Error: {error_msg}")
        return False, error_msg, {}


async def get_stimulus_generation_result(
    stimulus_py_path: str,
    task_folder: str,
    timeout: float = 60.0,
    ray_actor: Any | None = None,
) -> Tuple[bool, str]:
    """
    Execute stimulus generation and return the result.
    
    Uses Ray worker for execution with proper timeout handling for concurrent rollouts.
    
    Args:
        stimulus_py_path: Path to the stimulus generation Python file
        task_folder: Task folder for execution
        timeout: Execution timeout in seconds
        ray_actor: Ray actor for execution
        
    Returns:
        (success, error_message) tuple
        
    Raises:
        ValueError: If ray_actor is None
    """
    try:
        if ray_actor is None:
            raise ValueError("ray_actor is required")
        
        timeout_buffer = max(timeout * 1.5, 30.0)
        total_timeout = timeout + timeout_buffer
        
        obj_ref = ray_actor.run_stimulus_generation.remote(
            stimulus_py_path, task_folder, timeout
        )
        result = await _await_ray_object_ref(obj_ref, total_timeout)
        
        return result
        
    except asyncio.TimeoutError as e:
        error_msg = f"Stimulus generation timed out after {total_timeout}s"
        print(f"Error: {error_msg}")
        return False, error_msg
    except Exception as e:
        error_msg = f"Stimulus generation failed: {e}"
        print(f"Error: {error_msg}")
        return False, error_msg
