"""Ray initialization utilities"""
import os
import json
import subprocess
import ray


def _cleanup_stale_ray():
    """Clean up stale Ray processes and temp dirs from previous runs"""
    subprocess.run(["ray", "stop", "--force"], capture_output=True, check=False)
    for pattern in ["raylet", "gcs_server"]:
        subprocess.run(["pkill", "-9", "-f", pattern], capture_output=True, check=False)
    import shutil
    import glob
    for d in glob.glob("/tmp/verl_ray_*") + glob.glob("/tmp/verl_spill_*") + ["/tmp/ray"]:
        shutil.rmtree(d, ignore_errors=True)
    import time
    time.sleep(2)


def _set_ray_health_check_env():
    """Increase Ray health check timeouts for AFS/slow-storage environments.

    When Python and Ray packages are on a network filesystem (e.g. AFS),
    the dashboard agent and worker processes take longer to start.
    Default health check timeouts (30s) cause false-positive failures.
    """
    health_env = {
        "RAY_health_check_failure_threshold": "100",
        "RAY_health_check_period_ms": "60000",
        "RAY_health_check_initial_delay_ms": "60000",
        "RAY_gcs_server_request_timeout_seconds": "120",
        "RAY_raylet_death_check_interval_milliseconds": "60000",
        # Disable worker pre-starting: Ray tries to pre-start ~112 Python workers,
        # which all hang when loading from AFS (network filesystem) simultaneously
        "RAY_prestart_worker_first_driver": "0",
        # Reduce idle worker overhead on AFS
        "RAY_min_spilled_worker_count": "0",
        "RAY_idle_worker_killing_time_threshold_ms": "10000",
        # Avoid OOM kills (default 0.95 threshold)
        "RAY_memory_usage_threshold": "0.95",
    }
    for key, value in health_env.items():
        os.environ.setdefault(key, value)


def init_ray_with_temp_dirs(config=None, n_gpus_per_node=None):
    """
    Initialize Ray with temporary directories and spilling configuration

    Args:
        config: Optional config object with resource settings
        n_gpus_per_node: Number of GPUs per node (overrides config if provided)

    Returns:
        Tuple of (ray_tmp_dir, ray_spill_dir)
    """
    from .clean_up import register_temp_dirs

    if ray.is_initialized():
        print("Ray is already initialized")
        return None, None

    # Clean up stale Ray processes from previous runs
    _cleanup_stale_ray()

    # Increase health check timeouts for AFS environments
    _set_ray_health_check_env()

    # Create experiment-specific temporary directories using process ID
    pid = os.getpid()
    ray_tmp_dir = f"/tmp/verl_ray_{pid}"
    ray_spill_dir = f"/tmp/verl_spill_{pid}"
    os.makedirs(ray_tmp_dir, exist_ok=True)
    os.makedirs(ray_spill_dir, exist_ok=True)

    # Register directories for cleanup
    register_temp_dirs(ray_tmp_dir, ray_spill_dir)

    # Configure spilling
    spilling_conf = {"type": "filesystem", "params": {"directory_path": [ray_spill_dir]}}
    system_config = {"object_spilling_config": json.dumps(spilling_conf)}

    # Determine GPU count
    if n_gpus_per_node is None:
        n_gpus_per_node = getattr(config.resource, 'n_gpus_per_node', 1) if config and hasattr(config, 'resource') else 1

    # Validate GPU availability
    cuda_visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if cuda_visible_devices:
        available_gpu_count = len(cuda_visible_devices.split(','))
        n_gpus_per_node = min(n_gpus_per_node, available_gpu_count)

    # Set env vars directly instead of using runtime_env (which causes raylet crash)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
    os.environ.setdefault("NCCL_DEBUG", "WARN")

    print(f"Initializing Ray with {n_gpus_per_node} GPUs")
    ray.init(
        num_gpus=n_gpus_per_node,
        _temp_dir=ray_tmp_dir,
        _system_config=system_config
    )

    return ray_tmp_dir, ray_spill_dir
