#!/bin/bash
set -e

echo "============================================"
echo "  Ray Debugging Script for PettingLLMs"
echo "  $(date)"
echo "============================================"

# ---- Section 1: System Resources ----
echo ""
echo "===== 1. SYSTEM RESOURCES ====="

echo "--- /dev/shm (shared memory) ---"
df -h /dev/shm
echo ""
echo "Ray uses /dev/shm heavily. If 'Available' is < 2GB, Ray workers WILL fail to register."
echo "Fix: ask admin to increase shm size, or set RAY_OBJECT_STORE_MEMORY to limit usage."

echo ""
echo "--- /tmp space ---"
df -h /tmp
echo ""
echo "Ray stores temp files in /tmp. If full, raylet crashes silently."

echo ""
echo "--- Memory ---"
free -h

echo ""
echo "--- CPU count ---"
nproc
echo "Available CPUs to this process:"
python3 -c "import os; print(f'  os.cpu_count() = {os.cpu_count()}')" 2>/dev/null || true
python3 -c "import multiprocessing; print(f'  multiprocessing.cpu_count() = {multiprocessing.cpu_count()}')" 2>/dev/null || true

echo ""
echo "--- GPU status ---"
nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free --format=csv,noheader 2>/dev/null || echo "nvidia-smi failed!"
echo ""
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-not set}"

# ---- Section 2: Zombie Ray processes ----
echo ""
echo "===== 2. EXISTING RAY PROCESSES ====="
echo "--- ray processes ---"
ps aux | grep -E "(raylet|gcs_server|ray::|plasma)" | grep -v grep || echo "No ray processes found."

echo ""
echo "--- verl_ray temp dirs ---"
ls -la /tmp/verl_ray_* 2>/dev/null || echo "No verl_ray temp dirs found."
ls -la /tmp/verl_spill_* 2>/dev/null || echo "No verl_spill temp dirs found."

echo ""
echo "--- ray default temp dirs ---"
ls -la /tmp/ray/ 2>/dev/null && echo "WARNING: /tmp/ray/ exists - may conflict!" || echo "/tmp/ray/ does not exist (good)."

# ---- Section 3: Ray cluster check ----
echo ""
echo "===== 3. RAY CLUSTER STATUS ====="
python3 -c "import ray; ray.init(ignore_reinit_error=True); print(ray.cluster_resources()); ray.shutdown()" 2>&1 || echo "Failed to init a test Ray cluster!"

# ---- Section 4: Python / Ray / vLLM versions ----
echo ""
echo "===== 4. PACKAGE VERSIONS ====="
python3 -c "
import sys
print(f'Python: {sys.version}')
print(f'Python executable: {sys.executable}')
try:
    import ray; print(f'Ray: {ray.__version__}')
except: print('Ray: NOT INSTALLED')
try:
    import torch; print(f'PyTorch: {torch.__version__}')
    print(f'CUDA available: {torch.cuda.is_available()}')
    print(f'CUDA device count: {torch.cuda.device_count()}')
    if torch.cuda.is_available():
        print(f'CUDA version: {torch.version.cuda}')
except: print('PyTorch: NOT INSTALLED')
try:
    import vllm; print(f'vLLM: {vllm.__version__}')
except: print('vLLM: NOT INSTALLED')
try:
    import verl; print(f'verl: {verl.__version__}')
except Exception as e: print(f'verl: could not get version ({e})')
try:
    import pettingllms; print(f'pettingllms: found at {pettingllms.__file__}')
except Exception as e: print(f'pettingllms: {e}')
"

# ---- Section 5: Detailed Ray init test ----
echo ""
echo "===== 5. DETAILED RAY INIT TEST ====="
python3 << 'PYEOF'
import os
import sys
import json
import traceback

print("--- Test 1: Basic Ray init with 1 GPU ---")
try:
    import ray
    ray.shutdown()  # ensure clean state

    pid = os.getpid()
    ray_tmp_dir = f"/tmp/verl_ray_debug_{pid}"
    ray_spill_dir = f"/tmp/verl_spill_debug_{pid}"
    os.makedirs(ray_tmp_dir, exist_ok=True)
    os.makedirs(ray_spill_dir, exist_ok=True)

    spilling_conf = {"type": "filesystem", "params": {"directory_path": [ray_spill_dir]}}
    system_config = {"object_spilling_config": json.dumps(spilling_conf)}

    ray.init(
        num_gpus=1,
        _temp_dir=ray_tmp_dir,
        _system_config=system_config,
    )
    print(f"  SUCCESS: Ray initialized with 1 GPU")
    print(f"  Cluster resources: {ray.cluster_resources()}")
    ray.shutdown()
    print(f"  Ray shutdown OK")

    # Cleanup
    import shutil
    shutil.rmtree(ray_tmp_dir, ignore_errors=True)
    shutil.rmtree(ray_spill_dir, ignore_errors=True)
except Exception as e:
    print(f"  FAILED: {e}")
    traceback.print_exc()
    try:
        ray.shutdown()
    except:
        pass

print()
print("--- Test 2: Ray init with 8 GPUs ---")
try:
    import ray
    ray.shutdown()

    pid = os.getpid()
    ray_tmp_dir = f"/tmp/verl_ray_debug2_{pid}"
    ray_spill_dir = f"/tmp/verl_spill_debug2_{pid}"
    os.makedirs(ray_tmp_dir, exist_ok=True)
    os.makedirs(ray_spill_dir, exist_ok=True)

    spilling_conf = {"type": "filesystem", "params": {"directory_path": [ray_spill_dir]}}
    system_config = {"object_spilling_config": json.dumps(spilling_conf)}

    ray.init(
        num_gpus=8,
        runtime_env={"env_vars": {"TOKENIZERS_PARALLELISM": "true", "NCCL_DEBUG": "WARN"}},
        _temp_dir=ray_tmp_dir,
        _system_config=system_config,
    )
    print(f"  SUCCESS: Ray initialized with 8 GPUs")
    print(f"  Cluster resources: {ray.cluster_resources()}")
    ray.shutdown()
    print(f"  Ray shutdown OK")

    import shutil
    shutil.rmtree(ray_tmp_dir, ignore_errors=True)
    shutil.rmtree(ray_spill_dir, ignore_errors=True)
except Exception as e:
    print(f"  FAILED: {e}")
    traceback.print_exc()
    try:
        ray.shutdown()
    except:
        pass

print()
print("--- Test 3: Submit a remote task (reproduces the actual failure) ---")
try:
    import ray
    import time
    ray.shutdown()

    pid = os.getpid()
    ray_tmp_dir = f"/tmp/verl_ray_debug3_{pid}"
    ray_spill_dir = f"/tmp/verl_spill_debug3_{pid}"
    os.makedirs(ray_tmp_dir, exist_ok=True)
    os.makedirs(ray_spill_dir, exist_ok=True)

    spilling_conf = {"type": "filesystem", "params": {"directory_path": [ray_spill_dir]}}
    system_config = {"object_spilling_config": json.dumps(spilling_conf)}

    ray.init(
        num_gpus=8,
        runtime_env={"env_vars": {"TOKENIZERS_PARALLELISM": "true", "NCCL_DEBUG": "WARN"}},
        _temp_dir=ray_tmp_dir,
        _system_config=system_config,
    )
    print(f"  Ray initialized. Cluster resources: {ray.cluster_resources()}")

    # This mimics what train.py does: create a remote task requesting many CPUs
    cpu_count = ray.cluster_resources().get("CPU", 0)
    num_cpus_request = max(8, int(cpu_count * 0.1))
    print(f"  Total CPUs available: {cpu_count}")
    print(f"  Requesting num_cpus={num_cpus_request} for remote task (same as train.py)")

    @ray.remote(num_cpus=num_cpus_request)
    def dummy_task():
        import socket
        return f"Worker running on {socket.gethostname()}, PID={os.getpid()}"

    print(f"  Submitting remote task...")
    future = dummy_task.remote()
    result = ray.get(future, timeout=60)
    print(f"  SUCCESS: {result}")

    ray.shutdown()
    import shutil
    shutil.rmtree(ray_tmp_dir, ignore_errors=True)
    shutil.rmtree(ray_spill_dir, ignore_errors=True)
except Exception as e:
    print(f"  FAILED: {e}")
    traceback.print_exc()
    try:
        ray.shutdown()
    except:
        pass

print()
print("--- Test 4: Import the actual training modules ---")
try:
    from verl.single_controller.ray import RayWorkerGroup
    print("  verl.single_controller.ray.RayWorkerGroup: OK")
except Exception as e:
    print(f"  FAILED to import RayWorkerGroup: {e}")

try:
    from verl.workers.fsdp_workers import AsyncActorRolloutRefWorker
    print("  verl.workers.fsdp_workers.AsyncActorRolloutRefWorker: OK")
except Exception as e:
    print(f"  FAILED to import AsyncActorRolloutRefWorker: {e}")

try:
    from pettingllms.trainer.multi_agents_ppo_trainer import MultiAgentsPPOTrainer
    print("  pettingllms.trainer.multi_agents_ppo_trainer.MultiAgentsPPOTrainer: OK")
except Exception as e:
    print(f"  FAILED to import MultiAgentsPPOTrainer: {e}")

print()
print("--- Test 5: Check if ray.remote(max_concurrency=2048) works ---")
try:
    import ray
    ray.shutdown()

    pid = os.getpid()
    ray_tmp_dir = f"/tmp/verl_ray_debug5_{pid}"
    os.makedirs(ray_tmp_dir, exist_ok=True)

    ray.init(num_gpus=8, _temp_dir=ray_tmp_dir)

    @ray.remote(max_concurrency=2048)
    class TestActor:
        def ping(self):
            return "pong"

    actor = TestActor.remote()
    result = ray.get(actor.ping.remote(), timeout=30)
    print(f"  SUCCESS: max_concurrency=2048 actor works, got: {result}")
    ray.shutdown()
    import shutil
    shutil.rmtree(ray_tmp_dir, ignore_errors=True)
except Exception as e:
    print(f"  FAILED: {e}")
    traceback.print_exc()
    try:
        ray.shutdown()
    except:
        pass

PYEOF

# ---- Section 6: Check raylet logs from last crash ----
echo ""
echo "===== 6. RECENT RAYLET CRASH LOGS ====="
echo "Checking for recent ray logs..."
for dir in /tmp/verl_ray_*/logs /tmp/ray/session_latest/logs; do
    if [ -d "$dir" ]; then
        echo "--- Logs in $dir ---"
        # Check raylet logs
        if ls "$dir"/raylet.* 2>/dev/null; then
            echo "--- Last 50 lines of raylet stderr ---"
            tail -50 "$dir"/raylet.err 2>/dev/null || true
            echo ""
            echo "--- Last 50 lines of raylet stdout ---"
            tail -50 "$dir"/raylet.out 2>/dev/null || true
        fi
        # Check GCS logs
        if ls "$dir"/gcs_server.* 2>/dev/null; then
            echo "--- Last 30 lines of gcs_server stderr ---"
            tail -30 "$dir"/gcs_server.err 2>/dev/null || true
        fi
        # Check for SIGTERM/SIGSEGV/OOM in any log
        echo "--- Searching for crash signals ---"
        grep -l -i "SIGTERM\|SIGSEGV\|SIGKILL\|OOM\|out of memory\|killed\|core dumped" "$dir"/* 2>/dev/null || echo "  No crash signals found."
    fi
done

# ---- Section 7: ulimit ----
echo ""
echo "===== 7. ULIMITS ====="
echo "Open files limit:"
ulimit -n
echo "Max user processes:"
ulimit -u
echo "Stack size:"
ulimit -s
echo ""
echo "If 'open files' < 65536 or 'max user processes' < 4096, Ray may fail."
echo "Fix: ulimit -n 65536; ulimit -u unlimited"

# ---- Section 8: Network ----
echo ""
echo "===== 8. NETWORK ====="
echo "Hostname: $(hostname)"
echo "Checking if localhost resolves..."
python3 -c "
import socket
try:
    ip = socket.gethostbyname(socket.gethostname())
    print(f'  Hostname resolves to: {ip}')
except Exception as e:
    print(f'  FAILED: {e}')
    print('  This can cause Ray worker registration failures!')
try:
    ip = socket.gethostbyname('localhost')
    print(f'  localhost resolves to: {ip}')
except Exception as e:
    print(f'  localhost resolution FAILED: {e}')
"

echo ""
echo "============================================"
echo "  Debugging complete!"
echo "============================================"
echo ""
echo "COMMON FIXES:"
echo "  1. /dev/shm too small: export RAY_OBJECT_STORE_MEMORY=1000000000  (1GB)"
echo "  2. /tmp full: clean up /tmp or set RAY_TMPDIR to another dir"
echo "  3. Stale ray: ray stop --force && rm -rf /tmp/ray /tmp/verl_ray_*"
echo "  4. ulimits too low: ulimit -n 65536; ulimit -u unlimited"
echo "  5. Hostname not resolving: add entry to /etc/hosts"
echo "  6. Too many CPUs requested: reduce num_cpus in ray.remote()"
