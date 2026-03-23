#!/bin/bash
# Debug v2: Fix hang caused by Ray pre-starting too many workers on AFS
# Root cause: Ray pre-starts 112 Python workers, all loading from AFS simultaneously = hang
cd /mnt/afs/zhangyaolun/safe_model/tool/PettingLLMs
source pettingllms_venv/bin/activate

echo "============================================"
echo "  Debug Hang v2 - Worker Pre-start Fix"
echo "  $(date)"
echo "============================================"

echo ""
echo "===== 1. Check GPU usage (is something else using GPUs?) ====="
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader 2>/dev/null || echo "  nvidia-smi failed"
echo ""
echo "Processes on GPUs:"
nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory --format=csv,noheader 2>/dev/null || echo "  none"

echo ""
echo "===== 2. Clean up any stale Ray processes ====="
ray stop --force 2>/dev/null || true
pkill -9 -f "raylet|gcs_server|ray::" 2>/dev/null || true
rm -rf /tmp/ray /tmp/verl_ray_* /tmp/verl_spill_* 2>/dev/null || true
sleep 3
echo "  Cleaned up"

echo ""
echo "===== 3. Test: Ray with worker pre-start DISABLED ====="
echo "  Setting RAY_prestart_worker_first_driver=0 to disable worker pre-starting"
echo "  This prevents Ray from pre-loading 112 Python workers from AFS"

timeout 120 python << 'PYEOF'
import os, time, ray

# KEY FIX: Disable worker pre-starting (prevents AFS hang)
os.environ["RAY_prestart_worker_first_driver"] = "0"

# Also reduce idle worker count
os.environ["RAY_min_spilled_worker_count"] = "0"
os.environ["RAY_idle_worker_killing_time_threshold_ms"] = "10000"

# Health check timeouts for AFS
os.environ.setdefault("RAY_health_check_failure_threshold", "100")
os.environ.setdefault("RAY_health_check_period_ms", "60000")
os.environ.setdefault("RAY_health_check_initial_delay_ms", "60000")
os.environ.setdefault("RAY_gcs_server_request_timeout_seconds", "120")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")

# Memory threshold (avoid OOM kills)
os.environ.setdefault("RAY_memory_usage_threshold", "0.95")

pid = os.getpid()
tmp = f"/tmp/verl_ray_{pid}"
os.makedirs(tmp, exist_ok=True)

print(f"[{time.strftime('%H:%M:%S')}] Starting ray.init()...")
t0 = time.time()
ray.init(num_gpus=8, _temp_dir=tmp)
print(f"[{time.strftime('%H:%M:%S')}] Ray init done in {time.time()-t0:.1f}s")
print(f"  Resources: {ray.cluster_resources()}")
print(f"  Available: {ray.available_resources()}")

# Test 1: simple task
print(f"\n[{time.strftime('%H:%M:%S')}] Test 1: simple remote task...")
t0 = time.time()
result = ray.get(ray.remote(lambda: "pong").remote(), timeout=30)
print(f"  OK: {result} ({time.time()-t0:.1f}s)")

# Test 2: heavy CPU task (simulates train_multi_agents)
cpu_count = ray.cluster_resources().get("CPU", 0)
num_cpus = max(8, int(cpu_count * 0.1))
print(f"\n[{time.strftime('%H:%M:%S')}] Test 2: remote task with num_cpus={num_cpus}...")

@ray.remote(num_cpus=num_cpus)
def heavy_import_task():
    t0 = __import__('time').time()
    print("  [worker] Starting heavy imports...", flush=True)
    from omegaconf import OmegaConf, DictConfig
    from verl.single_controller.ray import RayWorkerGroup
    from verl.workers.fsdp_workers import AsyncActorRolloutRefWorker
    from pettingllms.trainer.multi_agents_ppo_trainer import MultiAgentsPPOTrainer
    from verl.utils import hf_tokenizer
    elapsed = __import__('time').time() - t0
    print(f"  [worker] All imports done in {elapsed:.1f}s", flush=True)
    return f"imports done in {elapsed:.1f}s"

t0 = time.time()
result = ray.get(heavy_import_task.remote(), timeout=300)
print(f"  OK: {result} (wall: {time.time()-t0:.1f}s)")

# Test 3: OmegaConf config serialization
print(f"\n[{time.strftime('%H:%M:%S')}] Test 3: passing OmegaConf config...")
from omegaconf import OmegaConf
mock_cfg = OmegaConf.create({"a": 1, "b": {"c": 2}})

@ray.remote(num_cpus=num_cpus)
def config_task(cfg):
    return f"config type={type(cfg).__name__}"

t0 = time.time()
result = ray.get(config_task.remote(mock_cfg), timeout=60)
print(f"  OK: {result} ({time.time()-t0:.1f}s)")

print(f"\n[{time.strftime('%H:%M:%S')}] ALL TESTS PASSED")
ray.shutdown()

import shutil
shutil.rmtree(tmp, ignore_errors=True)
PYEOF

EXIT_CODE=$?
if [ $EXIT_CODE -ne 0 ]; then
    echo ""
    echo "TESTS FAILED (exit code $EXIT_CODE)"
    echo "Check raylet logs:"
    for dir in /tmp/verl_ray_*/session_latest/logs; do
        if [ -d "$dir" ]; then
            echo "--- raylet.err ---"
            tail -20 "$dir/raylet.err" 2>/dev/null | sed 's/^/  /'
            echo "--- raylet.out ---"
            tail -10 "$dir/raylet.out" 2>/dev/null | sed 's/^/  /'
        fi
    done
else
    echo ""
    echo "============================================"
    echo "  SUCCESS! The fix works."
    echo "  Apply these env vars to ray_utils.py:"
    echo "    RAY_prestart_worker_first_driver=0"
    echo "    RAY_min_spilled_worker_count=0"
    echo "    RAY_memory_usage_threshold=0.95"
    echo "============================================"
fi
