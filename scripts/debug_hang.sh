#!/bin/bash
# Debug: check why training is hanging after Ray init
cd /mnt/afs/zhangyaolun/safe_model/tool/PettingLLMs
source pettingllms_venv/bin/activate

echo "============================================"
echo "  Debug Hang After Ray Init"
echo "  $(date)"
echo "============================================"

echo ""
echo "===== 1. Check if there's a running training process ====="
ps aux | grep -E "pettingllms.trainer.train|train_multi_agents" | grep -v grep || echo "  No training process found"

echo ""
echo "===== 2. Check Ray processes ====="
ps aux | grep -E "raylet|gcs_server|ray::" | grep -v grep | grep -v defunct | head -20 || echo "  No live ray processes"
echo ""
echo "Zombie ray processes:"
ps aux | grep -E "raylet|gcs_server" | grep defunct | head -10 || echo "  None"

echo ""
echo "===== 3. Check Ray dashboard ====="
timeout 5 python -c "
import ray
try:
    ray.init(address='auto', ignore_reinit_error=True)
    print(f'Connected to Ray. Resources: {ray.cluster_resources()}')
    print(f'Available resources: {ray.available_resources()}')
    # Check pending tasks
    import subprocess
    result = subprocess.run(['ray', 'status'], capture_output=True, text=True, timeout=10)
    print(result.stdout)
except Exception as e:
    print(f'Cannot connect to Ray: {e}')
" 2>&1 || echo "  Failed to connect"

echo ""
echo "===== 4. Check Ray logs for errors ====="
# Find most recent ray session
for dir in /tmp/verl_ray_*/session_latest/logs /tmp/ray/session_latest/logs; do
    if [ -d "$dir" ]; then
        echo "--- Checking $dir ---"

        # Raylet alive?
        echo "Raylet status:"
        if [ -f "$dir/raylet.out" ]; then
            echo "  Last 5 lines of raylet.out:"
            tail -5 "$dir/raylet.out" 2>/dev/null | sed 's/^/    /'
        fi
        if [ -f "$dir/raylet.err" ] && [ -s "$dir/raylet.err" ]; then
            echo "  raylet.err (last 10 lines):"
            tail -10 "$dir/raylet.err" 2>/dev/null | sed 's/^/    /'
        fi

        # GCS alive?
        echo "GCS status:"
        if [ -f "$dir/gcs_server.out" ]; then
            echo "  Last 5 lines of gcs_server.out:"
            tail -5 "$dir/gcs_server.out" 2>/dev/null | sed 's/^/    /'
        fi

        # Worker errors
        echo "Worker logs:"
        for f in "$dir"/worker-*.err; do
            if [ -s "$f" ] 2>/dev/null; then
                echo "  === $(basename $f) (last 10 lines) ==="
                tail -10 "$f" | sed 's/^/    /'
            fi
        done 2>/dev/null

        # Worker stdout (for train_multi_agents output)
        echo "Worker stdout (looking for train_multi_agents):"
        for f in "$dir"/worker-*.out; do
            if [ -s "$f" ] 2>/dev/null; then
                if grep -q "train_multi_agents\|Processing model\|Agent mapping\|pettingllms" "$f" 2>/dev/null; then
                    echo "  === $(basename $f) (last 20 lines) ==="
                    tail -20 "$f" | sed 's/^/    /'
                fi
            fi
        done 2>/dev/null

        break
    fi
done

echo ""
echo "===== 5. Quick test: can remote tasks run? ====="
ray stop --force 2>/dev/null || true
rm -rf /tmp/ray /tmp/verl_ray_test_* 2>/dev/null || true
sleep 2

timeout 120 python << 'PYEOF'
import os, time, ray

os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")

tmp = f"/tmp/verl_ray_test_{os.getpid()}"
os.makedirs(tmp, exist_ok=True)

ray.init(num_gpus=8, _temp_dir=tmp)
print(f"Ray init OK. Resources: {ray.cluster_resources()}")
print(f"Available: {ray.available_resources()}")

# Test 1: simple task
print("\nTest 1: simple remote task...")
result = ray.get(ray.remote(lambda: "pong").remote(), timeout=30)
print(f"  Result: {result}")

# Test 2: task requesting many CPUs (like train.py does)
cpu_count = ray.cluster_resources().get("CPU", 0)
num_cpus = max(8, int(cpu_count * 0.1))
print(f"\nTest 2: remote task with num_cpus={num_cpus}...")

@ray.remote(num_cpus=num_cpus)
def heavy_task():
    import time
    time.sleep(1)
    return f"heavy task done, PID={os.getpid()}"

t0 = time.time()
result = ray.get(heavy_task.remote(), timeout=60)
print(f"  Result: {result} ({time.time()-t0:.1f}s)")

# Test 3: task with heavy imports (like train_multi_agents)
print(f"\nTest 3: remote task with heavy imports + num_cpus={num_cpus}...")

@ray.remote(num_cpus=num_cpus)
def import_task():
    t0 = __import__('time').time()
    print("  [worker] Starting imports...", flush=True)
    from omegaconf import OmegaConf, DictConfig
    print(f"  [worker] OmegaConf imported ({__import__('time').time()-t0:.1f}s)", flush=True)
    from verl.single_controller.ray import RayWorkerGroup
    print(f"  [worker] RayWorkerGroup imported ({__import__('time').time()-t0:.1f}s)", flush=True)
    from verl.workers.fsdp_workers import AsyncActorRolloutRefWorker
    print(f"  [worker] AsyncActorRolloutRefWorker imported ({__import__('time').time()-t0:.1f}s)", flush=True)
    from pettingllms.trainer.multi_agents_ppo_trainer import MultiAgentsPPOTrainer
    print(f"  [worker] MultiAgentsPPOTrainer imported ({__import__('time').time()-t0:.1f}s)", flush=True)
    from verl.utils import hf_tokenizer
    print(f"  [worker] hf_tokenizer imported ({__import__('time').time()-t0:.1f}s)", flush=True)
    return f"all imports done in {__import__('time').time()-t0:.1f}s"

t0 = time.time()
result = ray.get(import_task.remote(), timeout=300)
print(f"  Result: {result} (total wall: {time.time()-t0:.1f}s)")

# Test 4: Simulate passing OmegaConf config (serialization test)
print(f"\nTest 4: passing OmegaConf DictConfig to remote task...")
from omegaconf import OmegaConf
mock_config = OmegaConf.create({"a": 1, "b": {"c": 2}})

@ray.remote(num_cpus=num_cpus)
def config_task(cfg):
    from omegaconf import OmegaConf
    return f"received config type={type(cfg).__name__}, keys={list(cfg.keys()) if hasattr(cfg, 'keys') else 'N/A'}"

t0 = time.time()
result = ray.get(config_task.remote(mock_config), timeout=60)
print(f"  Result: {result} ({time.time()-t0:.1f}s)")

print("\nAll tests PASSED")
ray.shutdown()

import shutil
shutil.rmtree(tmp, ignore_errors=True)
PYEOF

echo ""
echo "===== 6. Check GPU memory (is something else using the GPUs?) ====="
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader 2>/dev/null || echo "  nvidia-smi failed"

echo ""
echo "===== 7. Check AFS responsiveness ====="
t0=$(date +%s%N)
ls /mnt/afs/zhangyaolun/safe_model/tool/PettingLLMs/pettingllms/trainer/train.py > /dev/null 2>&1
t1=$(date +%s%N)
echo "AFS file access: $(( (t1 - t0) / 1000000 ))ms"

t0=$(date +%s%N)
python -c "import pettingllms" 2>/dev/null
t1=$(date +%s%N)
echo "import pettingllms: $(( (t1 - t0) / 1000000 ))ms"

echo ""
echo "============================================"
echo "  Done. Key things:"
echo "  - Section 4: Ray worker logs show where it's stuck"
echo "  - Section 5: Tests if remote tasks work at all"
echo "  - Section 6: GPU memory (someone else using GPUs?)"
echo "  - Section 7: AFS speed (slow AFS = slow imports)"
echo "============================================"
