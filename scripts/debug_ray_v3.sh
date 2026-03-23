#!/bin/bash
set -e

cd /mnt/afs/zhangyaolun/safe_model/tool/PettingLLMs
source pettingllms_venv/bin/activate

echo "============================================"
echo "  Ray Debug V3 - Root Cause Isolation"
echo "  $(date)"
echo "============================================"

# Clean up first
ray stop --force 2>/dev/null || true
rm -rf /tmp/verl_ray_* /tmp/verl_spill_* /tmp/ray 2>/dev/null || true
sleep 2

echo ""
echo "===== TEST 1: Is it a Ray version issue? ====="
echo "Virtualenv Ray version:"
python -c "import ray; print(f'  Ray {ray.__version__}')"
echo "System Ray version:"
/opt/conda/bin/python3 -c "import ray; print(f'  Ray {ray.__version__}')" 2>/dev/null || echo "  Not installed in system python"

echo ""
echo "===== TEST 2: Ray init with verbose logging ====="
echo "This will show WHY the raylet crashes..."
RAY_BACKEND_LOG_LEVEL=debug python << 'PYEOF'
import os, sys, json, ray, time, traceback

ray.shutdown()
pid = os.getpid()
ray_tmp_dir = f"/tmp/verl_ray_debug_{pid}"
ray_spill_dir = f"/tmp/verl_spill_debug_{pid}"
os.makedirs(ray_tmp_dir, exist_ok=True)
os.makedirs(ray_spill_dir, exist_ok=True)

spilling_conf = {"type": "filesystem", "params": {"directory_path": [ray_spill_dir]}}
system_config = {"object_spilling_config": json.dumps(spilling_conf)}

try:
    ray.init(
        num_gpus=8,
        runtime_env={"env_vars": {"TOKENIZERS_PARALLELISM": "true", "NCCL_DEBUG": "WARN"}},
        _temp_dir=ray_tmp_dir,
        _system_config=system_config,
    )
    print(f"Ray init OK. Resources: {ray.cluster_resources()}")

    # Check if raylet is alive after init
    time.sleep(2)
    import subprocess
    result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
    raylet_lines = [l for l in result.stdout.split('\n') if 'raylet' in l and 'grep' not in l]
    print(f"Raylet processes after init: {len(raylet_lines)}")
    for l in raylet_lines:
        print(f"  {l}")

    # Try simple remote task
    @ray.remote(num_cpus=1)
    def ping():
        return "pong"

    print("Submitting ping task...")
    future = ping.remote()
    result = ray.get(future, timeout=60)
    print(f"Result: {result}")
except Exception as e:
    print(f"FAILED: {e}")
    traceback.print_exc()

    # Dump raylet logs
    import glob
    log_dir = f"{ray_tmp_dir}/session_latest/logs"
    print(f"\n--- Raylet stderr ({log_dir}/raylet.err) ---")
    try:
        with open(f"{log_dir}/raylet.err") as f:
            print(f.read()[-3000:] or "(empty)")
    except: print("  Could not read")

    print(f"\n--- Raylet stdout (last 50 lines) ---")
    try:
        with open(f"{log_dir}/raylet.out") as f:
            lines = f.readlines()
            print(''.join(lines[-50:]) or "(empty)")
    except: print("  Could not read")

    print(f"\n--- GCS server stderr ---")
    try:
        with open(f"{log_dir}/gcs_server.err") as f:
            print(f.read()[-2000:] or "(empty)")
    except: print("  Could not read")

    print(f"\n--- GCS server stdout (last 30 lines) ---")
    try:
        with open(f"{log_dir}/gcs_server.out") as f:
            lines = f.readlines()
            print(''.join(lines[-30:]) or "(empty)")
    except: print("  Could not read")
finally:
    try: ray.shutdown()
    except: pass

import shutil
shutil.rmtree(ray_tmp_dir, ignore_errors=True)
shutil.rmtree(ray_spill_dir, ignore_errors=True)
PYEOF

echo ""
echo "===== TEST 3: Try without object spilling config ====="
ray stop --force 2>/dev/null || true
rm -rf /tmp/ray /tmp/verl_ray_* 2>/dev/null || true
sleep 2

python << 'PYEOF'
import os, ray, traceback

ray.shutdown()
try:
    print("Init Ray with minimal config (no spilling, no custom temp dir)...")
    ray.init(num_gpus=8)
    print(f"Resources: {ray.cluster_resources()}")

    @ray.remote(num_cpus=1)
    def ping():
        return "pong"

    result = ray.get(ping.remote(), timeout=60)
    print(f"Result: {result}")
    print("PASSED")
except Exception as e:
    print(f"FAILED: {e}")
    traceback.print_exc()
finally:
    try: ray.shutdown()
    except: pass
PYEOF

echo ""
echo "===== TEST 4: Try with local /tmp temp dir (not AFS) ====="
ray stop --force 2>/dev/null || true
rm -rf /tmp/ray /tmp/verl_ray_* 2>/dev/null || true
sleep 2

python << 'PYEOF'
import os, ray, traceback

ray.shutdown()
try:
    tmp = f"/tmp/ray_local_test_{os.getpid()}"
    os.makedirs(tmp, exist_ok=True)
    print(f"Init Ray with local temp dir: {tmp}")
    ray.init(num_gpus=8, _temp_dir=tmp)
    print(f"Resources: {ray.cluster_resources()}")

    @ray.remote(num_cpus=1)
    def ping():
        return "pong"

    result = ray.get(ping.remote(), timeout=60)
    print(f"Result: {result}")
    print("PASSED")
except Exception as e:
    print(f"FAILED: {e}")
    traceback.print_exc()
finally:
    try: ray.shutdown()
    except: pass
    import shutil
    shutil.rmtree(tmp, ignore_errors=True)
PYEOF

echo ""
echo "===== TEST 5: Try with runtime_env worker setup hook ====="
ray stop --force 2>/dev/null || true
rm -rf /tmp/ray /tmp/verl_ray_* 2>/dev/null || true
sleep 2

python << 'PYEOF'
import os, ray, traceback

ray.shutdown()
try:
    print("Init Ray with runtime_env (same as train script)...")
    ray.init(
        num_gpus=8,
        runtime_env={"env_vars": {"TOKENIZERS_PARALLELISM": "true", "NCCL_DEBUG": "WARN"}},
    )
    print(f"Resources: {ray.cluster_resources()}")

    @ray.remote(num_cpus=1)
    def ping():
        return "pong"

    result = ray.get(ping.remote(), timeout=60)
    print(f"Result: {result}")
    print("PASSED")
except Exception as e:
    print(f"FAILED: {e}")
    traceback.print_exc()
finally:
    try: ray.shutdown()
    except: pass
PYEOF

echo ""
echo "===== TEST 6: Check if AFS causes slow worker startup ====="
python << 'PYEOF'
import time, subprocess

# Time how long it takes to start a Python process from AFS
start = time.time()
result = subprocess.run(
    ['/mnt/afs/zhangyaolun/safe_model/tool/PettingLLMs/pettingllms_venv/bin/python', '-c', 'print("ok")'],
    capture_output=True, text=True, timeout=30
)
elapsed = time.time() - start
print(f"AFS Python startup time: {elapsed:.2f}s (output: {result.stdout.strip()})")

# Compare with local Python
start = time.time()
result = subprocess.run(
    ['/opt/conda/bin/python3', '-c', 'print("ok")'],
    capture_output=True, text=True, timeout=30
)
elapsed = time.time() - start
print(f"Local Python startup time: {elapsed:.2f}s (output: {result.stdout.strip()})")

# Time import of ray in AFS Python
start = time.time()
result = subprocess.run(
    ['/mnt/afs/zhangyaolun/safe_model/tool/PettingLLMs/pettingllms_venv/bin/python', '-c', 'import ray; print("ok")'],
    capture_output=True, text=True, timeout=60
)
elapsed = time.time() - start
print(f"AFS Python 'import ray' time: {elapsed:.2f}s")

PYEOF

echo ""
echo "============================================"
echo "  Debug V3 complete!"
echo "  Key things to look at:"
echo "  - TEST 2: Raylet crash logs (the WHY)"
echo "  - TEST 3 vs 4 vs 5: Which config causes the failure"
echo "  - TEST 6: AFS I/O speed"
echo "============================================"
