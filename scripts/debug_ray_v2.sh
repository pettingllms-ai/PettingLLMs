#!/bin/bash
set -e

cd /mnt/afs/zhangyaolun/safe_model/tool/PettingLLMs
source pettingllms_venv/bin/activate

echo "============================================"
echo "  Ray Debug V2 - In Correct Virtualenv"
echo "  $(date)"
echo "============================================"

echo ""
echo "===== 1. ENVIRONMENT ====="
echo "Python: $(which python)"
echo "Python version: $(python --version)"
echo "VIRTUAL_ENV: $VIRTUAL_ENV"

echo ""
echo "===== 2. PACKAGE IMPORTS ====="
python -c "
import sys
print(f'sys.executable: {sys.executable}')

# Core packages
for pkg in ['ray', 'torch', 'vllm', 'verl', 'pettingllms']:
    try:
        mod = __import__(pkg)
        ver = getattr(mod, '__version__', 'no version attr')
        print(f'{pkg}: {ver}')
    except Exception as e:
        print(f'{pkg}: IMPORT FAILED - {e}')

# Critical submodules
for mod_path in [
    'verl.single_controller.ray',
    'verl.workers.fsdp_workers',
    'pettingllms.trainer.multi_agents_ppo_trainer',
    'pettingllms.utils.ray_utils',
    'pettingllms.utils.clean_up',
]:
    try:
        __import__(mod_path)
        print(f'{mod_path}: OK')
    except Exception as e:
        print(f'{mod_path}: FAILED - {e}')
"

echo ""
echo "===== 3. CLEAN UP ZOMBIE RAY PROCESSES ====="
echo "Zombie ray processes before cleanup:"
ps aux | grep -E "(raylet|gcs_server|ray::)" | grep -v grep || echo "  None"

echo "Running: ray stop --force"
ray stop --force 2>/dev/null || true
echo "Cleaning stale temp dirs..."
rm -rf /tmp/verl_ray_* /tmp/verl_spill_* /tmp/ray 2>/dev/null || true

echo "Zombie ray processes after cleanup:"
ps aux | grep -E "(raylet|gcs_server|ray::)" | grep -v grep || echo "  None (good)"

echo ""
echo "===== 4. RAY INIT + REMOTE TASK TEST ====="
python << 'PYEOF'
import os
import sys
import json
import traceback
import ray

# ---- Test A: Basic ray init + remote task ----
print("--- Test A: Ray init + simple remote task ---")
try:
    ray.shutdown()
    pid = os.getpid()
    ray_tmp_dir = f"/tmp/verl_ray_debug_{pid}"
    ray_spill_dir = f"/tmp/verl_spill_debug_{pid}"
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
    print(f"  Ray init OK. Resources: {ray.cluster_resources()}")

    @ray.remote(num_cpus=1)
    def simple_task():
        return f"OK from PID={os.getpid()}, python={sys.executable}"

    result = ray.get(simple_task.remote(), timeout=30)
    print(f"  Simple task OK: {result}")
    ray.shutdown()
    import shutil
    shutil.rmtree(ray_tmp_dir, ignore_errors=True)
    shutil.rmtree(ray_spill_dir, ignore_errors=True)
    print("  PASSED")
except Exception as e:
    print(f"  FAILED: {e}")
    traceback.print_exc()
    try: ray.shutdown()
    except: pass

# ---- Test B: Remote task with heavy imports (like train.py) ----
print()
print("--- Test B: Remote task with heavy imports (mimics train.py) ---")
try:
    ray.shutdown()
    pid = os.getpid()
    ray_tmp_dir = f"/tmp/verl_ray_debug_{pid}"
    ray_spill_dir = f"/tmp/verl_spill_debug_{pid}"
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

    # Mimic train.py: request many CPUs like the real code does
    cpu_count = ray.cluster_resources().get("CPU", 0)
    num_cpus_request = max(8, int(cpu_count * 0.1))
    print(f"  Requesting {num_cpus_request} CPUs (same as train.py)")

    @ray.remote(num_cpus=num_cpus_request)
    def heavy_import_task():
        """This mimics what train_multi_agents does on startup"""
        import os, sys
        results = {}
        results['python'] = sys.executable
        results['pid'] = os.getpid()

        # Try importing exactly what train_multi_agents imports
        try:
            from verl.utils.fs import copy_local_path_from_hdfs
            results['verl.utils.fs'] = 'OK'
        except Exception as e:
            results['verl.utils.fs'] = str(e)

        try:
            from verl.utils import hf_tokenizer, hf_processor
            results['verl.utils.hf'] = 'OK'
        except Exception as e:
            results['verl.utils.hf'] = str(e)

        try:
            from pettingllms.verl.ray_trainer import ResourcePoolManager, Role
            results['pettingllms.verl.ray_trainer'] = 'OK'
        except Exception as e:
            results['pettingllms.verl.ray_trainer'] = str(e)

        try:
            from verl.single_controller.ray import RayWorkerGroup
            results['RayWorkerGroup'] = 'OK'
        except Exception as e:
            results['RayWorkerGroup'] = str(e)

        try:
            from verl.workers.fsdp_workers import AsyncActorRolloutRefWorker
            results['AsyncActorRolloutRefWorker'] = 'OK'
        except Exception as e:
            results['AsyncActorRolloutRefWorker'] = str(e)

        try:
            from pettingllms.trainer.multi_agents_ppo_trainer import MultiAgentsPPOTrainer
            results['MultiAgentsPPOTrainer'] = 'OK'
        except Exception as e:
            results['MultiAgentsPPOTrainer'] = str(e)

        return results

    print("  Submitting heavy import task...")
    result = ray.get(heavy_import_task.remote(), timeout=120)
    print("  Results:")
    for k, v in result.items():
        status = "OK" if v == "OK" else f"FAILED: {v}"
        print(f"    {k}: {status}")
    ray.shutdown()
    import shutil
    shutil.rmtree(ray_tmp_dir, ignore_errors=True)
    shutil.rmtree(ray_spill_dir, ignore_errors=True)
    print("  PASSED")
except Exception as e:
    print(f"  FAILED: {e}")
    traceback.print_exc()
    try: ray.shutdown()
    except: pass

# ---- Test C: Check if Ray workers use the same Python ----
print()
print("--- Test C: Verify Ray workers use correct Python/virtualenv ---")
try:
    ray.shutdown()
    pid = os.getpid()
    ray_tmp_dir = f"/tmp/verl_ray_debug_{pid}"
    os.makedirs(ray_tmp_dir, exist_ok=True)

    ray.init(num_gpus=8, _temp_dir=ray_tmp_dir)

    @ray.remote
    def check_env():
        import sys, os
        return {
            'executable': sys.executable,
            'sys.path[:5]': sys.path[:5],
            'VIRTUAL_ENV': os.environ.get('VIRTUAL_ENV', 'NOT SET'),
            'PATH_first': os.environ.get('PATH', '').split(':')[0],
            'cwd': os.getcwd(),
        }

    result = ray.get(check_env.remote(), timeout=30)
    print(f"  Driver python: {sys.executable}")
    print(f"  Driver VIRTUAL_ENV: {os.environ.get('VIRTUAL_ENV', 'NOT SET')}")
    print(f"  Worker python: {result['executable']}")
    print(f"  Worker VIRTUAL_ENV: {result['VIRTUAL_ENV']}")
    print(f"  Worker PATH first: {result['PATH_first']}")
    print(f"  Worker cwd: {result['cwd']}")
    print(f"  Worker sys.path[:5]: {result['sys.path[:5]']}")

    if result['executable'] != sys.executable:
        print("  WARNING: Worker uses DIFFERENT Python than driver!")
        print("  This can cause import failures in remote tasks!")
    else:
        print("  OK: Worker uses same Python as driver")

    ray.shutdown()
    import shutil
    shutil.rmtree(ray_tmp_dir, ignore_errors=True)
    print("  PASSED")
except Exception as e:
    print(f"  FAILED: {e}")
    traceback.print_exc()
    try: ray.shutdown()
    except: pass

# ---- Test D: Full simulation of train.py's run_ppo ----
print()
print("--- Test D: Full simulation of train.py flow ---")
try:
    ray.shutdown()
    from pettingllms.utils.ray_utils import init_ray_with_temp_dirs
    from pettingllms.utils.clean_up import cleanup_ray_runtime

    # Create a minimal mock config
    from omegaconf import OmegaConf
    mock_config = OmegaConf.create({
        'resource': {'n_gpus_per_node': 8}
    })

    print("  Calling init_ray_with_temp_dirs (same as train.py)...")
    init_ray_with_temp_dirs(mock_config)
    print(f"  Ray init OK. Resources: {ray.cluster_resources()}")

    # This is exactly what train.py does
    num_cpus = max(8, int(ray.cluster_resources()["CPU"] * 0.1))
    print(f"  Creating remote function with num_cpus={num_cpus}")

    @ray.remote(num_cpus=num_cpus)
    def fake_train():
        import os, sys
        # Import the real training dependencies
        from pettingllms.trainer.multi_agents_ppo_trainer import MultiAgentsPPOTrainer
        from verl.single_controller.ray import RayWorkerGroup
        from verl.workers.fsdp_workers import AsyncActorRolloutRefWorker
        return f"All imports OK in worker PID={os.getpid()}"

    print("  Submitting fake_train.remote()...")
    result = ray.get(fake_train.remote(), timeout=120)
    print(f"  Result: {result}")
    print("  PASSED - The training flow should work!")
    cleanup_ray_runtime()
except Exception as e:
    print(f"  FAILED: {e}")
    traceback.print_exc()
    try:
        cleanup_ray_runtime()
    except:
        try: ray.shutdown()
        except: pass

PYEOF

echo ""
echo "===== 5. CHECK CRASHED RUN LOGS ====="
for dir in /tmp/verl_ray_1136/session_*/logs /tmp/verl_ray_31/session_*/logs; do
    if [ -d "$dir" ]; then
        echo "--- $dir ---"
        echo "Raylet stderr:"
        cat "$dir/raylet.err" 2>/dev/null | tail -30 || echo "  (empty)"
        echo "Raylet stdout (last 30):"
        cat "$dir/raylet.out" 2>/dev/null | tail -30 || echo "  (empty)"
        echo "Worker errors:"
        for f in "$dir"/worker-*.err; do
            [ -s "$f" ] && echo "=== $(basename $f) ===" && tail -20 "$f" && echo
        done 2>/dev/null || echo "  No worker error logs"
        echo ""
    fi
done

echo ""
echo "============================================"
echo "  Debug V2 complete!"
echo "============================================"
