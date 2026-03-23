#!/bin/bash
cd /mnt/afs/zhangyaolun/safe_model/tool/PettingLLMs
source pettingllms_venv/bin/activate

echo "============================================"
echo "  Ray Debug V6 - Reproduce the actual failure"
echo "  $(date)"
echo "============================================"

echo ""
echo "===== TEST 1: Simulate what happens when training script runs without cleanup ====="
echo "First, start a Ray instance and kill it abruptly (like a failed training run)..."

# Start ray and kill it badly to create zombies
timeout 30 python -c "
import ray, os, signal, time
ray.init(num_gpus=8)
print(f'Ray started. PID={os.getpid()}')
time.sleep(3)
# Kill self without proper cleanup (simulates crash)
os.kill(os.getpid(), signal.SIGKILL)
" 2>&1 || true

sleep 3
echo ""
echo "Zombie/orphan processes after crash:"
ps aux | grep -E "(raylet|gcs_server)" | grep -v grep || echo "  None"

echo ""
echo "Now try starting Ray AGAIN without cleanup (this is what your training script does)..."
timeout 90 python -c "
import ray, os, json
tmp = f'/tmp/verl_ray_{os.getpid()}'
spill = f'/tmp/verl_spill_{os.getpid()}'
os.makedirs(tmp, exist_ok=True)
os.makedirs(spill, exist_ok=True)
conf = {'object_spilling_config': json.dumps({'type': 'filesystem', 'params': {'directory_path': [spill]}})}
ray.init(
    num_gpus=8,
    runtime_env={'env_vars': {'TOKENIZERS_PARALLELISM': 'true', 'NCCL_DEBUG': 'WARN'}},
    _temp_dir=tmp,
    _system_config=conf,
)
print(f'Ray init OK')
result = ray.get(ray.remote(lambda: 'pong').remote(), timeout=60)
print(f'Result: {result}')
print('TEST 1 PASSED - no cleanup needed, stale processes are NOT the issue')
ray.shutdown()
" 2>&1
TEST1=$?
if [ $TEST1 -ne 0 ]; then
    echo "TEST 1 FAILED - Stale processes DO cause the issue!"
fi

echo ""
echo "===== TEST 2: Clean up, then try again ====="
ray stop --force 2>/dev/null || true
# Kill any remaining ray processes
pkill -9 -f "raylet" 2>/dev/null || true
pkill -9 -f "gcs_server" 2>/dev/null || true
rm -rf /tmp/verl_ray_* /tmp/verl_spill_* /tmp/ray 2>/dev/null || true
sleep 3

timeout 90 python -c "
import ray, os, json
tmp = f'/tmp/verl_ray_{os.getpid()}'
spill = f'/tmp/verl_spill_{os.getpid()}'
os.makedirs(tmp, exist_ok=True)
os.makedirs(spill, exist_ok=True)
conf = {'object_spilling_config': json.dumps({'type': 'filesystem', 'params': {'directory_path': [spill]}})}
ray.init(
    num_gpus=8,
    runtime_env={'env_vars': {'TOKENIZERS_PARALLELISM': 'true', 'NCCL_DEBUG': 'WARN'}},
    _temp_dir=tmp,
    _system_config=conf,
)
print(f'Ray init OK')
result = ray.get(ray.remote(lambda: 'pong').remote(), timeout=60)
print(f'Result: {result}')
print('TEST 2 PASSED - cleanup fixes it!')
ray.shutdown()
" 2>&1
TEST2=$?
if [ $TEST2 -ne 0 ]; then
    echo "TEST 2 FAILED - even with cleanup, still broken"
fi

echo ""
echo "===== TEST 3: Simulate nohup launch (how training script runs) ====="
ray stop --force 2>/dev/null || true
pkill -9 -f "raylet" 2>/dev/null || true
pkill -9 -f "gcs_server" 2>/dev/null || true
rm -rf /tmp/verl_ray_* /tmp/verl_spill_* /tmp/ray 2>/dev/null || true
sleep 3

echo "Running via nohup (like the real training script)..."
nohup python -c "
import ray, os, json, sys
sys.stdout = open('/tmp/ray_nohup_test.log', 'w')
sys.stderr = sys.stdout
tmp = f'/tmp/verl_ray_{os.getpid()}'
spill = f'/tmp/verl_spill_{os.getpid()}'
os.makedirs(tmp, exist_ok=True)
os.makedirs(spill, exist_ok=True)
conf = {'object_spilling_config': json.dumps({'type': 'filesystem', 'params': {'directory_path': [spill]}})}
ray.init(
    num_gpus=8,
    runtime_env={'env_vars': {'TOKENIZERS_PARALLELISM': 'true', 'NCCL_DEBUG': 'WARN'}},
    _temp_dir=tmp,
    _system_config=conf,
)
print(f'Ray init OK', flush=True)
result = ray.get(ray.remote(lambda: 'pong').remote(), timeout=60)
print(f'Result: {result}', flush=True)
print('TEST 3 PASSED', flush=True)
ray.shutdown()
" </dev/null > /tmp/ray_nohup_test_out.log 2>&1 &

NOHUP_PID=$!
echo "Nohup PID: $NOHUP_PID"
echo "Waiting up to 90 seconds..."

for i in $(seq 1 30); do
    sleep 3
    if ! kill -0 $NOHUP_PID 2>/dev/null; then
        wait $NOHUP_PID 2>/dev/null
        EXIT_CODE=$?
        echo "Process exited after $((i*3))s with code $EXIT_CODE"
        break
    fi
    echo "  t=$((i*3))s: still running..."
done

echo ""
echo "--- nohup output ---"
cat /tmp/ray_nohup_test_out.log 2>/dev/null || echo "(no output)"
echo ""
echo "--- nohup internal log ---"
cat /tmp/ray_nohup_test.log 2>/dev/null || echo "(no log)"

echo ""
echo "===== TEST 4: Run the ACTUAL training script with cleanup prefix ====="
echo "(dry run - just test Ray init portion)"
ray stop --force 2>/dev/null || true
pkill -9 -f "raylet" 2>/dev/null || true
pkill -9 -f "gcs_server" 2>/dev/null || true
rm -rf /tmp/verl_ray_* /tmp/verl_spill_* /tmp/ray 2>/dev/null || true
sleep 3

echo "Setting up same env vars as training script..."
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export VLLM_USE_FLASHINFER_SAMPLER=0
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:False"
export VLLM_USE_V1=1
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_ENGINE_ITERATION_TIMEOUT_S=100000000000
export NCCL_IB_DISABLE=1
export NCCL_NET_GDR_LEVEL=0
export NCCL_TIMEOUT=3600
export NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=WARN
export NCCL_NVLS_ENABLE=0
export MAX_ROLLOUT_CONCURRENCY=16
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export VLLM_CUDAGRAPH_MODE=piecewise
export MAX_ROLLOUT_RETRIES=3

timeout 90 python -c "
import ray, os, json
print('Env vars set. Testing Ray init with same config as train.py...')
tmp = f'/tmp/verl_ray_{os.getpid()}'
spill = f'/tmp/verl_spill_{os.getpid()}'
os.makedirs(tmp, exist_ok=True)
os.makedirs(spill, exist_ok=True)
conf = {'object_spilling_config': json.dumps({'type': 'filesystem', 'params': {'directory_path': [spill]}})}
ray.init(
    num_gpus=8,
    runtime_env={'env_vars': {'TOKENIZERS_PARALLELISM': 'true', 'NCCL_DEBUG': 'WARN'}},
    _temp_dir=tmp,
    _system_config=conf,
)
print(f'Ray init OK. Resources: {ray.cluster_resources()}')

# Exact same pattern as train.py
num_cpus = max(8, int(ray.cluster_resources()['CPU'] * 0.1))
print(f'Creating remote task with num_cpus={num_cpus}')

@ray.remote(num_cpus=num_cpus)
def train_sim():
    from pettingllms.trainer.multi_agents_ppo_trainer import MultiAgentsPPOTrainer
    from verl.single_controller.ray import RayWorkerGroup
    from verl.workers.fsdp_workers import AsyncActorRolloutRefWorker
    import os
    return f'All imports OK, PID={os.getpid()}'

result = ray.get(train_sim.remote(), timeout=120)
print(f'Result: {result}')
print('TEST 4 PASSED')
ray.shutdown()
" 2>&1
TEST4=$?
if [ $TEST4 -ne 0 ]; then
    echo "TEST 4 FAILED (exit code $TEST4)"
fi

echo ""
echo "============================================"
echo "  RESULTS:"
echo "  TEST 1 (no cleanup, stale procs): $([ ${TEST1:-1} -eq 0 ] && echo 'PASSED' || echo 'FAILED')"
echo "  TEST 2 (with cleanup):            $([ ${TEST2:-1} -eq 0 ] && echo 'PASSED' || echo 'FAILED')"
echo "  TEST 3 (via nohup):               check output above"
echo "  TEST 4 (full env + imports):       $([ ${TEST4:-1} -eq 0 ] && echo 'PASSED' || echo 'FAILED')"
echo ""
echo "  If TEST 1 FAILED but TEST 2 PASSED:"
echo "    -> Add cleanup to your training script (see fix below)"
echo "  If TEST 3 FAILED:"
echo "    -> nohup causes the issue, run interactively instead"
echo "  If TEST 4 FAILED:"
echo "    -> Specific env vars or imports cause the issue"
echo ""
echo "  RECOMMENDED FIX: Add these lines at the TOP of train_design_tree_mix.sh:"
echo '    ray stop --force 2>/dev/null || true'
echo '    pkill -9 -f "raylet" 2>/dev/null || true'
echo '    pkill -9 -f "gcs_server" 2>/dev/null || true'
echo '    rm -rf /tmp/verl_ray_* /tmp/verl_spill_* /tmp/ray 2>/dev/null || true'
echo '    sleep 2'
echo "============================================"
