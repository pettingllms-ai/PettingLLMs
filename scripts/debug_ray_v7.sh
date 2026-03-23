#!/bin/bash
cd /mnt/afs/zhangyaolun/safe_model/tool/PettingLLMs
source pettingllms_venv/bin/activate

echo "============================================"
echo "  Ray Debug V7 - Isolate _system_config"
echo "  $(date)"
echo "============================================"

cleanup() {
    ray stop --force 2>/dev/null || true
    rm -rf /tmp/verl_ray_* /tmp/verl_spill_* /tmp/ray /tmp/ray_test_* 2>/dev/null || true
    sleep 3
}

cleanup

echo "===== TEST 1: Bare ray.init(num_gpus=8) ====="
timeout 90 python -c "
import ray
ray.init(num_gpus=8)
print(f'Resources: {ray.cluster_resources()}')
result = ray.get(ray.remote(lambda: 'pong').remote(), timeout=60)
print(f'Result: {result}')
print('TEST 1 PASSED')
ray.shutdown()
" 2>&1 || echo "TEST 1 FAILED"

cleanup

echo ""
echo "===== TEST 2: + _temp_dir only ====="
timeout 90 python -c "
import ray, os
tmp = f'/tmp/ray_test_{os.getpid()}'
os.makedirs(tmp, exist_ok=True)
ray.init(num_gpus=8, _temp_dir=tmp)
print(f'Resources: {ray.cluster_resources()}')
result = ray.get(ray.remote(lambda: 'pong').remote(), timeout=60)
print(f'Result: {result}')
print('TEST 2 PASSED')
ray.shutdown()
" 2>&1 || echo "TEST 2 FAILED"

cleanup

echo ""
echo "===== TEST 3: + runtime_env only ====="
timeout 90 python -c "
import ray
ray.init(num_gpus=8, runtime_env={'env_vars': {'TOKENIZERS_PARALLELISM': 'true', 'NCCL_DEBUG': 'WARN'}})
print(f'Resources: {ray.cluster_resources()}')
result = ray.get(ray.remote(lambda: 'pong').remote(), timeout=60)
print(f'Result: {result}')
print('TEST 3 PASSED')
ray.shutdown()
" 2>&1 || echo "TEST 3 FAILED"

cleanup

echo ""
echo "===== TEST 4: + _system_config (spilling) only ====="
timeout 90 python -c "
import ray, os, json
spill = f'/tmp/verl_spill_{os.getpid()}'
os.makedirs(spill, exist_ok=True)
conf = {'object_spilling_config': json.dumps({'type': 'filesystem', 'params': {'directory_path': [spill]}})}
ray.init(num_gpus=8, _system_config=conf)
print(f'Resources: {ray.cluster_resources()}')
result = ray.get(ray.remote(lambda: 'pong').remote(), timeout=60)
print(f'Result: {result}')
print('TEST 4 PASSED')
ray.shutdown()
" 2>&1 || echo "TEST 4 FAILED"

cleanup

echo ""
echo "===== TEST 5: + _temp_dir + _system_config (no runtime_env) ====="
timeout 90 python -c "
import ray, os, json
tmp = f'/tmp/verl_ray_{os.getpid()}'
spill = f'/tmp/verl_spill_{os.getpid()}'
os.makedirs(tmp, exist_ok=True)
os.makedirs(spill, exist_ok=True)
conf = {'object_spilling_config': json.dumps({'type': 'filesystem', 'params': {'directory_path': [spill]}})}
ray.init(num_gpus=8, _temp_dir=tmp, _system_config=conf)
print(f'Resources: {ray.cluster_resources()}')
result = ray.get(ray.remote(lambda: 'pong').remote(), timeout=60)
print(f'Result: {result}')
print('TEST 5 PASSED')
ray.shutdown()
" 2>&1 || echo "TEST 5 FAILED"

cleanup

echo ""
echo "===== TEST 6: FULL config (same as train.py) ====="
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
print(f'Resources: {ray.cluster_resources()}')
result = ray.get(ray.remote(lambda: 'pong').remote(), timeout=60)
print(f'Result: {result}')
print('TEST 6 PASSED')
ray.shutdown()
" 2>&1 || echo "TEST 6 FAILED"

cleanup

echo ""
echo "===== TEST 7: Use stable API (object_spilling_directory) instead of _system_config ====="
timeout 90 python -c "
import ray, os
tmp = f'/tmp/verl_ray_{os.getpid()}'
spill = f'/tmp/verl_spill_{os.getpid()}'
os.makedirs(tmp, exist_ok=True)
os.makedirs(spill, exist_ok=True)
ray.init(
    num_gpus=8,
    runtime_env={'env_vars': {'TOKENIZERS_PARALLELISM': 'true', 'NCCL_DEBUG': 'WARN'}},
    _temp_dir=tmp,
    object_spilling_directory=spill,
)
print(f'Resources: {ray.cluster_resources()}')
result = ray.get(ray.remote(lambda: 'pong').remote(), timeout=60)
print(f'Result: {result}')
print('TEST 7 PASSED')
ray.shutdown()
" 2>&1 || echo "TEST 7 FAILED"

cleanup

echo ""
echo "===== TEST 8: _temp_dir + runtime_env, NO spilling at all ====="
timeout 90 python -c "
import ray, os
tmp = f'/tmp/verl_ray_{os.getpid()}'
os.makedirs(tmp, exist_ok=True)
ray.init(
    num_gpus=8,
    runtime_env={'env_vars': {'TOKENIZERS_PARALLELISM': 'true', 'NCCL_DEBUG': 'WARN'}},
    _temp_dir=tmp,
)
print(f'Resources: {ray.cluster_resources()}')
result = ray.get(ray.remote(lambda: 'pong').remote(), timeout=60)
print(f'Result: {result}')
print('TEST 8 PASSED')
ray.shutdown()
" 2>&1 || echo "TEST 8 FAILED"

echo ""
echo "============================================"
echo "  SUMMARY: Which param combination breaks Ray?"
echo "  TEST 1: bare                    → check above"
echo "  TEST 2: +_temp_dir              → check above"
echo "  TEST 3: +runtime_env            → check above"
echo "  TEST 4: +_system_config         → check above"
echo "  TEST 5: +_temp_dir+_sys_config  → check above"
echo "  TEST 6: ALL (same as train.py)  → check above"
echo "  TEST 7: stable spilling API     → check above"
echo "  TEST 8: no spilling at all      → check above"
echo "============================================"
