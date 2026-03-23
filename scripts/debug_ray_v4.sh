#!/bin/bash
cd /mnt/afs/zhangyaolun/safe_model/tool/PettingLLMs
source pettingllms_venv/bin/activate

echo "============================================"
echo "  Ray Debug V4 - Isolated Subprocess Tests"
echo "  $(date)"
echo "============================================"

# Cleanup between tests
cleanup() {
    ray stop --force 2>/dev/null || true
    rm -rf /tmp/verl_ray_* /tmp/verl_spill_* /tmp/ray /tmp/ray_local_* 2>/dev/null || true
    sleep 2
}

cleanup

echo ""
echo "===== TEST 1: Minimal ray.init (no options) ====="
timeout 90 python -c "
import ray
ray.init()
print(f'Resources: {ray.cluster_resources()}')
result = ray.get(ray.remote(lambda: 'pong').remote(), timeout=60)
print(f'Result: {result}')
print('PASSED')
ray.shutdown()
" 2>&1 || echo "FAILED (exit code $?)"

cleanup

echo ""
echo "===== TEST 2: ray.init with num_gpus=8 only ====="
timeout 90 python -c "
import ray
ray.init(num_gpus=8)
print(f'Resources: {ray.cluster_resources()}')
result = ray.get(ray.remote(lambda: 'pong').remote(), timeout=60)
print(f'Result: {result}')
print('PASSED')
ray.shutdown()
" 2>&1 || echo "FAILED (exit code $?)"

cleanup

echo ""
echo "===== TEST 3: ray.init with _temp_dir=/tmp/... ====="
timeout 90 python -c "
import ray, os
tmp = f'/tmp/ray_test_{os.getpid()}'
os.makedirs(tmp, exist_ok=True)
ray.init(num_gpus=8, _temp_dir=tmp)
print(f'Resources: {ray.cluster_resources()}')
result = ray.get(ray.remote(lambda: 'pong').remote(), timeout=60)
print(f'Result: {result}')
print('PASSED')
ray.shutdown()
" 2>&1 || echo "FAILED (exit code $?)"

cleanup

echo ""
echo "===== TEST 4: ray.init with runtime_env ====="
timeout 90 python -c "
import ray
ray.init(num_gpus=8, runtime_env={'env_vars': {'TOKENIZERS_PARALLELISM': 'true'}})
print(f'Resources: {ray.cluster_resources()}')
result = ray.get(ray.remote(lambda: 'pong').remote(), timeout=60)
print(f'Result: {result}')
print('PASSED')
ray.shutdown()
" 2>&1 || echo "FAILED (exit code $?)"

cleanup

echo ""
echo "===== TEST 5: ray.init with _system_config (spilling) ====="
timeout 90 python -c "
import ray, os, json
spill = f'/tmp/verl_spill_{os.getpid()}'
os.makedirs(spill, exist_ok=True)
conf = {'object_spilling_config': json.dumps({'type': 'filesystem', 'params': {'directory_path': [spill]}})}
ray.init(num_gpus=8, _system_config=conf)
print(f'Resources: {ray.cluster_resources()}')
result = ray.get(ray.remote(lambda: 'pong').remote(), timeout=60)
print(f'Result: {result}')
print('PASSED')
ray.shutdown()
" 2>&1 || echo "FAILED (exit code $?)"

cleanup

echo ""
echo "===== TEST 6: ray.init with FULL config (same as train.py) ====="
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
print('PASSED')
ray.shutdown()
" 2>&1 || echo "FAILED (exit code $?)"

cleanup

echo ""
echo "===== TEST 7: Same test with system Python (Ray 2.46.0) ====="
timeout 90 /opt/conda/bin/python3 -c "
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
print('PASSED')
ray.shutdown()
" 2>&1 || echo "FAILED (exit code $?)"

cleanup

echo ""
echo "===== TEST 8: AFS vs local Python startup speed ====="
python -c "
import time, subprocess

cmds = [
    ('AFS venv python', ['/mnt/afs/zhangyaolun/safe_model/tool/PettingLLMs/pettingllms_venv/bin/python', '-c', 'print(\"ok\")']),
    ('Local conda python', ['/opt/conda/bin/python3', '-c', 'print(\"ok\")']),
    ('AFS python import ray', ['/mnt/afs/zhangyaolun/safe_model/tool/PettingLLMs/pettingllms_venv/bin/python', '-c', 'import ray; print(\"ok\")']),
    ('Local python import ray', ['/opt/conda/bin/python3', '-c', 'import ray; print(\"ok\")']),
]
for label, cmd in cmds:
    start = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    elapsed = time.time() - start
    print(f'{label}: {elapsed:.2f}s (status={r.returncode})')
"

echo ""
echo "===== TEST 9: Check raylet lifespan ====="
echo "Starting Ray and monitoring raylet for 45 seconds..."
timeout 90 python -c "
import ray, os, time, subprocess

ray.init(num_gpus=8)
print(f'Ray started. Resources: {ray.cluster_resources()}')

for i in range(9):
    time.sleep(5)
    result = subprocess.run(['pgrep', '-a', 'raylet'], capture_output=True, text=True)
    lines = [l for l in result.stdout.strip().split('\n') if l and 'defunct' not in l]
    zombies = [l for l in result.stdout.strip().split('\n') if 'defunct' in l]
    elapsed = (i + 1) * 5
    print(f'  t={elapsed:2d}s: {len(lines)} alive raylet(s), {len(zombies)} zombie(s)')
    if not lines:
        print('  RAYLET DIED! This is the root cause.')
        # Check logs
        import glob
        for logdir in glob.glob('/tmp/ray/session_latest/logs'):
            err_file = os.path.join(logdir, 'raylet.err')
            out_file = os.path.join(logdir, 'raylet.out')
            if os.path.exists(err_file):
                print(f'  --- raylet.err ---')
                with open(err_file) as f:
                    print(f.read()[-2000:] or '(empty)')
            if os.path.exists(out_file):
                print(f'  --- raylet.out (last 30 lines) ---')
                with open(out_file) as f:
                    for line in f.readlines()[-30:]:
                        print(f'  {line.rstrip()}')
        break
else:
    print('  Raylet survived 45 seconds - trying remote task...')
    try:
        result = ray.get(ray.remote(lambda: 'pong').remote(), timeout=30)
        print(f'  Remote task result: {result}')
    except Exception as e:
        print(f'  Remote task failed: {e}')

ray.shutdown()
" 2>&1 || echo "FAILED (exit code $?)"

echo ""
echo "============================================"
echo "  Summary: Check which tests PASSED vs FAILED"
echo "  - If TEST 1-6 all fail: Ray 2.48.0 + AFS python is broken"
echo "  - If TEST 7 passes: Ray version issue (2.48 vs 2.46)"
echo "  - If TEST 1 passes but later tests fail: specific config triggers it"
echo "  - TEST 9 shows exactly when raylet dies"
echo "============================================"
