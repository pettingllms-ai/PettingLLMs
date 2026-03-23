#!/bin/bash
cd /mnt/afs/zhangyaolun/safe_model/tool/PettingLLMs
source pettingllms_venv/bin/activate

echo "============================================"
echo "  Ray Debug V5 - WHY does raylet die?"
echo "  $(date)"
echo "============================================"

cleanup() {
    ray stop --force 2>/dev/null || true
    rm -rf /tmp/verl_ray_* /tmp/verl_spill_* /tmp/ray /tmp/ray_local_* 2>/dev/null || true
    sleep 2
}

echo ""
echo "===== INFO ====="
echo "Python: $(which python)"
echo "Ray: $(python -c 'import ray; print(ray.__version__)')"
RAY_DIR=$(python -c "import ray, os; print(os.path.dirname(ray.__file__))")
echo "Ray package dir: $RAY_DIR"
echo "Raylet binary: $(ls -la $RAY_DIR/core/src/ray/raylet/raylet 2>/dev/null || echo 'NOT FOUND at expected path')"
# Find actual raylet binary
RAYLET_BIN=$(find $RAY_DIR -name "raylet" -type f 2>/dev/null | head -1)
echo "Raylet binary found: $RAYLET_BIN"
echo "GCS binary: $(find $RAY_DIR -name "gcs_server" -type f 2>/dev/null | head -1)"

echo ""
echo "===== TEST 1: Start Ray, monitor raylet+gcs, dump logs after death ====="
cleanup
timeout 120 python << 'PYEOF'
import ray, os, time, subprocess, glob

pid = os.getpid()
tmp = f"/tmp/verl_ray_debug_{pid}"
os.makedirs(tmp, exist_ok=True)

ray.init(num_gpus=8, _temp_dir=tmp)
print(f"Ray init OK. PID={pid}")

# Find the raylet and gcs pids
def get_ray_procs():
    result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
    procs = {'raylet': [], 'gcs': [], 'raylet_zombie': [], 'gcs_zombie': []}
    for line in result.stdout.split('\n'):
        if 'raylet' in line and 'grep' not in line and 'python' not in line:
            if 'defunct' in line:
                procs['raylet_zombie'].append(line.split()[1])
            else:
                procs['raylet'].append(line.split()[1])
        if 'gcs_server' in line and 'grep' not in line and 'python' not in line:
            if 'defunct' in line:
                procs['gcs_zombie'].append(line.split()[1])
            else:
                procs['gcs'].append(line.split()[1])
    return procs

initial_procs = get_ray_procs()
print(f"Initial: raylet PIDs={initial_procs['raylet']}, gcs PIDs={initial_procs['gcs']}")

# Monitor every 3 seconds
for i in range(15):
    time.sleep(3)
    procs = get_ray_procs()
    elapsed = (i + 1) * 3
    status = f"t={elapsed:2d}s: raylet={procs['raylet']} gcs={procs['gcs']} " \
             f"raylet_zombie={procs['raylet_zombie']} gcs_zombie={procs['gcs_zombie']}"
    print(status)

    # Check if raylet died
    if not procs['raylet'] and initial_procs['raylet']:
        print(f"\n*** RAYLET DIED between t={elapsed-3}s and t={elapsed}s ***")

        # Check dmesg for OOM
        try:
            dmesg = subprocess.run(['dmesg', '-T'], capture_output=True, text=True, timeout=5)
            oom_lines = [l for l in dmesg.stdout.split('\n') if 'oom' in l.lower() or 'killed' in l.lower()]
            if oom_lines:
                print(f"\n--- dmesg OOM/killed entries ---")
                for l in oom_lines[-10:]:
                    print(f"  {l}")
        except:
            pass

        # Dump logs
        log_dir = f"{tmp}/session_latest/logs"
        print(f"\n--- {log_dir}/raylet.err ---")
        try:
            with open(f"{log_dir}/raylet.err") as f:
                content = f.read()
                print(content[-3000:] if content else "(empty)")
        except Exception as e:
            print(f"  Error reading: {e}")

        print(f"\n--- {log_dir}/raylet.out (last 60 lines) ---")
        try:
            with open(f"{log_dir}/raylet.out") as f:
                lines = f.readlines()
                for l in lines[-60:]:
                    print(f"  {l.rstrip()}")
        except Exception as e:
            print(f"  Error reading: {e}")

        print(f"\n--- {log_dir}/gcs_server.err ---")
        try:
            with open(f"{log_dir}/gcs_server.err") as f:
                content = f.read()
                print(content[-2000:] if content else "(empty)")
        except Exception as e:
            print(f"  Error reading: {e}")

        print(f"\n--- {log_dir}/gcs_server.out (last 30 lines) ---")
        try:
            with open(f"{log_dir}/gcs_server.out") as f:
                lines = f.readlines()
                for l in lines[-30:]:
                    print(f"  {l.rstrip()}")
        except Exception as e:
            print(f"  Error reading: {e}")

        # Also check monitor.err and dashboard logs
        for logname in ['monitor.err', 'monitor.log', 'dashboard.log', 'dashboard_agent.log']:
            logpath = f"{log_dir}/{logname}"
            if os.path.exists(logpath) and os.path.getsize(logpath) > 0:
                print(f"\n--- {logname} (last 20 lines) ---")
                with open(logpath) as f:
                    lines = f.readlines()
                    for l in lines[-20:]:
                        print(f"  {l.rstrip()}")

        # Check worker logs
        worker_errs = sorted(glob.glob(f"{log_dir}/worker-*.err"))
        for wf in worker_errs:
            if os.path.getsize(wf) > 0:
                print(f"\n--- {os.path.basename(wf)} (last 20 lines) ---")
                with open(wf) as f:
                    lines = f.readlines()
                    for l in lines[-20:]:
                        print(f"  {l.rstrip()}")

        break

    # Check if gcs died
    if not procs['gcs'] and initial_procs['gcs']:
        print(f"\n*** GCS SERVER DIED between t={elapsed-3}s and t={elapsed}s ***")
        print("GCS dying first would cause raylet to die too!")
        # Same log dump as above
        log_dir = f"{tmp}/session_latest/logs"
        for logname in ['gcs_server.err', 'gcs_server.out', 'raylet.err', 'raylet.out']:
            logpath = f"{log_dir}/{logname}"
            if os.path.exists(logpath):
                print(f"\n--- {logname} ---")
                with open(logpath) as f:
                    content = f.read()
                    print(content[-2000:] if content else "(empty)")
        break
else:
    print("\nRaylet survived 45 seconds! Trying remote task...")
    try:
        result = ray.get(ray.remote(lambda: "pong").remote(), timeout=30)
        print(f"Remote task: {result}")
    except Exception as e:
        print(f"Remote task failed: {e}")

try:
    ray.shutdown()
except:
    pass
PYEOF

echo ""
echo "===== TEST 2: Check /proc for raylet exit reason ====="
cleanup
echo "Starting Ray and using strace on raylet..."
timeout 120 python << 'PYEOF'
import ray, os, time, subprocess, signal

pid = os.getpid()
tmp = f"/tmp/verl_ray_debug2_{pid}"
os.makedirs(tmp, exist_ok=True)

ray.init(num_gpus=8, _temp_dir=tmp)
print(f"Ray init OK")

# Find raylet PID
result = subprocess.run(['pgrep', '-f', 'raylet/raylet'], capture_output=True, text=True)
raylet_pids = result.stdout.strip().split('\n')
raylet_pids = [p for p in raylet_pids if p]
print(f"Raylet PIDs: {raylet_pids}")

if raylet_pids:
    raylet_pid = raylet_pids[0]
    # Check /proc for raylet info
    try:
        with open(f"/proc/{raylet_pid}/status") as f:
            for line in f:
                if any(k in line for k in ['Name', 'Pid', 'VmRSS', 'VmSize', 'Threads']):
                    print(f"  {line.strip()}")
    except:
        print(f"  Could not read /proc/{raylet_pid}/status")

    # Wait for it to die and check exit status
    print(f"Waiting for raylet (PID {raylet_pid}) to die...")
    for i in range(15):
        time.sleep(3)
        try:
            # Check if process still exists
            os.kill(int(raylet_pid), 0)
            # Still alive, check memory
            try:
                with open(f"/proc/{raylet_pid}/status") as f:
                    for line in f:
                        if 'VmRSS' in line:
                            print(f"  t={((i+1)*3):2d}s: raylet alive, {line.strip()}")
                            break
            except:
                pass
        except ProcessLookupError:
            print(f"  t={((i+1)*3):2d}s: Raylet DEAD")
            # Try to get exit info
            try:
                result = subprocess.run(['wait', raylet_pid], capture_output=True, text=True, shell=True)
            except:
                pass
            break
        except PermissionError:
            print(f"  t={((i+1)*3):2d}s: raylet alive (permission error on signal)")

try:
    ray.shutdown()
except:
    pass
PYEOF

echo ""
echo "===== TEST 3: Try with RAY_TMPDIR and different socket dir ====="
cleanup
export RAY_TMPDIR=/tmp
export RAY_SOCKET_DIR=/tmp/ray_sockets_$$
mkdir -p $RAY_SOCKET_DIR
timeout 90 python -c "
import ray, os
print(f'RAY_TMPDIR={os.environ.get(\"RAY_TMPDIR\")}')
ray.init(num_gpus=8)
print(f'Ray init OK. Resources: {ray.cluster_resources()}')
result = ray.get(ray.remote(lambda: 'pong').remote(), timeout=60)
print(f'Result: {result}')
print('TEST 3 PASSED')
ray.shutdown()
" 2>&1 || echo "TEST 3 FAILED (exit code $?)"
unset RAY_TMPDIR RAY_SOCKET_DIR

echo ""
echo "===== TEST 4: Try with raylet binary copied to local disk ====="
cleanup
RAY_DIR=$(python -c "import ray, os; print(os.path.dirname(ray.__file__))")
echo "Copying ray core binaries to /tmp/ray_local_bin/..."
mkdir -p /tmp/ray_local_bin
# Copy the key binaries
for bin in raylet gcs_server; do
    src=$(find $RAY_DIR -name "$bin" -type f 2>/dev/null | head -1)
    if [ -n "$src" ]; then
        cp "$src" "/tmp/ray_local_bin/$bin"
        chmod +x "/tmp/ray_local_bin/$bin"
        echo "  Copied $src -> /tmp/ray_local_bin/$bin"
    fi
done

# Patch PATH so ray uses local binaries
export PATH="/tmp/ray_local_bin:$PATH"
timeout 90 python -c "
import ray, os
ray.init(num_gpus=8)
print(f'Ray init OK. Resources: {ray.cluster_resources()}')
result = ray.get(ray.remote(lambda: 'pong').remote(), timeout=60)
print(f'Result: {result}')
print('TEST 4 PASSED')
ray.shutdown()
" 2>&1 || echo "TEST 4 FAILED (exit code $?)"

echo ""
echo "===== TEST 5: Use system python's ray but our virtualenv packages ====="
cleanup
echo "Testing if system python + ray works with PYTHONPATH hack..."
export PYTHONPATH="/mnt/afs/zhangyaolun/safe_model/tool/PettingLLMs/pettingllms_venv/lib/python3.11/site-packages:$PYTHONPATH"
timeout 90 /opt/conda/bin/python3 -c "
import ray, os, sys
print(f'Python: {sys.executable}')
print(f'Ray: {ray.__version__}')
ray.init(num_gpus=8)
print(f'Ray init OK. Resources: {ray.cluster_resources()}')
result = ray.get(ray.remote(lambda: 'pong').remote(), timeout=60)
print(f'Result: {result}')
print('TEST 5 PASSED')
ray.shutdown()
" 2>&1 || echo "TEST 5 FAILED (exit code $?)"
unset PYTHONPATH

echo ""
echo "============================================"
echo "  DONE. Key things to check:"
echo "  - TEST 1: Shows EXACTLY what killed raylet"
echo "  - TEST 2: Memory usage before raylet dies"
echo "  - TEST 3: Does RAY_TMPDIR fix it?"
echo "  - TEST 4: Does copying binaries to local fix it?"
echo "  - TEST 5: Does system python's ray work?"
echo "============================================"
