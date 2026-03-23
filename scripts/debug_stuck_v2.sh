#!/bin/bash
# Debug: why execution phase is stuck (GPU 0% after designs complete)
echo "============================================"
echo "  Debug Stuck Execution Phase v2"
echo "  $(date)"
echo "============================================"

echo ""
echo "===== 1. Training process alive? ====="
ps aux | grep train_multi_agents | grep -v grep

echo ""
echo "===== 2. Find tmp_auto_mas ====="
find / -name "tmp_auto_mas" -type d 2>/dev/null | head -5

echo ""
echo "===== 3. Check latest output.txt files ====="
MAS_DIR=$(find / -path "*/tmp_auto_mas/*/train/rollout_0/output.txt" 2>/dev/null | head -1)
if [ -n "$MAS_DIR" ]; then
    BASE=$(dirname $(dirname "$MAS_DIR"))
    echo "Found: $BASE"
    echo ""
    echo "Latest output files:"
    ls -lt "$BASE"/rollout_*/output.txt 2>/dev/null | head -10
    echo ""
    echo "Content of latest output.txt:"
    LATEST=$(ls -t "$BASE"/rollout_*/output.txt 2>/dev/null | head -1)
    if [ -n "$LATEST" ]; then
        echo "File: $LATEST ($(wc -c < "$LATEST") bytes)"
        head -30 "$LATEST"
    fi
    echo ""
    echo "Content of rollout_0/mas.py _api_base line:"
    grep "_api_base" "$BASE/rollout_0/mas.py" 2>/dev/null || echo "  no mas.py found"
else
    echo "  No output.txt found anywhere"
fi

echo ""
echo "===== 4. Ray worker logs - errors ====="
for dir in /tmp/verl_ray_*/session_latest/logs; do
    if [ -d "$dir" ]; then
        echo "Log dir: $dir"
        echo ""
        echo "--- Worker stdout with errors/exceptions ---"
        for f in "$dir"/worker-*.out; do
            if [ -s "$f" ] 2>/dev/null; then
                if grep -qiE "Error|Exception|Traceback|timeout|failed" "$f" 2>/dev/null; then
                    echo ""
                    echo "=== $(basename $f) (last 30 lines) ==="
                    tail -30 "$f"
                fi
            fi
        done 2>/dev/null

        echo ""
        echo "--- Worker stderr (non-empty) ---"
        for f in "$dir"/worker-*.err; do
            if [ -s "$f" ] 2>/dev/null; then
                echo ""
                echo "=== $(basename $f) (last 10 lines) ==="
                tail -10 "$f"
            fi
        done 2>/dev/null

        echo ""
        echo "--- Raylet errors ---"
        tail -10 "$dir/raylet.err" 2>/dev/null

        break
    fi
done

echo ""
echo "===== 5. Check env_worker Ray actors ====="
# env_workers run mas.py subprocesses
ps aux | grep -E "ray::EnvWorker|ray::SandboxWorker|ray::MathWorker" | grep -v grep | head -10 || echo "  No env_worker actors found"

echo ""
echo "===== 6. Any Python subprocesses running? ====="
ps aux | grep python | grep -v grep | grep -v defunct | grep -v jupyter | grep -v tensorboard | head -20

echo ""
echo "===== 7. GPU status ====="
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader 2>/dev/null

echo ""
echo "============================================"
echo "  Done"
echo "============================================"
