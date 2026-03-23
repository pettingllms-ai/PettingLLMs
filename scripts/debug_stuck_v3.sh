#!/bin/bash
# Debug: why execution phase stuck, GPU 0% after designs complete
echo "============================================"
echo "  Debug Stuck v3 - $(date)"
echo "============================================"

echo ""
echo "===== 1. Training process ====="
ps aux | grep train_multi_agents | grep -v grep

TRAIN_PID=$(ps aux | grep "ray::train_multi_agents" | grep -v grep | awk '{print $2}' | head -1)
if [ -n "$TRAIN_PID" ]; then
    echo "PID: $TRAIN_PID"
    echo "Working dir: $(readlink /proc/$TRAIN_PID/cwd 2>/dev/null)"
fi

echo ""
echo "===== 2. Find tmp_auto_mas ====="
CWD=$(readlink /proc/$TRAIN_PID/cwd 2>/dev/null)
for d in "$CWD/tmp_auto_mas" "/mnt/afs/zhangyaolun/safe_model/tool/PettingLLMs/tmp_auto_mas" "/app/tmp_auto_mas" "/tmp/tmp_auto_mas"; do
    if [ -d "$d" ]; then
        echo "Found: $d"
        ls -lt "$d"/*/train/rollout_*/output.txt 2>/dev/null | head -5
        echo ""
        LATEST=$(ls -t "$d"/*/train/rollout_*/output.txt 2>/dev/null | head -1)
        if [ -n "$LATEST" ]; then
            echo "Latest output: $LATEST ($(wc -c < "$LATEST") bytes)"
            head -20 "$LATEST"
        fi
        echo ""
        grep "_api_base" "$d"/*/train/rollout_0/mas.py 2>/dev/null | head -1
        break
    fi
done

echo ""
echo "===== 3. Ray worker logs ====="
for dir in /tmp/verl_ray_*/session_latest/logs; do
    if [ -d "$dir" ]; then
        echo "Log dir: $dir"
        echo ""

        # Find workers with train_multi_agents or executor output
        echo "--- train_multi_agents worker (last 30 lines) ---"
        for f in "$dir"/worker-*.out; do
            if [ -s "$f" ] 2>/dev/null; then
                if grep -q "train_multi_agents\|EXECUTOR\|TREE DESIGN\|env_worker\|MAS execution" "$f" 2>/dev/null; then
                    echo "File: $(basename $f)"
                    tail -30 "$f"
                    echo ""
                    break
                fi
            fi
        done 2>/dev/null

        # Find workers with errors
        echo "--- Workers with errors ---"
        for f in "$dir"/worker-*.out; do
            if [ -s "$f" ] 2>/dev/null; then
                if grep -qiE "Traceback|Exception|Error" "$f" 2>/dev/null; then
                    echo "File: $(basename $f)"
                    grep -A5 "Traceback\|Exception\|Error" "$f" 2>/dev/null | tail -20
                    echo ""
                fi
            fi
        done 2>/dev/null

        # Stderr
        echo "--- Non-empty stderr ---"
        for f in "$dir"/worker-*.err; do
            if [ -s "$f" ] 2>/dev/null; then
                sz=$(wc -c < "$f")
                if [ "$sz" -gt 100 ]; then
                    echo "File: $(basename $f) ($sz bytes)"
                    tail -10 "$f"
                    echo ""
                fi
            fi
        done 2>/dev/null

        break
    fi
done

echo ""
echo "===== 4. All Ray actors ====="
ps aux | grep "ray::" | grep -v grep | grep -v defunct | head -20

echo ""
echo "===== 5. GPU status ====="
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader 2>/dev/null

echo ""
echo "============================================"
echo "  Done - $(date)"
echo "============================================"
