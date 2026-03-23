#!/bin/bash
# Debug: check what the training process is stuck on during execution phase
cd /mnt/afs/zhangyaolun/safe_model/tool/PettingLLMs

echo "============================================"
echo "  Debug Stuck Execution Phase"
echo "  $(date)"
echo "============================================"

echo ""
echo "===== 1. Is the training process alive? ====="
ps aux | grep "train_multi_agents\|pettingllms.trainer.train" | grep -v grep | head -5

echo ""
echo "===== 2. Last log output (when did it stop?) ====="
echo "Last 5 lines:"
tail -5 0321_mix_try_run.log
echo ""
echo "Last modified:"
stat -c '%y' 0321_mix_try_run.log 2>/dev/null || ls -la 0321_mix_try_run.log

echo ""
echo "===== 3. Check Ray worker logs for executor activity ====="
for dir in /tmp/verl_ray_*/session_latest/logs; do
    if [ -d "$dir" ]; then
        echo "--- Ray log dir: $dir ---"

        # Check worker stdout for recent activity
        echo ""
        echo "Worker stdout with recent MAS/executor activity:"
        for f in "$dir"/worker-*.out; do
            if [ -s "$f" ] 2>/dev/null; then
                last_line=$(tail -1 "$f" 2>/dev/null)
                if echo "$last_line" | grep -qiE "executor|MAS|TREE|rollout|step|HTTP|API|timeout|error|SUCCESSSAVED"; then
                    echo "  === $(basename $f) ==="
                    tail -5 "$f" | sed 's/^/    /'
                    echo ""
                fi
            fi
        done 2>/dev/null

        # Check worker stderr for errors
        echo "Worker stderr (recent errors):"
        for f in "$dir"/worker-*.err; do
            if [ -s "$f" ] 2>/dev/null; then
                recent=$(tail -3 "$f" 2>/dev/null)
                if [ -n "$recent" ]; then
                    echo "  === $(basename $f) ==="
                    echo "$recent" | sed 's/^/    /'
                fi
            fi
        done 2>/dev/null

        break
    fi
done

echo ""
echo "===== 4. Check if env_workers (sandbox execution) are stuck ====="
# env_workers are Ray actors that run generated MAS code
# They call subprocess to execute Python scripts
ps aux | grep -E "sandbox|mas\.py|env_worker" | grep -v grep | head -10 || echo "  No sandbox/mas.py processes found"

echo ""
echo "===== 5. Check network connectivity to model server ====="
# The API server used during execution
echo "Testing API server at 10.119.17.93:8001..."
timeout 5 curl -s -o /dev/null -w "HTTP %{http_code} (%{time_total}s)" http://10.119.17.93:8001/v1/models 2>/dev/null || echo "  Cannot reach API server!"

echo ""
echo ""
echo "===== 6. Check for stuck Python subprocesses ====="
# MAS execution spawns Python subprocesses
ps aux | grep python | grep -v grep | grep -v "pettingllms.trainer.train" | grep -v "defunct" | head -20

echo ""
echo "===== 7. GPU utilization (is anything running on GPUs?) ====="
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader 2>/dev/null

echo ""
echo "===== 8. Thread dump of training process ====="
TRAIN_PID=$(ps aux | grep "train_multi_agents\|pettingllms.trainer.train" | grep -v grep | awk '{print $2}' | head -1)
if [ -n "$TRAIN_PID" ]; then
    echo "Training PID: $TRAIN_PID"
    echo "Threads:"
    ls /proc/$TRAIN_PID/task/ 2>/dev/null | wc -l
    echo ""
    echo "Open files (network sockets):"
    ls -la /proc/$TRAIN_PID/fd/ 2>/dev/null | grep socket | wc -l
    echo " sockets open"
    echo ""
    echo "Stack trace (py-spy or /proc):"
    cat /proc/$TRAIN_PID/stack 2>/dev/null | head -20 || echo "  Cannot read kernel stack"
else
    echo "  Training process not found!"
fi

echo ""
echo "===== 9. Check tmp_auto_mas for execution progress ====="
MAS_DIR=$(ls -td ./tmp_auto_mas/*/train/rollout_* 2>/dev/null | head -1)
if [ -n "$MAS_DIR" ]; then
    echo "Latest rollout dir: $MAS_DIR"
    echo "Files:"
    ls -la "$MAS_DIR" 2>/dev/null | head -10

    # Check output.txt for execution results
    if [ -f "$MAS_DIR/output.txt" ]; then
        echo ""
        echo "output.txt (last 10 lines):"
        tail -10 "$MAS_DIR/output.txt" | sed 's/^/  /'
    fi
else
    echo "  No rollout dirs found in tmp_auto_mas"
    # List what we have
    ls -d ./tmp_auto_mas/*/ 2>/dev/null | head -5
fi

echo ""
echo "============================================"
echo "  Summary: Check sections 2, 3, 5"
echo "  - Section 2: When did log output stop?"
echo "  - Section 3: Are Ray workers doing anything?"
echo "  - Section 5: Can we reach the API server?"
echo "============================================"
