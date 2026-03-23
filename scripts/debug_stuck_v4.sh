#!/bin/bash
# Debug v4: lightweight, avoid slow AFS operations
echo "=== Debug v4 - $(date) ==="

echo ""
echo "=== 1. Ray actors ==="
ps aux | grep "ray::" | grep -v grep | grep -v defunct

echo ""
echo "=== 2. GPU ==="
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader 2>/dev/null

echo ""
echo "=== 3. Ray worker logs ==="
LOGDIR=$(ls -d /tmp/verl_ray_*/session_latest/logs 2>/dev/null | head -1)
if [ -z "$LOGDIR" ]; then
    echo "No ray log dir found"
    exit 1
fi
echo "Log dir: $LOGDIR"

echo ""
echo "--- Main worker (train_multi_agents) last 40 lines ---"
# Find the worker file for PID 11730
MAIN_LOG=$(grep -l "TREE DESIGN\|train_multi_agents\|EXECUTOR" "$LOGDIR"/worker-*.out 2>/dev/null | head -1)
if [ -n "$MAIN_LOG" ]; then
    echo "File: $MAIN_LOG"
    tail -40 "$MAIN_LOG"
else
    echo "Not found, showing largest worker log:"
    ls -lS "$LOGDIR"/worker-*.out 2>/dev/null | head -3
    BIGGEST=$(ls -S "$LOGDIR"/worker-*.out 2>/dev/null | head -1)
    [ -n "$BIGGEST" ] && tail -40 "$BIGGEST"
fi

echo ""
echo "--- Workers with Traceback/Error (last 20 lines each) ---"
grep -l "Traceback" "$LOGDIR"/worker-*.out 2>/dev/null | head -5 | while read f; do
    echo ""
    echo "FILE: $(basename $f)"
    tail -20 "$f"
done

echo ""
echo "--- Stderr (non-trivial) ---"
for f in "$LOGDIR"/worker-*.err; do
    [ -s "$f" ] 2>/dev/null || continue
    sz=$(wc -c < "$f")
    [ "$sz" -lt 200 ] && continue
    echo "FILE: $(basename $f) ($sz bytes) - last 10 lines:"
    tail -10 "$f"
    echo ""
done 2>/dev/null

echo ""
echo "=== Done - $(date) ==="
