#!/bin/bash
# Stop the IBR training run and clean up all residual ray/vLLM processes
# so GPU memory is fully released. Use this instead of raw pkill to avoid
# leaving zombie workers that will OOM the next run.

echo "=== Before cleanup ==="
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv 2>/dev/null | head

echo ""
echo "=== Stopping pettingllms training process tree ==="
# -TERM first (gentle), then -KILL for any survivors
pkill -TERM -f "pettingllms.trainer.train" 2>/dev/null
pkill -TERM -f "ray::" 2>/dev/null
pkill -TERM -f "AsyncvLLMServer" 2>/dev/null
pkill -TERM -f "WorkerDict" 2>/dev/null
sleep 3
pkill -KILL -f "pettingllms.trainer.train" 2>/dev/null
pkill -KILL -f "ray::" 2>/dev/null
pkill -KILL -f "AsyncvLLMServer" 2>/dev/null
pkill -KILL -f "WorkerDict" 2>/dev/null

echo ""
echo "=== ray stop ==="
"${RAY_BIN:-ray}" stop --force 2>&1 | tail -5

sleep 2

echo ""
echo "=== After cleanup ==="
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv 2>/dev/null | head

echo ""
echo "Done. If GPU still shows residual processes belonging to YOUR user, kill them by PID:"
echo "  kill -9 <PID>"
