#!/bin/bash
# Quick check: are MAS executions making progress?
echo "=== $(date) ==="

echo ""
echo "=== Active Python subprocesses (mas.py / script.py) ==="
ps aux | grep -E "script\.py|mas\.py" | grep -v grep | wc -l
echo " subprocess(es) running"
ps aux | grep -E "script\.py|mas\.py" | grep -v grep | head -5

echo ""
echo "=== Temp execution dirs ==="
TMPCOUNT=$(ls -d /mnt/afs/zhangyaolun/safe_model/tool/PettingLLMs/tmp/pllm_exec_* 2>/dev/null | wc -l)
echo "$TMPCOUNT temp dirs exist"
if [ "$TMPCOUNT" -gt 0 ]; then
    echo "Latest 5:"
    ls -ltd /mnt/afs/zhangyaolun/safe_model/tool/PettingLLMs/tmp/pllm_exec_* 2>/dev/null | head -5
    echo ""
    LATEST=$(ls -td /mnt/afs/zhangyaolun/safe_model/tool/PettingLLMs/tmp/pllm_exec_* 2>/dev/null | head -1)
    if [ -n "$LATEST" ]; then
        echo "Latest dir: $LATEST"
        ls -la "$LATEST"/ 2>/dev/null
        echo ""
        if [ -f "$LATEST/stdout.txt" ]; then
            SZ=$(wc -c < "$LATEST/stdout.txt")
            echo "stdout.txt: $SZ bytes"
            [ "$SZ" -gt 0 ] && tail -5 "$LATEST/stdout.txt"
        fi
        if [ -f "$LATEST/stderr.txt" ]; then
            SZ=$(wc -c < "$LATEST/stderr.txt")
            echo "stderr.txt: $SZ bytes"
            [ "$SZ" -gt 0 ] && tail -5 "$LATEST/stderr.txt"
        fi
    fi
fi

echo ""
echo "=== GPU utilization ==="
nvidia-smi --query-gpu=index,utilization.gpu --format=csv,noheader 2>/dev/null

echo ""
echo "=== vLLM server alive? ==="
VLLM_PORT=$(grep "Async vLLM Server running at" /mnt/afs/zhangyaolun/safe_model/tool/PettingLLMs/0321_mix_try_run.log | grep -oP ':\K\d+')
if [ -n "$VLLM_PORT" ]; then
    echo "vLLM port: $VLLM_PORT"
    timeout 3 curl -s -o /dev/null -w "HTTP %{http_code} (%{time_total}s)\n" "http://localhost:$VLLM_PORT/v1/models" 2>/dev/null || echo "Cannot reach vLLM"
else
    echo "Cannot find vLLM port"
fi
