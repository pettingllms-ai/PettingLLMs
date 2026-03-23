#!/bin/bash
# Debug: why mas.py subprocess times out after 1200s
echo "=== Debug Timeout - $(date) ==="

echo ""
echo "=== 1. Are any script.py subprocesses running? ==="
ps aux | grep "script\.py" | grep -v grep

echo ""
echo "=== 2. Temp exec dirs ==="
ls -ltd tmp/pllm_exec_* 2>/dev/null | head -5
LATEST=$(ls -td tmp/pllm_exec_* 2>/dev/null | head -1)
if [ -n "$LATEST" ]; then
    echo ""
    echo "Latest: $LATEST"
    ls -la "$LATEST"/ 2>/dev/null
    echo ""
    echo "--- stdout.txt ($(wc -c < "$LATEST/stdout.txt" 2>/dev/null || echo 0) bytes) ---"
    cat "$LATEST/stdout.txt" 2>/dev/null | head -30
    echo ""
    echo "--- stderr.txt ($(wc -c < "$LATEST/stderr.txt" 2>/dev/null || echo 0) bytes) ---"
    cat "$LATEST/stderr.txt" 2>/dev/null | head -30
fi

echo ""
echo "=== 3. vLLM server reachable from localhost? ==="
VLLM_ADDR=$(grep "Async vLLM Server running at" 0321_mix_try_run.log 2>/dev/null | grep -oP '\d+\.\d+\.\d+\.\d+:\d+')
if [ -n "$VLLM_ADDR" ]; then
    echo "vLLM at: $VLLM_ADDR"
    echo "Test localhost..."
    timeout 5 curl -s -w "\nHTTP %{http_code} (%{time_total}s)\n" "http://$VLLM_ADDR/v1/models" 2>/dev/null | head -5
    echo ""
    echo "Test 127.0.0.1..."
    PORT=$(echo $VLLM_ADDR | cut -d: -f2)
    timeout 5 curl -s -w "\nHTTP %{http_code} (%{time_total}s)\n" "http://127.0.0.1:$PORT/v1/models" 2>/dev/null | head -5
else
    echo "Cannot find vLLM address in log"
fi

echo ""
echo "=== 4. Quick test: can we call vLLM API? ==="
if [ -n "$VLLM_ADDR" ]; then
    timeout 30 python3 -c "
import requests, json, time
url = 'http://$VLLM_ADDR/v1/chat/completions'
data = {
    'model': 'Mercury7353/masrl_0228_mix_coldstart',
    'messages': [{'role': 'user', 'content': 'Say hello'}],
    'max_tokens': 10
}
print(f'Calling {url}...')
t0 = time.time()
try:
    r = requests.post(url, json=data, timeout=25)
    print(f'Status: {r.status_code} ({time.time()-t0:.1f}s)')
    print(f'Response: {r.text[:200]}')
except Exception as e:
    print(f'FAILED: {e} ({time.time()-t0:.1f}s)')
" 2>&1
fi

echo ""
echo "=== 5. Test: can a subprocess call vLLM? ==="
if [ -n "$VLLM_ADDR" ]; then
    timeout 60 python3 -c "
import subprocess, sys, tempfile, os, time

script = '''
import requests, json, time
url = 'http://$VLLM_ADDR/v1/chat/completions'
data = {
    \"model\": \"Mercury7353/masrl_0228_mix_coldstart\",
    \"messages\": [{\"role\": \"user\", \"content\": \"Say hi\"}],
    \"max_tokens\": 10
}
print(f'[subprocess] Calling {url}...')
t0 = time.time()
r = requests.post(url, json=data, timeout=30)
print(f'[subprocess] Status: {r.status_code} ({time.time()-t0:.1f}s)')
print(f'[subprocess] Response: {r.text[:200]}')
'''

tmpdir = tempfile.mkdtemp()
script_path = os.path.join(tmpdir, 'test.py')
with open(script_path, 'w') as f:
    f.write(script)

print(f'Running subprocess test...')
t0 = time.time()
result = subprocess.run([sys.executable, script_path], capture_output=True, text=True, timeout=50)
print(f'Exit code: {result.returncode} ({time.time()-t0:.1f}s)')
print(f'stdout: {result.stdout}')
if result.stderr:
    print(f'stderr: {result.stderr[:500]}')
" 2>&1
fi

echo ""
echo "=== Done - $(date) ==="
