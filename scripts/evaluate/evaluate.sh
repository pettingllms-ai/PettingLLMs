#!/bin/bash
# evaluate.sh - vLLM Launch and Evaluation Script
# Usage: bash scripts/evaluate/evaluate.sh
set -e

export TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export VLLM_USE_FLASHINFER_SAMPLER=0
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:False"
export VLLM_USE_V1=1
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_ENGINE_ITERATION_TIMEOUT_S=100000000000
export HYDRA_FULL_ERROR=1
export NCCL_IB_DISABLE=1
export NCCL_NET_GDR_LEVEL=0
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
export LD_LIBRARY_PATH=$CUDA_HOME/targets/x86_64-linux/lib:$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# ============================================
# Configuration - Edit these parameters
# ============================================
MODEL_PATHS=(
    "/home/nvidia/data/models/Qwen3-8B"
)

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(pwd)"
CONFIG_PATH="$REPO_ROOT/pettingllms/config/stateful"
CONFIG_NAME="stateful_single_policy"
BENCHMARK="sokoban"
BASE_VLLM_PORT=8201
BASE_PROXY_PORT=8220
GPU_START_ID=0
HOST="127.0.0.1"
GPU_MEM=0.8
TP_SIZE=1
MAX_LEN=32768
MAX_WAIT=180  # Maximum wait time in seconds
CHECK_INTERVAL=2  # Check interval in seconds
# ============================================

echo "Starting with ${#MODEL_PATHS[@]} models"

declare -a VLLM_PIDS PROXY_PIDS
CLEANUP_DONE=0

cleanup() {
    if [ $CLEANUP_DONE -eq 1 ]; then
        echo "Cleanup already in progress, force exiting..."
        exit 1
    fi
    CLEANUP_DONE=1
    
    echo "Cleaning up..."
    for pid in "${VLLM_PIDS[@]}" "${PROXY_PIDS[@]}"; do 
        kill $pid 2>/dev/null || true
    done
    sleep 1
    for ((i=0; i<${#MODEL_PATHS[@]}; i++)); do
        timeout 2 lsof -ti:$((BASE_VLLM_PORT + i)) 2>/dev/null | xargs -r kill -9 2>/dev/null || true
        timeout 2 lsof -ti:$((BASE_PROXY_PORT + i)) 2>/dev/null | xargs -r kill -9 2>/dev/null || true
    done
    
    echo "Cleanup completed"
}
trap cleanup EXIT INT TERM

# Function to wait for HTTP endpoint
wait_for_endpoint() {
    local host=$1
    local port=$2
    local name=$3
    local max_wait=$4
    
    echo -n "Waiting for $name at $host:$port"
    local elapsed=0
    while [ $elapsed -lt $max_wait ]; do
        if curl -s "http://$host:$port/v1/models" >/dev/null 2>&1; then
            echo " ✓ Ready (${elapsed}s)"
            return 0
        fi
        echo -n "."
        sleep $CHECK_INTERVAL
        elapsed=$((elapsed + CHECK_INTERVAL))
    done
    echo " ✗ Timeout after ${max_wait}s"
    return 1
}

# Kill existing processes
echo "Cleaning existing processes..."
for ((i=0; i<${#MODEL_PATHS[@]}; i++)); do
    timeout 2 lsof -ti:$((BASE_VLLM_PORT + i)) 2>/dev/null | xargs -r kill -9 2>/dev/null || true
    timeout 2 lsof -ti:$((BASE_PROXY_PORT + i)) 2>/dev/null | xargs -r kill -9 2>/dev/null || true
done
sleep 1

# Launch vLLM services
echo "Launching vLLM services..."
for ((i=0; i<${#MODEL_PATHS[@]}; i++)); do
    MODEL_PATH="${MODEL_PATHS[$i]}"
    # Generate model name: if contains "checkpoint", use full path; otherwise use last 2 segments
    if [[ "$MODEL_PATH" == *"checkpoint"* ]]; then
        SERVED_MODEL_NAME="$MODEL_PATH"
    else
        SERVED_MODEL_NAME="$(echo "$MODEL_PATH" | rev | cut -d'/' -f1-2 | rev)"
    fi
    
    echo "Starting model$((i+1)): ${MODEL_PATH}"
    echo "  Served model name: ${SERVED_MODEL_NAME}"
    CUDA_VISIBLE_DEVICES=$((GPU_START_ID + i)) python -m vllm.entrypoints.openai.api_server \
        --model "${MODEL_PATH}" \
        --served-model-name "${SERVED_MODEL_NAME}" \
        --host $HOST \
        --port $((BASE_VLLM_PORT + i)) \
        --gpu-memory-utilization $GPU_MEM \
        --tensor-parallel-size $TP_SIZE \
        --max-model-len $MAX_LEN > /tmp/vllm_model${i}.log 2>&1 &
    VLLM_PIDS[$i]=$!
    echo "  PID: ${VLLM_PIDS[$i]}, Port: $((BASE_VLLM_PORT + i))"
done

# Wait for all vLLM services to be ready
echo
echo "Waiting for vLLM services to initialize..."
for ((i=0; i<${#MODEL_PATHS[@]}; i++)); do
    # Check if process is still running
    if ! kill -0 ${VLLM_PIDS[$i]} 2>/dev/null; then
        echo "Error: model$((i+1)) vLLM process died"
        echo "Last 20 lines of log:"
        tail -n 20 /tmp/vllm_model${i}.log
        exit 1
    fi
    
    # Wait for HTTP endpoint
    if ! wait_for_endpoint "$HOST" "$((BASE_VLLM_PORT + i))" "model$((i+1)) vLLM" "$MAX_WAIT"; then
        echo "Error: model$((i+1)) vLLM failed to start"
        echo "Last 20 lines of log:"
        tail -n 20 /tmp/vllm_model${i}.log
        exit 1
    fi
done

echo "✓ All vLLM services ready"
echo

# Launch proxy services
echo "Launching proxy services..."
for ((i=0; i<${#MODEL_PATHS[@]}; i++)); do
    echo "Starting proxy for model$((i+1))"
    VLLM_BACKEND_ADDRESS="${HOST}:$((BASE_VLLM_PORT + i))" \
    PROXY_PORT=$((BASE_PROXY_PORT + i)) \
    python pettingllms/evaluate/vllm_id_token_proxy.py > /tmp/proxy_model${i}.log 2>&1 &
    PROXY_PIDS[$i]=$!
    echo "  PID: ${PROXY_PIDS[$i]}, Port: $((BASE_PROXY_PORT + i))"
done

# Wait for all proxy services to be ready
echo
echo "Waiting for proxy services to initialize..."
for ((i=0; i<${#MODEL_PATHS[@]}; i++)); do
    # Check if process is still running
    if ! kill -0 ${PROXY_PIDS[$i]} 2>/dev/null; then
        echo "Error: model$((i+1)) proxy process died"
        echo "Last 20 lines of log:"
        tail -n 20 /tmp/proxy_model${i}.log
        exit 1
    fi
    
    # Wait for HTTP endpoint
    if ! wait_for_endpoint "$HOST" "$((BASE_PROXY_PORT + i))" "model$((i+1)) proxy" "60"; then
        echo "Error: model$((i+1)) proxy failed to start"
        echo "Last 20 lines of log:"
        tail -n 20 /tmp/proxy_model${i}.log
        exit 1
    fi
done

echo "✓ All proxy services ready"
echo

# Build model args for evaluation
MODEL_ARGS=""
for ((i=0; i<${#MODEL_PATHS[@]}; i++)); do
    MODEL_ARGS="$MODEL_ARGS models.model_${i}.path=\"${MODEL_PATHS[$i]}\""
done

# Display service summary
echo "======================================"
echo "All services running successfully!"
echo "======================================"
for ((i=0; i<${#MODEL_PATHS[@]}; i++)); do
    echo "Model $((i+1)):"
    echo "  vLLM:  http://$HOST:$((BASE_VLLM_PORT + i))"
    echo "  Proxy: http://$HOST:$((BASE_PROXY_PORT + i))"
done
echo "======================================"
echo

# Run evaluation
echo "Starting evaluation..."
cd "$REPO_ROOT" || exit 1  
VLLM_ADDRESS="${HOST}:${BASE_PROXY_PORT}"

python3 -m pettingllms.evaluate.evaluate \
    --config-path "$CONFIG_PATH" \
    --config-name "$CONFIG_NAME" \
    +parallel=false \
    enable_thinking=false \
    +vllm_address="$VLLM_ADDRESS" \
    env.map_size=6 \
    $MODEL_ARGS \
    benchmark="$BENCHMARK" \

echo
echo "======================================"
echo "Evaluation completed successfully!"
echo "======================================"
