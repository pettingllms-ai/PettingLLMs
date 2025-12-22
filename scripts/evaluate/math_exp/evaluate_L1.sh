#!/bin/bash
# evaluate.sh - vLLM Launch and Evaluation Script
# 
# Usage: bash scripts/evaluate/math/evaluate_L1.sh
#
# Multi-GPU Examples:
#   Single GPU per model (default):
#     TP_SIZE=1, GPU_START_ID=0, 2 models → uses GPU 0, 1
#   
#   Two GPUs per model (Tensor Parallel):
#     TP_SIZE=2, GPU_START_ID=0, 2 models → uses GPU 0-1, 2-3
#   
#   Four GPUs per model:
#     TP_SIZE=4, GPU_START_ID=0, 1 model → uses GPU 0-3
#   
#   Start from GPU 2:
#     TP_SIZE=1, GPU_START_ID=2, 2 models → uses GPU 2, 3
#
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
    "your model path"
)

# Assuming execution from repository root
REPO_ROOT="$(pwd)"
CONFIG_PATH="${REPO_ROOT}/pettingllms/config/math"
CONFIG_NAME="math_L1_prompt"
BENCHMARK="AIME24"
MAX_TURNS=5
EVAL_TEMPERATURE=0
BASE_VLLM_PORT=8001
BASE_PROXY_PORT=8020
MAX_PROMPT_LENGTH=8192
MAX_RESPONSE_LENGTH=8192
eval_temperature=0.6
GPU_START_ID=0
ENABLE_THINKING=false
HOST="127.0.0.1"
GPU_MEM=0.8
TP_SIZE=1  # Tensor parallel size (number of GPUs per model)
           # This value will be used for:
           # - vLLM tensor_parallel_size
           # - config resource.n_gpus_per_node
           # - config tensor_model_parallel_size
MAX_LEN=32768
MAX_WAIT=180  # Maximum wait time in seconds
CHECK_INTERVAL=2  # Check interval in seconds

# Shutdown behavior
VLLM_SHUTDOWN=true  # If true, vLLM will be shut down when script exits; if false, vLLM will remain running

# Multi-GPU configuration
# If TP_SIZE > 1, each model will use TP_SIZE consecutive GPUs
# Example: TP_SIZE=2, GPU_START_ID=0 -> model0 uses GPU 0,1; model1 uses GPU 2,3
# ============================================

echo "Starting with ${#MODEL_PATHS[@]} models"
echo "=========================================="
echo "Multi-GPU Configuration:"
echo "  TP_SIZE: ${TP_SIZE}"
echo "  GPU_START_ID: ${GPU_START_ID}"
echo "  Total GPUs required: $((${#MODEL_PATHS[@]} * TP_SIZE))"
echo "=========================================="

# Check GPU availability
if command -v nvidia-smi &> /dev/null; then
    AVAILABLE_GPUS=$(nvidia-smi --list-gpus | wc -l)
    REQUIRED_GPUS=$((GPU_START_ID + ${#MODEL_PATHS[@]} * TP_SIZE))
    echo "Available GPUs: ${AVAILABLE_GPUS}"
    echo "Required GPUs: ${REQUIRED_GPUS}"
    if [ $REQUIRED_GPUS -gt $AVAILABLE_GPUS ]; then
        echo "ERROR: Not enough GPUs available!"
        echo "  Required: ${REQUIRED_GPUS} (GPU_START_ID=${GPU_START_ID} + ${#MODEL_PATHS[@]} models × ${TP_SIZE} GPUs/model)"
        echo "  Available: ${AVAILABLE_GPUS}"
        exit 1
    fi
    echo "✓ GPU availability check passed"
else
    echo "WARNING: nvidia-smi not found, skipping GPU availability check"
fi
echo "=========================================="
echo

declare -a VLLM_PIDS PROXY_PIDS
CLEANUP_DONE=0

cleanup() {
    if [ $CLEANUP_DONE -eq 1 ]; then
        exit 1
    fi
    CLEANUP_DONE=1
    
    # Always cleanup proxy processes
    for pid in "${PROXY_PIDS[@]}"; do 
        kill $pid 2>/dev/null || true
    done
    for ((i=0; i<${#MODEL_PATHS[@]}; i++)); do
        timeout 2 lsof -ti:$((BASE_PROXY_PORT + i)) 2>/dev/null | xargs -r kill -9 2>/dev/null || true
    done
    
    # Only cleanup vLLM if VLLM_SHUTDOWN is true
    if [ "$VLLM_SHUTDOWN" = true ]; then
        echo "Shutting down vLLM..."
        for pid in "${VLLM_PIDS[@]}"; do 
            kill $pid 2>/dev/null || true
        done
        for ((i=0; i<${#MODEL_PATHS[@]}; i++)); do
            timeout 2 lsof -ti:$((BASE_VLLM_PORT + i)) 2>/dev/null | xargs -r kill -9 2>/dev/null || true
        done
    fi
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
echo "Configuration: TP_SIZE=${TP_SIZE}, GPU_START_ID=${GPU_START_ID}"
echo "Total GPUs required: $((${#MODEL_PATHS[@]} * TP_SIZE))"
echo

for ((i=0; i<${#MODEL_PATHS[@]}; i++)); do
    MODEL_PATH="${MODEL_PATHS[$i]}"
    # Generate model name: if contains "checkpoint", use full path; otherwise use last 2 segments
    if [[ "$MODEL_PATH" == *"checkpoint"* ]]; then
        SERVED_MODEL_NAME="$MODEL_PATH"
    else
        SERVED_MODEL_NAME="$(echo "$MODEL_PATH" | rev | cut -d'/' -f1-2 | rev)"
    fi
    
    # Calculate GPU IDs for this model
    FIRST_GPU=$((GPU_START_ID + i * TP_SIZE))
    LAST_GPU=$((FIRST_GPU + TP_SIZE - 1))
    
    # Build CUDA_VISIBLE_DEVICES string
    if [ $TP_SIZE -eq 1 ]; then
        GPU_IDS="$FIRST_GPU"
    else
        GPU_IDS="$FIRST_GPU"
        for ((g=1; g<$TP_SIZE; g++)); do
            GPU_IDS="$GPU_IDS,$((FIRST_GPU + g))"
        done
    fi
    
    echo "Starting model$((i+1)): ${MODEL_PATH}"
    echo "  Served model name: ${SERVED_MODEL_NAME}"
    echo "  GPUs: $GPU_IDS (TP_SIZE=${TP_SIZE})"
    echo "  Log file: /tmp/vllm_model${i}.log"
    CUDA_VISIBLE_DEVICES=$GPU_IDS python -m vllm.entrypoints.openai.api_server \
        --model "${MODEL_PATH}" \
        --served-model-name "${SERVED_MODEL_NAME}" \
        --host $HOST \
        --port $((BASE_VLLM_PORT + i)) \
        --gpu-memory-utilization $GPU_MEM \
        --tensor-parallel-size $TP_SIZE \
        --max-model-len $MAX_LEN > /tmp/vllm_model${i}.log 2>&1 &
    VLLM_PIDS[$i]=$!
    echo "  PID: ${VLLM_PIDS[$i]}, Port: $((BASE_VLLM_PORT + i))"
    
    # Wait a moment for vLLM to start and output initial logs
    sleep 3
    
    # Check if process is still running and show initial logs
    if ! kill -0 ${VLLM_PIDS[$i]} 2>/dev/null; then
        echo "  ✗ ERROR: vLLM process died immediately!"
        echo "  ====== Full log output ======"
        cat /tmp/vllm_model${i}.log
        echo "  ============================"
        exit 1
    else
        # Show first 15 lines of log to see initialization
        if [ -f /tmp/vllm_model${i}.log ]; then
            echo "  ====== Initial vLLM output (first 15 lines) ======"
            head -n 15 /tmp/vllm_model${i}.log
            echo "  =================================================="
        fi
    fi
    echo
done

# Wait for all vLLM services to be ready
echo
echo "Waiting for vLLM services to initialize..."
for ((i=0; i<${#MODEL_PATHS[@]}; i++)); do
    # Check if process is still running
    if ! kill -0 ${VLLM_PIDS[$i]} 2>/dev/null; then
        echo "Error: model$((i+1)) vLLM process died"
        echo "====== Full vLLM log ======"
        cat /tmp/vllm_model${i}.log
        echo "=========================="
        exit 1
    fi
    
    # Wait for HTTP endpoint
    if ! wait_for_endpoint "$HOST" "$((BASE_VLLM_PORT + i))" "model$((i+1)) vLLM" "$MAX_WAIT"; then
        echo "Error: model$((i+1)) vLLM failed to start"
        echo "====== Last 50 lines of vLLM log ======"
        tail -n 50 /tmp/vllm_model${i}.log
        echo "======================================="
        echo ""
        echo "To view full log: cat /tmp/vllm_model${i}.log"
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
echo "GPU Configuration for evaluation:"
echo "  resource.n_gpus_per_node: $TP_SIZE"
echo "  tensor_model_parallel_size: $TP_SIZE"
echo "======================================"
VLLM_ADDRESS="${HOST}:${BASE_PROXY_PORT}"

python3 -m pettingllms.evaluate.evaluate \
    --config-path "$CONFIG_PATH" \
    --config-name "$CONFIG_NAME" \
    base_models.policy_0.path="${MODEL_PATHS[0]}" \
    +parallel=true \
    +vllm_address="$VLLM_ADDRESS" \
    env.max_turns=$MAX_TURNS \
    $MODEL_ARGS \
    training.max_prompt_length=$MAX_PROMPT_LENGTH \
    training.max_response_length=$MAX_RESPONSE_LENGTH \
    env.benchmark="$BENCHMARK" \
    resource.n_gpus_per_node=$TP_SIZE \
    agent_policy_configs.agent_configs.agent_0.val_temperature=$EVAL_TEMPERATURE \
    agent_policy_configs.agent_configs.agent_1.val_temperature=$EVAL_TEMPERATURE \
    agent_policy_configs.agent_configs.agent_0.enable_thinking=$ENABLE_THINKING \
    agent_policy_configs.agent_configs.agent_1.enable_thinking=$ENABLE_THINKING \
    resource.nnodes=1 \
    models.model_0.ppo_trainer_config.actor_rollout_ref.rollout.tensor_model_parallel_size=$TP_SIZE \
    models.model_0.ppo_trainer_config.actor_rollout_ref.trainer.n_gpus_per_node=$TP_SIZE \
    models.model_0.ppo_trainer_config.actor_rollout_ref.trainer.n_training_gpus_per_node=$TP_SIZE

echo
echo "======================================"
echo "Evaluation completed successfully!"
echo "======================================"
