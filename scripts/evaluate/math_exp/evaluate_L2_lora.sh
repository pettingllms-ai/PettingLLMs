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
# Base model path (required)
BASE_MODEL_PATH="/home/lah003/models/Qwen3-1.7B"

# LoRA adapter paths (required for LoRA evaluation)
# Each agent will load its corresponding LoRA adapter
# The number of LoRA paths must match the number of agents in the config
LORA_PATHS=(
    "checkpoints/20251216/math_1.7B_lora/global_step_2/actor/lora_1"
    "checkpoints/20251216/math_1.7B_lora/global_step_2/actor/lora_2"
)
# Assuming execution from repository root
REPO_ROOT="$(pwd)"
CONFIG_PATH="${REPO_ROOT}/pettingllms/config/math"
CONFIG_NAME="math_L2_lora"  
BENCHMARK="AIME24"
MAX_TURNS=1
EVAL_TEMPERATURE=0
BASE_VLLM_PORT=8001
BASE_PROXY_PORT=8020
MAX_PROMPT_LENGTH=8192
MAX_RESPONSE_LENGTH=8192
GPU_START_ID=4
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

# Multi-GPU configuration
# If TP_SIZE > 1, each model will use TP_SIZE consecutive GPUs
# Example: TP_SIZE=2, GPU_START_ID=0 -> model0 uses GPU 0,1; model1 uses GPU 2,3
# ============================================

echo "Starting LoRA Evaluation"
echo "=========================================="
echo "Base Model: ${BASE_MODEL_PATH}"
echo "Number of LoRA adapters: ${#LORA_PATHS[@]}"
for ((i=0; i<${#LORA_PATHS[@]}; i++)); do
    echo "  LoRA $i: ${LORA_PATHS[$i]}"
done
echo "=========================================="
echo "Multi-GPU Configuration:"
echo "  TP_SIZE: ${TP_SIZE}"
echo "  GPU_START_ID: ${GPU_START_ID}"
echo "  Total GPUs required: ${TP_SIZE} (single base model with LoRA)"
echo "=========================================="

# Check GPU availability
if command -v nvidia-smi &> /dev/null; then
    AVAILABLE_GPUS=$(nvidia-smi --list-gpus | wc -l)
    REQUIRED_GPUS=$((GPU_START_ID + TP_SIZE))
    if [ $REQUIRED_GPUS -gt $AVAILABLE_GPUS ]; then
        echo "ERROR: Not enough GPUs available!"
        echo "  Required: ${REQUIRED_GPUS} (GPU_START_ID=${GPU_START_ID} + ${TP_SIZE} GPUs for base model)"
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
        echo "Cleanup already in progress, force exiting..."
        exit 1
    fi
    CLEANUP_DONE=1

    echo "Cleaning up..."
    for pid in "${VLLM_PIDS[@]}" "${PROXY_PIDS[@]}"; do
        kill $pid 2>/dev/null || true
    done
    sleep 1
    timeout 2 lsof -ti:$BASE_VLLM_PORT 2>/dev/null | xargs -r kill -9 2>/dev/null || true
    timeout 2 lsof -ti:$BASE_PROXY_PORT 2>/dev/null | xargs -r kill -9 2>/dev/null || true

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
timeout 2 lsof -ti:$((BASE_VLLM_PORT)) 2>/dev/null | xargs -r kill -9 2>/dev/null || true
timeout 2 lsof -ti:$((BASE_PROXY_PORT)) 2>/dev/null | xargs -r kill -9 2>/dev/null || true
sleep 1

# Launch vLLM service with LoRA support
echo "Launching vLLM service with LoRA adapters..."
echo "Configuration: TP_SIZE=${TP_SIZE}, GPU_START_ID=${GPU_START_ID}"
echo

# Generate model name for base model
if [[ "$BASE_MODEL_PATH" == *"checkpoint"* ]]; then
    SERVED_MODEL_NAME="$BASE_MODEL_PATH"
else
    SERVED_MODEL_NAME="$(echo "$BASE_MODEL_PATH" | rev | cut -d'/' -f1-2 | rev)"
fi

# Build CUDA_VISIBLE_DEVICES string
if [ $TP_SIZE -eq 1 ]; then
    GPU_IDS="$GPU_START_ID"
else
    GPU_IDS="$GPU_START_ID"
    for ((g=1; g<$TP_SIZE; g++)); do
        GPU_IDS="$GPU_IDS,$((GPU_START_ID + g))"
    done
fi

# Build LoRA modules arguments (vLLM expects all modules after a single --lora-modules)
LORA_MODULES_ARG=""          # for logging only
LORA_MODULES_ARGS=(--lora-modules)  # actual CLI args - start with the flag
for ((i=0; i<${#LORA_PATHS[@]}; i++)); do
    name="lora_$((i+1))"
    path="${LORA_PATHS[$i]}"
    if [ -z "$LORA_MODULES_ARG" ]; then
        LORA_MODULES_ARG="${name}=${path}"
    else
        LORA_MODULES_ARG="${LORA_MODULES_ARG},${name}=${path}"
    fi
    LORA_MODULES_ARGS+=("${name}=${path}")  # Add each module as a separate array element
done

echo "Starting base model with LoRA adapters: ${BASE_MODEL_PATH}"
echo "  Served model name: ${SERVED_MODEL_NAME}"
echo "  GPUs: $GPU_IDS (TP_SIZE=${TP_SIZE})"
echo "  LoRA modules: ${LORA_MODULES_ARG}"
echo "  Log file: /tmp/vllm_lora.log"

CUDA_VISIBLE_DEVICES=$GPU_IDS python -m vllm.entrypoints.openai.api_server \
    --model "${BASE_MODEL_PATH}" \
    --served-model-name "${SERVED_MODEL_NAME}" \
    --host $HOST \
    --port $BASE_VLLM_PORT \
    --gpu-memory-utilization $GPU_MEM \
    --tensor-parallel-size $TP_SIZE \
    --max-model-len $MAX_LEN \
    --enable-lora \
    "${LORA_MODULES_ARGS[@]}" \
    --max-lora-rank 64 > /tmp/vllm_lora.log 2>&1 &

VLLM_PIDS[0]=$!
echo "  PID: ${VLLM_PIDS[0]}, Port: $BASE_VLLM_PORT"

# Wait a moment for vLLM to start and output initial logs
sleep 5

# Check if process is still running and show initial logs
if ! kill -0 ${VLLM_PIDS[0]} 2>/dev/null; then
    echo "  ✗ ERROR: vLLM process died immediately!"
    echo "  ====== Full log output ======"
    cat /tmp/vllm_lora.log
    echo "  ============================"
    exit 1
else
    # Show first 20 lines of log to see initialization
    if [ -f /tmp/vllm_lora.log ]; then
        echo "  ====== Initial vLLM output (first 20 lines) ======"
        head -n 20 /tmp/vllm_lora.log
        echo "  =================================================="
    fi
fi
echo

# Wait for vLLM service to be ready
echo
echo "Waiting for vLLM service to initialize..."
# Check if process is still running
if ! kill -0 ${VLLM_PIDS[0]} 2>/dev/null; then
    echo "Error: vLLM process died"
    echo "====== Full vLLM log ======"
    cat /tmp/vllm_lora.log
    echo "=========================="
    exit 1
fi

# Wait for HTTP endpoint
if ! wait_for_endpoint "$HOST" "$BASE_VLLM_PORT" "vLLM with LoRA" "$MAX_WAIT"; then
    echo "Error: vLLM failed to start"
    echo "====== Last 50 lines of vLLM log ======"
    tail -n 50 /tmp/vllm_lora.log
    echo "======================================="
    echo ""
    echo "To view full log: cat /tmp/vllm_lora.log"
    exit 1
fi

echo "✓ vLLM service ready with ${#LORA_PATHS[@]} LoRA adapters"
echo

# Launch proxy service
echo "Launching proxy service..."
echo "Starting proxy"
VLLM_BACKEND_ADDRESS="${HOST}:${BASE_VLLM_PORT}" \
PROXY_PORT=${BASE_PROXY_PORT} \
python pettingllms/evaluate/vllm_id_token_proxy.py > /tmp/proxy_lora.log 2>&1 &
PROXY_PIDS[0]=$!
echo "  PID: ${PROXY_PIDS[0]}, Port: ${BASE_PROXY_PORT}"

# Wait for proxy service to be ready
echo
echo "Waiting for proxy service to initialize..."
# Check if process is still running
if ! kill -0 ${PROXY_PIDS[0]} 2>/dev/null; then
    echo "Error: proxy process died"
    echo "Last 20 lines of log:"
    tail -n 20 /tmp/proxy_lora.log
    exit 1
fi

# Wait for HTTP endpoint
if ! wait_for_endpoint "$HOST" "$BASE_PROXY_PORT" "proxy" "60"; then
    echo "Error: proxy failed to start"
    echo "Last 20 lines of log:"
    tail -n 20 /tmp/proxy_lora.log
    exit 1
fi

echo "✓ Proxy service ready"
echo

# Build LoRA paths argument for evaluation
LORA_PATHS_ARG=""
for ((i=0; i<${#LORA_PATHS[@]}; i++)); do
    if [ -z "$LORA_PATHS_ARG" ]; then
        LORA_PATHS_ARG="${LORA_PATHS[$i]}"
    else
        LORA_PATHS_ARG="${LORA_PATHS_ARG},${LORA_PATHS[$i]}"
    fi
done

# Display service summary
echo "======================================"
echo "All services running successfully!"
echo "======================================"
echo "Base Model:"
echo "  vLLM:  http://$HOST:$BASE_VLLM_PORT"
echo "  Proxy: http://$HOST:$BASE_PROXY_PORT"
echo "LoRA Adapters: ${#LORA_PATHS[@]}"
for ((i=0; i<${#LORA_PATHS[@]}; i++)); do
    echo "  lora_${i}: ${LORA_PATHS[$i]}"
done
echo "======================================"
echo

# Run evaluation
echo "Starting evaluation..."
echo "GPU Configuration for evaluation:"
echo "  resource.n_gpus_per_node: $TP_SIZE"
echo "  tensor_model_parallel_size: $TP_SIZE"
echo "======================================"
echo "LoRA Configuration:"
echo "  Number of LoRA adapters: ${#LORA_PATHS[@]}"
echo "  LoRA paths: $LORA_PATHS_ARG"
echo "======================================"

VLLM_ADDRESS="${HOST}:${BASE_PROXY_PORT}"

python3 -m pettingllms.evaluate.evaluate \
    --config-path "$CONFIG_PATH" \
    --config-name "$CONFIG_NAME" \
    base_models.policy_0.path="${BASE_MODEL_PATH}" \
    +parallel=true \
    +vllm_address="$VLLM_ADDRESS" \
    "+lora_paths='$LORA_PATHS_ARG'" \
    env.max_turns=$MAX_TURNS \
    models.model_0.path="${BASE_MODEL_PATH}" \
    training.max_prompt_length=$MAX_PROMPT_LENGTH \
    training.max_response_length=$MAX_RESPONSE_LENGTH \
    env.benchmark="$BENCHMARK" \
    resource.n_gpus_per_node=$TP_SIZE \
    +agent_policy_configs.agent_configs.agent_0.val_temperature=$EVAL_TEMPERATURE \
    +agent_policy_configs.agent_configs.agent_1.val_temperature=$EVAL_TEMPERATURE \
    +agent_policy_configs.agent_configs.agent_0.enable_thinking=$ENABLE_THINKING \
    +agent_policy_configs.agent_configs.agent_1.enable_thinking=$ENABLE_THINKING \
    resource.nnodes=1 \
    models.model_0.ppo_trainer_config.actor_rollout_ref.rollout.tensor_model_parallel_size=$TP_SIZE \
    models.model_0.ppo_trainer_config.actor_rollout_ref.trainer.n_gpus_per_node=$TP_SIZE \
    models.model_0.ppo_trainer_config.actor_rollout_ref.trainer.n_training_gpus_per_node=$TP_SIZE

echo
echo "======================================"
echo "Evaluation completed successfully!"
echo "======================================"