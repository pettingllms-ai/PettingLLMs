#!/bin/bash
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
    "/raid/lah003/mas_rl_cold_start"
)
EXPERIMENT_NAME="mas_graph_test"
# Assuming execution from repository root
REPO_ROOT="$(pwd)"
CONFIG_PATH="${REPO_ROOT}/pettingllms/config/autoevol"
CONFIG_NAME="math_L1_prompt"
BENCHMARK="AIME24"
BASE_VLLM_PORT=8301
BASE_PROXY_PORT=8320
GPU_START_ID=3
HOST="127.0.0.1"
GPU_MEM=0.15  # Reduced from 0.8 to fit in available memory (33.38 GiB available)
VLLM_SHUTDOWN=false  # If true, vLLM will be shut down when script exits; if false, vLLM will remain running
TP_SIZE=1
MAX_PROMPT_LENGTH=8192
MAX_RESPONSE_LENGTH=4096
MAX_LEN=32768
MAX_WAIT=180  # Maximum wait time in seconds
CHECK_INTERVAL=2  # Check interval in seconds

# Multi-GPU configuration
# If TP_SIZE > 1, each model will use TP_SIZE consecutive GPUs
# Example: TP_SIZE=2, GPU_START_ID=0 -> model0 uses GPU 0,1; model1 uses GPU 2,3
# ============================================

# Check GPU availability
if command -v nvidia-smi &> /dev/null; then
    AVAILABLE_GPUS=$(nvidia-smi --list-gpus | wc -l)
    REQUIRED_GPUS=$((GPU_START_ID + ${#MODEL_PATHS[@]} * TP_SIZE))
    if [ $REQUIRED_GPUS -gt $AVAILABLE_GPUS ]; then
        echo "ERROR: Not enough GPUs available!"
        echo "  Required: ${REQUIRED_GPUS}, Available: ${AVAILABLE_GPUS}"
        exit 1
    fi
fi

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

# Kill existing proxy processes
echo "Cleaning existing proxy processes..."
for ((i=0; i<${#MODEL_PATHS[@]}; i++)); do
    timeout 2 lsof -ti:$((BASE_PROXY_PORT + i)) 2>/dev/null | xargs -r kill -9 2>/dev/null || true
done
sleep 1

# Try to connect to existing vLLM first (5 seconds timeout)
echo "Checking for existing vLLM services..."
VLLM_ALREADY_RUNNING=false
VLLM_CHECK_TIMEOUT=5
for ((i=0; i<${#MODEL_PATHS[@]}; i++)); do
    echo -n "Trying to connect to vLLM at port $((BASE_VLLM_PORT + i))..."
    if timeout $VLLM_CHECK_TIMEOUT curl -s "http://$HOST:$((BASE_VLLM_PORT + i))/v1/models" >/dev/null 2>&1; then
        echo " ✓ Connected"
        VLLM_ALREADY_RUNNING=true
    else
        echo " ✗ Not available"
        VLLM_ALREADY_RUNNING=false
        break
    fi
done

# If vLLM not running or connection failed, clean up ports and prepare for new launch
if [ "$VLLM_ALREADY_RUNNING" = false ]; then
    echo "vLLM not detected, will start new instances..."
    for ((i=0; i<${#MODEL_PATHS[@]}; i++)); do
        timeout 2 lsof -ti:$((BASE_VLLM_PORT + i)) 2>/dev/null | xargs -r kill -9 2>/dev/null || true
    done
    sleep 1
else
    echo "✓ Existing vLLM instances detected, will reuse them"
fi

# Launch vLLM services (only if not already running)
if [ "$VLLM_ALREADY_RUNNING" = false ]; then
    echo "Starting vLLM services..."
else
    echo "Skipping vLLM launch (already running)..."
fi
if [ "$VLLM_ALREADY_RUNNING" = false ]; then
    for ((i=0; i<${#MODEL_PATHS[@]}; i++)); do
        MODEL_PATH="${MODEL_PATHS[$i]}"
        if [[ "$MODEL_PATH" == *"checkpoint"* ]]; then
            SERVED_MODEL_NAME="$MODEL_PATH"
        else
            SERVED_MODEL_NAME="$(echo "$MODEL_PATH" | rev | cut -d'/' -f1-2 | rev)"
        fi

        # Calculate GPU IDs
        FIRST_GPU=$((GPU_START_ID + i * TP_SIZE))
        if [ $TP_SIZE -eq 1 ]; then
            GPU_IDS="$FIRST_GPU"
        else
            GPU_IDS="$FIRST_GPU"
            for ((g=1; g<$TP_SIZE; g++)); do
                GPU_IDS="$GPU_IDS,$((FIRST_GPU + g))"
            done
        fi

        echo "vLLM model$((i+1)): ${SERVED_MODEL_NAME} | GPU: $GPU_IDS | Port: $((BASE_VLLM_PORT + i))"
        CUDA_VISIBLE_DEVICES=$GPU_IDS python -m vllm.entrypoints.openai.api_server \
            --model "${MODEL_PATH}" \
            --served-model-name "${SERVED_MODEL_NAME}" \
            --host $HOST \
            --port $((BASE_VLLM_PORT + i)) \
            --gpu-memory-utilization $GPU_MEM \
            --tensor-parallel-size $TP_SIZE \
            --max-model-len $MAX_LEN \
            --enable-auto-tool-choice \
            --tool-call-parser hermes > /tmp/vllm_model${i}.log 2>&1 &
        VLLM_PIDS[$i]=$!

        sleep 3
        if ! kill -0 ${VLLM_PIDS[$i]} 2>/dev/null; then
            echo "ERROR: vLLM died. Log: /tmp/vllm_model${i}.log"
            tail -n 30 /tmp/vllm_model${i}.log
            exit 1
        fi
    done
fi

if [ "$VLLM_ALREADY_RUNNING" = false ]; then
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
else
    echo "Verifying existing vLLM services..."
    for ((i=0; i<${#MODEL_PATHS[@]}; i++)); do
        if ! curl -s "http://$HOST:$((BASE_VLLM_PORT + i))/v1/models" >/dev/null 2>&1; then
            echo "Error: Expected vLLM at port $((BASE_VLLM_PORT + i)) but it's not responding"
            echo "Please check if vLLM is running or set VLLM_SHUTDOWN=true to restart"
            exit 1
        fi
        echo "✓ vLLM model$((i+1)) at port $((BASE_VLLM_PORT + i)) is ready"
    done
fi

echo "✓ All vLLM services ready"
echo

# Launch proxy services
echo "Launching proxy services..."
for ((i=0; i<${#MODEL_PATHS[@]}; i++)); do
    VLLM_BACKEND_ADDRESS="${HOST}:$((BASE_VLLM_PORT + i))" \
    PROXY_PORT=$((BASE_PROXY_PORT + i)) \
    python pettingllms/evaluate/vllm_id_token_proxy.py > /tmp/proxy_model${i}.log 2>&1 &
    PROXY_PIDS[$i]=$!
done

# Wait for proxy ready
echo "Waiting for proxy services to initialize..."
for ((i=0; i<${#MODEL_PATHS[@]}; i++)); do
    if ! kill -0 ${PROXY_PIDS[$i]} 2>/dev/null; then
        echo "ERROR: Proxy model$((i+1)) died. Log: /tmp/proxy_model${i}.log"
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

# Build model args for both base_models and models
MODEL_ARGS=""
for ((i=0; i<${#MODEL_PATHS[@]}; i++)); do
    MODEL_ARGS="$MODEL_ARGS models.model_${i}.path=\"${MODEL_PATHS[$i]}\""
    MODEL_ARGS="$MODEL_ARGS base_models.policy_${i}.path=\"${MODEL_PATHS[$i]}\""
done

# Run evaluation
echo "Running evaluation..."
# Use BASE_VLLM_PORT instead of BASE_PROXY_PORT for chat/completions endpoint
VLLM_ADDRESS="${HOST}:${BASE_VLLM_PORT}"

python3 -m pettingllms.evaluate.evaluate \
    --config-path "$CONFIG_PATH" \
    --config-name "$CONFIG_NAME" \
    +parallel=true \
    +vllm_address="$VLLM_ADDRESS" \
    env.benchmark="$BENCHMARK" \
    $MODEL_ARGS \
    training.max_prompt_length=$MAX_PROMPT_LENGTH \
    training.max_response_length=$MAX_RESPONSE_LENGTH \
    training.experiment_name="$EXPERIMENT_NAME" \
    +traning.validate_sample_num=1 \
    resource.n_gpus_per_node=$TP_SIZE \
    resource.nnodes=1 \
    models.model_0.ppo_trainer_config.actor_rollout_ref.rollout.tensor_model_parallel_size=$TP_SIZE \
    models.model_0.ppo_trainer_config.actor_rollout_ref.trainer.n_gpus_per_node=$TP_SIZE \
    models.model_0.ppo_trainer_config.actor_rollout_ref.trainer.n_training_gpus_per_node=$TP_SIZE
    

echo "Evaluation completed"
