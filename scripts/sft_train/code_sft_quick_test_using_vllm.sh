set -x

# ==================== GPU Configuration ====================
# Configure which GPU(s) to use for both vLLM server and SFT training
GPU_START_ID=3        # Starting GPU ID (e.g., 3 for GPU:3)
GPU_NUM=1             # Number of GPUs to use

# Automatically set CUDA_VISIBLE_DEVICES based on GPU configuration
if [ $GPU_NUM -eq 1 ]; then
    export CUDA_VISIBLE_DEVICES=$GPU_START_ID
else
    # For multiple GPUs, create a comma-separated list
    GPU_IDS=""
    for ((i=0; i<$GPU_NUM; i++)); do
        if [ $i -eq 0 ]; then
            GPU_IDS="$((GPU_START_ID + i))"
        else
            GPU_IDS="$GPU_IDS,$((GPU_START_ID + i))"
        fi
    done
    export CUDA_VISIBLE_DEVICES=$GPU_IDS
fi

echo "=========================================="
echo "GPU Configuration"
echo "=========================================="
echo "GPU_START_ID: $GPU_START_ID"
echo "GPU_NUM: $GPU_NUM"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "Note: vLLM server and SFT training use the same GPU(s)"
echo "=========================================="
echo ""

# ==================== Environment Variables ====================
# VLLM and CUDA Environment Variables
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

# CUDA Environment
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
export LD_LIBRARY_PATH=$CUDA_HOME/targets/x86_64-linux/lib:${LD_LIBRARY_PATH}
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH}

# ==================== Model and Output Configuration ====================
VLLM_MODEL_PATH="VLLM model path here"  # Model for vLLM inference
TRAIN_MODEL_PATH="your trained model path here"  # Model for SFT training
EXPERIMENT_NAME="code_sft_quick_test"
OUTPUT_DIR="./sft_data_test"
SFT_MODEL_OUTPUT="./sft_model_test"

# ==================== vLLM Server Configuration ====================
VLLM_HOST="127.0.0.1"
VLLM_PORT=8500
VLLM_GPU_MEM=0.85  # Reduced from 0.9 to avoid OOM (GPU mem insufficient error)
VLLM_MAX_LEN=8192
VLLM_TP_SIZE=$GPU_NUM  # Tensor parallel size = number of GPUs

# ==================== API Configuration for SFT ====================
API_BASE_URL="http://${VLLM_HOST}:${VLLM_PORT}/v1"
API_MODEL="Qwen3-8B"
API_TEMPERATURE=0.7
API_MAX_TOKENS=2048
API_TIMEOUT=120.0

# ==================== Model Resource Configuration ====================
model_0_config_path="models.model_0.ppo_trainer_config"
model_0_resource="resource.n_gpus_per_node=$GPU_NUM $model_0_config_path.trainer.n_gpus_per_node=$GPU_NUM $model_0_config_path.trainer.nnodes=1 $model_0_config_path.actor_rollout_ref.rollout.tensor_model_parallel_size=$GPU_NUM"

# ==================== Start vLLM Server ====================
echo "=========================================="
echo "Starting vLLM server..."
echo "=========================================="
echo "Model: $VLLM_MODEL_PATH"
echo "Host: $VLLM_HOST"
echo "Port: $VLLM_PORT"
echo "GPU Memory: $VLLM_GPU_MEM"
echo "Max Length: $VLLM_MAX_LEN"
echo "Tensor Parallel Size: $VLLM_TP_SIZE"
echo ""

# Function to wait for endpoint to be ready
wait_for_endpoint() {
    local host=$1
    local port=$2
    local max_wait=${3:-300}
    local elapsed=0

    echo "Waiting for vLLM server at $host:$port (max ${max_wait}s)..."
    while [ $elapsed -lt $max_wait ]; do
        if curl -s "http://$host:$port/v1/models" > /dev/null 2>&1; then
            echo "✓ vLLM server is ready!"
            return 0
        fi
        sleep 10
        elapsed=$((elapsed + 2))
        if [ $((elapsed % 10)) -eq 0 ]; then
            echo "  Still waiting... (${elapsed}s elapsed)"
        fi
    done
    echo "✗ Timeout waiting for vLLM server"
    return 1
}

# Check GPU availability before starting vLLM
echo "Checking GPU availability..."
nvidia-smi --query-gpu=index,name,memory.free,memory.total --format=csv,noheader,nounits | while read line; do
    gpu_id=$(echo $line | cut -d',' -f1)
    if [ "$gpu_id" = "$CUDA_VISIBLE_DEVICES" ]; then
        echo "GPU $gpu_id: $line"
    fi
done
echo ""

# Start vLLM server with verbose logging
echo "Starting vLLM with the following command:"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python -m vllm.entrypoints.openai.api_server \\"
echo "  --model \"$VLLM_MODEL_PATH\" \\"
echo "  --served-model-name \"$API_MODEL\" \\"
echo "  --host $VLLM_HOST \\"
echo "  --port $VLLM_PORT \\"
echo "  --gpu-memory-utilization $VLLM_GPU_MEM \\"
echo "  --tensor-parallel-size $VLLM_TP_SIZE \\"
echo "  --max-model-len $VLLM_MAX_LEN"
echo ""

CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python -m vllm.entrypoints.openai.api_server \
    --model "$VLLM_MODEL_PATH" \
    --served-model-name "$API_MODEL" \
    --host $VLLM_HOST \
    --port $VLLM_PORT \
    --gpu-memory-utilization $VLLM_GPU_MEM \
    --tensor-parallel-size $VLLM_TP_SIZE \
    --max-model-len $VLLM_MAX_LEN > /tmp/vllm_sft_quick_test.log 2>&1 &

VLLM_PID=$!
echo "vLLM server started with PID: $VLLM_PID"
echo "Log file: /tmp/vllm_sft_quick_test.log"
echo ""

# Monitor vLLM startup in real-time
echo "Monitoring vLLM startup (will check for 30 seconds)..."
for i in {1..6}; do
    sleep 5

    # Check if process died
    if ! kill -0 $VLLM_PID 2>/dev/null; then
        echo ""
        echo "✗ ERROR: vLLM process died after $((i * 5)) seconds!"
        echo "=========================================="
        echo "Full vLLM log output:"
        echo "=========================================="
        cat /tmp/vllm_sft_quick_test.log
        echo "=========================================="
        echo ""
        echo "Common issues and solutions:"
        echo "  1. GPU memory insufficient:"
        echo "     - Current VLLM_GPU_MEM: $VLLM_GPU_MEM"
        echo "     - Try reducing to 0.7 or 0.6"
        echo "  2. Model path incorrect:"
        echo "     - Current VLLM_MODEL_PATH: $VLLM_MODEL_PATH"
        echo "     - Check if model exists: ls -la $VLLM_MODEL_PATH"
        echo "  3. Port already in use:"
        echo "     - Current VLLM_PORT: $VLLM_PORT"
        echo "     - Check with: lsof -i :$VLLM_PORT"
        echo "  4. CUDA/GPU errors:"
        echo "     - Run: nvidia-smi"
        echo "     - Check CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
        exit 1
    fi

    echo "  [${i}/6] vLLM process still running (PID: $VLLM_PID)..."
done

echo ""
echo "====== vLLM initialization log (last 50 lines) ======"
tail -n 50 /tmp/vllm_sft_quick_test.log
echo "====================================================="
echo ""

# Wait for vLLM service to be ready
if ! wait_for_endpoint "$VLLM_HOST" "$VLLM_PORT" 300; then
    echo "Error: vLLM server failed to start"
    echo "====== Last 50 lines of vLLM log ======"
    tail -n 50 /tmp/vllm_sft_quick_test.log
    echo "======================================="
    kill $VLLM_PID 2>/dev/null
    exit 1
fi

echo ""
echo "=========================================="
echo "✓ vLLM server ready at $API_BASE_URL"
echo "=========================================="
echo ""

# Cleanup function to kill vLLM server on exit
cleanup() {
    echo ""
    echo "=========================================="
    echo "Cleaning up..."
    echo "=========================================="
    if [ ! -z "$VLLM_PID" ]; then
        echo "Stopping vLLM server (PID: $VLLM_PID)..."
        kill $VLLM_PID 2>/dev/null
        wait $VLLM_PID 2>/dev/null
        echo "✓ vLLM server stopped"
    fi
    echo "=========================================="
}

# Register cleanup function to run on script exit
trap cleanup EXIT INT TERM

# ==================== Run SFT Data Collection ====================
echo "=========================================="
echo "Starting SFT Data Collection..."
echo "=========================================="
echo ""

# Run quick test with only 10 episodes and 1 training epoch
python3 -m pettingllms.sft_train.train \
    --config-path ../config/code \
    --config-name code_L0_single_agent \
    $model_0_resource \
    base_models.policy_0.path="$TRAIN_MODEL_PATH" \
    models.model_0.path="$TRAIN_MODEL_PATH" \
    training.experiment_name=$EXPERIMENT_NAME \
    training.max_prompt_length=2048 \
    training.max_response_length=1024 \
    env.dataset=code_contests \
    env.benchmark=code_contests \
    env.max_turns=2 \
    +training.sft_mode=train \
    +training.sft_num_episodes=10 \
    +training.sft_data_dir="$OUTPUT_DIR" \
    +training.collect_mode=env \
    +training.use_api=true \
    +training.api_type=openai \
    +training.api_model="$API_MODEL" \
    +training.api_base_url="$API_BASE_URL" \
    +training.api_temperature=$API_TEMPERATURE \
    +training.api_max_tokens=$API_MAX_TOKENS \
    +training.api_timeout=$API_TIMEOUT \
    +training.run_sft_training=True \
    +training.sft_config.model_name_or_path="$TRAIN_MODEL_PATH" \
    +training.sft_config.output_dir="$SFT_MODEL_OUTPUT" \
    +training.sft_config.num_train_epochs=1 \
    +training.sft_config.per_device_train_batch_size=1 \
    +training.sft_config.gradient_accumulation_steps=2 \
    +training.sft_config.learning_rate=5e-5 \
    +training.sft_config.use_lora=True \
    +training.sft_config.lora_r=32 \
    +training.sft_config.lora_alpha=16

echo ""
echo "=========================================="
echo "✓ Quick test completed!"
echo "=========================================="
echo "Collected data saved to: $OUTPUT_DIR"
echo "Trained model saved to: $SFT_MODEL_OUTPUT"
echo "vLLM log saved to: /tmp/vllm_sft_quick_test.log"
echo "=========================================="
