#!/bin/bash
# SFT Data Collection Script for Code Environment (Collection Only)
# This script only collects SFT data without training
# Supports external API inference (OpenAI, DeepSeek, Claude) or local model

set -x

# GPU Configuration
export CUDA_VISIBLE_DEVICES=2,3
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

# Configuration
GPU_num=2
MODEL_PATH="your trained model path here"  # Model for vLLM inference
EXPERIMENT_NAME="code_sft_collection_only"
OUTPUT_DIR="./sft_data_code"

# API Configuration (optional - for using external APIs instead of local model)
# Set USE_API=true to enable API mode
USE_API=${USE_API:-false}
API_TYPE=${API_TYPE:-"openai"}  # "openai", "deepseek", or "claude"
API_MODEL=${API_MODEL:-"gpt-4o"}  # Model name for the API
API_BASE_URL=${API_BASE_URL:-""}  # Optional custom base URL
API_TEMPERATURE=${API_TEMPERATURE:-0.7}
API_MAX_TOKENS=${API_MAX_TOKENS:-2048}
API_TIMEOUT=${API_TIMEOUT:-60.0}

# API Keys (set via environment variables or here)
# export OPENAI_API_KEY="your-key-here"
# export DEEPSEEK_API_KEY="your-key-here"
# export ANTHROPIC_API_KEY="your-key-here"

# Model resource configuration
model_0_config_path="models.model_0.ppo_trainer_config"
model_0_resource="resource.n_gpus_per_node=$GPU_num $model_0_config_path.trainer.n_gpus_per_node=$GPU_num $model_0_config_path.trainer.nnodes=1 $model_0_config_path.actor_rollout_ref.rollout.tensor_model_parallel_size=$GPU_num"

# Build API arguments if API mode is enabled
API_ARGS=""
if [ "$USE_API" = "true" ]; then
    echo "Using API mode: $API_TYPE"
    API_ARGS="training.use_api=true training.api_type=$API_TYPE training.api_model=$API_MODEL"

    if [ -n "$API_BASE_URL" ]; then
        API_ARGS="$API_ARGS training.api_base_url=$API_BASE_URL"
    fi

    API_ARGS="$API_ARGS training.api_temperature=$API_TEMPERATURE"
    API_ARGS="$API_ARGS training.api_max_tokens=$API_MAX_TOKENS"
    API_ARGS="$API_ARGS training.api_timeout=$API_TIMEOUT"
else
    echo "Using local model: $MODEL_PATH"
fi

# Run SFT data collection only (no training)
python3 -m pettingllms.sft_train.train \
    --config-path ../config/code \
    --config-name code_L0_single_agent \
    $model_0_resource \
    base_models.policy_0.path="$MODEL_PATH" \
    models.model_0.path="$MODEL_PATH" \
    training.experiment_name=$EXPERIMENT_NAME \
    training.max_prompt_length=4096 \
    training.max_response_length=2048 \
    env.dataset=code_contests \
    env.benchmark=code_contests \
    env.max_turns=3 \
    +training.sft_mode=train \
    +training.sft_num_episodes=100 \
    +training.sft_data_dir="$OUTPUT_DIR" \
    +training.collect_mode=env \
    +training.run_sft_training=False \
    $API_ARGS

echo "SFT data collection completed!"
echo "Collected data saved to: $OUTPUT_DIR"
echo "To train on this data, use code_sft_train_only.sh"
