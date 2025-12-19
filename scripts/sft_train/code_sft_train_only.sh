#!/bin/bash
# SFT Training Script for Code Environment (Training Only)
# This script trains on already collected SFT data

set -x

# GPU Configuration
export CUDA_VISIBLE_DEVICES=2,3
export TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:False"
export HYDRA_FULL_ERROR=1
export NCCL_IB_DISABLE=1
export NCCL_NET_GDR_LEVEL=0

# CUDA Environment
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
export LD_LIBRARY_PATH=$CUDA_HOME/targets/x86_64-linux/lib:${LD_LIBRARY_PATH}
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH}

# Configuration
MODEL_PATH="your trained model path here"  # Model for SFT training
TRAIN_DATA_PATH="./sft_data_code/sft_data_20241203_000000.jsonl"  # Update with your actual data file
SFT_MODEL_OUTPUT="./sft_model_code"

# Check if training data exists
if [ ! -f "$TRAIN_DATA_PATH" ]; then
    echo "Error: Training data file not found: $TRAIN_DATA_PATH"
    echo "Please update TRAIN_DATA_PATH in the script or run data collection first."
    exit 1
fi

# Run SFT training only
python3 -m pettingllms.sft_train.train_sft \
    --model-name "$MODEL_PATH" \
    --train-file "$TRAIN_DATA_PATH" \
    --output-dir "$SFT_MODEL_OUTPUT" \
    --num-epochs 3 \
    --batch-size 2 \
    --gradient-accumulation 8 \
    --learning-rate 5e-5 \
    --use-lora \
    --lora-r 64 \
    --lora-alpha 16 \
    --bf16 \
    --max-seq-length 4096

echo "SFT training completed!"
echo "Trained model saved to: $SFT_MODEL_OUTPUT"
