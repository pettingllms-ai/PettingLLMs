set -x

# === Auto-fix: regenerate datasets if aime_past has empty questions ===
SCRIPT_DIR="$(cd "$(dirname "$0")/../../.." && pwd)"
AIME_PARQUET="$SCRIPT_DIR/data/math/train/aime_past.parquet"
if [ -f "$AIME_PARQUET" ]; then
    EMPTY_Q=$(python3 -c "
import pandas as pd
df = pd.read_parquet('$AIME_PARQUET')
print((df['question'].str.len() == 0).sum())
" 2>/dev/null)
    if [ "$EMPTY_Q" -gt 0 ] 2>/dev/null; then
        echo "[AUTO-FIX] aime_past.parquet has $EMPTY_Q empty questions, regenerating..."
        python3 "$SCRIPT_DIR/scripts/dataprocess/load_math.py"
        echo "[AUTO-FIX] Done."
    fi
fi

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export VLLM_USE_FLASHINFER_SAMPLER=0
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:False"
export VLLM_USE_V1=1
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_ENGINE_ITERATION_TIMEOUT_S=100000000000
export HYDRA_FULL_ERROR=1
export NCCL_IB_DISABLE=1
export NCCL_NET_GDR_LEVEL=0
# Fix for 8 GPU NCCL hanging
export NCCL_TIMEOUT=3600
export NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=WARN
# Set WANDB_API_KEY in the shell environment when using wandb logging.
export NCCL_NVLS_ENABLE=0
# Auto-detect CUDA: prefer conda env, fallback to system CUDA
if [ -n "$CONDA_PREFIX" ] && [ -d "$CONDA_PREFIX" ]; then
    # Try to find CUDA in conda env
    CONDA_CUDA_BIN=$(find "$CONDA_PREFIX" -name "ptxas" -type f 2>/dev/null | head -1 | xargs dirname 2>/dev/null)
    if [ -n "$CONDA_CUDA_BIN" ] && [ -f "$CONDA_CUDA_BIN/ptxas" ]; then
        export TRITON_PTXAS_PATH="$CONDA_CUDA_BIN/ptxas"
        export CUDA_HOME=$(dirname "$CONDA_CUDA_BIN" 2>/dev/null)
        echo "Using CUDA from conda env: $CUDA_HOME"
    else
        # Fallback to system CUDA
        export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
        export TRITON_PTXAS_PATH=${TRITON_PTXAS_PATH:-$CUDA_HOME/bin/ptxas}
        echo "Using system CUDA: $CUDA_HOME"
    fi
else
    # No conda env, use system CUDA
    export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
    export TRITON_PTXAS_PATH=${TRITON_PTXAS_PATH:-$CUDA_HOME/bin/ptxas}
    echo "Using system CUDA: $CUDA_HOME"
fi

# Set library paths (only if CUDA_HOME is set)
if [ -n "$CUDA_HOME" ]; then
    [ -d "$CUDA_HOME/targets/x86_64-linux/lib" ] && export LD_LIBRARY_PATH=$CUDA_HOME/targets/x86_64-linux/lib:${LD_LIBRARY_PATH}
    [ -d "$CUDA_HOME/lib64" ] && export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH}
fi

# select gpus
GPU_num=8


model_0_config_path="models.model_0.ppo_trainer_config"
model_0_resource="resource.n_gpus_per_node=$GPU_num  $model_0_config_path.trainer.n_gpus_per_node=$GPU_num $model_0_config_path.trainer.nnodes=1 $model_0_config_path.actor_rollout_ref.rollout.tensor_model_parallel_size=$GPU_num"

# Baseline: 8x2 sampling, fixed LR (no alternating)
# Compare with train_design_tree.sh which uses lr_alternate_steps=10
python -m pettingllms.trainer.train --config-path ../config/autoevol --config-name math_L1_prompt \
    $model_0_resource \
    base_models.policy_0.path="Mercury7353/masrl_0228_mix_coldstart"\
    lora_rank=0\
    lora_alpha=16\
    training.experiment_name=autoeval_8x2_fixed_lr_baseline\
    training.total_training_steps=400\
    training.train_batch_size=8\
    training.design_sample_num=8\
    training.execute_sample_num=2\
    training.validate_sample_num=1\
    training.max_prompt_length=4096\
    training.max_response_length=8192\
    training.val_freq=10\
    training.save_freq=10\
    training.train_data_mode=all\
    training.designer_lr=5e-6\
    training.executor_lr=1e-6\
    env.name=mixed_env\
    env.dataset_code=code_contests\
    env.benchmark_code=code_contests\
    'env.benchmark_math=[AIME25]'\
    $model_0_config_path.trainer.val_before_train=False\
    $model_0_config_path.actor.ppo_micro_batch_size=null\
    $model_0_config_path.actor.ppo_micro_batch_size_per_gpu=1\
    $model_0_config_path.actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1\
    $model_0_config_path.actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=true\
    +$model_0_config_path.actor.optim.lr=5e-6\
    +$model_0_config_path.actor.use_kl_loss=false\
    +$model_0_config_path.actor.kl_loss_coef=0.0\
    +$model_0_config_path.actor.entropy_coeff=0.00\
    $model_0_config_path.actor_rollout_ref.rollout.gpu_memory_utilization=0.8\
