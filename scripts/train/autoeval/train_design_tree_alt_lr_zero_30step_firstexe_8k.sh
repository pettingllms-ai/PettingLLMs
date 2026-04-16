set -x
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# Auto-install flashinfer if missing (avoids TMA descriptor crash on Hopper GPUs)
# Also verify ninja is available for FlashInfer JIT compilation
if ! python3 -c "import flashinfer" 2>/dev/null; then
    echo "[INFO] flashinfer not found, installing flashinfer-python..."
    pip install flashinfer-python 2>&1 | tail -3
    if ! python3 -c "import flashinfer" 2>/dev/null; then
        echo "[WARN] flashinfer install failed, falling back to FLASH_ATTN backend"
        export VLLM_ATTENTION_BACKEND=FLASH_ATTN
    else
        echo "[INFO] flashinfer installed successfully"
        export VLLM_ATTENTION_BACKEND=FLASHINFER
    fi
else
    export VLLM_ATTENTION_BACKEND=FLASHINFER
fi

# FlashInfer JIT requires ninja for kernel compilation — install if missing
if [ "$VLLM_ATTENTION_BACKEND" = "FLASHINFER" ]; then
    if ! command -v ninja &>/dev/null; then
        echo "[INFO] ninja not found, installing for FlashInfer JIT..."
        pip install ninja 2>&1 | tail -3
        if ! command -v ninja &>/dev/null; then
            echo "[WARN] ninja install failed, falling back to FLASH_ATTN backend"
            export VLLM_ATTENTION_BACKEND=FLASH_ATTN
        fi
    fi
    # Clear stale FlashInfer JIT cache that may reference wrong environment paths
    if [ -d "/root/.cache/flashinfer" ]; then
        echo "[INFO] Clearing stale FlashInfer JIT cache at /root/.cache/flashinfer"
        rm -rf /root/.cache/flashinfer
    fi
fi
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
export WANDB_API_KEY=e58969ddb292f80e531902b9a0e741b05d22f4ee
export NCCL_NVLS_ENABLE=0
export MAX_ROLLOUT_CONCURRENCY=64
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export VLLM_CUDAGRAPH_MODE=piecewise
export MAX_ROLLOUT_RETRIES=3
#pip install ray==2.46.0
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

# /mnt/afs/zhangyaolun/safe_model/tool/LLaMA-Factory/saves/masrl/0128_math_designer_only_wo_think/sft/checkpoint-1854
# /mnt/afs/zhangyaolun/safe_model/tool/LLaMA-Factory/saves/masrl/0128_math_designer_only_wo_think/sft/checkpoint-838 # this is 0206 no tool
# Mercury7353/masrlnothink0128
# Mercury7353/masrl0206_notool
# /mnt/afs/zhangyaolun/safe_model/tool/LLaMA-Factory/saves/masrl/0226_math_code_mix_wo_think/sft/checkpoint-968
# autoeval_mixcoldstart_4design_8execution_4gpu
# train data: designer_only, executor_only, all
# Mercury7353/masrl_0228_mix_coldstart 
#
# /mnt/afs/zhangyaolun/safe_model/tool/PettingLLMs/checkpoints/autoeval_mixcoldstart_8design_1execution_designonly_8gpus/global_step_20/actor/checkpoint
# --- Configurable parameters ---
DESIGN_SAMPLE_NUM=${DESIGN_SAMPLE_NUM:-4}
EXECUTE_SAMPLE_NUM=${EXECUTE_SAMPLE_NUM:-4}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-8}
# executor_group_mode: "question" = all executors per problem, "design" = executors per design, "null" = auto
EXECUTOR_GROUP_MODE=${EXECUTOR_GROUP_MODE:-question}
MODEL_PATH=${MODEL_PATH:-"Mercury7353/masrl_0228_mix_coldstart"}
APPS_RATIO=${APPS_RATIO:-0.7}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-"autoeval_mix_${DESIGN_SAMPLE_NUM}d_${EXECUTE_SAMPLE_NUM}e_mix_question_grouping_altlr_zerolr_30steps_firstexe"}

python -m pettingllms.trainer.train --config-path ../config/autoevol --config-name math_L1_prompt \
    $model_0_resource \
    base_models.policy_0.path="$MODEL_PATH"\
    lora_rank=0\
    lora_alpha=16\
    training.experiment_name=$EXPERIMENT_NAME\
    training.total_training_steps=400\
    training.train_batch_size=$TRAIN_BATCH_SIZE\
    training.design_sample_num=$DESIGN_SAMPLE_NUM\
    training.execute_sample_num=$EXECUTE_SAMPLE_NUM\
    training.executor_group_mode=$EXECUTOR_GROUP_MODE\
    training.validate_sample_num=1\
    training.max_prompt_length=8192\
    training.max_response_length=8192\
    training.val_freq=10\
    training.save_freq=10\
    training.train_data_mode=all\
    training.designer_lr=1e-9\
    training.executor_lr=5e-6\
    training.lr_alternate_steps=30\
    env.name=mixed_env\
    env.dataset_code=code_contests\
    env.benchmark_code=livecodebench\
    +env.max_code_val=50\
    env.apps_ratio=$APPS_RATIO\
    'env.benchmark_math=[AIME25]'\
    $model_0_config_path.trainer.resume_mode=auto\
    $model_0_config_path.trainer.experiment_name=$EXPERIMENT_NAME\
    $model_0_config_path.trainer.val_before_train=False\
    $model_0_config_path.actor.ppo_micro_batch_size_per_gpu=1\
    $model_0_config_path.actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1\
    $model_0_config_path.actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=true\
    +$model_0_config_path.actor.optim.lr=5e-6\
    +$model_0_config_path.actor.use_kl_loss=false\
    +$model_0_config_path.actor.kl_loss_coef=0.0\
    +$model_0_config_path.actor.entropy_coeff=0.00\
    $model_0_config_path.actor_rollout_ref.rollout.gpu_memory_utilization=0.8


# Usage examples:
#   Default 4x4 auto grouping:
#     bash train_design_tree_mix.sh
#   4x4 with question grouping (all executors per problem in one GRPO group):
#     EXECUTOR_GROUP_MODE=question bash train_design_tree_mix.sh
#   4x4 with design grouping (executors per design in one GRPO group):
#     EXECUTOR_GROUP_MODE=design bash train_design_tree_mix.sh
#   8x2 with question grouping:
#     DESIGN_SAMPLE_NUM=8 EXECUTE_SAMPLE_NUM=2 EXECUTOR_GROUP_MODE=question bash train_design_tree_mix.sh