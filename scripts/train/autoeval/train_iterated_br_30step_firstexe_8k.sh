set -x
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# Hard-pin FLASH_ATTN backend. Avoids FLASHINFER JIT nvcc path crashes in ray workers.
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export VLLM_USE_FLASHINFER_SAMPLER=0
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:False"
export VLLM_USE_V1=1
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_ENGINE_ITERATION_TIMEOUT_S=100000000000
export HYDRA_FULL_ERROR=1
export NCCL_IB_DISABLE=1
export NCCL_NET_GDR_LEVEL=0
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
# HuggingFace mirror (avoid direct hf.co access issues)
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_ENABLE_HF_TRANSFER=0

# Auto-detect CUDA: prefer conda env, fallback to system CUDA
if [ -n "$CONDA_PREFIX" ] && [ -d "$CONDA_PREFIX" ]; then
    CONDA_CUDA_BIN=$(find "$CONDA_PREFIX" -name "ptxas" -type f 2>/dev/null | head -1 | xargs dirname 2>/dev/null)
    if [ -n "$CONDA_CUDA_BIN" ] && [ -f "$CONDA_CUDA_BIN/ptxas" ]; then
        export TRITON_PTXAS_PATH="$CONDA_CUDA_BIN/ptxas"
        export CUDA_HOME=$(dirname "$CONDA_CUDA_BIN" 2>/dev/null)
        echo "Using CUDA from conda env: $CUDA_HOME"
    else
        export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
        export TRITON_PTXAS_PATH=${TRITON_PTXAS_PATH:-$CUDA_HOME/bin/ptxas}
        echo "Using system CUDA: $CUDA_HOME"
    fi
else
    export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
    export TRITON_PTXAS_PATH=${TRITON_PTXAS_PATH:-$CUDA_HOME/bin/ptxas}
    echo "Using system CUDA: $CUDA_HOME"
fi

if [ -n "$CUDA_HOME" ]; then
    [ -d "$CUDA_HOME/targets/x86_64-linux/lib" ] && export LD_LIBRARY_PATH=$CUDA_HOME/targets/x86_64-linux/lib:${LD_LIBRARY_PATH}
    [ -d "$CUDA_HOME/lib64" ] && export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH}
fi

# --- Split-policy resource layout: 2 models, TP=4 each ---
GPU_num=8
GPU_per_model=$((GPU_num / 2))

model_0_config_path="models.model_0.ppo_trainer_config"
model_0_resource="resource.n_gpus_per_node=$GPU_num  $model_0_config_path.trainer.n_gpus_per_node=$GPU_per_model $model_0_config_path.trainer.nnodes=1 $model_0_config_path.actor_rollout_ref.rollout.tensor_model_parallel_size=$GPU_per_model"

model_1_config_path="models.model_1.ppo_trainer_config"
model_1_resource="$model_1_config_path.trainer.n_gpus_per_node=$GPU_per_model $model_1_config_path.trainer.nnodes=1 $model_1_config_path.actor_rollout_ref.rollout.tensor_model_parallel_size=$GPU_per_model"

# --- Configurable parameters ---
DESIGN_SAMPLE_NUM=${DESIGN_SAMPLE_NUM:-4}
EXECUTE_SAMPLE_NUM=${EXECUTE_SAMPLE_NUM:-4}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-8}
EXECUTOR_GROUP_MODE=${EXECUTOR_GROUP_MODE:-question}
MODEL_PATH=${MODEL_PATH:-"Mercury7353/masrl_0228_mix_coldstart"}
APPS_RATIO=${APPS_RATIO:-0.7}
PHASE_ALTERNATE_STEPS=${PHASE_ALTERNATE_STEPS:-30}
FIRST_PHASE_TRAINS=${FIRST_PHASE_TRAINS:-executor}
DESIGNER_LR=${DESIGNER_LR:-5e-6}
EXECUTOR_LR=${EXECUTOR_LR:-5e-6}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-"autoeval_iterated_2policy_30step_firstexe"}

PY="/mnt/afs/zhangyaolun/safe_model/tool/PettingLLMs/pettingllms_venv/bin/python"
echo "Using python: $PY"
exec "$PY" -u -m pettingllms.trainer.train --config-path ../config/autoevol --config-name math_L1_iterated_br \
    $model_0_resource \
    $model_1_resource \
    base_models.policy_0.path="$MODEL_PATH"\
    base_models.policy_1.path="$MODEL_PATH"\
    lora_rank=0\
    lora_alpha=16\
    training.experiment_name=$EXPERIMENT_NAME\
    training.total_training_steps=400\
    training.train_batch_size=$TRAIN_BATCH_SIZE\
    training.design_sample_num=$DESIGN_SAMPLE_NUM\
    training.execute_sample_num=$EXECUTE_SAMPLE_NUM\
    training.executor_group_mode=$EXECUTOR_GROUP_MODE\
    training.validate_sample_num=1\
    training.max_prompt_length=7000\
    training.max_response_length=8192\
    training.val_freq=10\
    training.save_freq=10\
    training.train_data_mode=all\
    training.phase_alternate_steps=$PHASE_ALTERNATE_STEPS\
    training.first_phase_trains=$FIRST_PHASE_TRAINS\
    training.lr_alternate_steps=0\
    training.designer_lr=$DESIGNER_LR\
    training.executor_lr=$EXECUTOR_LR\
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
    +$model_0_config_path.actor.optim.lr=$DESIGNER_LR\
    +$model_0_config_path.actor.use_kl_loss=false\
    +$model_0_config_path.actor.kl_loss_coef=0.0\
    +$model_0_config_path.actor.entropy_coeff=0.00\
    $model_0_config_path.actor_rollout_ref.actor.fsdp_config.optimizer_offload=true\
    $model_0_config_path.actor_rollout_ref.rollout.gpu_memory_utilization=0.6\
    $model_1_config_path.trainer.resume_mode=auto\
    $model_1_config_path.trainer.experiment_name=$EXPERIMENT_NAME\
    $model_1_config_path.trainer.val_before_train=False\
    $model_1_config_path.actor.ppo_micro_batch_size_per_gpu=1\
    $model_1_config_path.actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1\
    $model_1_config_path.actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=true\
    +$model_1_config_path.actor.optim.lr=$EXECUTOR_LR\
    +$model_1_config_path.actor.use_kl_loss=false\
    +$model_1_config_path.actor.kl_loss_coef=0.0\
    +$model_1_config_path.actor.entropy_coeff=0.00\
    $model_1_config_path.actor_rollout_ref.actor.fsdp_config.optimizer_offload=true\
    $model_1_config_path.actor_rollout_ref.rollout.gpu_memory_utilization=0.6


# Usage examples:
#   Default 4x4, phase_alternate_steps=30, firstexe:
#     bash train_iterated_br_30step_firstexe_8k.sh
#   Longer phases:
#     PHASE_ALTERNATE_STEPS=60 bash train_iterated_br_30step_firstexe_8k.sh
#   Train designer first instead of executor:
#     FIRST_PHASE_TRAINS=designer bash train_iterated_br_30step_firstexe_8k.sh
