set -x

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export VLLM_USE_FLASHINFER_SAMPLER=0
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:False"
export VLLM_USE_V1=1
export VLLM_CUDART_SO_PATH=/usr/local/cuda/targets/x86_64-linux/lib/libcudart.so.12
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_ENGINE_ITERATION_TIMEOUT_S=100000000000
export HYDRA_FULL_ERROR=1
export NCCL_IB_DISABLE=1
export NCCL_NET_GDR_LEVEL=0
export WANDB_API_KEY=e58969ddb292f80e531902b9a0e741b05d22f4ee
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
# For split_policy with 2 models, need at least 2 GPUs (one per model)
GPU_num=8
# For 2 models, each model gets GPU_num // 2 GPUs
GPU_per_model=$((GPU_num / 2))


model_0_config_path="models.model_0.ppo_trainer_config"
model_0_resource="resource.n_gpus_per_node=$GPU_num  $model_0_config_path.trainer.n_gpus_per_node=$GPU_per_model $model_0_config_path.trainer.nnodes=1 $model_0_config_path.actor_rollout_ref.rollout.tensor_model_parallel_size=$GPU_per_model"

model_1_config_path="models.model_1.ppo_trainer_config"
model_1_resource="$model_1_config_path.trainer.n_gpus_per_node=$GPU_per_model $model_1_config_path.trainer.nnodes=1 $model_1_config_path.actor_rollout_ref.rollout.tensor_model_parallel_size=$GPU_per_model"
#/mnt/afs/share_data/models_weights/external/Qwen/Qwen3/Qwen3-4B
#/mnt/afs/zhangyaolun/safe_model/tool/LLaMA-Factory/saves/masrl/0128_math_designer_only_wo_think/sft/checkpoint-1854
python -m pettingllms.trainer.train --config-path ../config/autoevol --config-name math_L1_split_policy \
    $model_0_resource \
    $model_1_resource \
    base_models.policy_0.path="Mercury7353/masrlnothink0128"\
    base_models.policy_1.path="Qwen/Qwen3-8B"\
    training.experiment_name=autoeval_L1_split_policy\
    training.total_training_steps=400\
    training.train_batch_size=32\
    training.train_sample_num=8\
    training.validate_sample_num=3\
    training.max_prompt_length=2048\
    training.max_response_length=1024\
    training.val_freq=10\
    env.dataset=dapo_math\
    env.benchmark=AIME24\
    $model_0_config_path.actor_rollout_ref.actor.ppo_micro_batch_size=null\
    $model_0_config_path.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2\
    $model_0_config_path.actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4\
    $model_0_config_path.actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=true\
    $model_0_config_path.actor_rollout_ref.rollout.gpu_memory_utilization=0.6\
    $model_1_config_path.actor_rollout_ref.actor.ppo_micro_batch_size=null\
    $model_1_config_path.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2\
    $model_1_config_path.actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4\
    $model_1_config_path.actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=true\
    $model_1_config_path.actor_rollout_ref.rollout.gpu_memory_utilization=0.6\
