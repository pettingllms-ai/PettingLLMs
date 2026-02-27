set -x
# 复制 cudart 到本地 /tmp 避免 AFS 缓存问题                                                                                                      
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
export WANDB_API_KEY=e58969ddb292f80e531902b9a0e741b05d22f4ee
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

# /mnt/afs/zhangyaolun/safe_model/tool/LLaMA-Factory/saves/masrl/0128_math_designer_only_wo_think/sft/checkpoint-1854
# /mnt/afs/zhangyaolun/safe_model/tool/LLaMA-Factory/saves/masrl/0128_math_designer_only_wo_think/sft/checkpoint-838 # this is 0206 no tool
# Mercury7353/masrlnothink0128
# Mercury7353/masrl0206_notool
python -m pettingllms.trainer.train --config-path ../config/autoevol --config-name math_L1_prompt \
    $model_0_resource \
    base_models.policy_0.path="Mercury7353/masrl0206_notool"\
    lora_rank=0\
    lora_alpha=16\
    training.experiment_name=autoeval_L1_prompt_5e_6_8k_warmup_10\
    training.total_training_steps=400\
    training.train_batch_size=32\
    training.train_sample_num=8\
    training.validate_sample_num=3\
    training.max_prompt_length=4096\
    training.max_response_length=8192\
    training.val_freq=10\
    training.save_freq=10\
    env.dataset=polaris\
    env.benchmark=AIME24\
    $model_0_config_path.trainer.val_before_train=False\
    $model_0_config_path.actor.ppo_micro_batch_size=null\
    $model_0_config_path.actor.ppo_micro_batch_size_per_gpu=2\
    $model_0_config_path.actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4\
    $model_0_config_path.actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=true\
    +$model_0_config_path.actor.optim.lr=5e-6\
    +$model_0_config_path.actor.use_kl_loss=false\
    +$model_0_config_path.actor.kl_loss_coef=0.0\
    +$model_0_config_path.actor.entropy_coeff=0.00\
    $model_0_config_path.actor_rollout_ref.rollout.gpu_memory_utilization=0.6\


# +$model_0_config_path.actor.clip_ratio_low=0.15\
#    +$model_0_config_path.actor.clip_ratio_high=0.28\