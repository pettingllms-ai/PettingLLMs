set -x

export CUDA_VISIBLE_DEVICES=4,5
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
export LD_LIBRARY_PATH=$CUDA_HOME/targets/x86_64-linux/lib:${LD_LIBRARY_PATH}

export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH}


GPU_num=2


model_0_config_path="models.model_0.ppo_trainer_config"
model_0_resource="resource.n_gpus_per_node=$GPU_num  $model_0_config_path.trainer.n_gpus_per_node=$GPU_num $model_0_config_path.trainer.nnodes=1 $model_0_config_path.actor_rollout_ref.rollout.tensor_model_parallel_size=$GPU_num"


python3 -m pettingllms.trainer.train --config-path ../config/stateful --config-name stateful_L1_prompt \
    $model_0_resource \
    lora_rank=0\
    env.max_turns=4\
    base_models.policy_0.path="/home/lah003/models/Qwen3-8B"\
    training.experiment_name=suduku_8B_grpo\
    training.total_training_steps=200\
    training.train_batch_size=32\
    training.train_sample_num=8\
    training.validate_sample_num=1\
    training.max_prompt_length=8192\
    training.max_response_length=4096\
    training.val_freq=10\
    +env.map_size=4\
    env.benchmark=suduku\