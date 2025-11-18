set -x

export CUDA_VISIBLE_DEVICES=0
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


GPU_num=1


model_0_config_path="models.model_0.ppo_trainer_config"
model_0_resource="resource.n_gpus_per_node=$GPU_num  $model_0_config_path.trainer.n_gpus_per_node=$GPU_num $model_0_config_path.trainer.nnodes=1 $model_0_config_path.actor_rollout_ref.rollout.tensor_model_parallel_size=$GPU_num"


python3 -m pettingllms.trainer.train --config-path ../config/pychecker_rl --config-name pychecker_rl_L1_prompt \
    $model_0_resource \
    base_models.policy_0.path="/home/lah003/models/Qwen3-1.7B"\
    training.experiment_name=pychecker_rl_1.7B_prompt\
    training.total_training_steps=200\
    training.epoch_size=20\
    training.train_batch_size=32\
    training.train_sample_num=8\
    training.validate_sample_num=5\
    training.max_prompt_length=4096\
    training.max_response_length=2048\
    training.val_freq=10\
    training.resample_freq=3\
    env.dataset=pychecker\
    env.benchmark=pychecker

