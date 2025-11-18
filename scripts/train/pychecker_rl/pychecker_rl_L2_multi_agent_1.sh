set -x

# GPU Configuration (GPU 3,4)
export CUDA_VISIBLE_DEVICES=3,4
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

# CPU Resource Configuration (112 CPUs for this task)
# With 384 workers: 0.2625 CPU per worker, 90% utilization
export RAY_NUM_CPUS=112


export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
export LD_LIBRARY_PATH=$CUDA_HOME/targets/x86_64-linux/lib:${LD_LIBRARY_PATH}

export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH}


GPU_num=2


model_0_config_path="models.model_0.ppo_trainer_config"
model_0_resource="resource.n_gpus_per_node=$GPU_num  $model_0_config_path.trainer.n_gpus_per_node=$GPU_num $model_0_config_path.trainer.nnodes=1 $model_0_config_path.actor_rollout_ref.rollout.tensor_model_parallel_size=$GPU_num"


python3 -m pettingllms.trainer.train --config-path ../config/pychecker_rl --config-name pychecker_rl_L2_multi_agent \
    $model_0_resource \
    base_models.policy_0.path="/home/lah003/models/PRO-V-R1"\
    training.experiment_name=pychecker_rl_after_stl_8B_gpu34\
    training.total_training_steps=200\
    training.train_batch_size=128\
    training.train_sample_num=4\
    training.validate_sample_num=1\
    training.max_prompt_length=8192\
    training.max_response_length=8192\
    $model_0_config_path.actor_rollout_ref.rollout.max_model_len=16384\
    $model_0_config_path.actor_rollout_ref.rollout.gpu_memory_utilization=0.7\
    +training.save_freq=20\
    training.val_freq=10\
    training.num_workers=512\
    agent_policy_configs.agent_configs.agent_0.enable_thinking=true\
    agent_policy_configs.agent_configs.agent_1.enable_thinking=true\
    env.dataset=pychecker\
    env.benchmark=pychecker\
    env.max_turns=1

