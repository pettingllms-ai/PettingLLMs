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

model_0_config_path="models.model_0.ppo_trainer_config"
model_0_data_dir=~/data/math/model_0



model_0_resource="resource.n_gpus_per_node=1  $model_0_config_path.trainer.n_gpus_per_node=1 $model_0_config_path.trainer.nnodes=1 $model_0_config_path.actor_rollout_ref.rollout.tensor_model_parallel_size=1"


python3 -m pettingllms.trainer.train --config-path ../config/stateful --config-name stateful_single_policy \
    $model_0_resource $model_0_data\
    models.model_0.path=/home/nvidia/data/models/Qwen3-1.7B\
    experiment_name=plan_path_single_policy_1.7B\
    if_dapo=True\
    benchmark=plan_path\
    trainer.total_training_steps=400\
    trainer.save_freq=20\
    data.epoch_size=20\
    data.gen_batch_size=128\
    data.gen_n_samples=4\
    data.max_prompt_length=12000\
    data.max_response_length=2048\
    data.resample_freq=1\
    data.filter_method=std\
    data.filter_ratio=0\
    sample_mode=tree\
    env.max_turns=3\