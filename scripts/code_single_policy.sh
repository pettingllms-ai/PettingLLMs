set -x
export RAY_TMPDIR="/home/lah003/workspace/PettingLLMs/tmp"
export CUDA_VISIBLE_DEVICES=2
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
model_0_data_dir=~/data/code/model_0


model_0_USE_GRPO="$model_0_config_path.algorithm.adv_estimator=grpo $model_0_config_path.actor_rollout_ref.actor.use_kl_loss=False"

model_0_resource="resource.n_gpus_per_node=1  $model_0_config_path.trainer.n_gpus_per_node=1 $model_0_config_path.trainer.nnodes=1 $model_0_config_path.actor_rollout_ref.rollout.tensor_model_parallel_size=1"

model_0_data="+$model_0_config_path.data.train_files=$model_0_data_dir/text/train.parquet +$model_0_config_path.data.val_files=$model_0_data_dir/text/test.parquet"

python3 -m pettingllms.trainer.train --config-path ../config/code --config-name code_eval \
    experiment_name=code_1.7B_single_policy_livecodebench \
    +if_dapo=True\
    +num_workers=10000\
    enable_thinking=false\
    models.model_0.path="/home/lah003/models/Qwen3-8B"\
    benchmark=livecodebench\
    models.model_0.path=Qwen/Qwen3-8B\
    +difficulty=easy\
    trainer.total_training_steps=400\
    trainer.save_freq=150\
    data.epoch_size=40\
    data.gen_batch_size=128\
    data.gen_n_samples=4\
    data.resample_freq=3\
    data.max_prompt_length=8192\
    data.max_response_length=4096\
    $total_resource \
    $model_0_USE_GRPO $model_0_resource $model_0_data \
    sample_mode=tree \
    data.filter_ratio=0.3\
    data.filter_method=mean\
    env.max_turns=3\