set -x
export RAY_TMPDIR="/home/lah003/workspace/PettingLLMs/tmp"
export CUDA_VISIBLE_DEVICES=4,5,6,7
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

model_1_config_path="models.model_1.ppo_trainer_config"
model_1_data_dir=~/data/code/model_1

model_0_USE_GRPO="$model_0_config_path.algorithm.adv_estimator=grpo $model_0_config_path.actor_rollout_ref.actor.use_kl_loss=False"
model_1_USE_GRPO="$model_1_config_path.algorithm.adv_estimator=grpo $model_1_config_path.actor_rollout_ref.actor.use_kl_loss=False"
total_resource="resource.n_gpus_per_node=4"

model_0_resource=" $model_0_config_path.trainer.n_gpus_per_node=2 $model_0_config_path.trainer.nnodes=1 $model_0_config_path.actor_rollout_ref.rollout.tensor_model_parallel_size=2"
model_1_resource="  $model_1_config_path.trainer.n_gpus_per_node=2 $model_1_config_path.trainer.nnodes=1 $model_1_config_path.actor_rollout_ref.rollout.tensor_model_parallel_size=2"

model_0_data="+$model_0_config_path.data.train_files=$model_0_data_dir/text/train.parquet +$model_0_config_path.data.val_files=$model_0_data_dir/text/test.parquet"
model_1_data="+$model_1_config_path.data.train_files=$model_1_data_dir/text/train.parquet +$model_1_config_path.data.val_files=$model_1_data_dir/text/test.parquet"
python3 -m pettingllms.trainer.train --config-path ../config/code --config-name code_two_policies \
    experiment_name=code_eval_two_policies \
    $total_resource \
    $model_0_USE_GRPO $model_0_resource $model_0_data \
    $model_1_USE_GRPO $model_1_resource $model_1_data\
    sample_mode=tree \
    data.filter_ratio=0.5\
