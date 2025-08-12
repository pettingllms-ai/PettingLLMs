set -x

export CUDA_VISIBLE_DEVICES=0,1

export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export VLLM_USE_FLASHINFER_SAMPLER=0
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:False"
export VLLM_USE_V1=1
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_ENGINE_ITERATION_TIMEOUT_S=100000000000
export HYDRA_FULL_ERROR=1


export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
export LD_LIBRARY_PATH=$CUDA_HOME/targets/x86_64-linux/lib:${LD_LIBRARY_PATH}

export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH}

model_0_config_path="models.model_0.ppo_trainer_config"
model_1_config_path="models.model_1.ppo_trainer_config"
train_data_size=256
val_data_size=128
model_0_data_dir=~/data/code/model_0
model_1_data_dir=~/data/code/model_1

model_0_USE_GRPO="$model_0_config_path.algorithm.adv_estimator=grpo $model_0_config_path.actor_rollout_ref.actor.use_kl_loss=True"
model_1_USE_GRPO="$model_1_config_path.algorithm.adv_estimator=grpo $model_1_config_path.actor_rollout_ref.actor.use_kl_loss=True"

model_0_resource="$model_0_config_path.trainer.n_gpus_per_node=1 $model_0_config_path.trainer.nnodes=1"
model_1_resource="$model_1_config_path.trainer.n_gpus_per_node=1 $model_1_config_path.trainer.nnodes=1"

model_0_data="+$model_0_config_path.data.train_files=$model_0_data_dir/text/train.parquet +$model_0_config_path.data.val_files=$model_0_data_dir/text/test.parquet"
model_1_data="+$model_1_config_path.data.train_files=$model_1_data_dir/text/train.parquet +$model_1_config_path.data.val_files=$model_1_data_dir/text/test.parquet"

python3 -m pettingllms.data.preprocess.prepare \
    --mode 'text' \
    --train_data_size $train_data_size \
    --local_dir $model_0_data_dir \
    --val_data_size $((val_data_size * 2)) # evaluate 2 × val_data_size tasks during each iteration

python3 -m pettingllms.data.preprocess.prepare \
    --mode 'text' \
    --train_data_size $train_data_size \
    --local_dir $model_1_data_dir \
    --val_data_size $((val_data_size * 2)) # evaluate 2 × val_data_size tasks during each iteration

python3 -m pettingllms.trainer.train --config-path ../config/code --config-name code_two_policies \
    $model_0_USE_GRPO $model_1_USE_GRPO $model_0_resource $model_1_resource $model_0_data $model_1_data