++ pwd
+ export RAY_TMPDIR=/home/lah003/workspace/PettingLLMs/tmp
+ RAY_TMPDIR=/home/lah003/workspace/PettingLLMs/tmp
+ export CUDA_VISIBLE_DEVICES=2,3
+ CUDA_VISIBLE_DEVICES=2,3
+ export TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas
+ TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas
+ export VLLM_ATTENTION_BACKEND=FLASH_ATTN
+ VLLM_ATTENTION_BACKEND=FLASH_ATTN
+ export VLLM_USE_FLASHINFER_SAMPLER=0
+ VLLM_USE_FLASHINFER_SAMPLER=0
+ export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False
+ PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False
+ export VLLM_USE_V1=1
+ VLLM_USE_V1=1
+ export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
+ VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
+ export VLLM_ENGINE_ITERATION_TIMEOUT_S=100000000000
+ VLLM_ENGINE_ITERATION_TIMEOUT_S=100000000000
+ export HYDRA_FULL_ERROR=1
+ HYDRA_FULL_ERROR=1
+ export NCCL_IB_DISABLE=1
+ NCCL_IB_DISABLE=1
+ export NCCL_NET_GDR_LEVEL=0
+ NCCL_NET_GDR_LEVEL=0
+ export CUDA_HOME=/usr/local/cuda
+ CUDA_HOME=/usr/local/cuda
+ export LD_LIBRARY_PATH=/usr/local/cuda/targets/x86_64-linux/lib:
+ LD_LIBRARY_PATH=/usr/local/cuda/targets/x86_64-linux/lib:
+ export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/cuda/targets/x86_64-linux/lib:
+ LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/cuda/targets/x86_64-linux/lib:
+ model_0_config_path=models.model_0.ppo_trainer_config
+ model_0_data_dir=/home/lah003/data/code/model_0
+ model_0_USE_GRPO='models.model_0.ppo_trainer_config.algorithm.adv_estimator=grpo models.model_0.ppo_trainer_config.actor_rollout_ref.actor.use_kl_loss=False'
+ model_0_resource='resource.n_gpus_per_node=2  models.model_0.ppo_trainer_config.trainer.n_gpus_per_node=2 models.model_0.ppo_trainer_config.trainer.nnodes=1 models.model_0.ppo_trainer_config.actor_rollout_ref.rollout.tensor_model_parallel_size=2'
+ model_0_data='+models.model_0.ppo_trainer_config.data.train_files=/home/lah003/data/code/model_0/text/train.parquet +models.model_0.ppo_trainer_config.data.val_files=/home/lah003/data/code/model_0/text/test.parquet'
+ python3 -m pettingllms.trainer.train --config-path ../config/code --config-name code_eval_single_agent models.model_0.ppo_trainer_config.algorithm.adv_estimator=grpo models.model_0.ppo_trainer_config.actor_rollout_ref.actor.use_kl_loss=False resource.n_gpus_per_node=2 models.model_0.ppo_trainer_config.trainer.n_gpus_per_node=2 models.model_0.ppo_trainer_config.trainer.nnodes=1 models.model_0.ppo_trainer_config.actor_rollout_ref.rollout.tensor_model_parallel_size=2 +models.model_0.ppo_trainer_config.data.train_files=/home/lah003/data/code/model_0/text/train.parquet +models.model_0.ppo_trainer_config.data.val_files=/home/lah003/data/code/model_0/text/test.parquet data.epoch_size=1 data.resample_freq=10 sample_mode=tree experiment_name=4B_base_max_tree_resample_10
[MainTraining] Starting main training function at 02:42:42
[MainTraining] Loading config | +0.00s | Total: 0.00s
[MainTraining] Config loaded, starting PPO training | +0.01s | Total: 0.01s
[PPORunner] Starting run_ppo function at 02:42:42
[PPORunner] Initializing Ray cluster | +0.00s | Total: 0.00s
Received signal 2, cleaning up...

==================================================
STARTING RAY CLEANUP...
==================================================
Ray is not initialized, but will force cleanup anyway...
[PPORunner] Executing cleanup in run_ppo | +6.55s | Total: 6.55s
Executing cleanup in run_ppo...

==================================================
STARTING RAY CLEANUP...
==================================================
Ray is not initialized, but will force cleanup anyway...
[PPORunner] run_ppo function completed | Total: 6.55s
[MainTraining] Executing final cleanup in main | +6.55s | Total: 6.56s
Executing final cleanup in main...

==================================================
STARTING RAY CLEANUP...
==================================================
Ray is not initialized, but will force cleanup anyway...
[MainTraining] Main training function completed | Total: 6.56s

==================================================
STARTING RAY CLEANUP...
==================================================
Ray is not initialized, but will force cleanup anyway...
