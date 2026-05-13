#!/bin/bash
# Smoke test for IBR (Iterated Best-Response) co-training.
# Runs 6 steps total with phase_alternate_steps=2 so you observe:
#   step 0,1: train executor, FREEZE designer_policy  -> logs "[IBR] ... designer_policy is frozen"
#   step 2,3: train designer, FREEZE executor_policy  -> logs "[IBR] ... executor_policy is frozen"
#   step 4,5: train executor again (second IBR round)
# After run, checkpoints/autoeval_iterated_br_smoketest/{designer_policy,executor_policy}/
# should have independent global_step_* folders.
#
# Expected signals in log:
#   [IBR] Step K: phase={0|1}, training={Designer|Executor}, frozen_policy=...
#   [IBR] Step K: designer_policy is frozen, skip update_actor   (OR executor_policy)

set -x
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
export NCCL_TIMEOUT=3600
export NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=WARN
# Set WANDB_API_KEY in the shell environment when using wandb logging.
export NCCL_NVLS_ENABLE=0
export MAX_ROLLOUT_CONCURRENCY=64
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export VLLM_CUDAGRAPH_MODE=piecewise
export MAX_ROLLOUT_RETRIES=3
# HuggingFace mirror (avoid direct hf.co access issues)
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_ENABLE_HF_TRANSFER=0

if [ -n "$CONDA_PREFIX" ] && [ -d "$CONDA_PREFIX" ]; then
    CONDA_CUDA_BIN=$(find "$CONDA_PREFIX" -name "ptxas" -type f 2>/dev/null | head -1 | xargs dirname 2>/dev/null)
    if [ -n "$CONDA_CUDA_BIN" ] && [ -f "$CONDA_CUDA_BIN/ptxas" ]; then
        export TRITON_PTXAS_PATH="$CONDA_CUDA_BIN/ptxas"
        export CUDA_HOME=$(dirname "$CONDA_CUDA_BIN" 2>/dev/null)
    else
        export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
        export TRITON_PTXAS_PATH=${TRITON_PTXAS_PATH:-$CUDA_HOME/bin/ptxas}
    fi
else
    export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
    export TRITON_PTXAS_PATH=${TRITON_PTXAS_PATH:-$CUDA_HOME/bin/ptxas}
fi

if [ -n "$CUDA_HOME" ]; then
    [ -d "$CUDA_HOME/targets/x86_64-linux/lib" ] && export LD_LIBRARY_PATH=$CUDA_HOME/targets/x86_64-linux/lib:${LD_LIBRARY_PATH}
    [ -d "$CUDA_HOME/lib64" ] && export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH}
fi

GPU_num=8
GPU_per_model=$((GPU_num / 2))

model_0_config_path="models.model_0.ppo_trainer_config"
model_0_resource="resource.n_gpus_per_node=$GPU_num  $model_0_config_path.trainer.n_gpus_per_node=$GPU_per_model $model_0_config_path.trainer.nnodes=1 $model_0_config_path.actor_rollout_ref.rollout.tensor_model_parallel_size=$GPU_per_model"
model_1_config_path="models.model_1.ppo_trainer_config"
model_1_resource="$model_1_config_path.trainer.n_gpus_per_node=$GPU_per_model $model_1_config_path.trainer.nnodes=1 $model_1_config_path.actor_rollout_ref.rollout.tensor_model_parallel_size=$GPU_per_model"

# --- Smoke-test knobs (small) ---
MODEL_PATH="Mercury7353/masrl_0228_mix_coldstart"
EXPERIMENT_NAME="autoeval_iterated_br_smoketest"
TOTAL_STEPS=6
PHASE_ALTERNATE_STEPS=2
TRAIN_BATCH_SIZE=2   # tiny, just to exercise the plumbing
DESIGN_SAMPLE_NUM=2
EXECUTE_SAMPLE_NUM=2

PY="${PYTHON_BIN:-python3}"
echo "Using python: $PY"
exec "$PY" -u -m pettingllms.trainer.train --config-path ../config/autoevol --config-name math_L1_iterated_br \
    $model_0_resource \
    $model_1_resource \
    base_models.policy_0.path="$MODEL_PATH"\
    base_models.policy_1.path="$MODEL_PATH"\
    lora_rank=0\
    lora_alpha=16\
    training.experiment_name=$EXPERIMENT_NAME\
    training.total_training_steps=$TOTAL_STEPS\
    training.train_batch_size=$TRAIN_BATCH_SIZE\
    training.design_sample_num=$DESIGN_SAMPLE_NUM\
    training.execute_sample_num=$EXECUTE_SAMPLE_NUM\
    training.executor_group_mode=question\
    training.validate_sample_num=1\
    training.max_prompt_length=4096\
    training.max_response_length=4096\
    training.val_freq=1000\
    training.save_freq=2\
    training.train_data_mode=all\
    training.phase_alternate_steps=$PHASE_ALTERNATE_STEPS\
    training.first_phase_trains=executor\
    training.lr_alternate_steps=0\
    training.designer_lr=5e-6\
    training.executor_lr=5e-6\
    env.name=mixed_env\
    env.dataset_code=code_contests\
    env.benchmark_code=livecodebench\
    +env.max_code_val=10\
    env.apps_ratio=0.7\
    'env.benchmark_math=[AIME25]'\
    $model_0_config_path.trainer.resume_mode=disable\
    $model_0_config_path.trainer.experiment_name=$EXPERIMENT_NAME\
    $model_0_config_path.trainer.val_before_train=False\
    $model_0_config_path.actor.ppo_micro_batch_size_per_gpu=1\
    $model_0_config_path.actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1\
    $model_0_config_path.actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=true\
    +$model_0_config_path.actor.optim.lr=5e-6\
    +$model_0_config_path.actor.use_kl_loss=false\
    +$model_0_config_path.actor.kl_loss_coef=0.0\
    +$model_0_config_path.actor.entropy_coeff=0.00\
    $model_0_config_path.actor_rollout_ref.rollout.gpu_memory_utilization=0.6\
    $model_1_config_path.trainer.resume_mode=disable\
    $model_1_config_path.trainer.experiment_name=$EXPERIMENT_NAME\
    $model_1_config_path.trainer.val_before_train=False\
    $model_1_config_path.actor.ppo_micro_batch_size_per_gpu=1\
    $model_1_config_path.actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1\
    $model_1_config_path.actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=true\
    +$model_1_config_path.actor.optim.lr=5e-6\
    +$model_1_config_path.actor.use_kl_loss=false\
    +$model_1_config_path.actor.kl_loss_coef=0.0\
    +$model_1_config_path.actor.entropy_coeff=0.00\
    $model_1_config_path.actor_rollout_ref.rollout.gpu_memory_utilization=0.6 2>&1 | tee smoketest_ibr.log

echo ""
echo "=============================================================="
echo "Smoke test done. Check:"
echo "  1. grep '[IBR]' smoketest_ibr.log      # phase/freeze log lines"
echo "  2. ls checkpoints/$EXPERIMENT_NAME/designer_policy/  # should have global_step_*"
echo "  3. ls checkpoints/$EXPERIMENT_NAME/executor_policy/  # should have global_step_*"
echo "  4. Expected pattern (phase_alternate_steps=2):"
echo "     step 0,1 -> frozen_policy=designer_policy"
echo "     step 2,3 -> frozen_policy=executor_policy"
echo "     step 4,5 -> frozen_policy=designer_policy"
echo "=============================================================="
