#!/usr/bin/env bash
set -euo pipefail

# Merge-ready MetaAgent-X co-training example.
#
# Paper features exposed here:
#   1. Hierarchical rollout: DESIGN_SAMPLE_NUM=M designs and
#      EXECUTE_SAMPLE_NUM=N executions per design.
#   2. Stage-wise co-training: DESIGNER_LR and EXECUTOR_LR alternate every
#      LR_ALTERNATE_STEPS optimization steps.
#
# This public example intentionally keeps the training mode fixed:
# shared policy, question-level executor reward grouping, and all rollout data
# used for optimization.

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}"
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:False}"
export VLLM_USE_V1="${VLLM_USE_V1:-1}"
export VLLM_ALLOW_LONG_MAX_MODEL_LEN="${VLLM_ALLOW_LONG_MAX_MODEL_LEN:-1}"
export VLLM_ENGINE_ITERATION_TIMEOUT_S="${VLLM_ENGINE_ITERATION_TIMEOUT_S:-100000000000}"
export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export NCCL_NET_GDR_LEVEL="${NCCL_NET_GDR_LEVEL:-0}"
export NCCL_TIMEOUT="${NCCL_TIMEOUT:-3600}"
export NCCL_ASYNC_ERROR_HANDLING="${NCCL_ASYNC_ERROR_HANDLING:-1}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-0}"
export MAX_ROLLOUT_CONCURRENCY="${MAX_ROLLOUT_CONCURRENCY:-64}"
export MAX_ROLLOUT_RETRIES="${MAX_ROLLOUT_RETRIES:-3}"
export VLLM_ENABLE_V1_MULTIPROCESSING="${VLLM_ENABLE_V1_MULTIPROCESSING:-0}"
export VLLM_CUDAGRAPH_MODE="${VLLM_CUDAGRAPH_MODE:-piecewise}"

CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export CUDA_HOME
export TRITON_PTXAS_PATH="${TRITON_PTXAS_PATH:-$CUDA_HOME/bin/ptxas}"
[ -d "$CUDA_HOME/targets/x86_64-linux/lib" ] && export LD_LIBRARY_PATH="$CUDA_HOME/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}"
[ -d "$CUDA_HOME/lib64" ] && export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
GPU_NUM="${GPU_NUM:-8}"
CONFIG_NAME="${CONFIG_NAME:-math_L1_prompt}"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-1.7B}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-autoeval_cotrain_example_mixed_4d4e_alt10_8k}"

DESIGN_SAMPLE_NUM="${DESIGN_SAMPLE_NUM:-4}"
EXECUTE_SAMPLE_NUM="${EXECUTE_SAMPLE_NUM:-4}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-8}"
TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-400}"
VALIDATE_SAMPLE_NUM="${VALIDATE_SAMPLE_NUM:-1}"
VAL_FREQ="${VAL_FREQ:-10}"
SAVE_FREQ="${SAVE_FREQ:-10}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-8192}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-8192}"

# "first executor" soft alternation: executor starts with the normal LR and
# designer starts near-zero; LRs swap every LR_ALTERNATE_STEPS.
DESIGNER_LR="${DESIGNER_LR:-1e-9}"
EXECUTOR_LR="${EXECUTOR_LR:-5e-6}"
LR_ALTERNATE_STEPS="${LR_ALTERNATE_STEPS:-10}"

ENV_NAME="${ENV_NAME:-mixed_env}"
DATASET_CODE="${DATASET_CODE:-code_contests}"
BENCHMARK_CODE="${BENCHMARK_CODE:-livecodebench}"
BENCHMARK_MATH="${BENCHMARK_MATH:-[AIME25,AIME24]}"
APPS_RATIO="${APPS_RATIO:-0.7}"
MAX_CODE_VAL="${MAX_CODE_VAL:-50}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.8}"
LOGGER="${LOGGER:-[console]}"
RESUME_MODE="${RESUME_MODE:-auto}"
VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-true}"

model_0_config_path="models.model_0.ppo_trainer_config"
model_0_resource="resource.n_gpus_per_node=$GPU_NUM $model_0_config_path.trainer.n_gpus_per_node=$GPU_NUM $model_0_config_path.trainer.nnodes=1 $model_0_config_path.actor_rollout_ref.rollout.tensor_model_parallel_size=$GPU_NUM"

echo "AutoEval co-training example"
echo "  model: $MODEL_PATH"
echo "  config: $CONFIG_NAME"
echo "  rollout tree: M=$DESIGN_SAMPLE_NUM designs, N=$EXECUTE_SAMPLE_NUM executions"
echo "  shared policy: enabled"
echo "  executor reward grouping: question"
echo "  stage-wise LR alternation: designer=$DESIGNER_LR executor=$EXECUTOR_LR every $LR_ALTERNATE_STEPS steps"

exec "$PYTHON_BIN" -u -m pettingllms.trainer.train --config-path ../config/autoevol --config-name "$CONFIG_NAME" \
    $model_0_resource \
    base_models.policy_0.path="$MODEL_PATH" \
    lora_rank=0 \
    lora_alpha=16 \
    "training.logger=$LOGGER" \
    training.experiment_name="$EXPERIMENT_NAME" \
    training.total_training_steps="$TOTAL_TRAINING_STEPS" \
    training.train_batch_size="$TRAIN_BATCH_SIZE" \
    training.design_sample_num="$DESIGN_SAMPLE_NUM" \
    training.execute_sample_num="$EXECUTE_SAMPLE_NUM" \
    training.executor_group_mode=question \
    training.validate_sample_num="$VALIDATE_SAMPLE_NUM" \
    training.max_prompt_length="$MAX_PROMPT_LENGTH" \
    training.max_response_length="$MAX_RESPONSE_LENGTH" \
    training.val_freq="$VAL_FREQ" \
    training.save_freq="$SAVE_FREQ" \
    training.train_data_mode=all \
    training.lr_alternate_steps="$LR_ALTERNATE_STEPS" \
    training.designer_lr="$DESIGNER_LR" \
    training.executor_lr="$EXECUTOR_LR" \
    env.name="$ENV_NAME" \
    env.dataset_code="$DATASET_CODE" \
    env.benchmark_code="$BENCHMARK_CODE" \
    +env.max_code_val="$MAX_CODE_VAL" \
    env.apps_ratio="$APPS_RATIO" \
    "env.benchmark_math=$BENCHMARK_MATH" \
    "$model_0_config_path.trainer.resume_mode=$RESUME_MODE" \
    "$model_0_config_path.trainer.experiment_name=$EXPERIMENT_NAME" \
    "$model_0_config_path.trainer.val_before_train=$VAL_BEFORE_TRAIN" \
    "$model_0_config_path.actor.ppo_micro_batch_size_per_gpu=1" \
    "$model_0_config_path.actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1" \
    "$model_0_config_path.actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=true" \
    "+$model_0_config_path.actor.optim.lr=$EXECUTOR_LR" \
    "+$model_0_config_path.actor.use_kl_loss=false" \
    "+$model_0_config_path.actor.kl_loss_coef=0.0" \
    "+$model_0_config_path.actor.entropy_coeff=0.00" \
    "$model_0_config_path.actor_rollout_ref.rollout.gpu_memory_utilization=$GPU_MEMORY_UTILIZATION"
