#!/usr/bin/env bash
set -euo pipefail

# Eval-first AutoEval script for a single open model served by vLLM.
#
# MAS wiring:
#   CONFIG_NAME=math_L1_prompt has workflow_type=autoevol and maps the
#   Designer agent to the shared_model policy.
#   pettingllms.evaluate.evaluate selects MultiAgentsExecutionEngineAutoEvol.
#   The engine instantiates MASGenerator/MASExecutor from
#   pettingllms.multi_agent_env.autoevol.gen_agent and injects the runtime
#   AIClient plus the AutoEval workflow base package into generated mas.py.
#   SERVED_MODEL_NAME must match base_models.policy_0.name/models.model_0.name.

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}"
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"
export VLLM_USE_V1="${VLLM_USE_V1:-1}"
export VLLM_ALLOW_LONG_MAX_MODEL_LEN="${VLLM_ALLOW_LONG_MAX_MODEL_LEN:-1}"
export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"
export LLM_MAX_CONCURRENT="${LLM_MAX_CONCURRENT:-8}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
MODEL_PATH="${MODEL_PATH:-Mercury7353/MetaAgent-X}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-shared_model}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8300}"
START_VLLM="${START_VLLM:-true}"
KEEP_VLLM="${KEEP_VLLM:-false}"
TP_SIZE="${TP_SIZE:-1}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.80}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
MAX_WAIT_SECONDS="${MAX_WAIT_SECONDS:-600}"

CONFIG_NAME="${CONFIG_NAME:-math_L1_prompt}"
ENV_NAME="${ENV_NAME:-math_env}"
DATASET="${DATASET:-dapo_math}"
BENCHMARK="${BENCHMARK:-AIME24}"
VALIDATE_SAMPLE_NUM="${VALIDATE_SAMPLE_NUM:-3}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-8192}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-4096}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-autoeval_eval_first_${BENCHMARK}}"
LOGGER="${LOGGER:-[console]}"
NUM_WORKERS="${NUM_WORKERS:-8}"
MAS_CONCURRENCY="${MAS_CONCURRENCY:-4}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
CONFIG_PATH="$REPO_ROOT/pettingllms/config/autoevol"
VLLM_PID=""

wait_for_vllm() {
    local waited=0
    until curl -fsS "http://$HOST:$PORT/v1/models" >/dev/null 2>&1; do
        if [ "$waited" -ge "$MAX_WAIT_SECONDS" ]; then
            echo "Timed out waiting for vLLM at $HOST:$PORT"
            return 1
        fi
        sleep 2
        waited=$((waited + 2))
    done
}

cleanup() {
    if [ -n "$VLLM_PID" ] && [ "$KEEP_VLLM" != "true" ]; then
        kill "$VLLM_PID" 2>/dev/null || true
        wait "$VLLM_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

cd "$REPO_ROOT"

if curl -fsS "http://$HOST:$PORT/v1/models" >/dev/null 2>&1; then
    echo "Reusing existing vLLM server at $HOST:$PORT"
elif [ "$START_VLLM" = "true" ]; then
    echo "Starting vLLM for $MODEL_PATH as $SERVED_MODEL_NAME on $HOST:$PORT"
    "$PYTHON_BIN" -m vllm.entrypoints.openai.api_server \
        --model "$MODEL_PATH" \
        --served-model-name "$SERVED_MODEL_NAME" \
        --host "$HOST" \
        --port "$PORT" \
        --tensor-parallel-size "$TP_SIZE" \
        --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
        --max-model-len "$MAX_MODEL_LEN" \
        --enable-auto-tool-choice \
        --tool-call-parser hermes \
        >"/tmp/pettingllms_autoeval_vllm_${PORT}.log" 2>&1 &
    VLLM_PID=$!
    wait_for_vllm || {
        tail -n 80 "/tmp/pettingllms_autoeval_vllm_${PORT}.log" || true
        exit 1
    }
else
    echo "No vLLM server found at $HOST:$PORT and START_VLLM=false"
    exit 1
fi

echo "Running AutoEval validation on $BENCHMARK"
"$PYTHON_BIN" -m pettingllms.evaluate.evaluate \
    --config-path "$CONFIG_PATH" \
    --config-name "$CONFIG_NAME" \
    +vllm_address="$HOST:$PORT" \
    "training.logger=$LOGGER" \
    training.experiment_name="$EXPERIMENT_NAME" \
    training.validate_sample_num="$VALIDATE_SAMPLE_NUM" \
    training.max_prompt_length="$MAX_PROMPT_LENGTH" \
    training.max_response_length="$MAX_RESPONSE_LENGTH" \
    ++training.num_workers="$NUM_WORKERS" \
    ++training.mas_concurrency="$MAS_CONCURRENCY" \
    env.name="$ENV_NAME" \
    env.dataset="$DATASET" \
    env.benchmark="$BENCHMARK" \
    base_models.policy_0.path="$MODEL_PATH" \
    base_models.policy_0.name="$SERVED_MODEL_NAME" \
    models.model_0.path="$MODEL_PATH" \
    models.model_0.name="$SERVED_MODEL_NAME" \
    resource.n_gpus_per_node="$TP_SIZE" \
    resource.nnodes=1 \
    models.model_0.ppo_trainer_config.trainer.n_gpus_per_node="$TP_SIZE" \
    models.model_0.ppo_trainer_config.trainer.n_training_gpus_per_node="$TP_SIZE" \
    models.model_0.ppo_trainer_config.actor_rollout_ref.rollout.tensor_model_parallel_size="$TP_SIZE"
