#!/usr/bin/env bash
set -euo pipefail

# Serve an open MetaAgent-X checkpoint with vLLM, then run one demo question
# through MAS design, execution, and Mermaid visualization.

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}"
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"
export VLLM_USE_V1="${VLLM_USE_V1:-1}"
export VLLM_ALLOW_LONG_MAX_MODEL_LEN="${VLLM_ALLOW_LONG_MAX_MODEL_LEN:-1}"
export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"

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
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-8192}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/autoeval_demo}"
QUESTION="${QUESTION:-Find the value of x if 2x + 3 = 17. Answer with a single number.}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
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
        >"/tmp/pettingllms_autoeval_demo_vllm_${PORT}.log" 2>&1 &
    VLLM_PID=$!
    wait_for_vllm || {
        tail -n 80 "/tmp/pettingllms_autoeval_demo_vllm_${PORT}.log" || true
        exit 1
    }
else
    echo "No vLLM server found at $HOST:$PORT and START_VLLM=false"
    exit 1
fi

"$PYTHON_BIN" -m pettingllms.multi_agent_env.autoevol.demo \
    --server-address "$HOST:$PORT" \
    --model-name "$SERVED_MODEL_NAME" \
    --model-path "$MODEL_PATH" \
    --question "$QUESTION" \
    --output-dir "$OUTPUT_DIR" \
    --max-response-length "$MAX_RESPONSE_LENGTH"
