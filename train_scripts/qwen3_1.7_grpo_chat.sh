#!/usr/bin/env bash
set -Eeuo pipefail

# Recommended topology for the inspected server:
#   GPU 0       -> dedicated vLLM server (NUMA 0)
#   GPU 3,4,5,6 -> four-process ZeRO-3 training (NUMA 1, all PIX links)
# Override either value at launch time without editing this file, for example:
#   VLLM_GPU=2 TRAIN_GPUS=3,4,5,6 bash train_scripts/qwen3_1.7_grpo_chat.sh
VLLM_GPU="${VLLM_GPU:-0}"
TRAIN_GPUS="${TRAIN_GPUS:-3,4,5,6}"
VLLM_PORT="${VLLM_PORT:-8000}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.82}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-4096}"
REWARD_BATCH_SIZE="${REWARD_BATCH_SIZE:-4}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-1024}"
MAX_COMPLETION_LENGTH="${MAX_COMPLETION_LENGTH:-1024}"
GENERATION_BATCH_SIZE="${GENERATION_BATCH_SIZE:-256}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-64}"
MAX_STEPS="${MAX_STEPS:-500}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-1.7B}"
CONFIG_FILE="${CONFIG_FILE:-recipes/Qwen3-1.7B/config_chat_regular_qrm_lora_5gpu.yaml}"
ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-recipes/accelerate_configs/zero3_4gpus.yaml}"
PORT="${PORT:-29501}"
MERGE_AFTER_TRAINING="${MERGE_AFTER_TRAINING:-1}"

: "${HF_USERNAME:?Please set HF_USERNAME to the owner of the preprocessed UltraChat dataset}"
DATASET_NAME="${DATASET_NAME:-${HF_USERNAME}/UltraChat-200k}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

for command_name in nvidia-smi accelerate trl envsubst curl python; do
    command -v "${command_name}" >/dev/null 2>&1 || {
        echo "Required command not found: ${command_name}" >&2
        exit 1
    }
done

for numeric_name in REWARD_BATCH_SIZE MAX_PROMPT_LENGTH MAX_COMPLETION_LENGTH \
    GENERATION_BATCH_SIZE GRADIENT_ACCUMULATION_STEPS MAX_STEPS; do
    numeric_value="${!numeric_name}"
    if [[ ! "${numeric_value}" =~ ^[1-9][0-9]*$ ]]; then
        echo "${numeric_name} must be a positive integer; got: ${numeric_value}" >&2
        exit 1
    fi
done
if (( GENERATION_BATCH_SIZE % 8 != 0 )); then
    echo "GENERATION_BATCH_SIZE must be divisible by num_generations=8." >&2
    exit 1
fi
EXPECTED_GENERATION_BATCH_SIZE=$((4 * GRADIENT_ACCUMULATION_STEPS))
if (( GENERATION_BATCH_SIZE != EXPECTED_GENERATION_BATCH_SIZE )); then
    echo "GENERATION_BATCH_SIZE must equal 4 training GPUs × batch 1 × GRADIENT_ACCUMULATION_STEPS." >&2
    echo "Expected ${EXPECTED_GENERATION_BATCH_SIZE}, got ${GENERATION_BATCH_SIZE}." >&2
    exit 1
fi

TRAIN_GPUS="${TRAIN_GPUS// /}"
IFS=',' read -r -a TRAIN_GPU_ARRAY <<< "${TRAIN_GPUS}"
if [[ "${#TRAIN_GPU_ARRAY[@]}" -ne 4 ]]; then
    echo "TRAIN_GPUS must contain exactly four comma-separated GPU IDs; got: ${TRAIN_GPUS}" >&2
    exit 1
fi
declare -A SEEN_TRAIN_GPUS=()
for gpu_id in "${TRAIN_GPU_ARRAY[@]}"; do
    if [[ ! "${gpu_id}" =~ ^[0-9]+$ ]]; then
        echo "Every TRAIN_GPUS entry must be a non-negative integer; got: ${gpu_id}" >&2
        exit 1
    fi
    if [[ -n "${SEEN_TRAIN_GPUS[${gpu_id}]:-}" ]]; then
        echo "TRAIN_GPUS contains duplicate GPU ID ${gpu_id}: ${TRAIN_GPUS}" >&2
        exit 1
    fi
    SEEN_TRAIN_GPUS["${gpu_id}"]=1
done
if [[ ! "${VLLM_GPU}" =~ ^[0-9]+$ ]]; then
    echo "VLLM_GPU must be a non-negative integer; got: ${VLLM_GPU}" >&2
    exit 1
fi
if [[ ",${TRAIN_GPUS}," == *",${VLLM_GPU},"* ]]; then
    echo "VLLM_GPU (${VLLM_GPU}) must not also appear in TRAIN_GPUS (${TRAIN_GPUS})." >&2
    exit 1
fi

VISIBLE_GPU_IDS="$(nvidia-smi --query-gpu=index --format=csv,noheader,nounits)"
for gpu_id in "${VLLM_GPU}" "${TRAIN_GPU_ARRAY[@]}"; do
    if ! grep -qx "${gpu_id}" <<< "${VISIBLE_GPU_IDS}"; then
        echo "GPU ${gpu_id} is not visible to nvidia-smi." >&2
        exit 1
    fi
done

DEFAULT_OUTPUT_ROOT="/data/${USER:-user}/gopo_runs"
GOPO_OUTPUT_ROOT="${GOPO_OUTPUT_ROOT:-${DEFAULT_OUTPUT_ROOT}}"
RUN_STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
RUN_NAME="${RUN_NAME:-qwen3-1.7b-ultrachat-grpo-studentization-qrm-lora-seed42-${RUN_STAMP}}"
OUTPUT_DIR="${RUN_DIR:-${GOPO_OUTPUT_ROOT}/${RUN_NAME}}"
RUN_NAME="$(basename -- "${OUTPUT_DIR}")"

mkdir -p "${OUTPUT_DIR}/logs" "${OUTPUT_DIR}/config" "${OUTPUT_DIR}/tmp" "${OUTPUT_DIR}/wandb"

VLLM_PID=""
cleanup() {
    if [[ -n "${VLLM_PID}" ]] && kill -0 "${VLLM_PID}" >/dev/null 2>&1; then
        kill "${VLLM_PID}" >/dev/null 2>&1 || true
        wait "${VLLM_PID}" >/dev/null 2>&1 || true
    fi
    VLLM_PID=""
}
record_exit() {
    local exit_code=$?
    trap - EXIT INT TERM
    cleanup
    if [[ "${exit_code}" -eq 0 ]]; then
        printf 'completed_utc=%s\nstatus=success\n' "$(date -u +%Y%m%dT%H%M%SZ)" > "${OUTPUT_DIR}/RUN_STATUS"
    else
        printf 'finished_utc=%s\nstatus=failed\nexit_code=%s\n' "$(date -u +%Y%m%dT%H%M%SZ)" "${exit_code}" > "${OUTPUT_DIR}/RUN_STATUS"
    fi
    exit "${exit_code}"
}
trap record_exit EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

export RUN_NAME OUTPUT_DIR DATASET_NAME MODEL_NAME VLLM_PORT REWARD_BATCH_SIZE
export MAX_PROMPT_LENGTH MAX_COMPLETION_LENGTH GENERATION_BATCH_SIZE
export GRADIENT_ACCUMULATION_STEPS MAX_STEPS
export WANDB_PROJECT="${WANDB_PROJECT:-GOPO-GRPO}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_DIR="${OUTPUT_DIR}/wandb"
export TMPDIR="${OUTPUT_DIR}/tmp"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"

GOPO_CACHE_ROOT="${GOPO_CACHE_ROOT:-/data/${USER:-user}/gopo_cache}"
mkdir -p "${GOPO_CACHE_ROOT}"
export HF_HOME="${HF_HOME:-${GOPO_CACHE_ROOT}/huggingface}"

PROCESSED_CONFIG="${OUTPUT_DIR}/config/resolved_training_config.yaml"
envsubst < "${CONFIG_FILE}" > "${PROCESSED_CONFIG}"
cp "${ACCELERATE_CONFIG}" "${OUTPUT_DIR}/config/accelerate_config.yaml"
python -m pip freeze > "${OUTPUT_DIR}/config/pip-freeze.txt"

{
    printf 'run_name=%s\n' "${RUN_NAME}"
    printf 'output_dir=%s\n' "${OUTPUT_DIR}"
    printf 'model_name=%s\n' "${MODEL_NAME}"
    printf 'dataset_name=%s\n' "${DATASET_NAME}"
    printf 'vllm_gpu=%s\n' "${VLLM_GPU}"
    printf 'train_gpus=%s\n' "${TRAIN_GPUS}"
    printf 'vllm_port=%s\n' "${VLLM_PORT}"
    printf 'wandb_mode=%s\n' "${WANDB_MODE}"
    printf 'max_steps=%s\n' "${MAX_STEPS}"
    printf 'generation_batch_size=%s\n' "${GENERATION_BATCH_SIZE}"
    printf 'gradient_accumulation_steps=%s\n' "${GRADIENT_ACCUMULATION_STEPS}"
    printf 'started_utc=%s\n' "${RUN_STAMP}"
    printf 'git_commit=%s\n' "$(git rev-parse HEAD 2>/dev/null || printf unknown)"
} > "${OUTPUT_DIR}/run.env"

{
    printf 'Kernel and OS:\n'
    uname -a
    printf '\nPython:\n'
    python --version
    printf '\nNVIDIA status:\n'
    nvidia-smi
    printf '\nGPU topology:\n'
    nvidia-smi topo -m
    printf '\nDisk usage:\n'
    df -h "${OUTPUT_DIR}"
    printf '\nGit status:\n'
    git status --short
} > "${OUTPUT_DIR}/logs/system_snapshot.txt" 2>&1

if curl --silent --fail "http://127.0.0.1:${VLLM_PORT}/health/" >/dev/null 2>&1; then
    echo "Port ${VLLM_PORT} already has a TRL vLLM server. Stop it or choose another VLLM_PORT." >&2
    exit 1
fi

echo "Starting vLLM on physical GPU ${VLLM_GPU}; log: ${OUTPUT_DIR}/logs/vllm_server.log"
CUDA_VISIBLE_DEVICES="${VLLM_GPU}" trl vllm-serve \
    --model "${MODEL_NAME}" \
    --host 127.0.0.1 \
    --port "${VLLM_PORT}" \
    --tensor_parallel_size 1 \
    --gpu_memory_utilization "${VLLM_GPU_MEMORY_UTILIZATION}" \
    --max_model_len "${VLLM_MAX_MODEL_LEN}" \
    --dtype bfloat16 \
    > "${OUTPUT_DIR}/logs/vllm_server.log" 2>&1 &
VLLM_PID=$!

for _ in $(seq 1 120); do
    if curl --silent --fail "http://127.0.0.1:${VLLM_PORT}/health/" >/dev/null 2>&1; then
        break
    fi
    if ! kill -0 "${VLLM_PID}" >/dev/null 2>&1; then
        echo "vLLM exited before becoming healthy. Inspect ${OUTPUT_DIR}/logs/vllm_server.log" >&2
        exit 1
    fi
    sleep 5
done
if ! curl --silent --fail "http://127.0.0.1:${VLLM_PORT}/health/" >/dev/null 2>&1; then
    echo "vLLM did not become healthy within 10 minutes." >&2
    exit 1
fi

echo "Starting four-GPU ZeRO-3 training on physical GPUs ${TRAIN_GPUS}"
CUDA_VISIBLE_DEVICES="${TRAIN_GPUS}" ACCELERATE_LOG_LEVEL=info accelerate launch \
    --config_file "${ACCELERATE_CONFIG}" \
    --main_process_port "${PORT}" \
    --num_processes 4 \
    src/open_r1/grpo.py \
    --config "${PROCESSED_CONFIG}" \
    2>&1 | tee "${OUTPUT_DIR}/logs/training.log"

cleanup

if [[ "${MERGE_AFTER_TRAINING}" == "1" ]]; then
    echo "Merging the final LoRA adapter for evaluation"
    CUDA_VISIBLE_DEVICES="${VLLM_GPU}" python generate/merge_lora_adapter.py \
        --adapter "${OUTPUT_DIR}" \
        --output "${OUTPUT_DIR}/merged_model" \
        2>&1 | tee "${OUTPUT_DIR}/logs/merge_lora.log"
fi

echo "Run complete: ${OUTPUT_DIR}"
