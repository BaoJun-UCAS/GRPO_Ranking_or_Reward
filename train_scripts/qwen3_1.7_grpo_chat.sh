#!/usr/bin/env bash
set -Eeuo pipefail
# Each setsid child owns a process group whose ID is $!. Job control must be
# disabled so Ctrl-C can clean up that run's workers and log pipeline.
set +m

usage() {
    cat <<'USAGE'
Usage: bash train_scripts/qwen3_1.7_grpo_chat.sh [--dry-run]

--dry-run validates and prints the configuration without creating files,
downloading models, importing ML packages, or using GPUs. Activate the project's
Python environment first (PyYAML is required). DRY_RUN=1 is equivalent.

Set environment variables before invoking this script:
  DATASET_NAME                 Prepared dataset ID or local path (required), or
  HF_USERNAME                  Fallback: DATASET_NAME=$HF_USERNAME/UltraChat-200k
  VLLM_GPU=0                   One dedicated generation GPU
  TRAIN_GPUS=4,5,6,7           Comma-separated training GPU indices; any count
  MODEL_NAME=Qwen/Qwen3-1.7B    Model ID or local path
  PER_DEVICE_TRAIN_BATCH_SIZE=1 NUM_GENERATIONS=8
  GRADIENT_ACCUMULATION_STEPS=64 MAX_STEPS=500
  GENERATION_BATCH_SIZE         Defaults to training GPU count × per-device
                               batch × gradient accumulation; must match it
  MAX_PROMPT_LENGTH=1024 MAX_COMPLETION_LENGTH=1024 REWARD_BATCH_SIZE=4
  VLLM_HTTP_PORT=8000 PORT=29501 VLLM_GROUP_PORT=51216
                               Distinct ports for HTTP, training, weight sync
  VLLM_PORT                    Deprecated HTTP alias; prefer VLLM_HTTP_PORT
  VLLM_MAX_MODEL_LEN=4096 VLLM_GPU_MEMORY_UTILIZATION=0.82
  VLLM_STARTUP_TIMEOUT=600      Total server startup deadline, seconds
  GRPO_OUTPUT_ROOT              Defaults to <repository>/grpo_runs
  RUN_NAME / RUN_DIR            Unique output name or explicit new directory
  RESUME_FROM_CHECKPOINT        Existing checkpoint directory; resume into a new run
  GRPO_CACHE_ROOT               Defaults to ${XDG_CACHE_HOME:-$HOME/.cache}/grpo
  HF_HOME / HF_HUB_CACHE        Explicit Hugging Face cache settings take priority
  TORCH_EXTENSIONS_DIR / TRITON_CACHE_DIR / VLLM_CACHE_ROOT
                               Build caches default to subdirectories of GRPO_CACHE_ROOT
  CONFIG_FILE                  Training YAML template
  ACCELERATE_CONFIG            Single-machine Accelerate config; process count
                               is resolved from TRAIN_GPUS
  MERGE_AFTER_TRAINING=1        Set 0 to retain only the LoRA adapter
  WANDB_MODE=offline WANDB_PROJECT=GRPO
  PYTHON=python                Interpreter from the active environment

Example (one generation GPU plus two training GPUs):
  DATASET_NAME=owner/UltraChat-200k VLLM_GPU=0 TRAIN_GPUS=1,2 \
    MAX_STEPS=2 GRADIENT_ACCUMULATION_STEPS=8 \
    bash train_scripts/qwen3_1.7_grpo_chat.sh --dry-run
USAGE
}

DRY_RUN="${DRY_RUN:-0}"
case "${1:-}" in
    --help|-h) usage; exit 0 ;;
    --dry-run) DRY_RUN=1; shift ;;
    "") ;;
    *) usage >&2; exit 2 ;;
esac
if (($#)); then usage >&2; exit 2; fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"
PYTHON="${PYTHON:-python}"

# The default GPU layout is the original five-GPU example, not a requirement.
VLLM_GPU="${VLLM_GPU:-0}"
TRAIN_GPUS="${TRAIN_GPUS:-4,5,6,7}"
TRAIN_GPUS="${TRAIN_GPUS// /}"
VLLM_HTTP_PORT="${VLLM_HTTP_PORT:-${VLLM_PORT:-8000}}"
if [[ -n "${VLLM_PORT+x}" ]]; then
    echo 'Warning: VLLM_PORT is deprecated as an HTTP port; use VLLM_HTTP_PORT instead.' >&2
fi
# vLLM reserves VLLM_PORT for its internal TCPStore. Inheriting the HTTP port
# makes the engine bind that port before Uvicorn can start the HTTP service.
unset VLLM_PORT
PORT="${PORT:-29501}"
VLLM_GROUP_PORT="${VLLM_GROUP_PORT:-51216}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.82}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-4096}"
VLLM_STARTUP_TIMEOUT="${VLLM_STARTUP_TIMEOUT:-600}"
REWARD_BATCH_SIZE="${REWARD_BATCH_SIZE:-4}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-1024}"
MAX_COMPLETION_LENGTH="${MAX_COMPLETION_LENGTH:-1024}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-1}"
NUM_GENERATIONS="${NUM_GENERATIONS:-8}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-64}"
MAX_STEPS="${MAX_STEPS:-500}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-1.7B}"
CONFIG_FILE="${CONFIG_FILE:-recipes/Qwen3-1.7B/config_chat_regular_qrm_lora_5gpu.yaml}"
ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-recipes/accelerate_configs/zero3_4gpus.yaml}"
MERGE_AFTER_TRAINING="${MERGE_AFTER_TRAINING:-1}"
if [[ -z "${DATASET_NAME:-}" ]]; then
    : "${HF_USERNAME:?Set DATASET_NAME to a prepared dataset ID/local path, or set HF_USERNAME}"
    DATASET_NAME="${HF_USERNAME}/UltraChat-200k"
fi

for numeric_name in VLLM_STARTUP_TIMEOUT REWARD_BATCH_SIZE MAX_PROMPT_LENGTH \
    MAX_COMPLETION_LENGTH PER_DEVICE_TRAIN_BATCH_SIZE NUM_GENERATIONS \
    GRADIENT_ACCUMULATION_STEPS MAX_STEPS VLLM_MAX_MODEL_LEN VLLM_HTTP_PORT PORT VLLM_GROUP_PORT; do
    if [[ ! "${!numeric_name}" =~ ^[1-9][0-9]*$ ]]; then
        echo "${numeric_name} must be a positive integer; got: ${!numeric_name}" >&2
        exit 2
    fi
done
if [[ ! "${TRAIN_GPUS}" =~ ^[0-9]+(,[0-9]+)*$ ]] || [[ ! "${VLLM_GPU}" =~ ^[0-9]+$ ]]; then
    echo 'VLLM_GPU and TRAIN_GPUS must contain GPU indices (for example 0 and 1,2).' >&2
    exit 2
fi
IFS=',' read -r -a TRAIN_GPU_ARRAY <<< "${TRAIN_GPUS}"
NUM_TRAIN_GPUS="${#TRAIN_GPU_ARRAY[@]}"
GENERATION_BATCH_SIZE="${GENERATION_BATCH_SIZE:-$((NUM_TRAIN_GPUS * PER_DEVICE_TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS))}"

GRPO_OUTPUT_ROOT="${GRPO_OUTPUT_ROOT:-${PROJECT_ROOT}/grpo_runs}"
RUN_STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
RUN_NAME="${RUN_NAME:-qwen3-1.7b-grpo-${RUN_STAMP}-$$}"
OUTPUT_DIR="${RUN_DIR:-${GRPO_OUTPUT_ROOT}/${RUN_NAME}}"
OUTPUT_DIR="$("${PYTHON}" -c 'import pathlib, sys; print(pathlib.Path(sys.argv[1]).expanduser().resolve())' "${OUTPUT_DIR}")"
RUN_NAME="$(basename -- "${OUTPUT_DIR}")"
GRPO_CACHE_ROOT="${GRPO_CACHE_ROOT:-${XDG_CACHE_HOME:-${HOME}/.cache}/grpo}"
export HF_HOME="${HF_HOME:-${GRPO_CACHE_ROOT}/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-${GRPO_CACHE_ROOT}/torch_extensions}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${GRPO_CACHE_ROOT}/triton}"
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-${GRPO_CACHE_ROOT}/vllm}"
export RUN_NAME OUTPUT_DIR DATASET_NAME MODEL_NAME VLLM_HTTP_PORT VLLM_GROUP_PORT REWARD_BATCH_SIZE
export MAX_PROMPT_LENGTH MAX_COMPLETION_LENGTH GENERATION_BATCH_SIZE
export GRADIENT_ACCUMULATION_STEPS MAX_STEPS NUM_GENERATIONS PER_DEVICE_TRAIN_BATCH_SIZE
export TRAIN_GPUS VLLM_GPU NUM_TRAIN_GPUS VLLM_MAX_MODEL_LEN VLLM_GPU_MEMORY_UTILIZATION PORT
export DRY_RUN MERGE_AFTER_TRAINING
export RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-}"
export WANDB_PROJECT="${WANDB_PROJECT:-GRPO}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_DIR="${OUTPUT_DIR}/wandb"
export TMPDIR="${OUTPUT_DIR}/tmp"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
# TRL's requests client also inherits proxy settings. Keep external downloads
# proxied if configured, but all loopback service traffic must stay local.
export NO_PROXY="127.0.0.1,localhost,::1${NO_PROXY:+,${NO_PROXY}}${no_proxy:+,${no_proxy}}"
export no_proxy="${NO_PROXY}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
# This service's weight-sync extension targets vLLM 0.8.5's V0 worker API.
export VLLM_USE_V1="${VLLM_USE_V1:-0}"
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"

HELPER="${SCRIPT_DIR}/run_helpers.py"
"${PYTHON}" "${HELPER}" resolve --config "${CONFIG_FILE}" --accelerate-config "${ACCELERATE_CONFIG}"
if [[ "${DRY_RUN}" == 1 ]]; then
    echo 'Dry run complete. No run directory was created and no GPU was used.'
    exit 0
fi
for command_name in curl setsid tee; do
    command -v "${command_name}" >/dev/null 2>&1 || {
        echo "Required command not found: ${command_name}" >&2; exit 1;
    }
done
"${PYTHON}" "${HELPER}" check-ports "${VLLM_HTTP_PORT}" "${PORT}" "${VLLM_GROUP_PORT}"
mkdir -p -- "$(dirname -- "${OUTPUT_DIR}")"
# A fresh directory prevents silently overwriting a previous run's evidence.
mkdir -- "${OUTPUT_DIR}" || { echo "Use a new RUN_NAME or RUN_DIR." >&2; exit 1; }
mkdir -p "${OUTPUT_DIR}/logs" "${OUTPUT_DIR}/config" "${OUTPUT_DIR}/tmp" "${OUTPUT_DIR}/wandb"

VLLM_PID=""
ACTIVE_PID=""
terminate_group() {
    local child_pid="${1:-}"
    [[ -n "${child_pid}" ]] || return 0
    kill -TERM -- "-${child_pid}" 2>/dev/null || true
    for _ in {1..20}; do
        kill -0 -- "-${child_pid}" 2>/dev/null || break
        sleep 0.1
    done
    kill -KILL -- "-${child_pid}" 2>/dev/null || true
    wait "${child_pid}" 2>/dev/null || true
}
record_exit() {
    local exit_code=$?
    trap - EXIT INT TERM
    terminate_group "${ACTIVE_PID}"
    terminate_group "${VLLM_PID}"
    if [[ "${exit_code}" -eq 0 ]]; then
        printf 'finished_utc=%s\nstatus=success\n' "$(date -u +%Y%m%dT%H%M%SZ)" > "${OUTPUT_DIR}/RUN_STATUS"
    else
        printf 'finished_utc=%s\nstatus=failed\nexit_code=%s\n' "$(date -u +%Y%m%dT%H%M%SZ)" "${exit_code}" > "${OUTPUT_DIR}/RUN_STATUS"
    fi
    exit "${exit_code}"
}
trap record_exit EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

"${PYTHON}" "${HELPER}" resolve --config "${CONFIG_FILE}" --accelerate-config "${ACCELERATE_CONFIG}" --write
PROCESSED_CONFIG="${OUTPUT_DIR}/config/resolved_training_config.yaml"
RESOLVED_ACCELERATE_CONFIG="${OUTPUT_DIR}/config/accelerate_config.yaml"
"${PYTHON}" -m pip freeze > "${OUTPUT_DIR}/config/pip-freeze.txt"
{
    for key in RUN_NAME OUTPUT_DIR MODEL_NAME DATASET_NAME VLLM_GPU TRAIN_GPUS \
        NUM_TRAIN_GPUS VLLM_HTTP_PORT PORT VLLM_GROUP_PORT VLLM_USE_V1 VLLM_WORKER_MULTIPROC_METHOD \
        HF_HOME HF_HUB_CACHE TORCH_EXTENSIONS_DIR TRITON_CACHE_DIR VLLM_CACHE_ROOT \
        WANDB_MODE MAX_STEPS GENERATION_BATCH_SIZE RESUME_FROM_CHECKPOINT \
        NUM_GENERATIONS PER_DEVICE_TRAIN_BATCH_SIZE GRADIENT_ACCUMULATION_STEPS; do
        printf '%s=%q\n' "${key}" "${!key}"
    done
    printf 'started_utc=%s\n' "${RUN_STAMP}"
    printf 'CUDA_HOME=%q\n' "${CUDA_HOME:-}"
    printf 'git_commit=%s\n' "$(git rev-parse HEAD 2>/dev/null || printf unknown)"
} > "${OUTPUT_DIR}/run.env"
{
    uname -a
    "${PYTHON}" --version
    if [[ -n "${CUDA_HOME:-}" && -x "${CUDA_HOME}/bin/nvcc" ]]; then
        "${CUDA_HOME}/bin/nvcc" --version
    elif command -v nvcc >/dev/null 2>&1; then
        nvcc --version
    fi
    if command -v nvidia-smi >/dev/null 2>&1; then
        nvidia-smi || true
        nvidia-smi topo -m || true
    fi
    df -h "${OUTPUT_DIR}"
    git status --short || true
} > "${OUTPUT_DIR}/logs/system_snapshot.txt" 2>&1

VLLM_HEALTH_URL="http://127.0.0.1:${VLLM_HTTP_PORT}/health/"
VLLM_LOG="${OUTPUT_DIR}/logs/vllm_server.log"
echo "Starting vLLM on physical GPU ${VLLM_GPU}; log: ${VLLM_LOG}"
setsid env CUDA_VISIBLE_DEVICES="${VLLM_GPU}" "${PYTHON}" -m open_r1.vllm_serve \
    --model "${MODEL_NAME}" --host 127.0.0.1 --port "${VLLM_HTTP_PORT}" \
    --tensor_parallel_size 1 --gpu_memory_utilization "${VLLM_GPU_MEMORY_UTILIZATION}" \
    --max_model_len "${VLLM_MAX_MODEL_LEN}" --dtype bfloat16 > "${VLLM_LOG}" 2>&1 &
VLLM_PID=$!

VLLM_WAIT_STARTED=${SECONDS}
VLLM_WAIT_DEADLINE=$((SECONDS + VLLM_STARTUP_TIMEOUT))
VLLM_NEXT_STATUS=$((SECONDS + 30))
server_healthy() {
    curl --silent --fail --noproxy '*' --connect-timeout 1 --max-time 2 \
        "${VLLM_HEALTH_URL}" >/dev/null 2>&1
}
server_failure() {
    echo "$1 Log: ${VLLM_LOG}" >&2
    tail -n 60 "${VLLM_LOG}" >&2
    exit 1
}
while true; do
    kill -0 "${VLLM_PID}" 2>/dev/null || server_failure 'vLLM exited before becoming healthy.'
    if server_healthy; then break; fi
    ((SECONDS < VLLM_WAIT_DEADLINE)) || server_failure "vLLM startup timed out after ${VLLM_STARTUP_TIMEOUT}s."
    if ((SECONDS >= VLLM_NEXT_STATUS)); then
        echo "Waiting for vLLM: $((SECONDS - VLLM_WAIT_STARTED))s; log: ${VLLM_LOG}"
        VLLM_NEXT_STATUS=$((SECONDS + 30))
    fi
    sleep 2
done

# Command and tee share one owned group. Waiting for a background group keeps
# Bash's signal traps responsive during long training.
run_logged() {
    local log_file="$1"
    shift
    setsid bash -c 'set -o pipefail; log_file=$1; shift; "$@" 2>&1 | tee "$log_file"' \
        bash "${log_file}" "$@" &
    ACTIVE_PID=$!
    local result=0
    wait "${ACTIVE_PID}" || result=$?
    terminate_group "${ACTIVE_PID}"
    ACTIVE_PID=""
    return "${result}"
}

echo "Starting ${NUM_TRAIN_GPUS}-GPU DeepSpeed training on physical GPUs ${TRAIN_GPUS}"
run_logged "${OUTPUT_DIR}/logs/training.log" \
    env CUDA_VISIBLE_DEVICES="${TRAIN_GPUS}" ACCELERATE_LOG_LEVEL=info "${PYTHON}" -m accelerate.commands.launch \
    --config_file "${RESOLVED_ACCELERATE_CONFIG}" --main_process_port "${PORT}" \
    --num_processes "${NUM_TRAIN_GPUS}" src/open_r1/grpo.py --config "${PROCESSED_CONFIG}"

terminate_group "${VLLM_PID}"
VLLM_PID=""
if [[ "${MERGE_AFTER_TRAINING}" == 1 ]]; then
    echo 'Merging the final LoRA adapter for evaluation'
    run_logged "${OUTPUT_DIR}/logs/merge_lora.log" \
        env CUDA_VISIBLE_DEVICES="${VLLM_GPU}" "${PYTHON}" generate/merge_lora_adapter.py \
        --adapter "${OUTPUT_DIR}" --output "${OUTPUT_DIR}/merged_model"
fi
echo "Run complete: ${OUTPUT_DIR}"
