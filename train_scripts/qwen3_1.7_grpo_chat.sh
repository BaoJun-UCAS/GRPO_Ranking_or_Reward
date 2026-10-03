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
  VLLM_GPUS=4                 Policy rollout GPU; count sets tensor parallelism
  QRM_GPU=5                   Dedicated 8B reward-model server GPU
  TRAIN_GPUS=6,7              Policy training GPUs
  MODEL_NAME=Qwen/Qwen3-1.7B    Model ID or local path
  MODEL_REVISION=main          Same policy/tokenizer revision for trainer and vLLM
  ADVANTAGE                   studentization (alias grpo), robust_pairwise, ranking,
                               rank_reward, or module:function; default: studentization
  ADVANTAGE_KWARGS             JSON object, e.g. '{"delta":0.01,"c":0.08}' for robust_pairwise
                               These defaults are uncalibrated weighted-reward units.
                               No group-std normalization; scale_rewards has no effect.
                               Nested delta is distinct from the top-level PPO delta.
  DATASET_CONFIG / DATASET_ADAPTER / DATASET_PROMPT_COLUMN
  DATASET_TRAIN_SPLIT / DATASET_TEST_SPLIT / SYSTEM_PROMPT
                               Optional overrides; otherwise use the YAML values
  DO_EVAL=0                   Set 1 for final GRPO evaluation; requires test split
  PER_DEVICE_EVAL_BATCH_SIZE   Global eval batch must divide into generation groups
  EVAL_STRATEGY / EVAL_STEPS    Optional evaluation during training
  MAX_TRAIN_SAMPLES / MAX_EVAL_SAMPLES  Optional deterministic smoke-test subsets
  ATTN_IMPLEMENTATION / LOSS_TYPE / GRADIENT_CHECKPOINTING
  OVERLAP_QRM_REFERENCE        Overlap external reward scoring with reference log-probs (0/1)
  LOG_COMPLETIONS / SAVE_REWARD_DATA  Optional training YAML overrides
  QRM_MODEL=friendshipkim/QRM-Llama3.1-8B-v2
  PER_DEVICE_TRAIN_BATCH_SIZE=1 NUM_GENERATIONS=8
  QRM_REVISION=main             Pin a commit hash for formal runs
  GRADIENT_ACCUMULATION_STEPS=128 MAX_STEPS=800
  GENERATION_BATCH_SIZE         Defaults to training GPU count × per-device
                               batch × gradient accumulation; must match it
  MAX_PROMPT_LENGTH=2048 MAX_COMPLETION_LENGTH=3072 REWARD_BATCH_SIZE=4
  VLLM_HTTP_PORT=8000 QRM_HTTP_PORT=8001 PORT=29501 VLLM_GROUP_PORT=51216
                               Distinct service/training/weight-sync ports
  QRM_MAX_LENGTH=6144 QRM_MAX_BATCH_TOKENS=6144
  QRM_REQUEST_TIMEOUT=900 QRM_STARTUP_TIMEOUT=1200
  VLLM_PORT                    Deprecated HTTP alias; prefer VLLM_HTTP_PORT
  VLLM_MAX_MODEL_LEN=6144 VLLM_GPU_MEMORY_UTILIZATION=0.82
  VLLM_MAX_NUM_SEQS=256        Concurrent rollout sequences; lower if KV-cache preemptions occur
  VLLM_STARTUP_TIMEOUT=600      Total server startup deadline, seconds
  SERVICE_SHUTDOWN_GRACE_SECONDS=15 Graceful shutdown before SIGKILL
  GRPO_OUTPUT_ROOT              Defaults to <repository>/grpo_runs
  RUN_NAME / RUN_DIR            Unique output name or explicit new directory
  RESUME_FROM_CHECKPOINT        Existing checkpoint directory; resume into a new run
  GRPO_CACHE_ROOT               Explicit cache root; otherwise XDG cache, then
                               writable /data/<user>/cache/grpo, then ~/.cache/grpo
  GRPO_DATA_ROOT                Override the auto-detected /data/<user> root
  VLLM_TMPDIR                  Optional short IPC directory; default uses mktemp
  HF_HOME / HF_HUB_CACHE        Explicit Hugging Face cache settings take priority
  TORCH_EXTENSIONS_DIR / TRITON_CACHE_DIR / VLLM_CACHE_ROOT
                               Build caches default to subdirectories of GRPO_CACHE_ROOT
  CONFIG_FILE                  Training YAML template
  ACCELERATE_CONFIG            Single-machine Accelerate config; process count
                               is resolved from TRAIN_GPUS; DEEPSPEED or MULTI_GPU
                               e.g. recipes/accelerate_configs/ddp_2gpus.yaml
  MERGE_AFTER_TRAINING=1        Set 0 to retain only the LoRA adapter
  WANDB_MODE=offline WANDB_PROJECT=GRPO
  PYTHON=python                Interpreter from the active environment
  TRAINING_ENTRYPOINT          Optional Python entry; defaults to src/open_r1/grpo.py

Example (GPU 4 rollout, GPU 5 QRM, GPUs 6-7 training):
  DATASET_NAME=owner/UltraChat-200k VLLM_GPUS=4 QRM_GPU=5 TRAIN_GPUS=6,7 \
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
TRAINING_ENTRYPOINT="${TRAINING_ENTRYPOINT:-src/open_r1/grpo.py}"
[[ -f "${TRAINING_ENTRYPOINT}" ]] || { echo "Training entry not found: ${TRAINING_ENTRYPOINT}" >&2; exit 2; }

# Default: GPU 4 serves vLLM, GPU 5 serves QRM, and GPUs 6-7 train.
# Reject the old singular alias so a stale VLLM_GPU=0 cannot silently defeat
# the four-GPU split. Intentional alternate layouts use VLLM_GPUS explicitly.
if [[ -n "${VLLM_GPU+x}" ]]; then
    echo 'VLLM_GPU is no longer supported; unset it and use VLLM_GPUS (default: 4).' >&2
    exit 2
fi
VLLM_GPUS="${VLLM_GPUS:-4}"
VLLM_GPUS="${VLLM_GPUS// /}"
QRM_GPU="${QRM_GPU:-5}"
TRAIN_GPUS="${TRAIN_GPUS:-6,7}"
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
QRM_HTTP_PORT="${QRM_HTTP_PORT:-8001}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.82}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-6144}"
VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-256}"
VLLM_STARTUP_TIMEOUT="${VLLM_STARTUP_TIMEOUT:-600}"
QRM_STARTUP_TIMEOUT="${QRM_STARTUP_TIMEOUT:-1200}"
QRM_REQUEST_TIMEOUT="${QRM_REQUEST_TIMEOUT:-900}"
SERVICE_SHUTDOWN_GRACE_SECONDS="${SERVICE_SHUTDOWN_GRACE_SECONDS:-15}"
QRM_MAX_LENGTH="${QRM_MAX_LENGTH:-6144}"
REWARD_BATCH_SIZE="${REWARD_BATCH_SIZE:-4}"
QRM_MAX_BATCH_TOKENS="${QRM_MAX_BATCH_TOKENS:-${QRM_MAX_LENGTH}}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-2048}"
MAX_COMPLETION_LENGTH="${MAX_COMPLETION_LENGTH:-3072}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-1}"
NUM_GENERATIONS="${NUM_GENERATIONS:-8}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-128}"
QRM_REVISION="${QRM_REVISION:-main}"
MAX_STEPS="${MAX_STEPS:-800}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-1.7B}"
MODEL_REVISION="${MODEL_REVISION:-main}"
QRM_MODEL="${QRM_MODEL:-friendshipkim/QRM-Llama3.1-8B-v2}"
CONFIG_FILE="${CONFIG_FILE:-recipes/Qwen3-1.7B/config_chat_regular_qrm_lora_4gpu_split.yaml}"
ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-recipes/accelerate_configs/zero2_2gpus.yaml}"
MERGE_AFTER_TRAINING="${MERGE_AFTER_TRAINING:-1}"
if [[ -z "${DATASET_NAME:-}" ]]; then
    : "${HF_USERNAME:?Set DATASET_NAME to a prepared dataset ID/local path, or set HF_USERNAME}"
    DATASET_NAME="${HF_USERNAME}/UltraChat-200k"
fi

for numeric_name in VLLM_STARTUP_TIMEOUT QRM_STARTUP_TIMEOUT QRM_REQUEST_TIMEOUT QRM_MAX_LENGTH \
    QRM_MAX_BATCH_TOKENS REWARD_BATCH_SIZE MAX_PROMPT_LENGTH QRM_HTTP_PORT \
    MAX_COMPLETION_LENGTH PER_DEVICE_TRAIN_BATCH_SIZE NUM_GENERATIONS \
    GRADIENT_ACCUMULATION_STEPS MAX_STEPS VLLM_MAX_MODEL_LEN VLLM_MAX_NUM_SEQS VLLM_HTTP_PORT PORT VLLM_GROUP_PORT; do
    if [[ ! "${!numeric_name}" =~ ^[1-9][0-9]*$ ]]; then
        echo "${numeric_name} must be a positive integer; got: ${!numeric_name}" >&2
        exit 2
    fi
done
if [[ ! "${TRAIN_GPUS}" =~ ^[0-9]+(,[0-9]+)*$ ]] || [[ ! "${VLLM_GPUS}" =~ ^[0-9]+(,[0-9]+)*$ ]] || [[ ! "${QRM_GPU}" =~ ^[0-9]+$ ]]; then
    echo 'VLLM_GPUS and TRAIN_GPUS must be GPU lists; QRM_GPU must be one GPU index.' >&2
    exit 2
fi
IFS=',' read -r -a VLLM_GPU_ARRAY <<< "${VLLM_GPUS}"
IFS=',' read -r -a TRAIN_GPU_ARRAY <<< "${TRAIN_GPUS}"
NUM_VLLM_GPUS="${#VLLM_GPU_ARRAY[@]}"
NUM_TRAIN_GPUS="${#TRAIN_GPU_ARRAY[@]}"
MERGE_GPU="${VLLM_GPU_ARRAY[0]}"
GENERATION_BATCH_SIZE="${GENERATION_BATCH_SIZE:-$((NUM_TRAIN_GPUS * PER_DEVICE_TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS))}"

GRPO_OUTPUT_ROOT="${GRPO_OUTPUT_ROOT:-${PROJECT_ROOT}/grpo_runs}"
RUN_STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
RUN_TOKEN="${RUN_STAMP}-$$"
RUN_NAME="${RUN_NAME:-qwen3-1.7b-grpo-${RUN_STAMP}-$$}"
OUTPUT_DIR="${RUN_DIR:-${GRPO_OUTPUT_ROOT}/${RUN_NAME}}"
OUTPUT_DIR="$("${PYTHON}" -c 'import pathlib, sys; print(pathlib.Path(sys.argv[1]).expanduser().resolve())' "${OUTPUT_DIR}")"
RUN_NAME="$(basename -- "${OUTPUT_DIR}")"
if [[ -n "${GRPO_CACHE_ROOT:-}" ]]; then
    GRPO_CACHE_ROOT="${GRPO_CACHE_ROOT}"
elif [[ -n "${XDG_CACHE_HOME:-}" ]]; then
    GRPO_CACHE_ROOT="${XDG_CACHE_HOME}/grpo"
else
    GRPO_DATA_ROOT="${GRPO_DATA_ROOT:-/data/${USER:-${HOME##*/}}}"
    if [[ -d "${GRPO_DATA_ROOT}" && -w "${GRPO_DATA_ROOT}" && -x "${GRPO_DATA_ROOT}" ]]; then
        GRPO_CACHE_ROOT="${GRPO_DATA_ROOT}/cache/grpo"
    else
        GRPO_CACHE_ROOT="${HOME}/.cache/grpo"
    fi
fi
export GRPO_CACHE_ROOT
export HF_HOME="${HF_HOME:-${GRPO_CACHE_ROOT}/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-${GRPO_CACHE_ROOT}/torch_extensions}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${GRPO_CACHE_ROOT}/triton}"
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-${GRPO_CACHE_ROOT}/vllm}"
export RUN_NAME RUN_TOKEN OUTPUT_DIR DATASET_NAME MODEL_NAME MODEL_REVISION QRM_MODEL QRM_REVISION VLLM_HTTP_PORT VLLM_GROUP_PORT QRM_HTTP_PORT
export QRM_GPU QRM_REQUEST_TIMEOUT QRM_MAX_LENGTH QRM_MAX_BATCH_TOKENS
export SERVICE_SHUTDOWN_GRACE_SECONDS REWARD_BATCH_SIZE
export MAX_PROMPT_LENGTH MAX_COMPLETION_LENGTH GENERATION_BATCH_SIZE
export GRADIENT_ACCUMULATION_STEPS MAX_STEPS NUM_GENERATIONS PER_DEVICE_TRAIN_BATCH_SIZE
export TRAIN_GPUS VLLM_GPUS NUM_VLLM_GPUS NUM_TRAIN_GPUS VLLM_MAX_MODEL_LEN VLLM_GPU_MEMORY_UTILIZATION PORT
export VLLM_MAX_NUM_SEQS
export DRY_RUN MERGE_AFTER_TRAINING
export RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-}"
export WANDB_PROJECT="${WANDB_PROJECT:-GRPO}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_DIR="${OUTPUT_DIR}/wandb"
export TMPDIR="${OUTPUT_DIR}/tmp"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export CUDA_DEVICE_ORDER="PCI_BUS_ID"
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
VALIDATOR="${PROJECT_ROOT}/scripts/validate_training_run.py"
"${PYTHON}" "${HELPER}" resolve --config "${CONFIG_FILE}" --accelerate-config "${ACCELERATE_CONFIG}"
if [[ "${DRY_RUN}" == 1 ]]; then
    echo 'Dry run complete. No run directory was created and no GPU was used.'
    exit 0
fi
for command_name in curl flock setsid tar tee mktemp; do
    command -v "${command_name}" >/dev/null 2>&1 || {
        echo "Required command not found: ${command_name}" >&2; exit 1;
    }
done

# Hold one advisory lock per physical GPU for the full launcher lifetime. This
# closes the long check-then-bind window while QRM/vLLM load their models and
# prevents a second cooperating launcher from cross-connecting to our ports.
GPU_LOCK_FDS=()
for selected_gpu in "${VLLM_GPU_ARRAY[@]}" "${QRM_GPU}" "${TRAIN_GPU_ARRAY[@]}"; do
    unset gpu_lock_fd
    exec {gpu_lock_fd}>"/tmp/grpo-gpu-${selected_gpu}.lock"
    flock --nonblock "${gpu_lock_fd}" || {
        echo "Physical GPU ${selected_gpu} is locked by another GRPO launcher; no process was started." >&2; exit 1;
    }
    GPU_LOCK_FDS+=("${gpu_lock_fd}")
done
"${PYTHON}" "${HELPER}" check-ports "${VLLM_HTTP_PORT}" "${QRM_HTTP_PORT}" "${PORT}" "${VLLM_GROUP_PORT}"
mkdir -p -- "$(dirname -- "${OUTPUT_DIR}")"
# A fresh directory prevents silently overwriting a previous run's evidence.
mkdir -- "${OUTPUT_DIR}" || { echo "Use a new RUN_NAME or RUN_DIR." >&2; exit 1; }
mkdir -p "${OUTPUT_DIR}/logs" "${OUTPUT_DIR}/config" "${OUTPUT_DIR}/tmp" "${OUTPUT_DIR}/wandb"

VLLM_PID=""
QRM_PID=""
ACTIVE_PID=""
VLLM_TMPDIR="${VLLM_TMPDIR:-}"
VLLM_TMPDIR_OWNED=0
terminate_group() {
    local child_pid="${1:-}"
    [[ -n "${child_pid}" ]] || return 0
    kill -TERM -- "-${child_pid}" 2>/dev/null || true
    local max_attempts=$((SERVICE_SHUTDOWN_GRACE_SECONDS * 10))
    for ((attempt = 0; attempt < max_attempts; attempt++)); do
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
    terminate_group "${QRM_PID}"
    if [[ "${VLLM_TMPDIR_OWNED}" == 1 && -n "${VLLM_TMPDIR}" && "${VLLM_TMPDIR}" == /tmp/grpo-vllm.* ]]; then
        rm -rf -- "${VLLM_TMPDIR}"
    fi
    if [[ "${exit_code}" -eq 0 ]]; then
        printf 'finished_utc=%s\nstatus=success\n' "$(date -u +%Y%m%dT%H%M%SZ)" > "${OUTPUT_DIR}/RUN_STATUS"
    else
        printf 'finished_utc=%s\nstatus=failed\nexit_code=%s\n' "$(date -u +%Y%m%dT%H%M%SZ)" "${exit_code}" > "${OUTPUT_DIR}/RUN_STATUS"
        if [[ -f "${VALIDATOR}" ]]; then
            "${PYTHON}" "${VALIDATOR}" "${OUTPUT_DIR}" >> "${OUTPUT_DIR}/logs/validation.log" 2>&1 || true
        fi
    fi
    exit "${exit_code}"
}
trap record_exit EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

if [[ -z "${VLLM_TMPDIR}" ]]; then
    VLLM_TMPDIR="$(mktemp -d /tmp/grpo-vllm.XXXXXX)"
    VLLM_TMPDIR_OWNED=1
else
    mkdir -p -- "${VLLM_TMPDIR}"
fi
if ((${#VLLM_TMPDIR} > 60)); then
    echo "VLLM_TMPDIR is too long for vLLM IPC sockets: ${VLLM_TMPDIR}" >&2
    exit 2
fi
export VLLM_TMPDIR

"${PYTHON}" "${HELPER}" resolve --config "${CONFIG_FILE}" --accelerate-config "${ACCELERATE_CONFIG}" --write
PROCESSED_CONFIG="${OUTPUT_DIR}/config/resolved_training_config.yaml"
RESOLVED_ACCELERATE_CONFIG="${OUTPUT_DIR}/config/accelerate_config.yaml"
"${PYTHON}" -m pip freeze > "${OUTPUT_DIR}/config/pip-freeze.txt"
git diff --binary --no-ext-diff > "${OUTPUT_DIR}/config/working-tree.patch" 2>/dev/null || true
git ls-files --others --exclude-standard -- '*.py' '*.sh' '*.yaml' '*.yml' '*.md' 'Makefile' \
    > "${OUTPUT_DIR}/config/untracked-files.txt" 2>/dev/null || true
if [[ -s "${OUTPUT_DIR}/config/untracked-files.txt" ]]; then
    tar --create --gzip --file "${OUTPUT_DIR}/config/untracked-files.tar.gz" \
        --verbatim-files-from --files-from "${OUTPUT_DIR}/config/untracked-files.txt"
fi
{
    for key in RUN_NAME RUN_TOKEN OUTPUT_DIR MODEL_NAME MODEL_REVISION QRM_MODEL QRM_REVISION DATASET_NAME VLLM_GPUS QRM_GPU TRAIN_GPUS CUDA_DEVICE_ORDER \
        NUM_VLLM_GPUS NUM_TRAIN_GPUS VLLM_HTTP_PORT QRM_HTTP_PORT PORT VLLM_GROUP_PORT \
        VLLM_MAX_MODEL_LEN VLLM_MAX_NUM_SEQS VLLM_GPU_MEMORY_UTILIZATION VLLM_STARTUP_TIMEOUT \
        QRM_MAX_LENGTH QRM_MAX_BATCH_TOKENS REWARD_BATCH_SIZE QRM_REQUEST_TIMEOUT QRM_STARTUP_TIMEOUT \
        SERVICE_SHUTDOWN_GRACE_SECONDS \
        MAX_PROMPT_LENGTH MAX_COMPLETION_LENGTH VLLM_USE_V1 VLLM_WORKER_MULTIPROC_METHOD \
        GRPO_CACHE_ROOT HF_HOME HF_HUB_CACHE TORCH_EXTENSIONS_DIR TRITON_CACHE_DIR VLLM_CACHE_ROOT VLLM_TMPDIR \
        WANDB_MODE MAX_STEPS GENERATION_BATCH_SIZE RESUME_FROM_CHECKPOINT \
        NUM_GENERATIONS PER_DEVICE_TRAIN_BATCH_SIZE GRADIENT_ACCUMULATION_STEPS; do
        printf '%s=%q\n' "${key}" "${!key}"
    done
    for key in ADVANTAGE ADVANTAGE_KWARGS DATASET_CONFIG DATASET_ADAPTER DATASET_PROMPT_COLUMN \
        DATASET_TRAIN_SPLIT DATASET_TEST_SPLIT SYSTEM_PROMPT DO_EVAL EVAL_STRATEGY EVAL_STEPS \
        MAX_TRAIN_SAMPLES MAX_EVAL_SAMPLES PER_DEVICE_EVAL_BATCH_SIZE ATTN_IMPLEMENTATION LOSS_TYPE \
        GRADIENT_CHECKPOINTING LOG_COMPLETIONS SAVE_REWARD_DATA OVERLAP_QRM_REFERENCE; do
        if [[ -v "${key}" ]]; then printf '%s=%q\n' "${key}" "${!key}"; fi
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

QRM_HEALTH_URL="http://127.0.0.1:${QRM_HTTP_PORT}/health/${RUN_TOKEN}/"
QRM_LOG="${OUTPUT_DIR}/logs/qrm_server.log"
echo "Starting QRM on physical GPU ${QRM_GPU}; log: ${QRM_LOG}"
setsid env CUDA_VISIBLE_DEVICES="${QRM_GPU}" "${PYTHON}" -m open_r1.reward_server \
    --model "${QRM_MODEL}" --revision "${QRM_REVISION}" --host 127.0.0.1 --port "${QRM_HTTP_PORT}" \
    --batch-size "${REWARD_BATCH_SIZE}" --max-batch-tokens "${QRM_MAX_BATCH_TOKENS}" \
    --max-length "${QRM_MAX_LENGTH}" --run-id "${RUN_TOKEN}" \
    --dtype bfloat16 > "${QRM_LOG}" 2>&1 &
QRM_PID=$!

QRM_WAIT_STARTED=${SECONDS}
QRM_WAIT_DEADLINE=$((SECONDS + QRM_STARTUP_TIMEOUT))
QRM_NEXT_STATUS=$((SECONDS + 30))
qrm_healthy() {
    curl --silent --fail --noproxy '*' --connect-timeout 1 --max-time 2 \
        "${QRM_HEALTH_URL}" >/dev/null 2>&1
}
qrm_failure() {
    echo "$1 Log: ${QRM_LOG}" >&2
    tail -n 60 "${QRM_LOG}" >&2
    exit 1
}
while true; do
    kill -0 "${QRM_PID}" 2>/dev/null || qrm_failure 'QRM server exited before becoming healthy.'
    if qrm_healthy; then break; fi
    ((SECONDS < QRM_WAIT_DEADLINE)) || qrm_failure "QRM startup timed out after ${QRM_STARTUP_TIMEOUT}s."
    if ((SECONDS >= QRM_NEXT_STATUS)); then
        echo "Waiting for QRM: $((SECONDS - QRM_WAIT_STARTED))s; log: ${QRM_LOG}"
        QRM_NEXT_STATUS=$((SECONDS + 30))
    fi
    sleep 2
done

VLLM_HEALTH_URL="http://127.0.0.1:${VLLM_HTTP_PORT}/health/${RUN_TOKEN}/"
VLLM_LOG="${OUTPUT_DIR}/logs/vllm_server.log"
echo "Starting TP=${NUM_VLLM_GPUS} vLLM on physical GPUs ${VLLM_GPUS}; log: ${VLLM_LOG}"
setsid env CUDA_VISIBLE_DEVICES="${VLLM_GPUS}" TMPDIR="${VLLM_TMPDIR}" "${PYTHON}" -m open_r1.vllm_serve \
    --model "${MODEL_NAME}" --revision "${MODEL_REVISION}" --host 127.0.0.1 --port "${VLLM_HTTP_PORT}" \
    --tensor_parallel_size "${NUM_VLLM_GPUS}" --gpu_memory_utilization "${VLLM_GPU_MEMORY_UTILIZATION}" \
    --max_model_len "${VLLM_MAX_MODEL_LEN}" --max_num_seqs "${VLLM_MAX_NUM_SEQS}" \
    --dtype bfloat16 --run-id "${RUN_TOKEN}" > "${VLLM_LOG}" 2>&1 &
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

echo "Starting ${NUM_TRAIN_GPUS}-GPU policy training on physical GPUs ${TRAIN_GPUS}"
run_logged "${OUTPUT_DIR}/logs/training.log" \
    env CUDA_VISIBLE_DEVICES="${TRAIN_GPUS}" ACCELERATE_LOG_LEVEL=info "${PYTHON}" -m accelerate.commands.launch \
    --config_file "${RESOLVED_ACCELERATE_CONFIG}" --main_process_port "${PORT}" \
    --num_processes "${NUM_TRAIN_GPUS}" "${TRAINING_ENTRYPOINT}" --config "${PROCESSED_CONFIG}"

echo "Validating training artifacts and pipeline timings"
"${PYTHON}" "${VALIDATOR}" "${OUTPUT_DIR}" --allow-running 2>&1 \
    | tee "${OUTPUT_DIR}/logs/validation.log"

terminate_group "${VLLM_PID}"
VLLM_PID=""
terminate_group "${QRM_PID}"
QRM_PID=""
if [[ "${MERGE_AFTER_TRAINING}" == 1 ]]; then
    echo 'Merging the final LoRA adapter for evaluation'
    run_logged "${OUTPUT_DIR}/logs/merge_lora.log" \
        env CUDA_VISIBLE_DEVICES="${MERGE_GPU}" "${PYTHON}" generate/merge_lora_adapter.py \
        --adapter "${OUTPUT_DIR}" --output "${OUTPUT_DIR}/merged_model"
fi
FINAL_VALIDATION_ARGS=()
if [[ "${MERGE_AFTER_TRAINING}" == 1 ]]; then
    FINAL_VALIDATION_ARGS+=(--require-merged)
    "${PYTHON}" "${VALIDATOR}" "${OUTPUT_DIR}" --allow-running "${FINAL_VALIDATION_ARGS[@]}" 2>&1 \
        | tee -a "${OUTPUT_DIR}/logs/validation.log"
fi
printf 'finished_utc=%s\nstatus=success\n' "$(date -u +%Y%m%dT%H%M%SZ)" > "${OUTPUT_DIR}/RUN_STATUS"
"${PYTHON}" "${VALIDATOR}" "${OUTPUT_DIR}" "${FINAL_VALIDATION_ARGS[@]}" 2>&1 \
    | tee -a "${OUTPUT_DIR}/logs/validation.log"
echo "Run complete: ${OUTPUT_DIR}"
