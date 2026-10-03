#!/usr/bin/env bash
set -Eeuo pipefail

if [[ $# -ne 1 ]]; then
    echo "Usage: $0 /absolute/path/to/training-run" >&2
    exit 2
fi

RUN_DIR="$(cd -- "$1" && pwd)"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

: "${DEEPSEEK_API_KEY:?Set DEEPSEEK_API_KEY before running evaluation}"
EVAL_GPU="${EVAL_GPU:-0}"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3-1.7B}"
TRAINED_MODEL="${TRAINED_MODEL:-${RUN_DIR}/merged_model}"
NUM_PROMPTS="${NUM_PROMPTS:-100}"
BOOTSTRAP_ITERATIONS="${BOOTSTRAP_ITERATIONS:-1000}"
GENERATION_TEMPERATURE="${GENERATION_TEMPERATURE:-0.7}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1024}"
JUDGE_MODEL="${JUDGE_MODEL:-deepseek-flash}"
JUDGE_BASE_URL="${JUDGE_BASE_URL:-https://api.deepseek.com}"
JUDGE_THINKING_MODE="${JUDGE_THINKING_MODE:-disabled}"
JUDGE_TEMPERATURE="${JUDGE_TEMPERATURE:-0}"
EVAL_SEED="${EVAL_SEED:-42}"
PYTHON="${PYTHON:-python}"
MAX_PROMPT_LENGTH="${EVAL_MAX_PROMPT_LENGTH:-2048}"
BASE_REVISION="${BASE_REVISION:-main}"
EVAL_DATASET_ID="${EVAL_DATASET_ID:-HuggingFaceH4/ultrachat_200k}"
EVAL_DATASET_SPLIT="${EVAL_DATASET_SPLIT:-test_sft}"
EVAL_PROMPT_COLUMN="${EVAL_PROMPT_COLUMN:-messages}"
EVAL_DATASET_REVISION="${EVAL_DATASET_REVISION:-main}"
TRAINING_CONFIG="${TRAINING_CONFIG:-${RUN_DIR}/config/resolved_training_config.yaml}"

EVAL_DIR="${EVAL_DIR:-${RUN_DIR}/evaluation/$(date -u +%Y%m%dT%H%M%SZ)-$$}"
COMPLETIONS_DIR="${EVAL_DIR}/completions"
RESULTS_DIR="${EVAL_DIR}/deepseek_judge"
LOG_DIR="${EVAL_DIR}/logs"
mkdir -p "${COMPLETIONS_DIR}" "${RESULTS_DIR}" "${LOG_DIR}"
# Prevent two launches writing the same explicit evaluation directory.
exec 9>"${EVAL_DIR}/.evaluation.lock"
flock -n 9 || { echo "Another evaluation owns ${EVAL_DIR}" >&2; exit 1; }

EVALUATION_STATUS_PATH="${EVAL_DIR}/EVALUATION_STATUS"
record_exit() {
    local exit_code=$?
    trap - EXIT INT TERM
    if [[ "${exit_code}" -eq 0 ]]; then
        printf 'status=success\ncompleted_utc=%s\n' "$(date -u +%Y%m%dT%H%M%SZ)" > "${EVALUATION_STATUS_PATH}"
    else
        printf 'status=failed\nfinished_utc=%s\nexit_code=%s\n' \
            "$(date -u +%Y%m%dT%H%M%SZ)" "${exit_code}" > "${EVALUATION_STATUS_PATH}"
    fi
    exit "${exit_code}"
}
trap record_exit EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

if [[ ! -f "${TRAINED_MODEL}/config.json" ]]; then
    echo "Merged model not found at ${TRAINED_MODEL}. Run merge_lora_adapter.py first." >&2
    exit 1
fi
if [[ ! -f "${TRAINING_CONFIG}" ]]; then
    echo "Resolved training config missing: ${TRAINING_CONFIG}; set TRAINING_CONFIG explicitly." >&2
    exit 1
fi

BASE_COMPLETIONS="${COMPLETIONS_DIR}/qwen3-1.7b-base-chat_N${NUM_PROMPTS}_seed${EVAL_SEED}_temp${GENERATION_TEMPERATURE}.json"
TRAINED_COMPLETIONS="${COMPLETIONS_DIR}/qwen3-1.7b-grpo-regular-lora-chat_N${NUM_PROMPTS}_seed${EVAL_SEED}_temp${GENERATION_TEMPERATURE}.json"

{
    printf 'run_dir=%s\n' "${RUN_DIR}"
    printf 'base_model=%s\n' "${BASE_MODEL}"
    printf 'trained_model=%s\n' "${TRAINED_MODEL}"
    printf 'eval_gpu=%s\n' "${EVAL_GPU}"
    printf 'num_prompts=%s\n' "${NUM_PROMPTS}"
    printf 'bootstrap_iterations=%s\n' "${BOOTSTRAP_ITERATIONS}"
    printf 'generation_temperature=%s\n' "${GENERATION_TEMPERATURE}"
    printf 'max_new_tokens=%s\n' "${MAX_NEW_TOKENS}"
    printf 'max_prompt_length=%s\n' "${MAX_PROMPT_LENGTH}"
    printf 'dataset_id=%s\ndataset_split=%s\ndataset_revision=%s\n' "${EVAL_DATASET_ID}" "${EVAL_DATASET_SPLIT}" "${EVAL_DATASET_REVISION}"
    printf 'base_revision=%s\ntraining_config=%s\n' "${BASE_REVISION}" "${TRAINING_CONFIG}"
    printf 'eval_seed=%s\n' "${EVAL_SEED}"
    printf 'judge_model=%s\n' "${JUDGE_MODEL}"
    printf 'judge_base_url=%s\n' "${JUDGE_BASE_URL}"
    printf 'judge_thinking_mode=%s\n' "${JUDGE_THINKING_MODE}"
    printf 'judge_temperature=%s\n' "${JUDGE_TEMPERATURE}"
    printf 'git_commit=%s\n' "$(git rev-parse HEAD 2>/dev/null || printf unknown)"
} > "${EVAL_DIR}/evaluation.env"

generate_if_missing() {
    local model_path=$1
    local output_path=$2
    local log_path=$3
    CUDA_VISIBLE_DEVICES="${EVAL_GPU}" "${PYTHON}" generate/generate_completions.py \
        --model "${model_path}" \
        --revision "${BASE_REVISION}" \
        --dataset chat \
        --dataset-id "${EVAL_DATASET_ID}" \
        --dataset-split "${EVAL_DATASET_SPLIT}" \
        --prompt-column "${EVAL_PROMPT_COLUMN}" \
        --dataset-revision "${EVAL_DATASET_REVISION}" \
        --training-config "${TRAINING_CONFIG}" \
        --max-prompt-length "${MAX_PROMPT_LENGTH}" \
        --reuse-existing \
        --num-prompts "${NUM_PROMPTS}" \
        --n-completions 1 \
        --temperature "${GENERATION_TEMPERATURE}" \
        --max-new-tokens "${MAX_NEW_TOKENS}" \
        --seed "${EVAL_SEED}" \
        --vllm-gpu-memory 0.85 \
        --output "${output_path}" \
        2>&1 | tee "${log_path}"
}

generate_if_missing "${BASE_MODEL}" "${BASE_COMPLETIONS}" "${LOG_DIR}/generate_base.log"
generate_if_missing "${TRAINED_MODEL}" "${TRAINED_COMPLETIONS}" "${LOG_DIR}/generate_trained.log"

"${PYTHON}" evaluate/bootstrap_judge.py \
    --completions1 "${BASE_COMPLETIONS}" \
    --completions2 "${TRAINED_COMPLETIONS}" \
    --api-provider deepseek \
    --base-url "${JUDGE_BASE_URL}" \
    --judge-model "${JUDGE_MODEL}" \
    --thinking-mode "${JUDGE_THINKING_MODE}" \
    --judge-temperature "${JUDGE_TEMPERATURE}" \
    --N "${NUM_PROMPTS}" \
    --B "${BOOTSTRAP_ITERATIONS}" \
    --seed "${EVAL_SEED}" \
    --no-ties \
    --output-dir "${RESULTS_DIR}" \
    --cache-path "${RESULTS_DIR}/judge_cache.jsonl" \
    2>&1 | tee "${LOG_DIR}/deepseek_judge.log"

echo "Evaluation complete: ${EVAL_DIR}"
