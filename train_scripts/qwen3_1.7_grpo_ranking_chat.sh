#!/usr/bin/env bash
set -Eeuo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export PORT="${PORT:-29500}"
: "${HF_USERNAME:?Please set HF_USERNAME environment variable}"
export HF_USERNAME

# Process the ranking-GRPO recipe with environment variables.
CONFIG_FILE="recipes/Qwen3-1.7B/config_chat_ranking_qrm_seed42.yaml"
PROCESSED_CONFIG="$(mktemp)"
cleanup() {
    rm -f -- "${PROCESSED_CONFIG}"
}
trap cleanup EXIT
envsubst < "${CONFIG_FILE}" > "${PROCESSED_CONFIG}"

ACCELERATE_LOG_LEVEL=info accelerate launch \
    --config_file recipes/accelerate_configs/zero3_4gpus.yaml \
    --main_process_port "${PORT}" \
    --num_processes=4 src/open_r1/grpo.py \
    --config "${PROCESSED_CONFIG}"
