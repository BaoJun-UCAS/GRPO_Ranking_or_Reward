#!/usr/bin/env bash
# Ranking shares the maintained four-GPU launcher and all of its overrides.
set -Eeuo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export ADVANTAGE="${ADVANTAGE:-ranking}"
exec bash "${SCRIPT_DIR}/qwen3_1.7_grpo_chat.sh" "$@"
