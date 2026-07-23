#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${PROJECT_ROOT}"

INPUT_DIR="${INPUT_DIR:-training_scripts/refine_segzero/Rl_data/data/query_reflect_grpo}"
OUTPUT_DIR="${OUTPUT_DIR:-training_scripts/refine_segzero/Rl_data/data/query_reflect_grpo_balanced}"
SHUFFLE_SEED="${SHUFFLE_SEED:-42}"

python -m training_scripts.refine_segzero.Rl_data.balance_unified_grpo_data \
  --train-path "${INPUT_DIR}/train.json" \
  --val-path "${INPUT_DIR}/val.json" \
  --output-dir "${OUTPUT_DIR}" \
  --shuffle-seed "${SHUFFLE_SEED}" \
  --overwrite-output
