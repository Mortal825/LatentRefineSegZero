#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${PROJECT_ROOT}"

STAGE1_OUTPUT_DIR="${STAGE1_OUTPUT_DIR:-training_scripts/refine_segzero/Rl_data/data/query_reflect_grpo_stage1}"
REFLECT_OUTPUT_DIR="${REFLECT_OUTPUT_DIR:-training_scripts/refine_segzero/Rl_data/data/query_reflect_grpo_reflect}"
OUTPUT_DIR="${OUTPUT_DIR:-training_scripts/refine_segzero/Rl_data/data/query_reflect_grpo}"
SHUFFLE_SEED="${SHUFFLE_SEED:-42}"

python -m training_scripts.refine_segzero.Rl_data.build_unified_grpo_data \
  --init-train-path "${STAGE1_OUTPUT_DIR}/init_box_train.json" \
  --init-val-path "${STAGE1_OUTPUT_DIR}/init_box_val.json" \
  --reflect-train-path "${REFLECT_OUTPUT_DIR}/reflect_train.json" \
  --reflect-val-path "${REFLECT_OUTPUT_DIR}/reflect_val.json" \
  --stage1-cache-train-path "${STAGE1_OUTPUT_DIR}/stage1_cache_train.json" \
  --stage1-cache-val-path "${STAGE1_OUTPUT_DIR}/stage1_cache_val.json" \
  --output-dir "${OUTPUT_DIR}" \
  --shuffle-seed "${SHUFFLE_SEED}" \
  --reflect-train-correct-min 2 \
  --reflect-train-correct-max 15 \
  --init-train-sample-count 2000 \
  --overwrite-output
