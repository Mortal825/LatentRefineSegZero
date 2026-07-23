#!/bin/bash

set -euo pipefail


SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${PROJECT_ROOT}"

OUTPUT_DIR="${OUTPUT_DIR:-training_scripts/refine_segzero/Rl_data/data/query_reflect_grpo_reflect}"
MLLM_MODEL_PATH="${MLLM_MODEL_PATH:-/path/to/geometric_export/mllm}"
STAGE1_TRAIN_PATHS_TEXT="${STAGE1_TRAIN_PATHS:-training_scripts/refine_segzero/Rl_data/data/query_reflect_grpo_stage1/stage1_cache_train.json}"
STAGE1_VAL_PATHS_TEXT="${STAGE1_VAL_PATHS:-training_scripts/refine_segzero/Rl_data/data/query_reflect_grpo_stage1/stage1_cache_val.json}"
read -r -a STAGE1_TRAIN_PATHS_ARRAY <<< "${STAGE1_TRAIN_PATHS_TEXT}"
read -r -a STAGE1_VAL_PATHS_ARRAY <<< "${STAGE1_VAL_PATHS_TEXT}"

torchrun --nproc_per_node=8 \
  -m training_scripts.refine_segzero.Rl_data.build_reflect_samples \
  --stage1-train-paths "${STAGE1_TRAIN_PATHS_ARRAY[@]}" \
  --stage1-val-paths "${STAGE1_VAL_PATHS_ARRAY[@]}" \
  --output-dir "${OUTPUT_DIR}" \
  --mllm-model-path "${MLLM_MODEL_PATH}" \
  --reflect-sample-count 16 \
  --reflect-temperature 1.2 \
  --reflect-top-p 1.0 \
  --debug-sample-output-limit 100 \
  --confidence-threshold 0.5 \
  --overwrite-output
