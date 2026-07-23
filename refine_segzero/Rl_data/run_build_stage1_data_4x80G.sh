#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${PROJECT_ROOT}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

IMAGE_ROOT="${IMAGE_ROOT:-/path/to/coco/train2014}"
STAGE1_EXPORT_DIR="${STAGE1_EXPORT_DIR:-}"
MLLM_MODEL_PATH="${MLLM_MODEL_PATH:-/path/to/base_mllm}"
PROCESSOR_PATH="${PROCESSOR_PATH:-${MLLM_MODEL_PATH}}"
SAM_CHECKPOINT_PATH="${SAM_CHECKPOINT_PATH:-/path/to/sam2_hiera_large.pt}"
SAM_MODEL_CFG="${SAM_MODEL_CFG:-sam2_hiera_l.yaml}"
OUTPUT_DIR="${OUTPUT_DIR:-training_scripts/refine_segzero/Rl_data/data/query_reflect_grpo_stage1}"
MAX_SAMPLE_RATIO_PER_FILE="${MAX_SAMPLE_RATIO_PER_FILE:-1}"
TRAIN_JSON_PATHS_TEXT="${TRAIN_JSON_PATHS:-/path/to/refcocog_train.json /path/to/refcoco_train.json /path/to/refcoco+_train.json}"
VAL_JSON_PATHS_TEXT="${VAL_JSON_PATHS:-/path/to/refcocog_val.json}"
read -r -a TRAIN_JSON_PATHS_ARRAY <<< "${TRAIN_JSON_PATHS_TEXT}"
read -r -a VAL_JSON_PATHS_ARRAY <<< "${VAL_JSON_PATHS_TEXT}"


BATCH_SIZE="${BATCH_SIZE:-64}"

if [[ -n "${STAGE1_EXPORT_DIR}" ]]; then
  MODEL_ARGS=(--stage1-export-dir "${STAGE1_EXPORT_DIR}")
else
  MODEL_ARGS=(
    --mllm-model-path "${MLLM_MODEL_PATH}"
    --processor-path "${PROCESSOR_PATH}"
    --sam-checkpoint-path "${SAM_CHECKPOINT_PATH}"
    --sam-model-cfg "${SAM_MODEL_CFG}"
  )
fi

torchrun --nproc_per_node=8 \
  -m training_scripts.refine_segzero.Rl_data.build_stage1_data \
  --train-json-paths "${TRAIN_JSON_PATHS_ARRAY[@]}" \
  --val-json-paths "${VAL_JSON_PATHS_ARRAY[@]}" \
  --image-root "${IMAGE_ROOT}" \
  "${MODEL_ARGS[@]}" \
  --output-dir "${OUTPUT_DIR}" \
  --max-sample-ratio-per-file "${MAX_SAMPLE_RATIO_PER_FILE}" \
  --sample-seed 42 \
  --resize-size 840 \
  --sam-image-size 1024 \
  --max-new-tokens 256 \
  --batch-size "${BATCH_SIZE}" \
  --overwrite-output
