#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

set -x

export NCCL_SOCKET_IFNAME=eth0
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export NCCL_TRANSPORT=socket
export NCCL_NSOCKS_PERTHREAD=8
export NCCL_SOCKET_NTHREADS=4

GEOMETRIC_EXPORT_DIR="${GEOMETRIC_EXPORT_DIR:-/path/to/geometric_export}"
SHARED_MLLM_PATH="${SHARED_MLLM_PATH:-}"
REF_JSON_PATH="${REF_JSON_PATH:-/path/to/refcocog_test.json}"
IMAGE_ROOT="${IMAGE_ROOT:-/path/to/coco/images}"
CONFIDENCE_THRESHOLD="${CONFIDENCE_THRESHOLD:-0.17}"
DECISION_MODE="${DECISION_MODE:-sign}"
MASTER_PORT="${MASTER_PORT:-29504}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
LIMIT="${LIMIT:--1}"
DRY_RUN="${DRY_RUN:-0}"
ENABLE_REFLECTION="${ENABLE_REFLECTION:-1}"
FORCE_DIRECT_AFTER_REFLECTION="${FORCE_DIRECT_AFTER_REFLECTION:-0}"
ENABLE_REFLECT_DECISION_LOGIT_THRESHOLD="${ENABLE_REFLECT_DECISION_LOGIT_THRESHOLD:-0}"
REFLECT_ACCEPT_PROBABILITY_THRESHOLD="${REFLECT_ACCEPT_PROBABILITY_THRESHOLD:-0.50}"

REFLECTION_ARGS=()
FORCE_DIRECT_ARGS=()
REFLECT_DECISION_LOGIT_ARGS=()
DEFAULT_OUTPUT_DIR="outputs/query_reflect_refcocog"
if [[ "${ENABLE_REFLECTION}" == "0" ]]; then
  REFLECTION_ARGS=(--no-enable-reflection)
  DEFAULT_OUTPUT_DIR="${DEFAULT_OUTPUT_DIR}_no_reflect"
fi
if [[ "${FORCE_DIRECT_AFTER_REFLECTION}" == "1" ]]; then
  if [[ "${ENABLE_REFLECTION}" == "0" ]]; then
    echo "FORCE_DIRECT_AFTER_REFLECTION=1 requires ENABLE_REFLECTION=1" >&2
    exit 2
  fi
  FORCE_DIRECT_ARGS=(--force-direct-after-reflection)
  DEFAULT_OUTPUT_DIR="${DEFAULT_OUTPUT_DIR}_force_direct"
fi
if [[ "${ENABLE_REFLECT_DECISION_LOGIT_THRESHOLD}" == "1" ]]; then
  if [[ "${ENABLE_REFLECTION}" == "0" ]]; then
    echo "ENABLE_REFLECT_DECISION_LOGIT_THRESHOLD=1 requires ENABLE_REFLECTION=1" >&2
    exit 2
  fi
  REFLECT_DECISION_LOGIT_ARGS=(
    --enable-reflect-decision-logit-threshold
    --reflect-accept-probability-threshold "${REFLECT_ACCEPT_PROBABILITY_THRESHOLD}"
  )
  DEFAULT_OUTPUT_DIR="${DEFAULT_OUTPUT_DIR}_decision_logit_threshold_${REFLECT_ACCEPT_PROBABILITY_THRESHOLD}"
fi
OUTPUT_DIR="${OUTPUT_DIR:-${DEFAULT_OUTPUT_DIR}}"

DIRECT_BRANCH_PROMPT_MODE="${DIRECT_BRANCH_PROMPT_MODE:-query_reflect_reason_only}"
PREDICT_COMMAND=(
  torchrun
  "--master_port=${MASTER_PORT}"
  "--nproc_per_node=${NPROC_PER_NODE}"
  -m training_scripts.eval.generate_query_reflect_predictions_refcocog
  --geometric-export-dir "${GEOMETRIC_EXPORT_DIR}"
  --shared-mllm-path "${SHARED_MLLM_PATH}"
  --ref-json-path "${REF_JSON_PATH}"
  --image-root "${IMAGE_ROOT}"
  --output-dir "${OUTPUT_DIR}"
  --confidence-threshold "${CONFIDENCE_THRESHOLD}"
  --decision-mode "${DECISION_MODE}"
  --save-branch-breakdown
  --reflect-max-new-tokens 256
  --stage1-max-new-tokens 256
)
if (( ${#REFLECTION_ARGS[@]} )); then
  PREDICT_COMMAND+=("${REFLECTION_ARGS[@]}")
fi
if (( ${#FORCE_DIRECT_ARGS[@]} )); then
  PREDICT_COMMAND+=("${FORCE_DIRECT_ARGS[@]}")
fi
if (( ${#REFLECT_DECISION_LOGIT_ARGS[@]} )); then
  PREDICT_COMMAND+=("${REFLECT_DECISION_LOGIT_ARGS[@]}")
fi
PREDICT_COMMAND+=(
  --use-direct-query-for-direct-branch
  --direct-branch-prompt-mode "${DIRECT_BRANCH_PROMPT_MODE}"
  --limit "${LIMIT}"
)

if [[ "${DRY_RUN}" == "1" ]]; then
  printf 'Prediction command:'
  printf ' %q' "${PREDICT_COMMAND[@]}"
  printf '\nMetrics command: python3 -m training_scripts.eval.calculate_query_reflect_metrics --output-dir %q\n' "${OUTPUT_DIR}"
  exit 0
fi

[[ -d "${GEOMETRIC_EXPORT_DIR}" ]] || { echo "Missing GEOMETRIC_EXPORT_DIR: ${GEOMETRIC_EXPORT_DIR}" >&2; exit 2; }
[[ -f "${REF_JSON_PATH}" ]] || { echo "Missing REF_JSON_PATH: ${REF_JSON_PATH}" >&2; exit 2; }
[[ -d "${IMAGE_ROOT}" ]] || { echo "Missing IMAGE_ROOT: ${IMAGE_ROOT}" >&2; exit 2; }
command -v torchrun >/dev/null || { echo "torchrun is not available in PATH" >&2; exit 2; }

"${PREDICT_COMMAND[@]}"

python3 -m training_scripts.eval.calculate_query_reflect_metrics \
  --output-dir "${OUTPUT_DIR}"
