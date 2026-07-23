#!/bin/bash

set -euo pipefail

export NCCL_SOCKET_IFNAME=eth0
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export NCCL_TRANSPORT=socket
export NCCL_NSOCKS_PERTHREAD=8
export NCCL_SOCKET_NTHREADS=4

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

CONFIG_PATH="${CONFIG_PATH:-training_scripts/refine_segzero/configs/geometric_query_branch_sft_7b.yaml}"
TRAIN_BRANCH="${TRAIN_BRANCH:-aligned}"

torchrun --master_port "${MASTER_PORT:-29502}" --nproc_per_node=8 \
  -m training_scripts.refine_segzero.geometric_query_train \
  --config "${CONFIG_PATH}" \
  train_branch="${TRAIN_BRANCH}" \
  "$@"
