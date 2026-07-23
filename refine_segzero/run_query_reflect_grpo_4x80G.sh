#!/bin/bash

set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

set -x
export NCCL_SOCKET_IFNAME=eth0
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export NCCL_TRANSPORT=socket
export NCCL_NSOCKS_PERTHREAD=8
export NCCL_SOCKET_NTHREADS=4

export VLLM_ATTENTION_BACKEND=XFORMERS

MODEL_PATH="${MODEL_PATH:-/path/to/geometric_export/mllm}"
TRAIN_DATA="${TRAIN_DATA:-training_scripts/refine_segzero/Rl_data/data/query_reflect_grpo/train.json}"
VAL_DATA="${VAL_DATA:-training_scripts/refine_segzero/Rl_data/data/query_reflect_grpo/val.json}"
SAVE_DIR="${SAVE_DIR:-outputs/query_reflect_grpo}"
RUN_NAME=$(basename "$0" .sh)

export QUERY_REFLECT_METRICS_DIR="${SAVE_DIR}"

python3 -m verl.trainer.main \
    config=training_scripts/refine_segzero/configs/query_reflect_grpo_7b.yaml \
    "data.train_files=${TRAIN_DATA}" \
    "data.val_files=${VAL_DATA}" \
    "worker.actor.model.model_path=${MODEL_PATH}" \
    "worker.actor.model.tokenizer_path=${MODEL_PATH}" \
    worker.actor.kl_loss_coef=1.0e-2 \
    worker.actor.use_kl_loss=false \
    algorithm.kl_coef=1.0e-2 \
    worker.actor.optim.lr=1.0e-6 \
    worker.actor.entropy_coeff=0.0 \
    worker.actor.micro_batch_size_per_device_for_update=4 \
    worker.actor.micro_batch_size_per_device_for_experience=8 \
    worker.rollout.tensor_parallel_size=2 \
    worker.rollout.gpu_memory_utilization=0.4 \
    worker.rollout.enable_chunked_prefill=false \
    worker.rollout.n=8 \
    worker.reward.compute_score=query_reflect_grpo \
    "trainer.experiment_name=${RUN_NAME}" \
    trainer.n_gpus_per_node=8 \
    trainer.total_episodes=1 \
    trainer.auto_resume=false \
    "trainer.save_checkpoint_path=${SAVE_DIR}"
