#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "$ROOT_DIR/scripts/common.sh"

require_value DATASET_REPO_ID
require_value DATASET_ROOT
require_value DATASET_EVAL_SPLIT
require_value DIFFUSION_BASELINE_OUTPUT_DIR
require_value DIFFUSION_BASELINE_JOB_NAME
require_value POLICY_DEVICE

mkdir -p "$ROOT_DIR/$(dirname "$DIFFUSION_BASELINE_OUTPUT_DIR")"

echo "Open-loop baseline: official Diffusion Policy with zero policy learning rate"
echo "The held-out eval_loss below is the pre-training reference."

args=(
  lerobot-train
  --dataset.repo_id="$DATASET_REPO_ID"
  --dataset.root="$DATASET_ROOT"
  --dataset.eval_split="$DATASET_EVAL_SPLIT"
  --policy.type=diffusion
  --policy.device="$POLICY_DEVICE"
  --output_dir="$DIFFUSION_BASELINE_OUTPUT_DIR"
  --job_name="$DIFFUSION_BASELINE_JOB_NAME"
  --batch_size="$DIFFUSION_BATCH_SIZE"
  --steps=1
  --eval_steps=1
  --max_eval_samples="$DIFFUSION_MAX_EVAL_SAMPLES"
  --save_freq=1
  --log_freq=1
  --num_workers="$DIFFUSION_NUM_WORKERS"
  --policy.optimizer_lr=0
  --policy.push_to_hub=false
  --wandb.enable=false
)

run_lerobot "${args[@]}"
