#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "$ROOT_DIR/scripts/common.sh"

require_value DATASET_REPO_ID
require_value DATASET_ROOT
require_value DIFFUSION_OUTPUT_DIR
require_value DIFFUSION_JOB_NAME
require_value POLICY_DEVICE

mkdir -p "$ROOT_DIR/$(dirname "$DIFFUSION_OUTPUT_DIR")"

args=(
  lerobot-train
  --dataset.repo_id="$DATASET_REPO_ID"
  --dataset.root="$DATASET_ROOT"
  --dataset.eval_split="$DATASET_EVAL_SPLIT"
  --policy.device="$POLICY_DEVICE"
  --output_dir="$DIFFUSION_OUTPUT_DIR"
  --job_name="$DIFFUSION_JOB_NAME"
  --batch_size="$DIFFUSION_BATCH_SIZE"
  --steps="$DIFFUSION_STEPS"
  --eval_steps="$DIFFUSION_EVAL_STEPS"
  --max_eval_samples="$DIFFUSION_MAX_EVAL_SAMPLES"
  --save_freq="$DIFFUSION_SAVE_FREQ"
  --log_freq="$DIFFUSION_LOG_FREQ"
  --num_workers="$DIFFUSION_NUM_WORKERS"
  --policy.push_to_hub="$POLICY_PUSH_TO_HUB"
  --wandb.enable="$WANDB_ENABLE"
  --wandb.project="$WANDB_PROJECT"
  --wandb.mode="$WANDB_MODE"
)

if [[ -n "${WANDB_ENTITY:-}" ]]; then
  args+=(--wandb.entity="$WANDB_ENTITY")
fi

if [[ -n "${DIFFUSION_PRETRAINED_PATH:-}" ]]; then
  # LeRobot accepts either a local `pretrained_model` directory or a Hub repo
  # id such as `org/compatible-diffusion-policy`.
  if [[ -e "$ROOT_DIR/$DIFFUSION_PRETRAINED_PATH" || -e "$DIFFUSION_PRETRAINED_PATH" ]]; then
    :
  elif [[ "$DIFFUSION_PRETRAINED_PATH" != */* ]]; then
    die "DIFFUSION_PRETRAINED_PATH must be a local checkpoint or a Hub repo id (org/name): $DIFFUSION_PRETRAINED_PATH"
  fi
  echo "Initializing from LeRobot policy checkpoint: $DIFFUSION_PRETRAINED_PATH"
  args+=(--policy.path="$DIFFUSION_PRETRAINED_PATH")
else
  args+=(--policy.type=diffusion)
fi

if [[ -n "${POLICY_REPO_ID:-}" ]]; then
  args+=(--policy.repo_id="$POLICY_REPO_ID")
fi

run_lerobot "${args[@]}"
