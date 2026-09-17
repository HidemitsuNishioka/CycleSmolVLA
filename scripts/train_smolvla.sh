#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "$ROOT_DIR/scripts/common.sh"

require_value DATASET_REPO_ID
require_value DATASET_ROOT
require_value DATASET_EVAL_SPLIT
require_value SMOLVLA_BASE_PATH
require_value SMOLVLA_OUTPUT_DIR
require_value SMOLVLA_JOB_NAME
require_value POLICY_DEVICE

mkdir -p "$ROOT_DIR/$(dirname "$SMOLVLA_OUTPUT_DIR")"

args=(
  lerobot-train
  --policy.path="$SMOLVLA_BASE_PATH"
  --dataset.repo_id="$DATASET_REPO_ID"
  --dataset.root="$DATASET_ROOT"
  --dataset.eval_split="$DATASET_EVAL_SPLIT"
  --rename_map="$SMOLVLA_RENAME_MAP"
  --policy.empty_cameras="$SMOLVLA_EMPTY_CAMERAS"
  --policy.device="$POLICY_DEVICE"
  --policy.use_amp="$SMOLVLA_USE_AMP"
  --policy.chunk_size="$SMOLVLA_CHUNK_SIZE"
  --policy.n_action_steps="$SMOLVLA_N_ACTION_STEPS"
  --policy.freeze_vision_encoder="$SMOLVLA_FREEZE_VISION_ENCODER"
  --policy.train_expert_only="$SMOLVLA_TRAIN_EXPERT_ONLY"
  --policy.train_state_proj="$SMOLVLA_TRAIN_STATE_PROJ"
  --policy.push_to_hub="$SMOLVLA_PUSH_TO_HUB"
  --output_dir="$SMOLVLA_OUTPUT_DIR"
  --job_name="$SMOLVLA_JOB_NAME"
  --batch_size="$SMOLVLA_BATCH_SIZE"
  --steps="$SMOLVLA_STEPS"
  --save_freq="$SMOLVLA_SAVE_FREQ"
  --log_freq="$SMOLVLA_LOG_FREQ"
  --num_workers="$SMOLVLA_NUM_WORKERS"
  --wandb.enable="$WANDB_ENABLE"
  --wandb.project="$WANDB_PROJECT"
  --wandb.mode="$WANDB_MODE"
)

if [[ -n "${WANDB_ENTITY:-}" ]]; then
  args+=(--wandb.entity="$WANDB_ENTITY")
fi

if [[ -n "${SMOLVLA_POLICY_REPO_ID:-}" ]]; then
  args+=(--policy.repo_id="$SMOLVLA_POLICY_REPO_ID")
fi

run_lerobot "${args[@]}"
