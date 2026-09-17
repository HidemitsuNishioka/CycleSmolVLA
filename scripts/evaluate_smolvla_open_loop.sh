#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "$ROOT_DIR/scripts/common.sh"

require_value DATASET_REPO_ID
require_value DATASET_ROOT
require_value DATASET_EVAL_SPLIT
require_value SMOLVLA_POLICY_PATH
require_value POLICY_DEVICE

OUTPUT_DIR="${SMOLVLA_OPEN_LOOP_OUTPUT_DIR:-outputs/eval/smolvla_so101_wrist_20k_open_loop}"
ACTION_MODE="${SMOLVLA_OPEN_LOOP_ACTION_MODE:-fresh}"
EPISODES="${SMOLVLA_OPEN_LOOP_EPISODES:-}"
STRIDE="${SMOLVLA_OPEN_LOOP_STRIDE:-1}"
MAX_FRAMES="${SMOLVLA_OPEN_LOOP_MAX_FRAMES:-0}"

mkdir -p "$ROOT_DIR/$OUTPUT_DIR"
echo "Official LeRobot open-loop evaluation with GT/prediction/error video"
echo "Checkpoint: $SMOLVLA_POLICY_PATH"
echo "Action mode: $ACTION_MODE"
echo "Output: $OUTPUT_DIR"

run_lerobot python /workspace/scripts/evaluate_smolvla_open_loop_video.py \
  --dataset-repo-id="$DATASET_REPO_ID" \
  --dataset-root="/workspace/$DATASET_ROOT" \
  --checkpoint="/workspace/$SMOLVLA_POLICY_PATH" \
  --output-dir="/workspace/$OUTPUT_DIR" \
  --device="$POLICY_DEVICE" \
  --eval-split="$DATASET_EVAL_SPLIT" \
  --episodes="$EPISODES" \
  --stride="$STRIDE" \
  --max-frames-per-episode="$MAX_FRAMES" \
  --action-mode="$ACTION_MODE"
