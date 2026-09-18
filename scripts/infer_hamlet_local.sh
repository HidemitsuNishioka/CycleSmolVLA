#!/usr/bin/env bash
# Run HAMLET inference/evaluation locally on the CUDA GPU.
# This is open-loop evaluation only; it never sends actions to the robot.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HAMLET_ROOT="${REPO_ROOT}/HAMLET-Isaac-GR00T"
cd "$REPO_ROOT"

PYTHON="${HAMLET_PYTHON:-${HAMLET_ROOT}/.venv/bin/python}"
CHECKPOINT="${HAMLET_CHECKPOINT:-${REPO_ROOT}/outputs/train/hamlet}"
DATASET_PATH="${HAMLET_DATASET_PATH:-${HAMLET_ROOT}/data/so101_shake_cup_gr00t}"
OUTPUT_DIR="${HAMLET_OUTPUT_DIR:-${REPO_ROOT}/outputs/eval/hamlet_local_open_loop}"
DEVICE="${HAMLET_DEVICE:-cuda}"
EVAL_FRACTION="${HAMLET_EVAL_FRACTION:-0.2}"
STRIDE="${HAMLET_STRIDE:-16}"
MAX_ANCHORS="${HAMLET_MAX_ANCHORS_PER_EPISODE:-0}"
SEED="${HAMLET_SEED:-6}"
EPISODES="${HAMLET_EPISODES:-}"

[[ -x "$PYTHON" ]] || {
  echo "ERROR: HAMLET Python not found: $PYTHON" >&2
  exit 1
}
[[ -d "$CHECKPOINT" ]] || {
  echo "ERROR: checkpoint not found: $CHECKPOINT" >&2
  exit 1
}
[[ -d "$DATASET_PATH" ]] || {
  echo "ERROR: dataset not found: $DATASET_PATH" >&2
  exit 1
}

EPISODE_ARGS=()
if [[ -n "$EPISODES" ]]; then
  EPISODE_ARGS+=(--episodes "$EPISODES")
fi

echo "Checkpoint: $CHECKPOINT"
echo "Dataset:    $DATASET_PATH"
echo "Output:     $OUTPUT_DIR"
echo "Device:     $DEVICE"

exec env "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}" \
  "$PYTHON" "$REPO_ROOT/scripts/evaluate_hamlet_so101_open_loop.py" \
  --checkpoint "$CHECKPOINT" \
  --dataset-path "$DATASET_PATH" \
  --output-dir "$OUTPUT_DIR" \
  --eval-fraction "$EVAL_FRACTION" \
  --stride "$STRIDE" \
  --max-anchors-per-episode "$MAX_ANCHORS" \
  --seed "$SEED" \
  --device "$DEVICE" \
  "${EPISODE_ARGS[@]}"
