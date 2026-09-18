#!/usr/bin/env bash
# Run HAMLET against the physical SO-101 follower.
# Default is dry-run. Set HAMLET_EXECUTE=1 to enable motor commands.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ -f "$REPO_ROOT/config/robot.env" ]]; then
  # shellcheck disable=SC1091
  source "$REPO_ROOT/config/robot.env"
fi
if [[ -f "$REPO_ROOT/config/dataset.env" ]]; then
  # shellcheck disable=SC1091
  source "$REPO_ROOT/config/dataset.env"
fi
if [[ -f "$REPO_ROOT/config/dataset.local.env" ]]; then
  # shellcheck disable=SC1091
  source "$REPO_ROOT/config/dataset.local.env"
fi

PYTHON="${HAMLET_PYTHON:-$REPO_ROOT/HAMLET-Isaac-GR00T/.venv/bin/python}"
MODEL_PATH="${HAMLET_MODEL_PATH:-$REPO_ROOT/outputs/train/hamlet}"
PORT="${HAMLET_PORT:-${FOLLOWER_PORT:-}}"
ROBOT_ID="${HAMLET_ROBOT_ID:-${FOLLOWER_ID:-so101_follower}}"
CAMERA="${HAMLET_CAMERA_DEVICE:-${CAMERA_TOP_DEVICE:-}}"
TASK="${HAMLET_TASK:-${DATASET_TASK:-Fold the cloth in half.}}"
DEVICE="${HAMLET_DEVICE:-cuda}"
FPS="${HAMLET_FPS:-30}"
MAX_STEPS="${HAMLET_MAX_STEPS:-1}"
MAX_DELTA="${HAMLET_MAX_RELATIVE_TARGET:-5}"
EXECUTE="${HAMLET_EXECUTE:-0}"

[[ -x "$PYTHON" ]] || { echo "ERROR: HAMLET Python not found: $PYTHON" >&2; exit 1; }
[[ -d "$MODEL_PATH" ]] || { echo "ERROR: model not found: $MODEL_PATH" >&2; exit 1; }
[[ -n "$PORT" ]] || { echo "ERROR: set FOLLOWER_PORT or HAMLET_PORT" >&2; exit 1; }
[[ -n "$CAMERA" ]] || { echo "ERROR: set CAMERA_TOP_DEVICE or HAMLET_CAMERA_DEVICE" >&2; exit 1; }
[[ -r "$PORT" && -w "$PORT" ]] || {
  echo "ERROR: serial device is not readable/writable: $PORT" >&2
  echo "Add the user to dialout, then log in again: sudo usermod -aG dialout \"$USER\"" >&2
  exit 1
}

EXECUTE_ARGS=()
if [[ "$EXECUTE" == "1" ]]; then
  EXECUTE_ARGS+=(--execute)
else
  echo "DRY-RUN: no motor commands will be sent. Set HAMLET_EXECUTE=1 to enable actuation."
fi

export PYTHONPATH="$REPO_ROOT/HAMLET-Isaac-GR00T:$REPO_ROOT/.lerobot-src/src${PYTHONPATH:+:$PYTHONPATH}"
export GR00T_INFERENCE_SEED="${HAMLET_SEED:-6}"
exec env "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}" \
  "$PYTHON" "$REPO_ROOT/scripts/hamlet_so101_rollout.py" \
  --model-path "$MODEL_PATH" \
  --port "$PORT" \
  --robot-id "$ROBOT_ID" \
  --camera-device "$CAMERA" \
  --camera-width "${CAMERA_WIDTH:-640}" \
  --camera-height "${CAMERA_HEIGHT:-480}" \
  --camera-fps "${CAMERA_FPS:-30}" \
  --task "$TASK" \
  --device "$DEVICE" \
  --fps "$FPS" \
  --max-steps "$MAX_STEPS" \
  --max-relative-target "$MAX_DELTA" \
  "${EXECUTE_ARGS[@]}"
