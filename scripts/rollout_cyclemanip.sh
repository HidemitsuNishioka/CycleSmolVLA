#!/usr/bin/env bash
# Run the cyclemanip SmolVLA checkpoint on the physical SO-101 follower.
#
# The checkpoint is a LeRobot/SmolVLA checkpoint, so this intentionally uses
# LeRobot's rollout path for the CycleManip policy.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "$ROOT_DIR/scripts/common.sh"

require_value FOLLOWER_PORT
require_value FOLLOWER_ID
require_value ROLLOUT_DEVICE

# Prefer an explicit CycleManip checkpoint.  Do not inherit the generic
# SMOLVLA_POLICY_PATH because it may point at a different trained task.
CHECKPOINT_INPUT="${CYCLEMANIP_CHECKPOINT:-$ROOT_DIR/outputs/train/smolvla_shake_cup_3times_cycle300/016000}"
if [[ "$CHECKPOINT_INPUT" = /* ]]; then
  CHECKPOINT_ROOT="$CHECKPOINT_INPUT"
else
  CHECKPOINT_ROOT="$ROOT_DIR/$CHECKPOINT_INPUT"
fi
if ! CHECKPOINT_DIR="$(realpath -e -- "$CHECKPOINT_ROOT")"; then
  die "checkpoint directory not found: $CHECKPOINT_ROOT"
fi

case "$CHECKPOINT_DIR" in
  "$ROOT_DIR"/*) CONTAINER_CHECKPOINT="/workspace/${CHECKPOINT_DIR#"$ROOT_DIR"/}" ;;
  *) die "CYCLEMANIP_CHECKPOINT must be inside the SO101 project: $CHECKPOINT_DIR" ;;
esac

[[ -f "$CHECKPOINT_DIR/config.json" ]] || die "checkpoint config.json not found: $CHECKPOINT_DIR"
[[ -f "$CHECKPOINT_DIR/model.safetensors" ]] || die "checkpoint model.safetensors not found: $CHECKPOINT_DIR"
grep -Eq '"type"[[:space:]]*:[[:space:]]*"smolvla"' "$CHECKPOINT_DIR/config.json" || {
  die "checkpoint is not a SmolVLA checkpoint: $CHECKPOINT_DIR"
}

TASK="${CYCLEMANIP_TASK:-${DATASET_TASK:-Shake the cup three times.}}"
# Prefer a CycleManip-specific override, then reuse the existing SMOLVLA
# rollout setting from config/dataset.local.env, and finally fall back to 10s.
DURATION="${CYCLEMANIP_DURATION:-${SMOLVLA_ROLLOUT_DURATION:-10}}"
FPS="${CYCLEMANIP_FPS:-${DATASET_FPS:-30}}"
DEVICE="${CYCLEMANIP_DEVICE:-$ROLLOUT_DEVICE}"
MAX_RELATIVE_TARGET="${CYCLEMANIP_MAX_RELATIVE_TARGET:-20.0}"
# Keep motor control decoupled from the comparatively slow VLA forward pass.
# The CycleManip checkpoint produces 50-step chunks, so guided RTC can keep a
# 30 Hz command stream alive while the next chunk is inferred in the background.
INFERENCE_TYPE="${CYCLEMANIP_INFERENCE_TYPE:-rtc}"
RTC_EXECUTION_HORIZON="${CYCLEMANIP_RTC_EXECUTION_HORIZON:-10}"
RTC_GUIDANCE_WEIGHT="${CYCLEMANIP_RTC_GUIDANCE_WEIGHT:-10.0}"
POLICY_ARGS=()
if [[ -n "${CYCLEMANIP_N_ACTION_STEPS:-}" ]]; then
  [[ "$INFERENCE_TYPE" == "sync" ]] || die "CYCLEMANIP_N_ACTION_STEPS requires CYCLEMANIP_INFERENCE_TYPE=sync"
  [[ "$CYCLEMANIP_N_ACTION_STEPS" =~ ^[1-9][0-9]*$ ]] || die "CYCLEMANIP_N_ACTION_STEPS must be a positive integer"
  POLICY_ARGS+=(--policy.n_action_steps="$CYCLEMANIP_N_ACTION_STEPS")
fi
CAMERAS="$(camera_config)"

if [[ -n "${CYCLEMANIP_RENAME_MAP:-}" ]]; then
  RENAME_MAP="$CYCLEMANIP_RENAME_MAP"
elif [[ -n "${SMOLVLA_RENAME_MAP:-}" ]]; then
  RENAME_MAP="$SMOLVLA_RENAME_MAP"
else
  RENAME_MAP='{"observation.images.top":"observation.images.camera1"}'
fi

echo "Checkpoint: $CHECKPOINT_DIR"
echo "Task:      $TASK"
echo "Device:    $DEVICE"
echo "Duration:  ${DURATION}s"
echo "Max delta: ${MAX_RELATIVE_TARGET} deg/step"
echo "Inference: ${INFERENCE_TYPE}"
echo "Keyboard:  R = reset history/actions and restart from the current pose (timer restarts)"
echo "Actions per inference (sync): ${CYCLEMANIP_N_ACTION_STEPS:-checkpoint default}"
echo "Follower:  $FOLLOWER_PORT ($FOLLOWER_ID)"
echo "Camera:    ${CAMERA_TOP_DEVICE:-}${CAMERA_WRIST_DEVICE:+, $CAMERA_WRIST_DEVICE}"

if [[ "${CYCLEMANIP_CONFIRM:-0}" != "1" ]]; then
  echo ""
  echo "WARNING: the next command sends policy actions to the real follower."
  read -r -p '実機へ送信する場合は MOVE と入力してください: ' confirmation
  [[ "$confirmation" == "MOVE" ]] || {
    echo "Actuation cancelled."
    exit 0
  }
fi

CAMERA_ARGS=(--robot.cameras="$CAMERAS")
INFERENCE_ARGS=(--inference.type="$INFERENCE_TYPE")
if [[ "$INFERENCE_TYPE" == "rtc" ]]; then
  INFERENCE_ARGS+=(
    --inference.rtc.mode=guided
    --inference.rtc.execution_horizon="$RTC_EXECUTION_HORIZON"
    --inference.rtc.max_guidance_weight="$RTC_GUIDANCE_WEIGHT"
  )
elif [[ "$INFERENCE_TYPE" != "sync" ]]; then
  die "CYCLEMANIP_INFERENCE_TYPE must be rtc or sync (got: $INFERENCE_TYPE)"
fi
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
run_lerobot bash -c '
  # The CycleManip model class lives in the workspace checkout.  Put it before
  # the image installed LeRobot so the checkpoint config and implementation
  # are interpreted by the same code used for training.
  export PYTHONPATH="/workspace/.lerobot-src/src:${PYTHONPATH:-}"
  runtime_python=python
  if [[ -x /workspace/.venv/bin/python ]]; then runtime_python=/workspace/.venv/bin/python; fi
  exec "$runtime_python" -m lerobot.scripts.lerobot_rollout "$@"
' rollout \
  --strategy.type=base \
  --strategy.keyboard_restart=true \
  "${INFERENCE_ARGS[@]}" \
  --policy.path="$CONTAINER_CHECKPOINT" \
  "${POLICY_ARGS[@]}" \
  --rename_map="$RENAME_MAP" \
  --device="$DEVICE" \
  --robot.type=so101_follower \
  --robot.port="$FOLLOWER_PORT" \
  --robot.id="$FOLLOWER_ID" \
  --robot.max_relative_target="$MAX_RELATIVE_TARGET" \
  "${CAMERA_ARGS[@]}" \
  --task="$TASK" \
  --duration="$DURATION" \
  --fps="$FPS" \
  --play_sounds=false
