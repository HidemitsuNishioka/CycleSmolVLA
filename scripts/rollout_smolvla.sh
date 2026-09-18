#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "$ROOT_DIR/scripts/common.sh"

CHECK_ONLY=false
while [[ $# -gt 0 ]]; do
  case "$1" in
    --check) CHECK_ONLY=true; shift ;;
    --checkpoint|--duration)
      [[ $# -ge 2 && -n "$2" ]] || die "$1 requires a value"
      if [[ "$1" == "--checkpoint" ]]; then SMOLVLA_POLICY_PATH="$2"; else SMOLVLA_ROLLOUT_DURATION="$2"; fi
      shift 2 ;;
    -h|--help)
      echo "Usage: $0 [--checkpoint PATH] [--duration SECONDS] [--check]"
      echo "--check validates the runtime and checkpoint without connecting to hardware."
      exit 0 ;;
    *) die "Unknown option: $1 (see --help)" ;;
  esac
done

require_value FOLLOWER_PORT
require_value FOLLOWER_ID
require_value DATASET_TASK
require_value SMOLVLA_POLICY_PATH
require_value ROLLOUT_DEVICE

# Resolve local checkpoints into the workspace mounted inside Docker.
checkpoint_host="$SMOLVLA_POLICY_PATH"
if [[ "$checkpoint_host" == /workspace/* ]]; then
  checkpoint_host="$ROOT_DIR/${checkpoint_host#/workspace/}"
elif [[ "$checkpoint_host" != /* ]]; then
  checkpoint_host="$ROOT_DIR/$checkpoint_host"
fi
checkpoint_host="$(realpath -m "$checkpoint_host")"
[[ "$checkpoint_host" == "$ROOT_DIR/"* ]] || die "Checkpoint must be inside $ROOT_DIR"
for file in config.json model.safetensors policy_preprocessor.json policy_postprocessor.json; do
  [[ -f "$checkpoint_host/$file" ]] || die "Missing $checkpoint_host/$file. Use --checkpoint outputs/train/smolvla_cyclemanip_wandb/checkpoints/last/pretrained_model"
done
checkpoint_container="/workspace/${checkpoint_host#"$ROOT_DIR/"}"

CAMERAS="$(camera_config)"
if [[ "$CHECK_ONLY" != true ]]; then
  [[ -e "$FOLLOWER_PORT" ]] || die "Follower port not found: $FOLLOWER_PORT. Connect the arm and update config/robot.env."
  for camera_device in "${CAMERA_TOP_DEVICE:-}" "${CAMERA_WRIST_DEVICE:-}"; do
    if [[ "$camera_device" == /dev/* && ! -e "$camera_device" ]]; then
      die "Camera not found: $camera_device. Update config/robot.env."
    fi
  done
fi

args=(
  --strategy.type=base
  --inference.type=sync
  --policy.path="$checkpoint_container"
  --rename_map="$SMOLVLA_RENAME_MAP"
  --device="$ROLLOUT_DEVICE"
  --robot.type=so101_follower
  --robot.port="$FOLLOWER_PORT"
  --robot.id="$FOLLOWER_ID"
  --robot.cameras="$CAMERAS"
  --task="$DATASET_TASK"
  --duration="$SMOLVLA_ROLLOUT_DURATION"
  --fps="$DATASET_FPS"
)
entrypoint=(-m lerobot.scripts.lerobot_rollout)
if [[ "$CHECK_ONLY" == true ]]; then
  entrypoint=(/workspace/scripts/check_smolvla_rollout.py)
fi

echo "Checkpoint: $checkpoint_container"
echo "Follower: $FOLLOWER_PORT ($FOLLOWER_ID); duration: ${SMOLVLA_ROLLOUT_DURATION}s; fps: $DATASET_FPS"
# Prefer the same Python environment used for training. Always import the
# workspace source first so the CycleManip inference fixes are used.
run_lerobot bash -c '
  export PYTHONPATH="/workspace/.lerobot-src/src:${PYTHONPATH:-}"
  runtime_python=python
  if [[ -x /workspace/.venv/bin/python ]]; then runtime_python=/workspace/.venv/bin/python; fi
  exec "$runtime_python" "$@"
' rollout "${entrypoint[@]}" "${args[@]}"
