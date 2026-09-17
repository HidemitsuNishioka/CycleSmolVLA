#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "$ROOT_DIR/scripts/common.sh"

require_value FOLLOWER_PORT
require_value FOLLOWER_ID
require_value DATASET_TASK
require_value SMOLVLA_POLICY_PATH
require_value ROLLOUT_DEVICE

CAMERAS="$(camera_config)"

run_lerobot lerobot-rollout \
  --strategy.type=base \
  --inference.type=sync \
  --policy.path="$SMOLVLA_POLICY_PATH" \
  --rename_map="$SMOLVLA_RENAME_MAP" \
  --device="$ROLLOUT_DEVICE" \
  --robot.type=so101_follower \
  --robot.port="$FOLLOWER_PORT" \
  --robot.id="$FOLLOWER_ID" \
  --robot.cameras="$CAMERAS" \
  --task="$DATASET_TASK" \
  --duration="$SMOLVLA_ROLLOUT_DURATION" \
  --fps="$DATASET_FPS"
