#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "$ROOT_DIR/scripts/common.sh"

require_value LEADER_PORT
require_value FOLLOWER_PORT
require_value LEADER_ID
require_value FOLLOWER_ID

CAMERAS="$(camera_config)"

run_lerobot lerobot-teleoperate \
  --robot.type=so101_follower \
  --robot.port="$FOLLOWER_PORT" \
  --robot.id="$FOLLOWER_ID" \
  --robot.cameras="$CAMERAS" \
  --teleop.type=so101_leader \
  --teleop.port="$LEADER_PORT" \
  --teleop.id="$LEADER_ID" \
  --fps="$CAMERA_FPS" \
  --display_data="$DISPLAY_DATA"
