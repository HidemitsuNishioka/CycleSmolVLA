#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "$ROOT_DIR/scripts/common.sh"

require_value LEADER_PORT
require_value FOLLOWER_PORT
require_value LEADER_ID
require_value FOLLOWER_ID

echo "Calibrating the SO-101 Follower with the official LeRobot command..."
run_lerobot lerobot-calibrate \
  --robot.type=so101_follower \
  --robot.port="$FOLLOWER_PORT" \
  --robot.id="$FOLLOWER_ID"

echo "Calibrating the SO-101 Leader with the official LeRobot command..."
run_lerobot lerobot-calibrate \
  --teleop.type=so101_leader \
  --teleop.port="$LEADER_PORT" \
  --teleop.id="$LEADER_ID"

