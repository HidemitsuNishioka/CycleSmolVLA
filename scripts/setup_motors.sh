#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "$ROOT_DIR/scripts/common.sh"

require_value LEADER_PORT
require_value FOLLOWER_PORT

echo "Setting up SO-101 Leader motors with the official LeRobot command..."
run_lerobot lerobot-setup-motors \
  --teleop.type=so101_leader \
  --teleop.port="$LEADER_PORT"

echo "Setting up SO-101 Follower motors with the official LeRobot command..."
run_lerobot lerobot-setup-motors \
  --robot.type=so101_follower \
  --robot.port="$FOLLOWER_PORT"

