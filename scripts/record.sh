#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "$ROOT_DIR/scripts/common.sh"

require_value LEADER_PORT
require_value FOLLOWER_PORT
require_value LEADER_ID
require_value FOLLOWER_ID
require_value DATASET_REPO_ID
require_value DATASET_ROOT
require_value DATASET_TASK

mkdir -p "$ROOT_DIR/$(dirname "$DATASET_ROOT")"
CAMERAS="$(camera_config)"

run_lerobot lerobot-record \
  --robot.type=so101_follower \
  --robot.port="$FOLLOWER_PORT" \
  --robot.id="$FOLLOWER_ID" \
  --robot.cameras="$CAMERAS" \
  --teleop.type=so101_leader \
  --teleop.port="$LEADER_PORT" \
  --teleop.id="$LEADER_ID" \
  --dataset.repo_id="$DATASET_REPO_ID" \
  --dataset.root="$DATASET_ROOT" \
  --dataset.single_task="$DATASET_TASK" \
  --dataset.fps="$DATASET_FPS" \
  --dataset.episode_time_s="$DATASET_EPISODE_TIME_S" \
  --dataset.reset_time_s="$DATASET_RESET_TIME_S" \
  --dataset.num_episodes="$DATASET_NUM_EPISODES" \
  --dataset.video=true \
  --dataset.push_to_hub="$DATASET_PUSH_TO_HUB" \
  --dataset.private="$DATASET_PRIVATE" \
  --dataset.streaming_encoding="$DATASET_STREAMING_ENCODING" \
  --dataset.encoder_threads="$DATASET_ENCODER_THREADS" \
  --dataset.no_stamp=true \
  --play_sounds=false \
  --display_data="$DISPLAY_DATA"
