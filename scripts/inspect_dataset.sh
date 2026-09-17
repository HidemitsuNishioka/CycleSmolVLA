#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "$ROOT_DIR/scripts/common.sh"

require_value DATASET_REPO_ID
require_value DATASET_ROOT

run_lerobot env \
  DATASET_REPO_ID="$DATASET_REPO_ID" \
  DATASET_ROOT="$DATASET_ROOT" \
  python -c '
from lerobot.datasets.lerobot_dataset import LeRobotDataset
import os

repo_id = os.environ["DATASET_REPO_ID"]
root = os.environ["DATASET_ROOT"]
dataset = LeRobotDataset(repo_id=repo_id, root=root)
print(f"repo_id: {dataset.repo_id}")
print(f"episodes: {dataset.num_episodes}")
print(f"frames: {dataset.num_frames}")
print(f"fps: {dataset.fps}")
print("features:")
for key, feature in dataset.features.items():
    print(f"  {key}: {feature}")
required = {"observation.state", "action"}
missing = sorted(required - set(dataset.features))
if missing:
    raise SystemExit(f"Missing required features: {missing}")
image_keys = sorted(key for key in dataset.features if key.startswith("observation.image"))
if not image_keys:
    raise SystemExit("No observation.image feature found")
print(f"camera_features: {image_keys}")
'
