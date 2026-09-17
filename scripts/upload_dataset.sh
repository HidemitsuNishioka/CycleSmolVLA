#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "$ROOT_DIR/scripts/common.sh"

# With no arguments, use DATASET_ROOT and DATASET_REPO_ID from the local config.
# Optional arguments make it possible to upload another existing local dataset:
#   ./scripts/upload_dataset.sh <local-root> <username/dataset-name>
if [[ "$#" -gt 2 ]]; then
  die "Usage: $0 [local_dataset_root] [username/dataset-name]"
fi

if [[ "$#" -ge 1 ]]; then
  DATASET_ROOT="$1"
fi
if [[ "$#" -ge 2 ]]; then
  DATASET_REPO_ID="$2"
fi

require_value DATASET_REPO_ID
require_value DATASET_ROOT
require_file_or_dir DATASET_ROOT

DATASET_PRIVATE="${DATASET_PRIVATE:-false}"

echo "Uploading existing LeRobot dataset"
echo "  local root: $DATASET_ROOT"
echo "  Hub repo:   $DATASET_REPO_ID"
echo "  private:    $DATASET_PRIVATE"

run_lerobot env \
  DATASET_REPO_ID="$DATASET_REPO_ID" \
  DATASET_ROOT="$DATASET_ROOT" \
  DATASET_PRIVATE="$DATASET_PRIVATE" \
  python - <<'PY'
import os
from pathlib import Path

from lerobot.datasets.lerobot_dataset import LeRobotDataset

repo_id = os.environ["DATASET_REPO_ID"]
root = Path(os.environ["DATASET_ROOT"])
private = os.environ["DATASET_PRIVATE"].strip().lower() == "true"

dataset = LeRobotDataset(repo_id=repo_id, root=root)
if dataset.num_episodes == 0:
    raise SystemExit(f"No episodes found in {root}")

print(f"episodes: {dataset.num_episodes}")
print(f"frames:   {dataset.num_frames}")
print("Uploading metadata, parquet files, videos, and the dataset card...")
dataset.push_to_hub(private=private, push_videos=True)
print(f"Upload complete: https://huggingface.co/datasets/{repo_id}")
PY
