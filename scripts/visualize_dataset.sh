#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "$ROOT_DIR/scripts/common.sh"

require_value DATASET_REPO_ID
require_value DATASET_ROOT
EPISODE_INDEX="${1:-0}"

run_lerobot lerobot-dataset-viz \
  --repo-id="$DATASET_REPO_ID" \
  --root="$DATASET_ROOT" \
  --episode-index="$EPISODE_INDEX" \
  --mode=local

