#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "$ROOT_DIR/scripts/common.sh"

echo "Pulling the official Hugging Face LeRobot GPU image..."
docker compose -f "$COMPOSE_FILE" pull lerobot

echo "Checking the official LeRobot installation and GPU access..."
run_lerobot lerobot-info
run_lerobot nvidia-smi

