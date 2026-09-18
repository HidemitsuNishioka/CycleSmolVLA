#!/usr/bin/env bash
# Run HAMLET SO-101 open-loop evaluation entirely inside the Thor Docker image.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="${HAMLET_DOCKER_IMAGE:-so101-hamlet-thor:26.03}"
HF_CACHE="${HF_CACHE:-${XDG_CACHE_HOME:-/home/kaina/.cache}/huggingface}"
HOST_UID="$(id -u)"
HOST_GID="$(id -g)"
DATASET_PATH="${DATASET_PATH:-/workspace/SO101/data/so101_shake_cup_gr00t}"
OUTPUT_DIR="${OUTPUT_DIR:-/workspace/SO101/outputs/eval/hamlet_so101_open_loop}"
EVAL_FRACTION="${EVAL_FRACTION:-0.2}"
STRIDE="${STRIDE:-16}"
MAX_ANCHORS_PER_EPISODE="${MAX_ANCHORS_PER_EPISODE:-0}"
SEED="${SEED:-6}"

CHECKPOINT="${HAMLET_CHECKPOINT:-}"
if [[ -z "$CHECKPOINT" ]]; then
  latest_number="$(find "$REPO_ROOT/outputs/train/hamlet_so101_shake_cup" \
    -maxdepth 1 -type d -name 'checkpoint-*' -printf '%f\n' \
    | awk -F- '$2 ~ /^[0-9]+$/ {print $2}' | sort -n | tail -1)"
  if [[ -z "$latest_number" ]]; then
    echo "ERROR: no HAMLET checkpoint found" >&2
    exit 1
  fi
  CHECKPOINT="/workspace/SO101/outputs/train/hamlet_so101_shake_cup/checkpoint-${latest_number}"
elif [[ "$CHECKPOINT" != /* ]]; then
  CHECKPOINT="/workspace/SO101/$CHECKPOINT"
fi

mkdir -p "$HF_CACHE"

docker build \
  --tag "$IMAGE" \
  --file "$REPO_ROOT/docker/Dockerfile.hamlet-thor" \
  "$REPO_ROOT"

DOCKER_ARGS=(
  --rm
  --user "$HOST_UID:$HOST_GID"
  --gpus all
  --runtime=nvidia
  --ipc=host
  --ulimit memlock=-1
  --ulimit stack=67108864
  --workdir /workspace/SO101
  --volume "$REPO_ROOT:/workspace/SO101"
  --volume "$HF_CACHE:/workspace/.cache/huggingface"
  --env HF_HOME=/workspace/.cache/huggingface
  --env TRANSFORMERS_CACHE=/workspace/.cache/huggingface/hub
  --env GR00T_INFERENCE_SEED="$SEED"
)

if [[ -f "$REPO_ROOT/config/dataset.local.env" ]]; then
  DOCKER_ARGS+=(--env-file "$REPO_ROOT/config/dataset.local.env")
fi

exec docker run "${DOCKER_ARGS[@]}" "$IMAGE" \
  /opt/venv/bin/python scripts/evaluate_hamlet_so101_open_loop.py \
  --checkpoint "$CHECKPOINT" \
  --dataset-path "$DATASET_PATH" \
  --output-dir "$OUTPUT_DIR" \
  --eval-fraction "$EVAL_FRACTION" \
  --stride "$STRIDE" \
  --max-anchors-per-episode "$MAX_ANCHORS_PER_EPISODE" \
  --seed "$SEED"
