#!/usr/bin/env bash
# Build/run HAMLET in NVIDIA's Thor-compatible PyTorch container.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="${HAMLET_DOCKER_IMAGE:-so101-hamlet-thor:26.03}"
HF_CACHE="${HF_CACHE:-${XDG_CACHE_HOME:-/home/kaina/.cache}/huggingface}"
HOST_UID="$(id -u)"
HOST_GID="$(id -g)"

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
)

if [[ -f "$REPO_ROOT/config/dataset.local.env" ]]; then
  DOCKER_ARGS+=(--env-file "$REPO_ROOT/config/dataset.local.env")
fi

MAX_STEPS="${MAX_STEPS:-1000}"
NUM_GPUS="${NUM_GPUS:-1}"
DOCKER_ARGS+=(--env "MAX_STEPS=$MAX_STEPS" --env "NUM_GPUS=$NUM_GPUS")

exec docker run "${DOCKER_ARGS[@]}" "$IMAGE" bash -lc \
  'PYTHON="/opt/venv/bin/python"; export PYTHON; bash scripts/train_hamlet_so101.sh'
