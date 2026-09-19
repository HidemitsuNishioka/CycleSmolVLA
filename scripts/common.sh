#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COMPOSE_FILE="$ROOT_DIR/docker-compose.yml"

if [[ ! -f "$COMPOSE_FILE" ]]; then
  echo "docker-compose.yml not found: $COMPOSE_FILE" >&2
  exit 1
fi

if [[ ! -f "$ROOT_DIR/config/robot.env" || ! -f "$ROOT_DIR/config/dataset.env" ]]; then
  echo "Missing config/robot.env or config/dataset.env" >&2
  exit 1
fi

# shellcheck disable=SC1091
source "$ROOT_DIR/config/robot.env"
# shellcheck disable=SC1091
source "$ROOT_DIR/config/dataset.env"
if [[ -f "$ROOT_DIR/config/dataset.local.env" ]]; then
  # shellcheck disable=SC1091
  source "$ROOT_DIR/config/dataset.local.env"
fi

# Optional experiment settings override defaults without copying credentials.
if [[ -n "${SO101_DATASET_CONFIG:-}" ]]; then
  source "$SO101_DATASET_CONFIG"
fi

# Pass W&B authentication to the container without putting the secret in a
# command-line argument. Keep the value out of logs and printed diagnostics.
if [[ -n "${WANDB_API_KEY:-}" ]]; then
  export WANDB_API_KEY
fi

# Pass Hugging Face authentication to the container without putting the token
# itself in a command-line argument. The token is normally kept in the
# git-ignored config/dataset.local.env; a mounted Hugging Face login cache also
# remains supported when HF_TOKEN is empty.
if [[ -n "${HF_TOKEN:-}" ]]; then
  export HF_TOKEN
fi

# Avoid an interactive W&B login failure when a local smoke run has no key.
# If an API key is supplied (normally through config/dataset.local.env), the
# requested online logging remains enabled.
if [[ "${WANDB_ENABLE:-false}" == "true" && "${WANDB_MODE:-online}" == "online" && -z "${WANDB_API_KEY:-}" ]]; then
  echo "WARNING: WANDB_ENABLE=true but WANDB_API_KEY is empty; disabling online W&B logging for this run." >&2
  WANDB_ENABLE="false"
fi

export ROOT_DIR COMPOSE_FILE
export SO101_HOST_UID="$(id -u)"
export SO101_HOST_GID="$(id -g)"
export DIALOUT_GID="$(getent group dialout 2>/dev/null | cut -d: -f3 || true)"
export VIDEO_GID="$(getent group video 2>/dev/null | cut -d: -f3 || true)"

die() {
  echo "ERROR: $*" >&2
  exit 1
}

require_value() {
  local name="$1"
  local value="${!name:-}"
  [[ -n "$value" ]] || die "$name is empty. Edit config/robot.env or config/dataset.env."
}

require_file_or_dir() {
  local name="$1"
  local value="${!name:-}"
  [[ -e "$ROOT_DIR/$value" || -e "$value" ]] || die "$name does not exist: $value"
}

require_docker() {
  command -v docker >/dev/null 2>&1 || die "docker is not installed"
  docker compose version >/dev/null 2>&1 || die "docker compose is not available"
}

camera_config() {
  local config="{"
  local has_camera=false

  if [[ -n "${CAMERA_TOP_DEVICE:-}" ]]; then
    config+=" top: {type: opencv, index_or_path: ${CAMERA_TOP_DEVICE}, width: ${CAMERA_WIDTH}, height: ${CAMERA_HEIGHT}, fps: ${CAMERA_FPS} }"
    has_camera=true
  fi

  if [[ -n "${CAMERA_WRIST_DEVICE:-}" ]]; then
    if [[ "$has_camera" == true ]]; then
      config+=","
    fi
    config+=" wrist: {type: opencv, index_or_path: ${CAMERA_WRIST_DEVICE}, width: ${CAMERA_WIDTH}, height: ${CAMERA_HEIGHT}, fps: ${CAMERA_FPS} }"
    has_camera=true
  fi

  [[ "$has_camera" == true ]] || die "Set CAMERA_TOP_DEVICE or CAMERA_WRIST_DEVICE in config/robot.env."
  config+=" }"
  printf '%s' "$config"
}

run_lerobot() {
  require_docker
  local passthrough_env=()
  if [[ -n "${WANDB_API_KEY:-}" ]]; then
    passthrough_env+=(--env WANDB_API_KEY)
  fi
  if [[ -n "${HF_TOKEN:-}" ]]; then
    passthrough_env+=(--env HF_TOKEN)
  fi
  docker compose -f "$COMPOSE_FILE" run --rm --service-ports \
    "${passthrough_env[@]}" \
    --user "$SO101_HOST_UID:$SO101_HOST_GID" lerobot "$@"
}

run_shell() {
  require_docker
  docker compose -f "$COMPOSE_FILE" run --rm --service-ports \
    --user "$SO101_HOST_UID:$SO101_HOST_GID" lerobot bash
}
