#!/usr/bin/env bash
# Train GR00T N1.6 + HAMLET on the local SO-101 shake-cup dataset.
#
# The raw LeRobot v3 dataset is converted to the v2-like layout expected by
# HAMLET in a separate output tree. The raw source remains untouched.
#
# Usage:
#   bash scripts/train_hamlet_so101.sh
#   NUM_GPUS=1 MAX_STEPS=1000 bash scripts/train_hamlet_so101.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HAMLET_ROOT="$REPO_ROOT/HAMLET-Isaac-GR00T"
RAW_DATASET="${RAW_DATASET:-$REPO_ROOT/data/so101_shake_cup}"
DATASET_PATH="${DATASET_PATH:-$REPO_ROOT/data/so101_shake_cup_gr00t}"
MODALITY_CONFIG="$REPO_ROOT/scripts/hamlet_so101_modality.py"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/outputs/train/hamlet_so101_shake_cup}"
BASE_MODEL="${BASE_MODEL:-nvidia/GR00T-N1.6-3B}"
NUM_GPUS="${NUM_GPUS:-1}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"
MAX_STEPS="${MAX_STEPS:-10000}"
SAVE_STEPS="${SAVE_STEPS:-1000}"
K="${K:-4}"
MEMORY_STRIDE="${MEMORY_STRIDE:-16}"
N_MOMENT_TOKENS="${N_MOMENT_TOKENS:-4}"
MEM_COND_TYPE="${MEM_COND_TYPE:-cross_attn}"
MEMORY_TYPE="${MEMORY_TYPE:-moment_token}"
USE_FLASH_ATTENTION="${USE_FLASH_ATTENTION:-false}"
MASTER_PORT="${MASTER_PORT:-$((20000 + RANDOM % 10000))}"

FLASH_ARGS=(--no-use-flash-attention)
if [[ "$USE_FLASH_ATTENTION" == "true" ]]; then
  FLASH_ARGS=(--use-flash-attention)
fi

if [[ ! -d "$HAMLET_ROOT" ]]; then
  echo "ERROR: HAMLET submodule not found: $HAMLET_ROOT" >&2
  exit 1
fi
if [[ ! -d "$RAW_DATASET" ]]; then
  echo "ERROR: raw dataset not found: $RAW_DATASET" >&2
  exit 1
fi
if [[ ! -f "$MODALITY_CONFIG" ]]; then
  echo "ERROR: modality config not found: $MODALITY_CONFIG" >&2
  exit 1
fi

# The local config is git-ignored and is the repository's supported place for
# the Hugging Face read token used to fetch the GR00T base model.
if [[ -f "$REPO_ROOT/config/dataset.local.env" ]]; then
  # shellcheck disable=SC1091
  source "$REPO_ROOT/config/dataset.local.env"
fi
if [[ -n "${HF_TOKEN:-}" ]]; then
  export HF_TOKEN
fi

WANDB_ARGS=()
if [[ "${WANDB_ENABLE:-false}" == "true" ]]; then
  WANDB_ARGS=(--use-wandb)
  if [[ -n "${WANDB_PROJECT:-}" ]]; then
    WANDB_ARGS+=(--wandb-project "$WANDB_PROJECT")
  fi
fi

PYTHON="${PYTHON:-$HAMLET_ROOT/.venv/bin/python}"
if [[ ! -x "$PYTHON" ]]; then
  cat >&2 <<EOF
ERROR: HAMLET Python environment not found: $PYTHON
Install it first:
  cd "$HAMLET_ROOT"
  ~/.local/bin/uv sync
  ~/.local/bin/uv pip install -e .
Then rerun this script.
EOF
  exit 1
fi

"$PYTHON" "$REPO_ROOT/scripts/prepare_hamlet_so101.py" \
  --source "$RAW_DATASET" \
  --output "$DATASET_PATH"

pushd "$HAMLET_ROOT" >/dev/null
"$PYTHON" -m gr00t.data.stats \
  --dataset-path "$DATASET_PATH" \
  --embodiment-tag NEW_EMBODIMENT \
  --modality-config-path "$MODALITY_CONFIG"
popd >/dev/null

cd "$HAMLET_ROOT"
exec "$PYTHON" -m torch.distributed.run \
  --nproc_per_node="$NUM_GPUS" \
  --master_port="$MASTER_PORT" \
  --module gr00t.experiment.launch_finetune \
  --base-model-path "$BASE_MODEL" \
  --dataset-path "$DATASET_PATH" \
  --embodiment-tag NEW_EMBODIMENT \
  --modality-config-path "$MODALITY_CONFIG" \
  --num-gpus "$NUM_GPUS" \
  --output-dir "$OUTPUT_DIR" \
  --max-steps "$MAX_STEPS" \
  --global-batch-size "$GLOBAL_BATCH_SIZE" \
  --gradient-accumulation-steps "$GRAD_ACCUM" \
  --save-steps "$SAVE_STEPS" \
  --hamlet-mode finetune \
  --n-moment-tokens "$N_MOMENT_TOKENS" \
  --memory-window "$K" \
  --memory-stride "$MEMORY_STRIDE" \
  --memory-num-layers 2 \
  --mem-cond-type "$MEM_COND_TYPE" \
  --memory-type "$MEMORY_TYPE" \
  "${FLASH_ARGS[@]}" \
  "${WANDB_ARGS[@]}" \
  --no-freeze-moment-tokens
