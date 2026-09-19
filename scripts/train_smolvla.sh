#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "$ROOT_DIR/scripts/common.sh"

require_value DATASET_REPO_ID
require_value DATASET_ROOT
require_value DATASET_EVAL_SPLIT
require_value SMOLVLA_BASE_PATH
require_value SMOLVLA_OUTPUT_DIR
require_value SMOLVLA_JOB_NAME
require_value POLICY_DEVICE

mkdir -p "$ROOT_DIR/$(dirname "$SMOLVLA_OUTPUT_DIR")"

args=(
  python
  -m
  lerobot.scripts.lerobot_train
  --policy.path="$SMOLVLA_BASE_PATH"
  --dataset.repo_id="$DATASET_REPO_ID"
  --dataset.root="$DATASET_ROOT"
  --dataset.eval_split="$DATASET_EVAL_SPLIT"
  --rename_map="$SMOLVLA_RENAME_MAP"
  --policy.empty_cameras="$SMOLVLA_EMPTY_CAMERAS"
  --policy.device="$POLICY_DEVICE"
  --policy.use_amp="$SMOLVLA_USE_AMP"
  --policy.chunk_size="$SMOLVLA_CHUNK_SIZE"
  --policy.n_action_steps="$SMOLVLA_N_ACTION_STEPS"
  --policy.cycle_enabled="$SMOLVLA_CYCLE_ENABLED"
  --policy.cycle_history_size="$SMOLVLA_CYCLE_HISTORY_SIZE"
  --policy.cycle_image_history_size="$SMOLVLA_CYCLE_IMAGE_HISTORY_SIZE"
  --policy.cycle_progress_loss_weight="$SMOLVLA_CYCLE_PROGRESS_LOSS_WEIGHT"
  --policy.freeze_vision_encoder="$SMOLVLA_FREEZE_VISION_ENCODER"
  --policy.train_expert_only="$SMOLVLA_TRAIN_EXPERT_ONLY"
  --policy.train_state_proj="$SMOLVLA_TRAIN_STATE_PROJ"
  --policy.push_to_hub="$SMOLVLA_PUSH_TO_HUB"
  --output_dir="$SMOLVLA_OUTPUT_DIR"
  --job_name="$SMOLVLA_JOB_NAME"
  --batch_size="$SMOLVLA_BATCH_SIZE"
  --steps="$SMOLVLA_STEPS"
  --save_freq="$SMOLVLA_SAVE_FREQ"
  --log_freq="$SMOLVLA_LOG_FREQ"
  --num_workers="$SMOLVLA_NUM_WORKERS"
  --eval_steps="${SMOLVLA_EVAL_STEPS:-0}"
  --max_eval_samples="${SMOLVLA_MAX_EVAL_SAMPLES:-0}"
  --wandb.enable="$WANDB_ENABLE"
  --wandb.project="$WANDB_PROJECT"
  --wandb.mode="$WANDB_MODE"
)

if [[ -n "${WANDB_ENTITY:-}" ]]; then
  args+=(--wandb.entity="$WANDB_ENTITY")
fi

if [[ -n "${SMOLVLA_POLICY_REPO_ID:-}" ]]; then
  args+=(--policy.repo_id="$SMOLVLA_POLICY_REPO_ID")
fi

if [[ "${LEROBOT_IMAGE:-}" == "so101-hamlet-thor:26.03" ]]; then
  printf -v train_command '%q ' "${args[@]}"
  run_lerobot bash -lc "if [[ ! -x /workspace/.venv/bin/python ]]; then python -m venv --system-site-packages /workspace/.venv; fi && /workspace/.venv/bin/python -m pip install --no-deps -q --upgrade 'huggingface-hub>=1.6,<2' 'datasets>=4.8,<5' 'transformers>=5.4,<5.6' 'peft>=0.18,<1' 'tokenizers>=0.22,<0.23.1' httpx httpcore h11 anyio sniffio certifi idna draccus num2words mergedeep docopt typing-inspect mypy-extensions toml && exec /workspace/.venv/bin/python -m lerobot.scripts.lerobot_train ${train_command#*lerobot.scripts.lerobot_train }"
else
  run_lerobot "${args[@]}"
fi
