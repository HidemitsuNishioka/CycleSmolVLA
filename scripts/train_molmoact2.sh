#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "$ROOT_DIR/scripts/common.sh"

require_value DATASET_REPO_ID
require_value DATASET_ROOT
require_value DATASET_EVAL_SPLIT
require_value MOLMOACT2_CHECKPOINT_PATH
require_value MOLMOACT2_NORM_TAG
require_value MOLMOACT2_OUTPUT_DIR
require_value MOLMOACT2_JOB_NAME
require_value POLICY_DEVICE

mkdir -p "$ROOT_DIR/$(dirname "$MOLMOACT2_OUTPUT_DIR")"

# Official MolmoAct2 augmentation recipe from the LeRobot documentation.
IMAGE_TRANSFORMS='{"crop":{"weight":1.0,"type":"RandomResizedCrop","kwargs":{"size":[224,224],"scale":[0.9025,0.9025],"ratio":[1.0,1.0]}},"rotation":{"weight":1.0,"type":"RandomRotation","kwargs":{"degrees":[-5.0,5.0],"interpolation":2}},"color":{"weight":1.0,"type":"ColorJitter","kwargs":{"brightness":0.2,"contrast":[0.8,1.2],"saturation":[0.8,1.2],"hue":0.05}}}'

args=(
  lerobot-train
  --dataset.repo_id="$DATASET_REPO_ID"
  --dataset.root="$DATASET_ROOT"
  --dataset.eval_split="$DATASET_EVAL_SPLIT"
  --dataset.image_transforms.enable=true
  --dataset.image_transforms.max_num_transforms=3
  --dataset.image_transforms.random_order=false
  --dataset.image_transforms.tfs="$IMAGE_TRANSFORMS"
  --policy.type=molmoact2
  --policy.checkpoint_path="$MOLMOACT2_CHECKPOINT_PATH"
  --policy.norm_tag="$MOLMOACT2_NORM_TAG"
  --policy.device="$POLICY_DEVICE"
  --policy.action_mode="$MOLMOACT2_ACTION_MODE"
  --policy.inference_action_mode="$MOLMOACT2_INFERENCE_ACTION_MODE"
  --policy.train_mode_vlm="$MOLMOACT2_TRAIN_MODE_VLM"
  --policy.dtype="$MOLMOACT2_DTYPE"
  --policy.gradient_checkpointing="$MOLMOACT2_GRADIENT_CHECKPOINTING"
  --policy.compile_model="$MOLMOACT2_COMPILE_MODEL"
  --policy.chunk_size="$MOLMOACT2_CHUNK_SIZE"
  --policy.n_action_steps="$MOLMOACT2_N_ACTION_STEPS"
  --policy.num_flow_timesteps="$MOLMOACT2_NUM_FLOW_TIMESTEPS"
  --policy.image_keys="$MOLMOACT2_IMAGE_KEYS"
  --policy.setup_type="$MOLMOACT2_SETUP_TYPE"
  --policy.control_mode="$MOLMOACT2_CONTROL_MODE"
  --policy.joint_signs="$MOLMOACT2_JOINT_SIGNS"
  --policy.joint_offsets="$MOLMOACT2_JOINT_OFFSETS"
  --policy.push_to_hub="$MOLMOACT2_PUSH_TO_HUB"
  --output_dir="$MOLMOACT2_OUTPUT_DIR"
  --job_name="$MOLMOACT2_JOB_NAME"
  --batch_size="$MOLMOACT2_BATCH_SIZE"
  --steps="$MOLMOACT2_STEPS"
  --save_freq="$MOLMOACT2_SAVE_FREQ"
  --log_freq="$MOLMOACT2_LOG_FREQ"
  --num_workers="$MOLMOACT2_NUM_WORKERS"
  --env_eval_freq=-1
  --wandb.enable="$WANDB_ENABLE"
  --wandb.project="$WANDB_PROJECT"
  --wandb.mode="$WANDB_MODE"
)

if [[ -n "${WANDB_ENTITY:-}" ]]; then
  args+=(--wandb.entity="$WANDB_ENTITY")
fi

if [[ -n "${MOLMOACT2_POLICY_REPO_ID:-}" ]]; then
  args+=(--policy.repo_id="$MOLMOACT2_POLICY_REPO_ID")
fi

run_lerobot "${args[@]}"
