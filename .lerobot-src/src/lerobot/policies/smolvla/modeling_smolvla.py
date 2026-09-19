#!/usr/bin/env python

# Copyright 2025 HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
SmolVLA:

[Paper](https://huggingface.co/papers/2506.01844)

Designed by Hugging Face.

Install smolvla extra dependencies:
```bash
pip install -e ".[smolvla]"
```

Example of finetuning the smolvla pretrained model (`smolvla_base`):
```bash
lerobot-train \
--policy.path=lerobot/smolvla_base \
--dataset.repo_id=<USER>/svla_so100_task1_v3 \
--batch_size=64 \
--steps=200000
```

Example of finetuning a smolVLA. SmolVLA is composed of a pretrained VLM,
and an action expert.
```bash
lerobot-train \
--policy.type=smolvla \
--dataset.repo_id=<USER>/svla_so100_task1_v3 \
--batch_size=64 \
--steps=200000
```

Example of using the smolvla pretrained model outside LeRobot training framework:
```python
policy = SmolVLAPolicy.from_pretrained("lerobot/smolvla_base")
```

"""

import math
from collections import deque
from typing import TypedDict, Unpack

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

from lerobot.utils.constants import ACTION, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS, OBS_STATE
from lerobot.utils.import_utils import require_package

from ..common.flow_matching import euler_integrate, sample_noise, sample_time_beta
from ..common.vla_utils import (
    create_sinusoidal_pos_embedding,
    make_att_2d_masks,
    pad_vector,
    resize_with_pad,
)
from ..pretrained import PreTrainedPolicy
from ..rtc.modeling_rtc import RTCProcessor
from ..utils import (
    populate_queues,
)
from .configuration_smolvla import SmolVLAConfig
from .cyclemanip import CycleHistoryEncoder, cycle_progress_targets
from .smolvlm_with_expert import SmolVLMWithExpertModel


class ActionSelectKwargs(TypedDict, total=False):
    inference_delay: int | None
    prev_chunk_left_over: Tensor | None
    execution_horizon: int | None


def normalize(x, min_val, max_val):
    return (x - min_val) / (max_val - min_val)


def unnormalize(x, min_val, max_val):
    return x * (max_val - min_val) + min_val


def safe_arcsin(value):
    # This ensures that the input stays within
    # [−1,1] to avoid invalid values for arcsin
    return torch.arcsin(torch.clamp(value, -1.0, 1.0))


def aloha_gripper_to_angular(value):
    # Aloha transforms the gripper positions into a linear space. The following code
    # reverses this transformation to be consistent with smolvla which is pretrained in
    # angular space.
    #
    # These values are coming from the Aloha code:
    # PUPPET_GRIPPER_POSITION_OPEN, PUPPET_GRIPPER_POSITION_CLOSED
    value = unnormalize(value, min_val=0.01844, max_val=0.05800)

    # This is the inverse of the angular to linear transformation inside the Interbotix code.
    def linear_to_radian(linear_position, arm_length, horn_radius):
        value = (horn_radius**2 + linear_position**2 - arm_length**2) / (2 * horn_radius * linear_position)
        return safe_arcsin(value)

    # The constants are taken from the Interbotix code.
    value = linear_to_radian(value, arm_length=0.036, horn_radius=0.022)

    # Normalize to [0, 1].
    # The values 0.4 and 1.5 were measured on an actual Trossen robot.
    return normalize(value, min_val=0.4, max_val=1.5)


def aloha_gripper_from_angular(value):
    # Convert from the gripper position used by smolvla to the gripper position that is used by Aloha.
    # Note that the units are still angular but the range is different.

    # The values 0.4 and 1.5 were measured on an actual Trossen robot.
    value = unnormalize(value, min_val=0.4, max_val=1.5)

    # These values are coming from the Aloha code:
    # PUPPET_GRIPPER_JOINT_OPEN, PUPPET_GRIPPER_JOINT_CLOSE
    return normalize(value, min_val=-0.6213, max_val=1.4910)


def aloha_gripper_from_angular_inv(value):
    # Directly inverts the gripper_from_angular function.
    value = unnormalize(value, min_val=-0.6213, max_val=1.4910)
    return normalize(value, min_val=0.4, max_val=1.5)


class SmolVLAPolicy(PreTrainedPolicy):
    """Wrapper class around VLAFlowMatching model to train and run inference within LeRobot."""

    config_class = SmolVLAConfig
    name = "smolvla"

    def supports_rtc(self) -> bool:
        return True

    def __init__(
        self,
        config: SmolVLAConfig,
        **kwargs,
    ):
        """
        Args:
            config: Policy configuration class instance or None, in which case the default instantiation of
                    the configuration class is used.
        """

        require_package("transformers", extra="smolvla")
        super().__init__(config)
        config.validate_features()
        self.config = config
        self.init_rtc_processor()
        self.model = VLAFlowMatching(config, rtc_processor=self.rtc_processor)
        self.reset()

    def reset(self):
        """This should be called whenever the environment is reset."""
        self._queues = {
            ACTION: deque(maxlen=self.config.n_action_steps),
        }
        self._cycle_state_history = deque(maxlen=self.config.cycle_history_size)
        image_indices = self.config.image_observation_delta_indices or [0]
        self._cycle_image_history = {
            key: deque(maxlen=1 - min(image_indices)) for key in self.config.image_features
        }

    def init_rtc_processor(self):
        """Initialize RTC processor if RTC is enabled in config."""
        self.rtc_processor = None

        # Lets create processor if the config provided
        # If RTC is not enabled - we still can track the denoising data
        if self.config.rtc_config is not None:
            self.rtc_processor = RTCProcessor(self.config.rtc_config)

            # In case of calling init_rtc_processor after the model is created
            # We need to set the rtc_processor to the model
            # During the normal initialization process the model is not created yet
            model_value = getattr(self, "model", None)
            if model_value is not None:
                model_value.rtc_processor = self.rtc_processor

    def get_optim_params(self) -> dict:
        return self.parameters()

    def _get_action_chunk(
        self, batch: dict[str, Tensor], noise: Tensor | None = None, **kwargs: Unpack[ActionSelectKwargs]
    ) -> Tensor:
        # TODO: Check if this for loop is needed.
        # Context: In fact, self.queues contains only ACTION field, and in inference, we don't have action in the batch
        # In the case of offline inference, we have the action in the batch
        # that why without the k != ACTION check, it will raise an error because we are trying to stack
        # on an empty container.
        for k in batch:
            if k in self._queues and k != ACTION:
                batch[k] = torch.stack(list(self._queues[k]), dim=1)

        batch = self._prepare_cycle_batch(batch)
        images, img_masks = self.prepare_images(batch)
        state = self.prepare_state(batch)
        cycle_state, cycle_state_is_pad = self.prepare_cycle_history(batch)
        lang_tokens = batch[f"{OBS_LANGUAGE_TOKENS}"]
        lang_masks = batch[f"{OBS_LANGUAGE_ATTENTION_MASK}"]

        actions = self.model.sample_actions(
            images,
            img_masks,
            lang_tokens,
            lang_masks,
            state,
            noise=noise,
            cycle_state=cycle_state,
            cycle_state_is_pad=cycle_state_is_pad,
            **kwargs,
        )

        # Unpad actions
        original_action_dim = self.config.action_feature.shape[0]
        actions = actions[:, :, :original_action_dim]

        if self.config.adapt_to_pi_aloha:
            actions = self._pi_aloha_encode_actions(actions)

        return actions

    def _prepare_batch(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        if self.config.adapt_to_pi_aloha:
            batch[OBS_STATE] = self._pi_aloha_decode_state(batch[OBS_STATE])

        return batch

    @torch.no_grad()
    def predict_action_chunk(
        self, batch: dict[str, Tensor], noise: Tensor | None = None, **kwargs: Unpack[ActionSelectKwargs]
    ) -> Tensor:
        self.eval()

        batch = self._prepare_batch(batch)
        self._queues = populate_queues(self._queues, batch, exclude_keys=[ACTION])

        actions = self._get_action_chunk(batch, noise, **kwargs)
        return actions

    @torch.no_grad()
    def predict_cycle_progress(self, batch: dict[str, Tensor]) -> Tensor | None:
        """Return progress logits for diagnostics when CycleManip is enabled."""
        if not self.config.cycle_enabled:
            return None
        self.eval()
        batch = self._prepare_batch(batch)
        batch = self._prepare_cycle_batch(batch)
        cycle_state, cycle_state_is_pad = self.prepare_cycle_history(batch)
        return self.model.predict_cycle_progress(cycle_state, cycle_state_is_pad)

    @torch.no_grad()
    def select_action(
        self, batch: dict[str, Tensor], noise: Tensor | None = None, **kwargs: Unpack[ActionSelectKwargs]
    ) -> Tensor:
        """Select a single action given environment observations.

        This method wraps `select_actions` in order to return one action at a time for execution in the
        environment. It works by managing the actions in a queue and only calling `select_actions` when the
        queue is empty.
        """

        assert not self._rtc_enabled(), (
            "RTC is not supported for select_action, use it with predict_action_chunk"
        )

        self.eval()
        batch = self._prepare_batch(batch)
        self._queues = populate_queues(self._queues, batch, exclude_keys=[ACTION])

        # Observe every control tick, including ticks served from the action queue.
        # _get_action_chunk receives temporal tensors, so it will not append twice.
        batch = self._prepare_cycle_batch(batch)
        if self._check_get_actions_condition():
            actions = self._get_action_chunk(batch, noise)

            # `self.predict_action_chunk` returns a (batch_size, n_action_steps, action_dim) tensor, but the queue
            # effectively has shape (n_action_steps, batch_size, *), hence the transpose.
            self._queues[ACTION].extend(actions.transpose(0, 1)[: self.config.n_action_steps])

        return self._queues[ACTION].popleft()

    def _check_get_actions_condition(self) -> bool:
        return len(self._queues[ACTION]) == 0

    def _rtc_enabled(self) -> bool:
        return self.config.rtc_config is not None and self.config.rtc_config.enabled

    def forward(
        self, batch: dict[str, Tensor], noise=None, time=None, reduction: str = "mean"
    ) -> dict[str, Tensor]:
        """Do a full training forward pass to compute the loss.

        Args:
            batch: Training batch containing observations and actions.
            noise: Optional noise tensor for flow matching.
            time: Optional time tensor for flow matching.
            reduction: How to reduce the loss. Options:
                - "mean": Return scalar mean loss (default, backward compatible)
                - "none": Return per-sample losses of shape (batch_size,) for RA-BC weighting
        """
        if self.config.adapt_to_pi_aloha:
            batch[OBS_STATE] = self._pi_aloha_decode_state(batch[OBS_STATE])
            batch[ACTION] = self._pi_aloha_encode_actions_inv(batch[ACTION])

        batch = self._prepare_cycle_batch(batch)
        images, img_masks = self.prepare_images(batch)
        state = self.prepare_state(batch)
        cycle_state, cycle_state_is_pad = self.prepare_cycle_history(batch)
        lang_tokens = batch[f"{OBS_LANGUAGE_TOKENS}"]
        lang_masks = batch[f"{OBS_LANGUAGE_ATTENTION_MASK}"]
        actions = self.prepare_action(batch)
        actions_is_pad = batch.get("action_is_pad")
        loss_dict = {}
        losses, progress_logits = self.model.forward(
            images,
            img_masks,
            lang_tokens,
            lang_masks,
            state,
            actions,
            noise,
            time,
            cycle_state=cycle_state,
            cycle_state_is_pad=cycle_state_is_pad,
            return_auxiliary=True,
        )
        original_action_dim = self.config.action_feature.shape[0]
        losses = losses[:, :, :original_action_dim]
        loss_dict["losses_after_forward"] = losses.clone().mean().item()

        if actions_is_pad is not None:
            in_episode_bound = ~actions_is_pad
            losses = losses * in_episode_bound.unsqueeze(-1)
            loss_dict["losses_after_in_ep_bound"] = losses.clone().mean().item()

        # Remove padding
        losses = losses[:, :, : self.config.max_action_dim]
        loss_dict["losses_after_rm_padding"] = losses.clone().mean().item()

        if actions_is_pad is None:
            per_sample_loss = losses.mean(dim=(1, 2))
        else:
            num_valid = ((~actions_is_pad).sum(dim=1) * losses.shape[-1]).clamp_min(1)
            per_sample_loss = losses.sum(dim=(1, 2)) / num_valid

        if progress_logits is not None:
            progress_targets = self._cycle_progress_targets(batch)
            progress_loss_per_sample = F.cross_entropy(progress_logits, progress_targets, reduction="none")
            per_sample_loss = per_sample_loss + self.config.cycle_progress_loss_weight * progress_loss_per_sample
            loss_dict["cycle_progress_loss"] = progress_loss_per_sample.mean().item()
            loss_dict["cycle_progress_accuracy"] = (
                (progress_logits.argmax(dim=-1) == progress_targets).float().mean().item()
            )

        if reduction == "none":
            # Return per-sample losses (B,) by averaging over valid (time, action) entries
            loss_dict["loss"] = per_sample_loss.mean().item()
            return per_sample_loss, loss_dict
        else:
            # Default: return scalar mean loss over valid (time, action) entries
            loss = per_sample_loss.mean()
            loss_dict["loss"] = loss.item()
            return loss, loss_dict

    def prepare_images(self, batch):
        """Apply SmolVLA preprocessing to the images, like resizing to 224x224 and padding to keep aspect ratio, and
        convert pixel range from [0.0, 1.0] to [-1.0, 1.0] as requested by SigLIP.
        """
        images = []
        img_masks = []
        present_img_keys = [key for key in self.config.image_features if key in batch]
        missing_img_keys = [key for key in self.config.image_features if key not in batch]

        if len(present_img_keys) == 0:
            raise ValueError(
                f"All image features are missing from the batch. At least one expected. (batch: {batch.keys()}) (image_features:{self.config.image_features})"
            )
        # Preprocess image features present in the batch
        for key in present_img_keys:
            img = batch[key]
            if self.config.cycle_enabled and img.ndim == 5:
                for history_index in range(img.shape[1]):
                    history_img = img[:, history_index]
                    if self.config.resize_imgs_with_padding is not None:
                        history_img = resize_with_pad(
                            history_img,
                            self.config.resize_imgs_with_padding[1],
                            self.config.resize_imgs_with_padding[0],
                            pad_value=0,
                        )
                    history_img = history_img * 2.0 - 1.0
                    images.append(history_img)
                    if f"{key}_is_pad" in batch:
                        img_masks.append(~batch[f"{key}_is_pad"][:, history_index].bool())
                    elif f"{key}_padding_mask" in batch:
                        img_masks.append(batch[f"{key}_padding_mask"][:, history_index].bool())
                    else:
                        img_masks.append(
                            torch.ones(history_img.shape[0], dtype=torch.bool, device=history_img.device)
                        )
                continue
            if self.config.resize_imgs_with_padding is not None:
                # SmolVLA stores the target as (width, height); the shared helper expects (height, width).
                img = resize_with_pad(
                    img,
                    self.config.resize_imgs_with_padding[1],
                    self.config.resize_imgs_with_padding[0],
                    pad_value=0,
                )

            # Normalize from range [0,1] to [-1,1] as expacted by siglip
            img = img * 2.0 - 1.0

            bsize = img.shape[0]
            device = img.device
            if f"{key}_padding_mask" in batch:
                mask = batch[f"{key}_padding_mask"].bool()
            else:
                mask = torch.ones(bsize, dtype=torch.bool, device=device)
            images.append(img)
            img_masks.append(mask)

        # Create image features not present in the batch
        # as fully 0 padded images.
        for num_empty_cameras in range(len(missing_img_keys)):
            if num_empty_cameras >= self.config.empty_cameras:
                break
            if self.config.cycle_enabled and img.ndim == 5:
                for _ in range(img.shape[1]):
                    empty_img = torch.ones_like(img[:, 0]) * -1
                    if self.config.resize_imgs_with_padding is not None:
                        empty_img = resize_with_pad(
                            empty_img,
                            self.config.resize_imgs_with_padding[1],
                            self.config.resize_imgs_with_padding[0],
                            pad_value=-1,
                        )
                    images.append(empty_img)
                    img_masks.append(torch.zeros(empty_img.shape[0], dtype=torch.bool, device=empty_img.device))
            else:
                img = torch.ones_like(img) * -1
                mask = torch.zeros_like(mask)
                images.append(img)
                img_masks.append(mask)
        return images, img_masks

    def _pi_aloha_decode_state(self, state):
        # Flip the joints.
        for motor_idx in [1, 2, 8, 9]:
            state[:, motor_idx] *= -1
        # Reverse the gripper transformation that is being applied by the Aloha runtime.
        for motor_idx in [6, 13]:
            state[:, motor_idx] = aloha_gripper_to_angular(state[:, motor_idx])
        return state

    def _pi_aloha_encode_actions(self, actions):
        # Flip the joints.
        for motor_idx in [1, 2, 8, 9]:
            actions[:, :, motor_idx] *= -1
        # Reverse the gripper transformation that is being applied by the Aloha runtime.
        for motor_idx in [6, 13]:
            actions[:, :, motor_idx] = aloha_gripper_from_angular(actions[:, :, motor_idx])
        return actions

    def _pi_aloha_encode_actions_inv(self, actions):
        # Flip the joints again.
        for motor_idx in [1, 2, 8, 9]:
            actions[:, :, motor_idx] *= -1
        # Reverse the gripper transformation that is being applied by the Aloha runtime.
        for motor_idx in [6, 13]:
            actions[:, :, motor_idx] = aloha_gripper_from_angular_inv(actions[:, :, motor_idx])
        return actions

    def prepare_state(self, batch):
        """Pad state"""
        state = batch[OBS_STATE][:, -1, :] if batch[OBS_STATE].ndim > 2 else batch[OBS_STATE]
        state = pad_vector(state, self.config.max_state_dim)
        return state

    def _prepare_cycle_batch(self, batch):
        """Build temporal windows for live inference when the caller supplies one frame."""
        if not self.config.cycle_enabled:
            return batch

        state = batch[OBS_STATE]
        state_is_pad = batch.get(f"{OBS_STATE}_is_pad")
        # The default processor only adds a batch dimension to single-frame
        # observations. Offline evaluation therefore reaches this method with
        # an unbatched temporal sample (T,D), while live inference uses (B,D).
        if state.ndim == 2 and state_is_pad is not None and state_is_pad.ndim == 1:
            batch[OBS_STATE] = state.unsqueeze(0)
            batch[f"{OBS_STATE}_is_pad"] = state_is_pad.unsqueeze(0)
            for key in self.config.image_features:
                if key in batch and batch[key].ndim == 4:
                    batch[key] = batch[key].unsqueeze(0)
                mask_key = f"{key}_is_pad"
                if mask_key in batch and batch[mask_key].ndim == 1:
                    batch[mask_key] = batch[mask_key].unsqueeze(0)
            return batch

        if state.ndim == 2:
            self._cycle_state_history.append(state.detach().clone())
            states = list(self._cycle_state_history)
            history = state.new_zeros(state.shape[0], self.config.cycle_history_size, state.shape[-1])
            history_is_pad = torch.ones(
                state.shape[0], self.config.cycle_history_size, dtype=torch.bool, device=state.device
            )
            history[:, -len(states) :] = torch.stack(states, dim=1)
            history_is_pad[:, -len(states) :] = False
            batch[OBS_STATE] = history
            batch[f"{OBS_STATE}_is_pad"] = history_is_pad

        for key in self.config.image_features:
            if key not in batch or batch[key].ndim != 4:
                continue
            image = batch[key]
            offsets = self.config.image_observation_delta_indices
            self._cycle_image_history.setdefault(key, deque(maxlen=1 - min(offsets))).append(
                image.detach().clone()
            )
            frames = list(self._cycle_image_history[key])
            history = image.new_zeros(
                image.shape[0],
                len(offsets),
                image.shape[1],
                image.shape[2],
                image.shape[3],
            )
            history_is_pad = torch.ones(
                image.shape[0], len(offsets), dtype=torch.bool, device=image.device
            )
            for slot, offset in enumerate(offsets):
                index = len(frames) - 1 + offset
                # Match the dataset's episode-start clamping and padding mask.
                history[:, slot] = frames[max(0, index)]
                history_is_pad[:, slot] = index < 0
            batch[key] = history
            batch[f"{key}_is_pad"] = history_is_pad
        return batch

    def prepare_cycle_history(self, batch):
        if not self.config.cycle_enabled:
            return None, None
        state_history = pad_vector(batch[OBS_STATE], self.config.max_state_dim)
        is_pad = batch.get(f"{OBS_STATE}_is_pad")
        if is_pad is None:
            is_pad = torch.zeros(state_history.shape[:2], dtype=torch.bool, device=state_history.device)
        return state_history, is_pad

    def _cycle_progress_targets(self, batch):
        frame_index = batch.get("frame_index")
        episode_index = batch.get("episode_index")
        if frame_index is None or episode_index is None:
            raise KeyError("CycleManip training requires frame_index and episode_index in the batch")
        return cycle_progress_targets(
            frame_index,
            episode_index,
            self.config.cycle_episode_lengths,
            self.config.cycle_progress_bins,
            self.config.cycle_progress_fallback_length,
        )

    def prepare_action(self, batch):
        """Pad action"""
        actions = pad_vector(batch[ACTION], self.config.max_action_dim)
        return actions

    def _get_default_peft_targets(self) -> dict[str, any]:
        """Return default PEFT target modules for SmolVLA fine-tuning."""
        common_projections = (
            "state_proj|action_in_proj|action_out_proj|action_time_mlp_in|action_time_mlp_out"
        )
        target_modules = rf"(model\.vlm_with_expert\.lm_expert\..*\.(q|v)_proj|model\.({common_projections}))"
        return {
            "target_modules": target_modules,
            "modules_to_save": [],
        }

    def _validate_peft_config(self, peft_config) -> None:
        """Validate PEFT configuration for SmolVLA."""
        super()._validate_peft_config(peft_config)
        if not self.config.load_vlm_weights:
            import logging

            logging.warning(
                "Training SmolVLA from scratch using PEFT. This is unlikely to yield good results. "
                "Set `load_vlm_weights=True` to fine-tune the existing policy."
            )


def pad_tensor(tensor, max_len, pad_value=0):
    """
    Efficiently pads a tensor along sequence dimension to match max_len.

    Args:
        tensor (torch.Tensor): Shape (B, L, ...) or (B, L).
        max_len (int): Fixed sequence length.
        pad_value (int/float): Value for padding.

    Returns:
        torch.Tensor: Shape (B, max_len, ...) or (B, max_len).
    """
    b, d = tensor.shape[:2]

    # Create a padded tensor of max_len and copy the existing values
    padded_tensor = torch.full(
        (b, max_len, *tensor.shape[2:]), pad_value, dtype=tensor.dtype, device=tensor.device
    )
    padded_tensor[:, :d] = tensor  # Efficient in-place copy

    return padded_tensor


class VLAFlowMatching(nn.Module):
    """
    SmolVLA

    [Paper]()

    Designed by Hugging Face.
    ┌──────────────────────────────┐
    │                 actions      │
    │                    ▲         │
    │ ┌─────────┐      ┌─|────┐    │
    │ |         │────► │      │    │
    │ |         │ kv   │      │    │
    │ |         │────► │Action│    │
    │ |   VLM   │cache │Expert│    |
    │ │         │────► |      │    │
    │ │         │      │      │    │
    │ └▲──▲───▲─┘      └───▲──┘    |
    │  │  |   |            │       |
    │  |  |   |          noise     │
    │  │  │ state                  │
    │  │ language tokens           │
    │  image(s)                    │
    └──────────────────────────────┘
    """

    def __init__(self, config: SmolVLAConfig, rtc_processor: RTCProcessor | None = None):
        super().__init__()
        self.config = config

        self.vlm_with_expert = SmolVLMWithExpertModel(
            model_id=self.config.vlm_model_name,
            freeze_vision_encoder=self.config.freeze_vision_encoder,
            train_expert_only=self.config.train_expert_only,
            load_vlm_weights=self.config.load_vlm_weights,
            attention_mode=self.config.attention_mode,
            num_expert_layers=self.config.num_expert_layers,
            num_vlm_layers=self.config.num_vlm_layers,
            self_attn_every_n_layers=self.config.self_attn_every_n_layers,
            expert_width_multiplier=self.config.expert_width_multiplier,
            device=self.config.device if self.config.device is not None else "auto",
        )
        self.state_proj = nn.Linear(
            self.config.max_state_dim, self.vlm_with_expert.config.text_config.hidden_size
        )
        self.action_in_proj = nn.Linear(self.config.max_action_dim, self.vlm_with_expert.expert_hidden_size)
        self.action_out_proj = nn.Linear(self.vlm_with_expert.expert_hidden_size, self.config.max_action_dim)

        self.action_time_mlp_in = nn.Linear(
            self.vlm_with_expert.expert_hidden_size * 2, self.vlm_with_expert.expert_hidden_size
        )
        self.action_time_mlp_out = nn.Linear(
            self.vlm_with_expert.expert_hidden_size, self.vlm_with_expert.expert_hidden_size
        )

        self.cycle_history_encoder = None
        self.cycle_progress_head = None
        if self.config.cycle_enabled:
            text_hidden_size = self.vlm_with_expert.config.text_config.hidden_size
            self.cycle_history_encoder = CycleHistoryEncoder(
                state_dim=self.config.max_state_dim,
                hidden_dim=self.config.cycle_history_hidden_dim,
                output_dim=text_hidden_size,
                max_history=self.config.cycle_history_size,
            )
            self.cycle_progress_head = nn.Linear(text_hidden_size, self.config.cycle_progress_bins)

        self.set_requires_grad()
        self.fake_image_token = self.vlm_with_expert.processor.tokenizer.fake_image_token_id
        self.global_image_token = self.vlm_with_expert.processor.tokenizer.global_image_token_id
        self.global_image_start_token = torch.tensor(
            [self.fake_image_token, self.global_image_token], dtype=torch.long
        )

        self.add_image_special_tokens = self.config.add_image_special_tokens
        self.image_end_token = torch.tensor([self.fake_image_token], dtype=torch.long)
        self.prefix_length = self.config.prefix_length
        self.rtc_processor = rtc_processor

        # Compile model if requested
        if config.compile_model:
            torch.set_float32_matmul_precision("high")
            self.sample_actions = torch.compile(self.sample_actions, mode=config.compile_mode)
            self.forward = torch.compile(self.forward, mode=config.compile_mode)

    def _rtc_enabled(self):
        return self.config.rtc_config is not None and self.config.rtc_config.enabled

    def set_requires_grad(self):
        for params in self.state_proj.parameters():
            params.requires_grad = self.config.train_state_proj

    def sample_noise(self, shape, device):
        return sample_noise(shape, device)

    def sample_time(self, bsize, device):
        return sample_time_beta(bsize, device, alpha=1.5, beta=1.0, scale=0.999, offset=0.001)

    def embed_prefix(
        self,
        images,
        img_masks,
        lang_tokens,
        lang_masks,
        state: torch.Tensor = None,
        cycle_state: torch.Tensor | None = None,
        cycle_state_is_pad: torch.Tensor | None = None,
        return_cycle_embedding: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor, Tensor | None]:
        """Embed images with SigLIP and language tokens with embedding layer to prepare
        for SmolVLM transformer processing.
        """
        embs = []
        pad_masks = []
        att_masks = []
        for _img_idx, (
            img,
            img_mask,
        ) in enumerate(zip(images, img_masks, strict=False)):
            if self.add_image_special_tokens:
                image_start_token = (
                    self.vlm_with_expert.embed_language_tokens(
                        self.global_image_start_token.to(device=self.vlm_with_expert.vlm.device)
                    )
                    .unsqueeze(0)
                    .expand(img.shape[0], -1, -1)
                )
                image_start_mask = torch.ones_like(
                    image_start_token[:, :, 0], dtype=torch.bool, device=image_start_token.device
                )
                att_masks += [0] * (image_start_mask.shape[-1])
                embs.append(image_start_token)
                pad_masks.append(image_start_mask)

            img_emb = self.vlm_with_expert.embed_image(img)
            img_emb = img_emb

            # Normalize image embeddings
            img_emb_dim = img_emb.shape[-1]
            img_emb = img_emb * torch.tensor(img_emb_dim**0.5, dtype=img_emb.dtype, device=img_emb.device)

            bsize, num_img_embs = img_emb.shape[:2]
            img_mask = img_mask[:, None].expand(bsize, num_img_embs)

            embs.append(img_emb)
            pad_masks.append(img_mask)

            att_masks += [0] * (num_img_embs)
            if self.add_image_special_tokens:
                image_end_token = (
                    self.vlm_with_expert.embed_language_tokens(
                        self.image_end_token.to(device=self.vlm_with_expert.vlm.device)
                    )
                    .unsqueeze(0)
                    .expand(img.shape[0], -1, -1)
                )
                image_end_mask = torch.ones_like(
                    image_end_token[:, :, 0], dtype=torch.bool, device=image_end_token.device
                )
                embs.append(image_end_token)
                pad_masks.append(image_end_mask)
                att_masks += [0] * (image_end_mask.shape[1])
        lang_emb = self.vlm_with_expert.embed_language_tokens(lang_tokens)
        # Normalize language embeddings
        lang_emb_dim = lang_emb.shape[-1]
        lang_emb = lang_emb * math.sqrt(lang_emb_dim)

        embs.append(lang_emb)
        pad_masks.append(lang_masks)

        num_lang_embs = lang_emb.shape[1]
        att_masks += [0] * num_lang_embs

        state_emb = self.state_proj(state)
        state_emb = state_emb[:, None, :] if state_emb.ndim == 2 else state_emb
        embs.append(state_emb)
        bsize = state_emb.shape[0]
        device = state_emb.device

        states_seq_len = state_emb.shape[1]
        state_mask = torch.ones(bsize, states_seq_len, dtype=torch.bool, device=device)
        pad_masks.append(state_mask)

        # Set attention masks so that image and language inputs do not attend to state or actions
        att_masks += [1] * (states_seq_len)
        cycle_embedding = None
        if self.config.cycle_enabled:
            if self.cycle_history_encoder is None or cycle_state is None:
                raise ValueError("CycleManip is enabled but cycle state history is missing")
            cycle_embedding = self.cycle_history_encoder(cycle_state, cycle_state_is_pad)
            embs.append(cycle_embedding[:, None, :])
            cycle_mask = torch.ones(bsize, 1, dtype=torch.bool, device=device)
            pad_masks.append(cycle_mask)
            att_masks += [1]
        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=torch.bool, device=pad_masks.device)
        att_masks = att_masks[None, :]

        seq_len = pad_masks.shape[1]
        if seq_len < self.prefix_length:
            embs = pad_tensor(embs, self.prefix_length, pad_value=0)
            pad_masks = pad_tensor(pad_masks, self.prefix_length, pad_value=0)
            att_masks = pad_tensor(att_masks, self.prefix_length, pad_value=0)

        att_masks = att_masks.expand(bsize, -1)

        if return_cycle_embedding:
            return embs, pad_masks, att_masks, cycle_embedding
        return embs, pad_masks, att_masks

    def embed_suffix(self, noisy_actions, timestep):
        """Embed state, noisy_actions, timestep to prepare for Expert Gemma processing."""
        embs = []
        pad_masks = []
        att_masks = []

        # Fuse timestep + action information using an MLP
        action_emb = self.action_in_proj(noisy_actions)
        device = action_emb.device
        bsize = action_emb.shape[0]
        dtype = action_emb.dtype
        # Embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
        time_emb = create_sinusoidal_pos_embedding(
            timestep,
            self.vlm_with_expert.expert_hidden_size,
            self.config.min_period,
            self.config.max_period,
            device=device,
        )
        time_emb = time_emb.type(dtype=dtype)

        time_emb = time_emb[:, None, :].expand_as(action_emb)
        action_time_emb = torch.cat([action_emb, time_emb], dim=2)

        action_time_emb = self.action_time_mlp_in(action_time_emb)
        action_time_emb = F.silu(action_time_emb)  # swish == silu
        action_time_emb = self.action_time_mlp_out(action_time_emb)

        # Add to input tokens
        embs.append(action_time_emb)

        bsize, action_time_dim = action_time_emb.shape[:2]
        action_time_mask = torch.ones(bsize, action_time_dim, dtype=torch.bool, device=device)
        pad_masks.append(action_time_mask)

        # Set attention masks so that image, language and state inputs do not attend to action tokens
        att_masks += [1] * self.config.chunk_size
        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=embs.dtype, device=embs.device)
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))
        return embs, pad_masks, att_masks

    def forward(
        self,
        images,
        img_masks,
        lang_tokens,
        lang_masks,
        state,
        actions,
        noise=None,
        time=None,
        cycle_state=None,
        cycle_state_is_pad=None,
        return_auxiliary: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor | None]:
        """Do a full training forward pass and compute the loss (batch_size x num_steps x num_motors)"""
        if noise is None:
            noise = self.sample_noise(actions.shape, actions.device)

        if time is None:
            time = self.sample_time(actions.shape[0], actions.device)

        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions
        prefix_embs, prefix_pad_masks, prefix_att_masks, cycle_embedding = self.embed_prefix(
            images,
            img_masks,
            lang_tokens,
            lang_masks,
            state=state,
            cycle_state=cycle_state,
            cycle_state_is_pad=cycle_state_is_pad,
            return_cycle_embedding=True,
        )
        suffix_embs, suffix_pad_masks, suffix_att_masks = self.embed_suffix(x_t, time)

        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)

        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1
        (_, suffix_out), _ = self.vlm_with_expert.forward(
            attention_mask=att_2d_masks,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, suffix_embs],
            use_cache=False,
        )
        suffix_out = suffix_out[:, -self.config.chunk_size :]
        # Original openpi code, upcast attention output
        suffix_out = suffix_out.to(dtype=torch.float32)
        v_t = self.action_out_proj(suffix_out)
        losses = F.mse_loss(u_t, v_t, reduction="none")
        progress_logits = self.cycle_progress_head(cycle_embedding) if cycle_embedding is not None else None
        if return_auxiliary:
            return losses, progress_logits
        return losses

    def sample_actions(
        self,
        images,
        img_masks,
        lang_tokens,
        lang_masks,
        state,
        noise=None,
        cycle_state=None,
        cycle_state_is_pad=None,
        **kwargs: Unpack[ActionSelectKwargs],
    ) -> Tensor:
        """Do a full inference forward and compute the action (batch_size x num_steps x num_motors)"""
        bsize = state.shape[0]
        device = state.device

        if noise is None:
            actions_shape = (bsize, self.config.chunk_size, self.config.max_action_dim)
            noise = self.sample_noise(actions_shape, device)

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images,
            img_masks,
            lang_tokens,
            lang_masks,
            state=state,
            cycle_state=cycle_state,
            cycle_state_is_pad=cycle_state_is_pad,
        )
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        # Compute image and language key value cache
        _, past_key_values = self.vlm_with_expert.forward(
            attention_mask=prefix_att_2d_masks,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=self.config.use_cache,
        )
        num_steps = self.config.num_steps

        return euler_integrate(
            lambda input_x_t, current_timestep: self.denoise_step(
                x_t=input_x_t,
                prefix_pad_masks=prefix_pad_masks,
                past_key_values=past_key_values,
                timestep=current_timestep,
            ),
            noise,
            num_steps,
            rtc_processor=self.rtc_processor,
            rtc_enabled=self._rtc_enabled(),
            inference_delay=kwargs.get("inference_delay"),
            prev_chunk_left_over=kwargs.get("prev_chunk_left_over"),
            execution_horizon=kwargs.get("execution_horizon"),
        )

    @torch.no_grad()
    def predict_cycle_progress(self, cycle_state, cycle_state_is_pad) -> Tensor | None:
        if self.cycle_history_encoder is None or self.cycle_progress_head is None:
            return None
        cycle_embedding = self.cycle_history_encoder(cycle_state, cycle_state_is_pad)
        return self.cycle_progress_head(cycle_embedding)

    def denoise_step(
        self,
        prefix_pad_masks,
        past_key_values,
        x_t,
        timestep,
    ):
        """Apply one denoising step of the noise `x_t` at a given timestep."""
        suffix_embs, suffix_pad_masks, suffix_att_masks = self.embed_suffix(x_t, timestep)

        suffix_len = suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]
        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)

        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)

        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)
        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1

        outputs_embeds, _ = self.vlm_with_expert.forward(
            attention_mask=full_att_2d_masks,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, suffix_embs],
            use_cache=self.config.use_cache,
        )
        if past_key_values is not None:
            # Self-attention layers append suffix K/V in place; restore the prefix for the next step.
            past_key_values.crop(prefix_len)
        suffix_out = outputs_embeds[1]
        suffix_out = suffix_out[:, -self.config.chunk_size :]
        suffix_out = suffix_out.to(dtype=torch.float32)
        v_t = self.action_out_proj(suffix_out)
        return v_t
