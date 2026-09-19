from types import SimpleNamespace

import torch

from lerobot.policies.smolvla.cyclemanip import (
    CycleHistoryEncoder,
    cycle_image_delta_indices,
    cycle_progress_targets,
    cycle_state_delta_indices,
)


def test_cycle_sampling_has_dense_recent_and_sparse_old_history():
    assert cycle_image_delta_indices(32, 6) == [-31, -16, -8, -2, -1, 0]
    assert cycle_image_delta_indices(300, 6) == [-299, -150, -75, -2, -1, 0]
    assert cycle_state_delta_indices(4) == [-3, -2, -1, 0]


def test_history_architecture_matches_kainajetson_checkpoint():
    # These choices do not change weight shapes: strict weight loading alone
    # cannot detect accidentally running the checkpoint with another architecture.
    encoder = CycleHistoryEncoder(state_dim=32, hidden_dim=256, output_dim=960, max_history=300)
    layer = encoder.encoder.layers[0]
    assert layer.self_attn.num_heads == 4
    assert layer.norm_first
    assert layer.dropout.p == 0
    assert isinstance(encoder.fusion[2], torch.nn.SiLU)


def test_cycle_progress_targets_use_episode_lengths():
    targets = cycle_progress_targets(
        torch.tensor([0, 224, 449]),
        torch.tensor([3, 3, 3]),
        {"3": 450},
    )
    assert targets.tolist() == [0, 4, 9]


def test_cycle_history_encoder_respects_padding_and_backpropagates():
    encoder = CycleHistoryEncoder(state_dim=6, hidden_dim=16, output_dim=12, max_history=4)
    history = torch.randn(2, 4, 6, requires_grad=True)
    is_pad = torch.tensor([[True, True, False, False], [False, False, False, False]])
    output = encoder(history, is_pad)
    assert output.shape == (2, 12)
    output.square().mean().backward()
    assert history.grad is not None
    assert torch.isfinite(history.grad).all()


def test_cycle_history_encoder_ignores_left_padding_values():
    torch.manual_seed(0)
    encoder = CycleHistoryEncoder(state_dim=6, hidden_dim=16, output_dim=12, max_history=4).eval()
    history = torch.randn(2, 4, 6, requires_grad=True)
    is_pad = torch.tensor([[True, True, False, False], [True, False, False, False]])
    changed = history.detach().clone()
    changed[is_pad] += 100
    output = encoder(history, is_pad)
    torch.testing.assert_close(output, encoder(changed, is_pad))
    output.square().sum().backward()
    assert torch.count_nonzero(history.grad[is_pad]) == 0


def _make_live_policy(enabled=True):
    # Exercise the real policy entry points without loading a pretrained VLM.
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    class CaptureModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = []

        def sample_actions(self, images, masks, tokens, token_masks, state, **kwargs):
            self.calls.append((images, masks, kwargs))
            return torch.zeros(state.shape[0], 50, 6)

    policy = SmolVLAPolicy.__new__(SmolVLAPolicy)
    torch.nn.Module.__init__(policy)
    policy.config = SimpleNamespace(
        cycle_enabled=enabled,
        cycle_history_size=32,
        cycle_image_history_size=6,
        image_observation_delta_indices=cycle_image_delta_indices() if enabled else None,
        image_features={"observation.images.camera1": None},
        action_feature=SimpleNamespace(shape=(6,)),
        n_action_steps=50,
        max_state_dim=6,
        adapt_to_pi_aloha=False,
        rtc_config=None,
        resize_imgs_with_padding=None,
        empty_cameras=0,
    )
    policy.model = CaptureModel()
    policy.reset()
    return policy


def _live_batch(frame):
    from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS

    return {
        "observation.state": torch.full((1, 6), float(frame)),
        "observation.images.camera1": torch.full((1, 3, 2, 2), float(frame) / 100),
        OBS_LANGUAGE_TOKENS: torch.zeros(1, 2, dtype=torch.long),
        OBS_LANGUAGE_ATTENTION_MASK: torch.ones(1, 2, dtype=torch.bool),
    }


def test_select_action_keeps_dense_history_while_serving_queued_actions():
    policy = _make_live_policy()
    for frame in range(101):
        assert policy.select_action(_live_batch(frame)).shape == (1, 6)
    assert len(policy.model.calls) == 3
    images, masks, kwargs = policy.model.calls[-1]
    assert kwargs["cycle_state"][0, :, 0].tolist() == list(range(69, 101))
    assert not kwargs["cycle_state_is_pad"].any()
    observed = torch.stack(images)[:, 0, 0, 0, 0]
    expected = torch.tensor([69, 84, 92, 98, 99, 100]) / 100 * 2 - 1
    torch.testing.assert_close(observed, expected)
    assert all(mask.all() for mask in masks)


def test_predict_chunk_uses_sparse_images_and_reset_clears_history():
    policy = _make_live_policy()
    for frame in range(40):
        policy.predict_action_chunk(_live_batch(frame))
    images, _, kwargs = policy.model.calls[-1]
    torch.testing.assert_close(
        torch.stack(images)[:, 0, 0, 0, 0],
        torch.tensor([8, 23, 31, 37, 38, 39]) / 100 * 2 - 1,
    )
    assert kwargs["cycle_state"][0, :, 0].tolist() == list(range(8, 40))
    policy.reset()
    policy.predict_action_chunk(_live_batch(80))
    images, masks, kwargs = policy.model.calls[-1]
    assert [mask.item() for mask in masks] == [False] * 5 + [True]
    assert kwargs["cycle_state_is_pad"][0].tolist() == [True] * 31 + [False]
    assert len(policy._cycle_state_history) == 1


def test_offline_temporal_batch_does_not_append_live_history():
    policy = _make_live_policy()
    batch = _live_batch(0)
    batch["observation.state"] = torch.randn(32, 6)
    batch["observation.state_is_pad"] = torch.zeros(32, dtype=torch.bool)
    batch["observation.images.camera1"] = torch.rand(6, 3, 2, 2)
    batch["observation.images.camera1_is_pad"] = torch.zeros(6, dtype=torch.bool)
    policy.predict_action_chunk(batch)
    assert len(policy._cycle_state_history) == 0
    assert len(policy._cycle_image_history["observation.images.camera1"]) == 0
    assert policy.model.calls[-1][2]["cycle_state"].shape == (1, 32, 6)


def test_cycle_disabled_preserves_single_frame_inference():
    policy = _make_live_policy(enabled=False)
    for frame in range(51):
        policy.select_action(_live_batch(frame))
    assert len(policy.model.calls) == 2
    images, _, kwargs = policy.model.calls[-1]
    assert len(images) == 1
    assert kwargs["cycle_state"] is None
    assert len(policy._cycle_state_history) == 0
