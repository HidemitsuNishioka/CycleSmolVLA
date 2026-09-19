"""Regression tests for training/live CycleManip observation parity (no hardware)."""

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.rtc import ActionQueue
from lerobot.policies.rtc.configuration_rtc import RTCConfig
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.utils import prepare_observation_for_inference
from lerobot.rollout.inference.observation_history import ObservationHistory
from lerobot.rollout.inference.rtc import RTCInferenceEngine
from lerobot.rollout.inference.sync import SyncInferenceEngine
from lerobot.utils.constants import OBS_STATE

IMAGE = "observation.images.top"
FEATURES = {
    OBS_STATE: {"dtype": "float32", "shape": (1,), "names": ["joint.pos"]},
    IMAGE: {"dtype": "video", "shape": (2, 2, 3)},
    "action": {"dtype": "float32", "shape": (1,), "names": ["joint.pos"]},
}


def config(image_count=6):
    return SmolVLAConfig(
        device="cpu",
        cycle_enabled=True,
        cycle_history_size=300,
        cycle_image_history_size=image_count,
        input_features={
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(1,)),
            IMAGE: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 2, 2)),
        },
    )


def frame(index):
    return {
        OBS_STATE: np.array([index], dtype=np.float32),
        IMAGE: np.full((2, 2, 3), index % 256, dtype=np.uint8),
    }


@pytest.mark.parametrize("last", [0, 35, 299, 350])
def test_history_matches_dataset_offsets_and_start_padding(last):
    cfg = config()
    history = ObservationHistory.from_config(cfg)
    for index in range(last + 1):
        history.append(frame(index))
    batch = prepare_observation_for_inference(
        ObservationHistory.stack(history.snapshot()), torch.device("cpu")
    )
    expected_state = torch.tensor([max(0, last + i) for i in range(-299, 1)]).float()
    expected_image = torch.tensor([max(0, last + i) % 256 for i in [-299, -150, -75, -2, -1, 0]])
    torch.testing.assert_close(batch[OBS_STATE][0, :, 0], expected_state)
    torch.testing.assert_close(batch[IMAGE][0, :, 0, 0, 0], expected_image.float() / 255)
    assert batch[IMAGE].shape == (1, 6, 3, 2, 2)
    torch.testing.assert_close(
        batch[f"{OBS_STATE}_is_pad"][0], torch.tensor([last + i < 0 for i in range(-299, 1)])
    )
    torch.testing.assert_close(
        batch[f"{IMAGE}_is_pad"][0], torch.tensor([last + i < 0 for i in [-299, -150, -75, -2, -1, 0]])
    )


def test_snapshot_owns_camera_data_and_survives_append_and_reset():
    history = ObservationHistory.from_config(config())
    observation = frame(7)
    history.append(observation)
    snapshot = history.snapshot()
    observation[IMAGE][:] = 99
    for index in range(400):
        history.append(frame(index))
    history.clear()
    history.append(frame(9))
    assert np.all(ObservationHistory.stack(snapshot)[IMAGE] == 7)
    assert np.all(ObservationHistory.stack(history.snapshot())[IMAGE] == 9)


@pytest.mark.parametrize("image_count", [3, 6])
def test_policy_fallback_samples_spaced_images_without_confusing_rgb_with_time(image_count):
    # Exercise the actual policy preprocessing without loading the VLM.
    policy = SmolVLAPolicy.__new__(SmolVLAPolicy)
    torch.nn.Module.__init__(policy)
    policy.config = config(image_count)
    policy.reset()
    for index in range(351):
        batch = prepare_observation_for_inference(frame(index), torch.device("cpu"))
        policy._prepare_cycle_batch(batch)
    assert batch[IMAGE].shape == (1, image_count, 3, 2, 2)
    expected = torch.tensor([(350 + i) % 256 for i in policy.config.image_observation_delta_indices])
    torch.testing.assert_close(batch[IMAGE][0, :, 0, 0, 0], expected.float() / 255)
    torch.testing.assert_close(batch[OBS_STATE][0, :, 0], torch.arange(51, 351).float())
    # Full offline/live windows must be passed through without advancing a FIFO.
    original = {key: value.clone() for key, value in batch.items() if isinstance(value, torch.Tensor)}
    policy._prepare_cycle_batch(batch)
    for key, value in original.items():
        torch.testing.assert_close(batch[key], value)


def make_rtc():
    policy = Mock(config=config())
    preprocessor = Mock(steps=[], side_effect=lambda batch: batch)
    postprocessor = Mock(side_effect=lambda action: action)
    engine = RTCInferenceEngine(
        policy=policy,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        robot_wrapper=SimpleNamespace(robot_type="so_follower"),
        rtc_config=RTCConfig(),
        hw_features=FEATURES,
        task="Shake the cup three times.",
        fps=30,
        device="cpu",
    )
    return engine, policy


def test_rtc_keeps_every_control_tick_even_when_inference_is_sparse():
    engine, policy = make_rtc()
    for index in range(351):
        observation = frame(index)
        engine.notify_observation({"joint.pos": float(index), "top": observation[IMAGE]})

    def predict(batch, **kwargs):
        torch.testing.assert_close(batch[OBS_STATE][0, :, 0], torch.arange(51, 351).float())
        expected = torch.tensor([51, 200, 275 % 256, 348 % 256, 349 % 256, 350 % 256]).float() / 255
        torch.testing.assert_close(batch[IMAGE][0, :, 0, 0, 0], expected)
        engine._shutdown_event.set()
        return torch.zeros(1, 50, 1)

    policy.predict_action_chunk.side_effect = predict
    engine._action_queue = ActionQueue(RTCConfig())
    engine.resume()
    engine._rtc_loop()
    assert not engine.failed, engine.failure_traceback
    policy.predict_action_chunk.assert_called_once()

    engine.reset()
    engine.notify_observation({"joint.pos": 999.0, "top": frame(9)[IMAGE]})
    batch = ObservationHistory.stack(engine._observation_history.snapshot())
    assert np.all(batch[OBS_STATE] == 999)
    assert np.all(batch[IMAGE] == 9)


def test_sync_passes_full_history_and_resets_between_episodes():
    policy = Mock(config=config())
    policy.select_action.return_value = torch.zeros(1, 2)
    features = {**FEATURES, "action": {"names": ["joint.pos", "other.pos"]}}
    engine = SyncInferenceEngine(
        policy,
        Mock(side_effect=lambda x: x),
        Mock(side_effect=lambda x: x),
        features,
        ["joint.pos", "other.pos"],
        "shake",
        "cpu",
        "so_follower",
    )
    for index in range(351):
        engine.get_action(frame(index))
    batch = policy.select_action.call_args.args[0]
    torch.testing.assert_close(batch[OBS_STATE][0, :, 0], torch.arange(51, 351).float())
    assert batch[IMAGE].shape == (1, 6, 3, 2, 2)
    engine.reset()
    engine.get_action(frame(9))
    assert torch.all(policy.select_action.call_args.args[0][OBS_STATE] == 9)


def test_non_cycle_policy_keeps_single_frame_input():
    cfg = config()
    cfg.cycle_enabled = False
    assert ObservationHistory.from_config(cfg) is None
    batch = prepare_observation_for_inference(frame(1), torch.device("cpu"))
    assert batch[OBS_STATE].shape == (1, 1)
    assert batch[IMAGE].shape == (1, 3, 2, 2)


def test_restart_waits_for_inflight_rtc_and_discards_old_actions():
    import threading
    import time

    from lerobot.rollout import BaseStrategyConfig
    from lerobot.rollout.strategies import BaseStrategy
    from lerobot.utils.action_interpolator import ActionInterpolator

    engine, policy = make_rtc()
    entered = threading.Event()
    release = threading.Event()
    stop_started = threading.Event()
    second_inference = threading.Event()
    errors = []

    def predict(batch, **kwargs):
        if not entered.is_set():
            entered.set()
            assert release.wait(3)
            return torch.full((1, 50, 1), 111.0)
        assert torch.all(batch[OBS_STATE] == 9)
        assert batch[f"{OBS_STATE}_is_pad"].sum() == 299
        second_inference.set()
        return torch.full((1, 50, 1), 999.0)

    policy.predict_action_chunk.side_effect = predict
    engine.start()
    engine.notify_observation({"joint.pos": 1.0, "top": frame(1)[IMAGE]})
    engine.resume()
    assert entered.wait(3)
    original_stop = engine.stop

    def stop():
        stop_started.set()
        original_stop()

    engine.stop = stop
    strategy = BaseStrategy(BaseStrategyConfig())
    strategy._engine = engine
    strategy._interpolator = ActionInterpolator(multiplier=1)
    ctx = SimpleNamespace(runtime=SimpleNamespace(shutdown_event=threading.Event()))

    def restart():
        try:
            strategy._restart_inference(ctx, Mock())
        except BaseException as exc:
            errors.append(exc)

    worker = threading.Thread(target=restart)
    try:
        worker.start()
        assert stop_started.wait(3)
        policy.reset.assert_not_called()
        release.set()
        worker.join(3)
        assert not worker.is_alive()
        assert not errors
        assert engine.get_action(None) is None
        assert engine._observation_history.snapshot() == {}
        policy.reset.assert_called_once()
        engine.notify_observation({"joint.pos": 9.0, "top": frame(9)[IMAGE]})
        assert second_inference.wait(3)
        deadline = time.monotonic() + 3
        action = engine.get_action(None)
        while action is None and time.monotonic() < deadline:
            time.sleep(0.01)
            action = engine.get_action(None)
        assert action is not None
        torch.testing.assert_close(action, torch.tensor([999.0]))
    finally:
        release.set()
        worker.join(3)
        original_stop()


def test_rtc_stop_retains_live_worker_on_timeout():
    engine, _ = make_rtc()
    thread = Mock()
    thread.is_alive.return_value = True
    engine._rtc_thread = thread
    with pytest.raises(TimeoutError, match="did not stop"):
        engine.stop()
    assert engine._rtc_thread is thread
