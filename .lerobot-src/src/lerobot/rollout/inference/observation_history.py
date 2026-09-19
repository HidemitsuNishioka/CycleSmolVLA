"""Control-tick observation windows for CycleManip rollout.

Keep images as uint8 on the CPU and select only the training-time offsets
before transferring to the policy device. The caller owns synchronization.
"""

from collections import deque

import numpy as np

from lerobot.utils.constants import OBS_IMAGES, OBS_STATE


class ObservationHistory:
    def __init__(self, state_offsets: list[int], image_offsets: list[int]):
        self.state_offsets = state_offsets
        self.image_offsets = image_offsets
        self._frames: dict[str, deque[np.ndarray]] = {}

    @classmethod
    def from_config(cls, config):
        if getattr(config, "cycle_enabled", False) is not True:
            return None
        return cls(config.state_observation_delta_indices, config.image_observation_delta_indices)

    def clear(self) -> None:
        self._frames.clear()

    def _offsets(self, key: str) -> list[int]:
        if key == OBS_STATE:
            return self.state_offsets
        if key.startswith(f"{OBS_IMAGES}."):
            return self.image_offsets
        return [0]

    def append(self, frame: dict[str, np.ndarray]) -> None:
        for key, value in frame.items():
            offsets = self._offsets(key)
            queue = self._frames.setdefault(key, deque(maxlen=1 - min(offsets)))
            # Camera backends may reuse their buffers after notify_observation.
            queue.append(value.copy())

    def snapshot(self) -> dict[str, tuple[np.ndarray, ...]]:
        """Freeze observations and the episode-start padding masks used in training."""
        snapshot = {}
        for key, queue in self._frames.items():
            offsets = self._offsets(key)
            indices = [len(queue) - 1 + offset for offset in offsets]
            snapshot[key] = tuple(queue[max(0, index)] for index in indices)
            if key == OBS_STATE or key.startswith(f"{OBS_IMAGES}."):
                snapshot[f"{key}_is_pad"] = tuple(np.asarray(index < 0) for index in indices)
        return snapshot

    @staticmethod
    def stack(snapshot: dict[str, tuple[np.ndarray, ...]]) -> dict[str, np.ndarray]:
        """Materialize a snapshot outside the control thread's observation lock."""
        return {key: np.stack(frames) for key, frames in snapshot.items()}
