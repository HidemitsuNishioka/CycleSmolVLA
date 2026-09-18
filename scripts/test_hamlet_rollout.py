"""Hardware-free regression checks. Run using the HAMLET virtualenv."""
import contextlib
import io
import unittest
from types import SimpleNamespace

import numpy as np

from hamlet_so101_rollout import JOINT_NAMES, run_chunks, _target_from_prediction


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


class Robot:
    def __init__(self, clock):
        self.clock = clock
        self.sent = []

    def get_observation(self):
        return {**{f"{j}.pos": 0.0 for j in JOINT_NAMES},
                "top": np.zeros((4, 4, 3), dtype=np.uint8)}

    def send_action(self, target):
        self.sent.append((self.clock(), target))
        return target


class Policy:
    def __init__(self, clock, latency):
        self.clock, self.latency = clock, latency
        self.calls = []
        self.model = SimpleNamespace(config=SimpleNamespace(memory_stride=16))
        self.modality_configs = {"action": SimpleNamespace(delta_indices=list(range(16)))}

    def get_action(self, obs, options):
        self.calls.append((self.clock(), options))
        self.clock.sleep(self.latency)
        return {"arm": np.repeat(np.arange(16)[None, :, None], 5, axis=2),
                "gripper": np.zeros((1, 16, 1))}, {}


class RolloutTests(unittest.TestCase):
    def run_case(self, latency=0, execute=True, count=20):
        clock = Clock()
        policy, robot = Policy(clock, latency), Robot(clock)
        args = SimpleNamespace(fps=30, max_steps=count, task="Shake the cup.",
                               execute=execute, max_relative_target=5)
        with contextlib.redirect_stdout(io.StringIO()):
            run_chunks(policy, robot, args, clock, clock.sleep)
        return policy, robot

    def test_chunk_execution_and_memory_cadence(self):
        policy, robot = self.run_case()
        self.assertEqual(len(robot.sent), 20)
        self.assertAlmostEqual(policy.calls[1][0] - policy.calls[0][0], 16 / 30)
        self.assertTrue(policy.calls[0][1]["reset_memory"][0])
        self.assertFalse(policy.calls[1][1]["reset_memory"][0])
        np.testing.assert_allclose([t for t, _ in robot.sent], np.arange(20) / 30)
        self.assertEqual(robot.sent[3][1]["shoulder_pan.pos"], 3)
        self.assertEqual(robot.sent[8][1]["shoulder_pan.pos"], 5)

    def test_expired_prefix_skipped(self):
        policy, robot = self.run_case(latency=0.1, count=1)
        self.assertEqual(robot.sent[0][1]["shoulder_pan.pos"], 3)

    def test_expired_chunk_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "chunk expired"):
            self.run_case(latency=0.6)

    def test_dry_run_sends_nothing(self):
        _, robot = self.run_case(execute=False)
        self.assertFalse(robot.sent)

    def test_nonfinite_state_rejected(self):
        predicted = {"arm": np.zeros((1,16,5)), "gripper": np.zeros((1,16,1))}
        with self.assertRaises(ValueError):
            _target_from_prediction(predicted, np.full(6, np.nan), 5)


if __name__ == "__main__":
    unittest.main()
