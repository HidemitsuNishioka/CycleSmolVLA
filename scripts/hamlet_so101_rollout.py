#!/usr/bin/env python3
"""Run HAMLET on a real SO-101 follower with an explicit actuation guard.

The default mode is dry-run: it reads the follower and camera, predicts an
action, and prints it without sending motor commands.  Pass ``--execute`` and
type ``MOVE`` after warm-up to enable actuation. --max-steps counts commands.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import time

import numpy as np

from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.robots import make_robot_from_config
from lerobot.robots.so_follower.config_so_follower import SO101FollowerConfig

from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.policy.gr00t_policy import Gr00tPolicy


JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]
LANGUAGE_KEY = "annotation.human.action.task_description"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--port", required=True)
    parser.add_argument("--robot-id", default="so101_follower")
    parser.add_argument("--camera-device", required=True)
    parser.add_argument("--camera-width", type=int, default=640)
    parser.add_argument("--camera-height", type=int, default=480)
    parser.add_argument("--camera-fps", type=int, default=30)
    parser.add_argument("--task", default="Fold the cloth in half.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--max-steps", type=int, default=1, help="Maximum motor commands (not chunks)")
    parser.add_argument("--max-relative-target", type=float, default=5.0)
    parser.add_argument("--execute", action="store_true", help="Actually send targets to the follower")
    return parser.parse_args()


def _camera_index_or_path(value: str) -> int | Path:
    try:
        return int(value)
    except ValueError:
        return Path(value)


def _observation(raw: dict, task: str) -> dict:
    image = np.asarray(raw["top"], dtype=np.uint8)
    arm = np.asarray([raw[f"{name}.pos"] for name in JOINT_NAMES[:5]], dtype=np.float32)
    gripper = np.asarray([raw["gripper.pos"]], dtype=np.float32)
    return {
        "video": {"top": image[None, None, ...]},
        "state": {
            "arm": arm[None, None, ...],
            "gripper": gripper[None, None, ...],
        },
        "language": {LANGUAGE_KEY: [[task]]},
    }


def _current_position(raw: dict) -> np.ndarray:
    return np.asarray([raw[f"{name}.pos"] for name in JOINT_NAMES], dtype=np.float32)


def _target_from_prediction(predicted: dict, current: np.ndarray, max_delta: float,
                            action_index: int = 0) -> dict[str, float]:
    arm = np.asarray(predicted["arm"][0, action_index], dtype=np.float32)
    gripper = np.asarray(predicted["gripper"][0, action_index], dtype=np.float32)
    target = np.concatenate((arm, gripper))
    if (target.shape != (6,) or current.shape != (6,)
            or not np.isfinite(target).all() or not np.isfinite(current).all()):
        raise ValueError(f"Invalid HAMLET target: shape={target.shape}, values={target}")

    # Keep every command close to the measured present position. The follower
    # applies the same protection, but clipping here makes the printed target
    # exactly match the intended command before it reaches the bus.
    target = current + np.clip(target - current, -max_delta, max_delta)
    if not (-180.0 <= target[:5]).all() or not (target[:5] <= 180.0).all():
        raise ValueError(f"Arm target outside safety range: {target[:5]}")
    if not (0.0 <= target[5] <= 100.0):
        raise ValueError(f"Gripper target outside safety range: {target[5]}")
    return {f"{name}.pos": float(value) for name, value in zip(JOINT_NAMES, target)}


def run_chunks(policy, robot, args, clock=time.monotonic, sleep=time.sleep):
    """Use observation-anchored deadlines; never replay expired commands in a burst.

    Inference runs once per memory_stride at the dataset's 30 Hz. During
    inference motors hold the previous goal. Drop the expired chunk prefix;
    refuse a chunk if inference consumed its entire time window.
    """
    stride = int(policy.model.config.memory_stride)
    horizon = len(policy.modality_configs["action"].delta_indices)
    if stride != horizon or stride <= 0:
        raise ValueError(f"Expected matching memory stride and action horizon: {stride}, {horizon}")
    period = 1.0 / args.fps
    sent_count = 0
    previous_anchor = None
    while sent_count < args.max_steps:
        raw = robot.get_observation()
        anchor = clock()
        reset = previous_anchor is None or anchor - previous_anchor > (stride + 1) * period
        if previous_anchor is not None and reset:
            print("Observation interval exceeded memory stride; resetting HAMLET history.", flush=True)
        predicted, _ = policy.get_action(
            _observation(raw, args.task),
            options={"session_ids": ["so101-real-rollout"], "reset_memory": [reset]},
        )
        for key, dim in (("arm", 5), ("gripper", 1)):
            values = np.asarray(predicted[key])
            if values.shape != (1, horizon, dim) or not np.isfinite(values).all():
                raise ValueError(f"Invalid {key} action chunk")
        elapsed = clock() - anchor
        if elapsed >= stride * period:
            raise RuntimeError(f"Inference took {elapsed:.3f}s; chunk expired ({stride * period:.3f}s). No stale targets sent.")
        print(f"chunk inference={elapsed:.3f}s, expired prefix={int(elapsed / period)}", flush=True)
        index = 0
        while index < stride and sent_count < args.max_steps:
            index = max(index, int((clock() - anchor) / period))
            if index >= stride:
                break
            sleep(max(0.0, anchor + index * period - clock()))
            current = _current_position(robot.get_observation())
            # Observation I/O may also have crossed a command deadline.
            index = max(index, int((clock() - anchor) / period))
            if index >= stride:
                break
            target = _target_from_prediction(predicted, current, args.max_relative_target, index)
            uncut = np.concatenate((predicted["arm"][0, index], predicted["gripper"][0, index]))
            print(f"command {sent_count + 1}/{args.max_steps}, chunk index={index}", flush=True)
            print("predicted:", " ".join(f"{v:8.2f}" for v in uncut), flush=True)
            _print_target(current, target)
            if args.execute:
                sent = robot.send_action(target)
                print("sent:", sent, flush=True)
            else:
                print("dry-run: action was not sent", flush=True)
            sent_count += 1
            index += 1
            sleep(max(0.0, anchor + index * period - clock()))
        previous_anchor = anchor
        if sent_count < args.max_steps:
            sleep(max(0.0, anchor + stride * period - clock()))


def _print_target(current: np.ndarray, target: dict[str, float]) -> None:
    values = np.asarray([target[f"{name}.pos"] for name in JOINT_NAMES], dtype=np.float32)
    print("current:", " ".join(f"{v:8.2f}" for v in current), flush=True)
    print("target: ", " ".join(f"{v:8.2f}" for v in values), flush=True)


def main() -> None:
    args = _parse_args()
    if not args.model_path.is_dir():
        raise FileNotFoundError(f"Model checkpoint not found: {args.model_path}")
    if args.fps != 30:
        raise ValueError("This SO-101 checkpoint uses 30 Hz actions; --fps must be 30")
    if args.max_steps <= 0:
        raise ValueError("--max-steps must be positive")
    if not math.isfinite(args.max_relative_target) or args.max_relative_target <= 0:
        raise ValueError("--max-relative-target must be positive")

    print(f"Loading HAMLET: {args.model_path}", flush=True)
    policy = Gr00tPolicy(
        embodiment_tag=EmbodimentTag.NEW_EMBODIMENT,
        model_path=str(args.model_path),
        device=args.device,
        strict=True,
    )

    camera = OpenCVCameraConfig(
        index_or_path=_camera_index_or_path(args.camera_device),
        fps=args.camera_fps,
        width=args.camera_width,
        height=args.camera_height,
    )
    robot_config = SO101FollowerConfig(
        port=args.port,
        id=args.robot_id,
        cameras={"top": camera},
        use_degrees=True,
        max_relative_target=args.max_relative_target,
        disable_torque_on_disconnect=True,
    )
    robot = make_robot_from_config(robot_config)
    actuated = False
    try:
        print(f"Connecting follower: {args.port}", flush=True)
        if not robot.calibration:
            raise RuntimeError("Existing follower calibration is required; calibrate separately.")
        # Avoid configure()/automatic calibration during dry-run. connect()
        # configures the motors and can change torque even without send_action.
        robot.bus.connect()
        if not robot.is_calibrated:
            raise RuntimeError("Motor calibration differs from the local calibration file.")
        for cam in robot.cameras.values():
            cam.connect()
        raw = robot.get_observation()
        print("Follower connected and camera frame received.", flush=True)
        # Warm up before establishing the observation/action timeline.
        policy.get_action(_observation(raw, args.task), options={
            "session_ids": ["so101-real-rollout"], "reset_memory": [True]})
        if args.execute:
            print("開始後は30Hzで目標を送信します。停止時はトルクが解除されます。", flush=True)
            if input('実機へ送信する場合は MOVE と入力: ').strip() != "MOVE":
                print("Actuation cancelled.", flush=True)
                return
            actuated = True
            robot.bus.disable_torque()
            present = robot.bus.sync_read("Present_Position")
            robot.bus.sync_write("Goal_Position", present)
            robot.configure()
        # Obtain fresh state/image after confirmation and reset warm-up history.
        run_chunks(policy, robot, args)
    finally:
        try:
            if robot.bus.is_connected:
                robot.bus.disconnect(disable_torque=actuated)
        finally:
            for cam in robot.cameras.values():
                if cam.is_connected:
                    cam.disconnect()
        print("Follower disconnected.", flush=True)


if __name__ == "__main__":
    main()
