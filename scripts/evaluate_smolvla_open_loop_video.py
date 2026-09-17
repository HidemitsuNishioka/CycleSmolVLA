#!/usr/bin/env python3
"""Open-loop action comparison for an official LeRobot SmolVLA checkpoint.

This intentionally uses LeRobotDataset, SmolVLAPolicy, and LeRobot's saved
pre/post-processors. The only local code here is the report renderer because
the official evaluation path reports losses but does not render GT/prediction
overlays into a video.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy


JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-repo-id", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--eval-split", type=float, default=0.2)
    parser.add_argument(
        "--episodes",
        default="",
        help="Comma-separated episode IDs. Empty means the last eval_split fraction.",
    )
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max-frames-per-episode", type=int, default=0)
    parser.add_argument(
        "--action-mode",
        choices=("fresh", "rollout"),
        default="fresh",
        help="fresh=official predict_action_chunk per frame; rollout=official select_action queue.",
    )
    return parser.parse_args()


def selected_episode_ids(dataset: LeRobotDataset, args: argparse.Namespace) -> list[int]:
    if args.episodes.strip():
        ids = [int(value.strip()) for value in args.episodes.split(",") if value.strip()]
    else:
        first = max(0, math.floor(dataset.meta.total_episodes * (1.0 - args.eval_split)))
        ids = list(range(first, dataset.meta.total_episodes))
    if not ids:
        raise ValueError("No episodes selected")
    invalid = [episode for episode in ids if episode < 0 or episode >= dataset.meta.total_episodes]
    if invalid:
        raise ValueError(f"Episode IDs out of range: {invalid}")
    return ids


def as_rgb_uint8(image: torch.Tensor) -> np.ndarray:
    array = image.detach().cpu().float().numpy()
    if array.ndim == 4:
        array = array[0]
    if array.ndim == 3 and array.shape[0] in (1, 3, 4):
        array = np.transpose(array[:3], (1, 2, 0))
    if array.max(initial=0) <= 1.5:
        array = array * 255.0
    return np.clip(array, 0, 255).astype(np.uint8)


def put_text(
    image: np.ndarray,
    text: str,
    origin: tuple[int, int],
    scale: float = 0.55,
    color=(235, 235, 235),
):
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def draw_series(
    image: np.ndarray,
    values: np.ndarray,
    plot_box: tuple[int, int, int, int],
    value_min: float,
    value_max: float,
    color: tuple[int, int, int],
    total_points: int,
) -> None:
    x0, y0, width, height = plot_box
    if len(values) == 0:
        return
    points = []
    denominator = max(1, total_points - 1)
    for index, value in enumerate(values):
        x = x0 + int(round((width - 1) * index / denominator))
        normalized = (float(value) - value_min) / (value_max - value_min)
        y = y0 + height - 1 - int(round((height - 1) * np.clip(normalized, 0.0, 1.0)))
        points.append((x, y))
    if len(points) > 1:
        cv2.polylines(image, [np.asarray(points, dtype=np.int32)], False, color, 1, cv2.LINE_AA)
    cv2.circle(image, points[-1], 2, color, -1, cv2.LINE_AA)


def draw_joint_graph(
    panel: np.ndarray,
    x: int,
    y: int,
    width: int,
    height: int,
    joint_name: str,
    gt_history: np.ndarray,
    pred_history: np.ndarray,
    error_history: np.ndarray,
    total_points: int,
    fps: float,
) -> None:
    cv2.rectangle(panel, (x, y), (x + width - 2, y + height - 2), (42, 46, 56), -1)
    put_text(panel, joint_name, (x + 5, y + 16), 0.43, (245, 245, 245))

    top_box = (x + 28, y + 23, width - 36, 74)
    error_box = (x + 28, y + 119, width - 36, 48)
    for box in (top_box, error_box):
        bx, by, bw, bh = box
        cv2.rectangle(panel, (bx, by), (bx + bw, by + bh), (20, 23, 29), 1)

    # Angle plot: fixed SO-101 joint-angle range keeps the scale stable through the video.
    ax, ay, aw, ah = top_box
    zero_y = ay + ah // 2
    cv2.line(panel, (ax, zero_y), (ax + aw, zero_y), (70, 74, 84), 1)
    put_text(panel, "+180", (x + 1, ay + 7), 0.30, (145, 150, 160))
    put_text(panel, "0", (x + 13, zero_y + 4), 0.30, (145, 150, 160))
    put_text(panel, "-180", (x, ay + ah), 0.30, (145, 150, 160))
    draw_series(panel, gt_history[:, JOINT_NAMES.index(joint_name)], top_box, -180.0, 180.0, (255, 190, 90), total_points)
    draw_series(panel, pred_history[:, JOINT_NAMES.index(joint_name)], top_box, -180.0, 180.0, (100, 230, 120), total_points)

    # Error plot: absolute error is positive and uses a stable 0-90 degree scale.
    ex, ey, ew, eh = error_box
    put_text(panel, "0", (x + 13, ey + eh), 0.30, (145, 150, 160))
    put_text(panel, "90", (x + 6, ey + 7), 0.30, (145, 150, 160))
    draw_series(panel, error_history[:, JOINT_NAMES.index(joint_name)], error_box, 0.0, 90.0, (80, 170, 255), total_points)
    duration_s = (total_points - 1) / fps if total_points > 1 else 0.0
    put_text(panel, "0s", (ex, ey + eh + 13), 0.30, (145, 150, 160))
    put_text(panel, f"{duration_s:.0f}s", (ex + ew - 22, ey + eh + 13), 0.30, (145, 150, 160))


def render_frame(
    rgb: np.ndarray,
    episode: int,
    frame_in_episode: int,
    episode_length: int,
    gt_history: np.ndarray,
    pred_history: np.ndarray,
    error_history: np.ndarray,
    fps: float,
) -> np.ndarray:
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    height, width = bgr.shape[:2]
    panel_width = 650
    panel = np.full((height, panel_width, 3), (28, 31, 38), dtype=np.uint8)
    put_text(panel, "SmolVLA open-loop: angle and error over time", (18, 25), 0.53, (255, 255, 255))
    put_text(panel, f"episode {episode}  t={frame_in_episode / fps:.2f}s  ({frame_in_episode + 1}/{episode_length})", (18, 48), 0.45)
    put_text(panel, "GT", (18, 68), 0.38, (255, 190, 90))
    put_text(panel, "Pred", (58, 68), 0.38, (100, 230, 120))
    put_text(panel, "|error|", (108, 68), 0.38, (80, 170, 255))
    put_text(panel, "top: angle [-180,180] deg   bottom: abs error [0,90] deg", (180, 68), 0.34, (175, 180, 190))

    graph_width = 210
    graph_height = 195
    for index, name in enumerate(JOINT_NAMES):
        graph_x = 8 + (index % 3) * graph_width
        graph_y = 78 + (index // 3) * graph_height
        draw_joint_graph(panel, graph_x, graph_y, graph_width, graph_height, name, gt_history, pred_history, error_history, episode_length, fps)

    return np.concatenate((bgr, panel), axis=1)


def save_episode_graph(
    path: Path,
    gt: np.ndarray,
    pred: np.ndarray,
    fps: float,
    episode: int,
) -> None:
    times = np.arange(len(gt), dtype=np.float32) / fps
    errors = np.abs(pred - gt)
    figure, axes = plt.subplots(2, 3, figsize=(16, 8), sharex=True)
    for index, (axis, name) in enumerate(zip(axes.flat, JOINT_NAMES)):
        axis.plot(times, gt[:, index], label="GT", color="#2f80ed", linewidth=1.3)
        axis.plot(times, pred[:, index], label="Pred", color="#27ae60", linewidth=1.3)
        axis.plot(times, errors[:, index], label="|error|", color="#f2994a", linewidth=1.1)
        axis.set_title(name)
        axis.set_xlabel("time [s]")
        axis.set_ylabel("angle / abs error [deg]")
        axis.set_ylim(-180, 180)
        axis.grid(alpha=0.25)
    axes.flat[0].legend(loc="upper right", fontsize=8)
    figure.suptitle(f"SmolVLA open-loop episode {episode}: GT, prediction, and absolute error")
    figure.tight_layout()
    figure.savefig(path, dpi=140)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    if args.stride < 1:
        raise ValueError("--stride must be >= 1")
    if not (0.0 < args.eval_split <= 1.0):
        raise ValueError("--eval-split must be in (0, 1]")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = Path(args.checkpoint)
    device = torch.device(args.device)

    dataset = LeRobotDataset(args.dataset_repo_id, root=args.dataset_root)
    episodes = selected_episode_ids(dataset, args)
    if not dataset.meta.camera_keys:
        raise ValueError("The dataset has no camera feature")
    image_key = dataset.meta.camera_keys[0]
    if image_key != "observation.images.top":
        print(f"Using first dataset camera feature: {image_key}")

    print(f"Loading official SmolVLAPolicy from {checkpoint}")
    policy = SmolVLAPolicy.from_pretrained(str(checkpoint)).to(device).eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy.config,
        str(checkpoint),
        preprocessor_overrides={"device_processor": {"device": str(device)}},
    )

    csv_path = output_dir / "action_comparison.csv"
    fieldnames = [
        "episode",
        "frame_in_episode",
        "dataset_index",
        "timestamp",
        *[f"gt_{name}" for name in JOINT_NAMES],
        *[f"pred_{name}" for name in JOINT_NAMES],
        *[f"abs_error_{name}" for name in JOINT_NAMES],
    ]
    all_gt: list[np.ndarray] = []
    all_pred: list[np.ndarray] = []
    episode_summaries = []

    with csv_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        for episode in episodes:
            episode_meta = dataset.meta.episodes[episode]
            start = int(episode_meta["dataset_from_index"])
            stop = int(episode_meta["dataset_to_index"])
            indices = list(range(start, stop, args.stride))
            if args.max_frames_per_episode:
                indices = indices[: args.max_frames_per_episode]
            if not indices:
                continue

            first_sample = dataset[indices[0]]
            first_rgb = as_rgb_uint8(first_sample[image_key])
            output_video = output_dir / f"episode_{episode:03d}_gt_pred_error.mp4"
            output_graph = output_dir / f"episode_{episode:03d}_angle_graph.png"
            output_fps = max(1.0, float(dataset.fps) / args.stride)
            video = cv2.VideoWriter(
                str(output_video),
                cv2.VideoWriter_fourcc(*"mp4v"),
                output_fps,
                (first_rgb.shape[1] + 650, first_rgb.shape[0]),
            )
            if not video.isOpened():
                raise RuntimeError(f"Could not open video writer: {output_video}")

            policy.reset()
            episode_gt: list[np.ndarray] = []
            episode_pred: list[np.ndarray] = []
            print(f"Episode {episode}: {len(indices)} frames -> {output_video}")

            for local_frame, dataset_index in enumerate(indices):
                sample = first_sample if dataset_index == indices[0] else dataset[dataset_index]
                raw_observation = {
                    image_key: sample[image_key],
                    "observation.state": sample["observation.state"],
                    "task": sample["task"],
                }
                processed = preprocessor(raw_observation)
                with torch.inference_mode():
                    if args.action_mode == "fresh":
                        action_chunk = policy.predict_action_chunk(processed)
                        predicted = action_chunk[:, 0]
                    else:
                        predicted = policy.select_action(processed)
                    predicted = postprocessor(predicted)

                gt = sample["action"].detach().float().cpu().reshape(-1).numpy()
                pred = predicted.detach().float().cpu().reshape(-1).numpy()
                if gt.shape[0] != len(JOINT_NAMES) or pred.shape[0] != len(JOINT_NAMES):
                    raise ValueError(f"Expected 6 action values, got GT={gt.shape} prediction={pred.shape}")
                error = np.abs(pred - gt)
                episode_gt.append(gt)
                episode_pred.append(pred)
                all_gt.append(gt)
                all_pred.append(pred)

                row = {
                    "episode": episode,
                    "frame_in_episode": local_frame,
                    "dataset_index": dataset_index,
                    "timestamp": float(sample["timestamp"]),
                }
                row.update({f"gt_{name}": float(value) for name, value in zip(JOINT_NAMES, gt)})
                row.update({f"pred_{name}": float(value) for name, value in zip(JOINT_NAMES, pred)})
                row.update({f"abs_error_{name}": float(value) for name, value in zip(JOINT_NAMES, error)})
                writer.writerow(row)

                running_errors = np.abs(np.asarray(episode_pred) - np.asarray(episode_gt))
                frame = render_frame(
                    as_rgb_uint8(sample[image_key]),
                    episode,
                    local_frame,
                    len(indices),
                    np.asarray(episode_gt),
                    np.asarray(episode_pred),
                    running_errors,
                    output_fps,
                )
                video.write(frame)
                if (local_frame + 1) % 25 == 0 or local_frame + 1 == len(indices):
                    print(f"  {local_frame + 1}/{len(indices)} frames, MAE={running_errors.mean():.3f} deg", flush=True)

            video.release()
            ep_gt = np.asarray(episode_gt)
            ep_pred = np.asarray(episode_pred)
            ep_error = ep_pred - ep_gt
            save_episode_graph(output_graph, ep_gt, ep_pred, output_fps, episode)
            episode_summaries.append(
                {
                    "episode": episode,
                    "frames": len(indices),
                    "video": str(output_video),
                    "graph": str(output_graph),
                    "mae_deg": float(np.abs(ep_error).mean()),
                    "rmse_deg": float(np.sqrt(np.square(ep_error).mean())),
                    "per_joint_mae_deg": {
                        name: float(value) for name, value in zip(JOINT_NAMES, np.abs(ep_error).mean(axis=0))
                    },
                }
            )

    gt_all = np.asarray(all_gt)
    pred_all = np.asarray(all_pred)
    error_all = pred_all - gt_all
    summary = {
        "checkpoint": str(checkpoint),
        "dataset_repo_id": args.dataset_repo_id,
        "dataset_root": args.dataset_root,
        "dataset_fps": dataset.fps,
        "camera_feature": image_key,
        "action_mode": args.action_mode,
        "episodes": episodes,
        "frames": int(len(gt_all)),
        "mae_deg": float(np.abs(error_all).mean()),
        "rmse_deg": float(np.sqrt(np.square(error_all).mean())),
        "per_joint_mae_deg": {
            name: float(value) for name, value in zip(JOINT_NAMES, np.abs(error_all).mean(axis=0))
        },
        "per_joint_rmse_deg": {
            name: float(value) for name, value in zip(JOINT_NAMES, np.sqrt(np.square(error_all).mean(axis=0)))
        },
        "episode_summaries": episode_summaries,
        "csv": str(csv_path),
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
