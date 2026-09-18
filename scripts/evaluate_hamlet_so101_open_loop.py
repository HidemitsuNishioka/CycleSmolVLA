#!/usr/bin/env python3
"""Offline open-loop evaluation for a HAMLET SO-101 checkpoint.

The evaluator replays observations at the HAMLET memory stride and compares the
predicted 16-step absolute action chunk with the recorded ground-truth chunk.
It never sends actions to a robot.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="HAMLET checkpoint directory")
    parser.add_argument(
        "--dataset-path",
        default="data/so101_shake_cup_gr00t",
        help="Converted HAMLET dataset directory",
    )
    parser.add_argument(
        "--modality-config-path",
        default="scripts/hamlet_so101_modality.py",
        help="SO-101 modality registration module",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/eval/hamlet_so101_open_loop",
        help="Evaluation artifact directory",
    )
    parser.add_argument(
        "--episodes",
        default="",
        help="Comma-separated dataset episode indices; empty means the last 20%%",
    )
    parser.add_argument("--eval-fraction", type=float, default=0.2)
    parser.add_argument("--stride", type=int, default=16)
    parser.add_argument("--max-anchors-per-episode", type=int, default=0)
    parser.add_argument("--seed", type=int, default=6)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def _load_modality_config(path: Path, hamlet_root: Path):
    import importlib.util

    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Modality config not found: {path}")

    sys.path.insert(0, str(hamlet_root))
    spec = importlib.util.spec_from_file_location("hamlet_so101_modality_eval", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load modality config: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.so101_joint


def _episode_indices(loader, requested: str, fraction: float) -> list[int]:
    if requested.strip():
        indices = [int(item.strip()) for item in requested.split(",") if item.strip()]
    else:
        count = max(1, int(np.ceil(len(loader) * fraction)))
        indices = list(range(max(0, len(loader) - count), len(loader)))
    invalid = [idx for idx in indices if idx < 0 or idx >= len(loader)]
    if invalid:
        raise IndexError(f"Episode indices out of range: {invalid}; dataset has {len(loader)} episodes")
    return indices


def _make_observation(row: Any, language_key: str) -> dict[str, Any]:
    frame = np.asarray(row["video.top"], dtype=np.uint8)
    if frame.ndim != 3 or frame.shape[-1] != 3:
        raise ValueError(f"Expected an HWC RGB frame, got shape={frame.shape}")

    arm = np.asarray(row["state.arm"], dtype=np.float32)
    gripper = np.asarray(row["state.gripper"], dtype=np.float32)
    text = str(row[f"language.{language_key}"])
    return {
        "video": {"top": frame[None, None, ...]},
        "state": {
            "arm": arm[None, None, ...],
            "gripper": gripper[None, None, ...],
        },
        "language": {language_key: [[text]]},
    }


def _ground_truth(episode, start: int, horizon: int) -> dict[str, np.ndarray]:
    result = {}
    for key in ("arm", "gripper"):
        values = episode[f"action.{key}"].iloc[start : start + horizon]
        result[key] = np.stack([np.asarray(value, dtype=np.float32) for value in values], axis=0)
    return result


def _metric(errors: list[np.ndarray]) -> dict[str, Any]:
    if not errors:
        return {"samples": 0}
    stacked = np.concatenate(errors, axis=0)
    abs_error = np.abs(stacked)
    return {
        "samples": int(stacked.shape[0]),
        "mae": float(abs_error.mean()),
        "rmse": float(np.sqrt(np.square(stacked).mean())),
        "max_abs_error": float(abs_error.max()),
        "mae_by_dim": [float(value) for value in abs_error.mean(axis=0)],
        "rmse_by_dim": [float(value) for value in np.sqrt(np.square(stacked).mean(axis=0))],
    }


def _put_text(image: np.ndarray, text: str, origin: tuple[int, int], scale: float = 0.45, color=(235, 235, 235)):
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def _draw_series(
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
    span = max(1e-6, value_max - value_min)
    for index, value in enumerate(values):
        x = x0 + int(round((width - 1) * index / denominator))
        normalized = (float(value) - value_min) / span
        y = y0 + height - 1 - int(round((height - 1) * np.clip(normalized, 0.0, 1.0)))
        points.append((x, y))
    if len(points) > 1:
        cv2.polylines(image, [np.asarray(points, dtype=np.int32)], False, color, 1, cv2.LINE_AA)
    cv2.circle(image, points[-1], 2, color, -1, cv2.LINE_AA)


def _plot_ranges(gt: np.ndarray, pred: np.ndarray) -> tuple[list[tuple[float, float]], list[float]]:
    ranges = []
    error_maxes = []
    errors = np.abs(pred - gt)
    for index in range(gt.shape[1]):
        values = np.concatenate((gt[:, index], pred[:, index]))
        low = float(values.min())
        high = float(values.max())
        margin = max(1.0, (high - low) * 0.12)
        ranges.append((low - margin, high + margin))
        error_maxes.append(max(1.0, float(errors[:, index].max()) * 1.15))
    return ranges, error_maxes


def _render_frame(
    rgb: np.ndarray,
    episode_id: int,
    anchor: int,
    total_anchors: int,
    gt_history: np.ndarray,
    pred_history: np.ndarray,
    ranges: list[tuple[float, float]],
    error_maxes: list[float],
    fps: float,
) -> np.ndarray:
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    height, width = bgr.shape[:2]
    panel_width = 660
    panel = np.full((height, panel_width, 3), (28, 31, 38), dtype=np.uint8)
    _put_text(panel, "HAMLET SO-101 open-loop: GT / Pred / error", (16, 24), 0.52, (255, 255, 255))
    _put_text(panel, f"episode {episode_id}  anchor={anchor}  ({len(gt_history)}/{total_anchors})", (16, 47), 0.42)
    _put_text(panel, "GT", (16, 67), 0.36, (255, 190, 90))
    _put_text(panel, "Pred", (55, 67), 0.36, (100, 230, 120))
    _put_text(panel, "|error|", (100, 67), 0.36, (80, 170, 255))
    _put_text(panel, f"one action per {1.0 / fps:.2f}s  |  units: dataset action", (174, 67), 0.33, (175, 180, 190))

    error_history = np.abs(pred_history - gt_history)
    graph_width = 220
    graph_height = 196
    for index, name in enumerate(JOINT_NAMES):
        x = 8 + (index % 3) * graph_width
        y = 78 + (index // 3) * graph_height
        cv2.rectangle(panel, (x, y), (x + graph_width - 3, y + graph_height - 3), (42, 46, 56), -1)
        _put_text(panel, name, (x + 5, y + 16), 0.38, (245, 245, 245))
        top_box = (x + 28, y + 23, graph_width - 37, 70)
        error_box = (x + 28, y + 115, graph_width - 37, 44)
        for box in (top_box, error_box):
            bx, by, bw, bh = box
            cv2.rectangle(panel, (bx, by), (bx + bw, by + bh), (20, 23, 29), 1)
        low, high = ranges[index]
        _draw_series(panel, gt_history[:, index], top_box, low, high, (255, 190, 90), total_anchors)
        _draw_series(panel, pred_history[:, index], top_box, low, high, (100, 230, 120), total_anchors)
        _draw_series(panel, error_history[:, index], error_box, 0.0, error_maxes[index], (80, 170, 255), total_anchors)
        _put_text(panel, f"{high:.0f}", (x + 1, y + 31), 0.28, (145, 150, 160))
        _put_text(panel, f"{error_maxes[index]:.0f}", (x + 1, y + 126), 0.28, (145, 150, 160))
        _put_text(panel, "0", (x + 15, y + 163), 0.28, (145, 150, 160))

    return np.concatenate((bgr, panel), axis=1)


def _save_graph(path: Path, gt: np.ndarray, pred: np.ndarray, fps: float, episode_id: int) -> None:
    times = np.arange(len(gt), dtype=np.float32) / fps
    errors = np.abs(pred - gt)
    figure, axes = plt.subplots(2, 3, figsize=(16, 8), sharex=True)
    for index, (axis, name) in enumerate(zip(axes.flat, JOINT_NAMES)):
        axis.plot(times, gt[:, index], label="GT", color="#f2b84b", linewidth=1.5)
        axis.plot(times, pred[:, index], label="Pred", color="#35c46b", linewidth=1.5)
        axis.plot(times, errors[:, index], label="|error|", color="#4da3ff", linewidth=1.2)
        axis.set_title(name)
        axis.set_xlabel("evaluation time [s]")
        axis.set_ylabel("dataset action units")
        axis.grid(alpha=0.25)
    axes.flat[0].legend(loc="upper right", fontsize=8)
    figure.suptitle(f"HAMLET SO-101 open-loop episode {episode_id}: GT, prediction, and absolute error")
    figure.tight_layout()
    figure.savefig(path, dpi=140)
    plt.close(figure)


def main() -> None:
    args = _parse_args()
    repo_root = Path.cwd().resolve()
    hamlet_root = repo_root / "HAMLET-Isaac-GR00T"
    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_absolute():
        checkpoint = (repo_root / checkpoint).resolve()
    dataset_path = Path(args.dataset_path)
    if not dataset_path.is_absolute():
        dataset_path = (repo_root / dataset_path).resolve()
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = (repo_root / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.stride <= 0:
        raise ValueError("--stride must be positive")
    if not checkpoint.is_dir():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    if not dataset_path.is_dir():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    np.random.seed(args.seed)
    os.environ.setdefault("GR00T_INFERENCE_SEED", str(args.seed))
    modality_configs = _load_modality_config(Path(args.modality_config_path), hamlet_root)

    from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.policy.gr00t_policy import Gr00tPolicy

    loader = LeRobotEpisodeLoader(
        dataset_path=dataset_path,
        modality_configs=modality_configs,
        video_backend="pyav",
    )
    episode_indices = _episode_indices(loader, args.episodes, args.eval_fraction)
    horizon = len(modality_configs["action"].delta_indices)
    language_key = modality_configs["language"].modality_keys[0]

    print(f"Checkpoint: {checkpoint}", flush=True)
    print(f"Dataset: {dataset_path}", flush=True)
    print(f"Episodes: {episode_indices} / {len(loader)}", flush=True)
    print(f"HAMLET stride={args.stride}, action_horizon={horizon}", flush=True)

    policy = Gr00tPolicy(
        embodiment_tag=EmbodimentTag.NEW_EMBODIMENT,
        model_path=str(checkpoint),
        device=args.device,
        strict=True,
    )

    errors: dict[str, list[np.ndarray]] = {"arm": [], "gripper": []}
    first_step_errors: dict[str, list[np.ndarray]] = {"arm": [], "gripper": []}
    records: list[dict[str, Any]] = []
    episode_summaries: list[dict[str, Any]] = []
    total_anchors = 0

    for episode_index in episode_indices:
        episode_meta = loader.episodes_metadata[episode_index]
        episode_id = int(episode_meta["episode_index"])
        episode = loader[episode_index]
        max_start = len(episode) - horizon
        starts = list(range(0, max_start + 1, args.stride))
        if args.max_anchors_per_episode > 0:
            starts = starts[: args.max_anchors_per_episode]
        session_id = f"open_loop_episode_{episode_id}"
        print(
            f"Episode index={episode_index} id={episode_id} frames={len(episode)} anchors={len(starts)}",
            flush=True,
        )
        episode_images: list[np.ndarray] = []
        episode_gt_first: list[np.ndarray] = []
        episode_pred_first: list[np.ndarray] = []

        for anchor_number, start in enumerate(starts, start=1):
            row = episode.iloc[start]
            observation = _make_observation(row, language_key)
            predicted, _ = policy.get_action(
                observation,
                options={
                    "session_ids": [session_id],
                    "reset_memory": [start == starts[0]],
                },
            )
            target = _ground_truth(episode, start, horizon)
            record: dict[str, Any] = {
                "episode_index": episode_index,
                "episode_id": episode_id,
                "anchor": start,
            }
            for key in ("arm", "gripper"):
                pred = np.asarray(predicted[key][0], dtype=np.float32)
                gt = target[key]
                if pred.shape != gt.shape:
                    raise ValueError(f"{key} shape mismatch: prediction={pred.shape}, gt={gt.shape}")
                error = pred - gt
                errors[key].append(error)
                first_step_errors[key].append(error[:1])
                record[f"{key}_mae"] = float(np.abs(error).mean())
                record[f"{key}_first_step_mae"] = float(np.abs(error[0]).mean())
            episode_images.append(np.asarray(row["video.top"], dtype=np.uint8))
            episode_gt_first.append(np.concatenate((target["arm"][0], target["gripper"][0])))
            episode_pred_first.append(
                np.concatenate((np.asarray(predicted["arm"][0, 0]), np.asarray(predicted["gripper"][0, 0])))
            )
            records.append(record)
            total_anchors += 1
            if anchor_number == 1 or anchor_number == len(starts) or anchor_number % 5 == 0:
                print(f"  {anchor_number}/{len(starts)} anchors", flush=True)

        episode_gt_array = np.asarray(episode_gt_first, dtype=np.float32)
        episode_pred_array = np.asarray(episode_pred_first, dtype=np.float32)
        episode_ranges, episode_error_maxes = _plot_ranges(episode_gt_array, episode_pred_array)
        video_path = output_dir / f"episode_{episode_id:03d}_gt_pred_error.mp4"
        graph_path = output_dir / f"episode_{episode_id:03d}_action_graph.png"
        fps = max(1.0, float(loader.fps) / args.stride)
        first_frame = _render_frame(
            episode_images[0], episode_id, starts[0], len(starts), episode_gt_array[:1], episode_pred_array[:1],
            episode_ranges, episode_error_maxes, fps
        )
        video = cv2.VideoWriter(
            str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (first_frame.shape[1], first_frame.shape[0])
        )
        if not video.isOpened():
            raise RuntimeError(f"Could not open video writer: {video_path}")
        for frame_index, (image, start) in enumerate(zip(episode_images, starts)):
            frame = _render_frame(
                image, episode_id, start, len(starts), episode_gt_array[: frame_index + 1],
                episode_pred_array[: frame_index + 1], episode_ranges, episode_error_maxes, fps
            )
            video.write(frame)
        video.release()
        _save_graph(graph_path, episode_gt_array, episode_pred_array, fps, episode_id)
        episode_error = episode_pred_array - episode_gt_array
        episode_summaries.append(
            {
                "episode_id": episode_id,
                "anchors": len(starts),
                "video": str(video_path),
                "graph": str(graph_path),
                "first_action_mae": float(np.abs(episode_error).mean()),
                "first_action_rmse": float(np.sqrt(np.square(episode_error).mean())),
            }
        )
        print(f"Saved visualization: {video_path}", flush=True)

    all_error = np.concatenate(errors["arm"] + errors["gripper"], axis=1)
    all_first = np.concatenate(first_step_errors["arm"] + first_step_errors["gripper"], axis=1)
    summary = {
        "checkpoint": str(checkpoint),
        "dataset": str(dataset_path),
        "episodes": episode_indices,
        "total_anchors": total_anchors,
        "stride": args.stride,
        "action_horizon": horizon,
        "open_loop": True,
        "robot_control": False,
        "visualizations": episode_summaries,
        "metrics": {
            "all_horizon": {
                "arm": _metric(errors["arm"]),
                "gripper": _metric(errors["gripper"]),
                "overall": {
                    "mae": float(np.abs(all_error).mean()),
                    "rmse": float(np.sqrt(np.square(all_error).mean())),
                },
            },
            "first_action": {
                "arm": _metric(first_step_errors["arm"]),
                "gripper": _metric(first_step_errors["gripper"]),
                "overall": {
                    "mae": float(np.abs(all_first).mean()),
                    "rmse": float(np.sqrt(np.square(all_first).mean())),
                },
            },
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (output_dir / "predictions.jsonl").open("w") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")

    print(json.dumps(summary["metrics"], indent=2), flush=True)
    print(f"Saved: {output_dir / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
