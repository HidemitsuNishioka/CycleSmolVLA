"""Replay recorded observations through the real RTC engine without hardware.

Observations remain ground truth, so this measures open-loop action prediction,
not physical task success. Decoding/rendering happen outside the timed replay.
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch
import transformers
from evaluate_smolvla_open_loop_video import JOINT_NAMES, put_text, render_frame, save_episode_graph

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.rtc.configuration_rtc import RTCConfig
from lerobot.policies.smolvla import modeling_smolvla
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.rollout.inference.rtc import RTCInferenceEngine
from lerobot.rollout.inference.sync import SyncInferenceEngine
from lerobot.utils.feature_utils import build_dataset_frame


def metrics(gt, pred):
    """Summarize errors in the dataset's original joint units."""
    error = np.asarray(pred) - np.asarray(gt)
    return {
        "mae": float(np.abs(error).mean()),
        "rmse": float(np.sqrt(np.square(error).mean())),
        "per_joint_mae": dict(zip(JOINT_NAMES, np.abs(error).mean(axis=0).tolist(), strict=True)),
    }


def main():
    """Run timed, ground-truth-fed RTC evaluation and render its results."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--dataset-repo-id", default="Hidemitsu-Nishioka/so101_shake_cup_3times")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--episodes", default="8,9")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-relative-target", type=float, default=10.0)
    parser.add_argument("--guidance-weight", type=float, default=10.0)
    parser.add_argument("--inference-type", choices=("rtc", "sync"), default="rtc")
    parser.add_argument("--n-action-steps", type=int, default=None)
    parser.add_argument(
        "--disable-guidance",
        action="store_true",
        help="Bypass the policy's RTC processor entirely; keep async queue/delay handling for comparison.",
    )
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    dataset = LeRobotDataset(args.dataset_repo_id, root=args.dataset_root)
    policy = SmolVLAPolicy.from_pretrained(args.checkpoint).to(args.device).eval()
    policy.config.device = args.device
    rtc_config = RTCConfig(mode="guided", execution_horizon=10, max_guidance_weight=args.guidance_weight)
    guidance_enabled = args.inference_type == "rtc" and not args.disable_guidance
    policy.config.rtc_config = rtc_config if guidance_enabled else None
    policy.init_rtc_processor()
    if args.n_action_steps is not None:
        if not 1 <= args.n_action_steps <= policy.config.chunk_size:
            raise ValueError("--n-action-steps must be between 1 and chunk_size")
        policy.config.n_action_steps = args.n_action_steps
        policy.reset()
    if not guidance_enabled:
        assert not policy._rtc_enabled() and not policy.model._rtc_enabled()
        assert policy.rtc_processor is None and policy.model.rtc_processor is None
        print("RTC guidance fully bypassed: no processor, no correction, no guidance gradients", flush=True)
    preprocessor, postprocessor = make_pre_post_processors(
        policy.config,
        args.checkpoint,
        preprocessor_overrides={"device_processor": {"device": args.device}},
    )
    summaries, all_gt, all_pred, all_safe = [], [], [], []
    state_names = dataset.meta.features["observation.state"]["names"]
    action_names = dataset.meta.features["action"]["names"]
    if state_names != action_names or [name.removesuffix(".pos") for name in action_names] != JOINT_NAMES:
        raise ValueError("Joint order differs from the SO101 report renderer")
    for episode in [int(value) for value in args.episodes.split(",")]:
        meta = dataset.meta.episodes[episode]
        samples = []
        print(f"Preloading episode {episode} outside timed replay", flush=True)
        for index in range(int(meta["dataset_from_index"]), int(meta["dataset_to_index"])):
            sample = dataset[index]
            state = sample["observation.state"].numpy().copy()
            raw = dict(zip(state_names, state.tolist(), strict=True))
            for key in dataset.meta.camera_keys:
                raw[key.removeprefix("observation.images.")] = (
                    sample[key].permute(1, 2, 0).mul(255).round().byte().numpy()
                )
            samples.append((raw, state, sample["action"].numpy().copy(), sample["task"]))

        if args.inference_type == "rtc":
            engine = RTCInferenceEngine(
                policy=policy,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                robot_wrapper=SimpleNamespace(robot_type=dataset.meta.robot_type),
                rtc_config=rtc_config,
                hw_features=dataset.meta.features,
                task=samples[0][3],
                fps=dataset.fps,
                device=args.device,
            )
        else:
            engine = SyncInferenceEngine(
                policy=policy,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                dataset_features=dataset.meta.features,
                ordered_action_keys=action_names,
                task=samples[0][3],
                device=args.device,
                robot_type=dataset.meta.robot_type,
            )
        engine.reset()
        rows, predictions, safe_predictions, available = [], [], [], []
        last_command = samples[0][1].copy()
        engine.start()
        engine.resume()
        start = deadline = time.perf_counter()
        print(f"Replaying episode {episode}: {len(samples)} ticks at {dataset.fps} Hz", flush=True)
        try:
            for frame, (raw, state, gt, _) in enumerate(samples):
                tick_time = time.perf_counter() - start
                engine.notify_observation(raw)
                obs_frame = (
                    build_dataset_frame(dataset.meta.features, raw, prefix="observation")
                    if args.inference_type == "sync"
                    else None
                )
                action = engine.get_action(obs_frame)
                if engine.failed:
                    raise RuntimeError(engine.failure_traceback)
                valid = action is not None
                if valid:
                    pred = action.detach().float().cpu().numpy()
                    if not np.isfinite(pred).all():
                        raise ValueError("RTC returned non-finite actions")
                    safe = state + np.clip(pred - state, -args.max_relative_target, args.max_relative_target)
                    last_command = safe.copy()
                else:
                    # Only for plotting held commands; excluded from prediction metrics.
                    pred = safe = last_command.copy()
                predictions.append(pred)
                safe_predictions.append(safe)
                available.append(valid)
                row = {
                    "episode": episode,
                    "frame": frame,
                    "timestamp": frame / dataset.fps,
                    "wall_time": tick_time,
                    "action_available": valid,
                    "queue_size": (
                        engine.action_queue.qsize()
                        if args.inference_type == "rtc"
                        else len(policy._queues["action"])
                    ),
                }
                for j, name in enumerate(JOINT_NAMES):
                    row[f"gt_{name}"] = float(gt[j])
                    row[f"pred_{name}"] = float(pred[j]) if valid else ""
                    row[f"safe_{name}"] = float(safe[j]) if valid else ""
                rows.append(row)
                # A slow sync inference must not trigger a burst of catch-up
                # commands. Keep each recorded frame and report the achieved Hz.
                deadline += 1 / dataset.fps
                if args.inference_type == "sync":
                    deadline = max(deadline, time.perf_counter())
                time.sleep(max(0, deadline - time.perf_counter()))
                if (frame + 1) % 100 == 0:
                    print(f"  episode {episode}: {frame + 1}/{len(samples)}", flush=True)
        finally:
            elapsed = time.perf_counter() - start
            engine.stop()
        if engine.failed:
            raise RuntimeError(engine.failure_traceback)
        valid = np.asarray(available)
        if not valid.any():
            raise RuntimeError("RTC produced no actions")
        gt = np.asarray([sample[2] for sample in samples])
        pred, safe = np.asarray(predictions), np.asarray(safe_predictions)
        all_gt.extend(gt[valid])
        all_pred.extend(pred[valid])
        all_safe.extend(safe[valid])
        csv_path = output / f"episode_{episode:03d}_actions.csv"
        with csv_path.open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        graph_path = output / f"episode_{episode:03d}_angle_graph.png"
        save_episode_graph(graph_path, gt, pred, dataset.fps, episode)
        video_path = output / f"episode_{episode:03d}_gt_pred_error.mp4"
        camera_name = dataset.meta.camera_keys[0].removeprefix("observation.images.")
        h, w = samples[0][0][camera_name].shape[:2]
        video = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), dataset.fps, (w + 650, h))
        if not video.isOpened():
            raise RuntimeError(f"Cannot create {video_path}")
        try:
            for index, sample in enumerate(samples):
                frame = render_frame(
                    sample[0][camera_name],
                    episode,
                    index,
                    len(samples),
                    gt[: index + 1],
                    pred[: index + 1],
                    np.abs(pred[: index + 1] - gt[: index + 1]),
                    dataset.fps,
                )
                put_text(frame, f"{args.inference_type.upper()} replay / GT observations", (10, 22))
                if not valid[index]:
                    put_text(frame, "WAIT: no action; showing held command", (10, 44), color=(80, 180, 255))
                video.write(frame)
        finally:
            video.release()
        summary = {
            "episode": episode,
            "frames": len(samples),
            "predicted_frames": int(valid.sum()),
            "startup_wait_frames": int(np.flatnonzero(valid)[0]),
            "empty_queue_frames_after_start": int((~valid[np.flatnonzero(valid)[0] :]).sum()),
            "elapsed_seconds": elapsed,
            "replay_fps": len(samples) / elapsed,
            **metrics(gt[valid], pred[valid]),
            "after_clamp": metrics(gt[valid], safe[valid]),
            "clamped_frames": int(np.any(np.abs(safe[valid] - pred[valid]) > 1e-4, axis=1).sum()),
            "video": str(video_path),
            "graph": str(graph_path),
            "csv": str(csv_path),
        }
        summaries.append(summary)
        print(json.dumps(summary), flush=True)
    summary = {
        "mode": f"{args.inference_type}_ground_truth_replay",
        "n_action_steps": policy.config.n_action_steps,
        "sync_use_amp": policy.config.use_amp if args.inference_type == "sync" else False,
        "checkpoint": args.checkpoint,
        "dataset_root": args.dataset_root,
        "seed": args.seed,
        "fps": dataset.fps,
        "rtc": {
            "queue_enabled": args.inference_type == "rtc",
            "guidance_enabled": guidance_enabled,
            "mode": "guided",
            "execution_horizon": 10,
            "max_guidance_weight": args.guidance_weight,
            "queue_threshold": 30,
        },
        "max_relative_target": args.max_relative_target,
        "units": "original dataset units; no-action ticks excluded; clamp uses recorded current state",
        "limitation": "Ground-truth observations at every tick; no physical robot feedback or task-success test",
        "runtime": {
            "python": sys.executable,
            "python_version": sys.version,
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "policy_source": modeling_smolvla.__file__,
            "device": args.device,
        },
        **metrics(all_gt, all_pred),
        "after_clamp": metrics(all_gt, all_safe),
        "episode_summaries": summaries,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Saved {output / 'summary.json'}; no hardware opened", flush=True)


if __name__ == "__main__":
    main()
