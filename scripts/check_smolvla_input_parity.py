"""Replay a recorded episode through RTC input handling without opening hardware.

Compare the actual checkpoint preprocessor inputs with LeRobotDataset's offline
temporal windows. Optionally compare action chunks using identical diffusion noise.
"""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.rtc.configuration_rtc import RTCConfig
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.utils import prepare_observation_for_inference
from lerobot.rollout.inference.observation_history import ObservationHistory
from lerobot.rollout.inference.rtc import RTCInferenceEngine


def main():
    """Compare recorded observations through the offline and live input paths."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--dataset-repo-id", default="Hidemitsu-Nishioka/so101_shake_cup_3times")
    parser.add_argument("--episode", type=int, default=8)
    parser.add_argument("--frames", default="0,120,299,350")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--compare-actions", action="store_true")
    parser.add_argument("--output", default="outputs/eval/smolvla_input_parity.json")
    args = parser.parse_args()
    frames = sorted({int(value) for value in args.frames.split(",")})
    config = SmolVLAConfig.from_pretrained(args.checkpoint)
    config.device = args.device
    if not config.cycle_enabled:
        raise ValueError("This check requires a CycleManip checkpoint")
    pipeline = json.loads((Path(args.checkpoint) / "policy_preprocessor.json").read_text())
    rename_map = next(
        step["config"]["rename_map"]
        for step in pipeline["steps"]
        if step["registry_name"] == "rename_observations_processor"
    )
    dataset = LeRobotDataset(args.dataset_repo_id, root=args.dataset_root, video_backend="pyav")
    offline = LeRobotDataset(
        args.dataset_repo_id,
        root=args.dataset_root,
        video_backend="pyav",
        delta_timestamps=resolve_delta_timestamps(config, dataset.meta, rename_map),
    )
    episode_indices = np.flatnonzero(np.asarray(dataset.hf_dataset["episode_index"]) == args.episode)
    if not len(episode_indices) or min(frames) < 0 or max(frames) >= len(episode_indices):
        raise ValueError("Episode/frame selection is outside the dataset")
    preprocessor, postprocessor = make_pre_post_processors(
        config,
        args.checkpoint,
        preprocessor_overrides={"device_processor": {"device": args.device}},
    )
    policy = (
        SmolVLAPolicy.from_pretrained(args.checkpoint, config=config).to(args.device).eval()
        if args.compare_actions
        else SimpleNamespace(config=config)
    )
    first = dataset[int(episode_indices[0])]
    engine = RTCInferenceEngine(
        policy=policy,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        robot_wrapper=SimpleNamespace(robot_type=dataset.meta.robot_type),
        rtc_config=RTCConfig(),
        hw_features=dataset.meta.features,
        task=first["task"],
        fps=dataset.meta.fps,
        device=args.device,
    )
    report = {
        "checkpoint": args.checkpoint,
        "episode": args.episode,
        "task": first["task"],
        "fps": dataset.meta.fps,
        "state_history": config.cycle_history_size,
        "image_offsets": config.image_observation_delta_indices,
        "checks": [],
    }
    state_names = dataset.meta.features["observation.state"]["names"]
    for local_index in range(max(frames) + 1):
        index = int(episode_indices[local_index])
        sample = first if local_index == 0 else dataset[index]
        raw = dict(zip(state_names, sample["observation.state"].tolist(), strict=True))
        for key in dataset.meta.camera_keys:
            raw[key.removeprefix("observation.images.")] = (
                sample[key].permute(1, 2, 0).mul(255).round().byte().numpy()
            )
        engine.notify_observation(raw)
        if local_index not in frames:
            continue
        live = ObservationHistory.stack(engine._observation_history.snapshot())
        live = prepare_observation_for_inference(
            live, torch.device(args.device), first["task"], dataset.meta.robot_type
        )
        live = preprocessor(live)
        offline_sample = offline[index]
        expected = preprocessor(
            {
                "observation.state": offline_sample["observation.state"],
                **{key: offline_sample[key] for key in dataset.meta.camera_keys},
                **{
                    key: value
                    for key, value in offline_sample.items()
                    if key.endswith("_is_pad") and key.startswith("observation.")
                },
                "task": offline_sample["task"],
            }
        )
        check = {"frame": local_index, "inputs": {}}
        for key in expected:
            if not isinstance(expected[key], torch.Tensor):
                continue
            if expected[key].ndim + 1 == live[key].ndim:
                expected[key] = expected[key].unsqueeze(0)
            # Dataset pixels are scaled on CPU; live uint8 pixels are scaled on
            # the selected device. CUDA /255 can differ by one float32 ULP.
            tolerance = 1e-7 if ".images." in key else 0
            torch.testing.assert_close(live[key], expected[key], rtol=0, atol=tolerance)
            difference = (live[key].float() - expected[key].float()).abs().max()
            check["inputs"][key] = {
                "shape": list(live[key].shape),
                "max_abs_diff": float(difference),
            }
        if args.compare_actions:
            noise = torch.randn(
                (1, config.chunk_size, config.max_action_dim),
                device=args.device,
                generator=torch.Generator(device=args.device).manual_seed(42),
            )
            with torch.inference_mode():
                expected_actions = postprocessor(
                    policy.predict_action_chunk(dict(expected), noise=noise.clone())
                )
                live_actions = postprocessor(policy.predict_action_chunk(dict(live), noise=noise.clone()))
            if not torch.isfinite(live_actions).all() or not torch.isfinite(expected_actions).all():
                raise AssertionError("Non-finite policy action")
            # The vision encoder uses reduced precision: one-ULP pixel differences
            # can cross its rounding boundaries. Report the resulting action
            # difference instead of claiming bit-identical model outputs.
            check["action_max_abs_diff"] = float((live_actions - expected_actions).abs().max())
            check["action_mean_abs_diff"] = float((live_actions - expected_actions).abs().mean())
        report["checks"].append(check)
        print(json.dumps(check), flush=True)
    report["passed"] = True
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"PASS: offline/live inputs match. Report: {output}")
    print("No robot or camera device was opened; no motor commands were sent.")


if __name__ == "__main__":
    main()
