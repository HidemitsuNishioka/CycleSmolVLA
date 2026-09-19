"""Measure each future action offset with RTC completely disabled (no hardware)."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy


def main():
    """Compare every chunk element to its correctly aligned future ground truth."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dataset-repo-id", default="Hidemitsu-Nishioka/so101_shake_cup_3times")
    parser.add_argument("--episodes", default="8,9")
    parser.add_argument("--stride", type=int, default=30)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.stride < 1:
        raise ValueError("stride must be positive")
    torch.manual_seed(42)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    policy = SmolVLAPolicy.from_pretrained(args.checkpoint).to(args.device).eval()
    policy.config.rtc_config = None
    policy.init_rtc_processor()
    assert not policy._rtc_enabled() and policy.model.rtc_processor is None
    pipeline = json.loads((Path(args.checkpoint) / "policy_preprocessor.json").read_text())
    rename = next(
        s["config"]["rename_map"]
        for s in pipeline["steps"]
        if s["registry_name"] == "rename_observations_processor"
    )
    dataset = LeRobotDataset(args.dataset_repo_id, root=args.dataset_root)
    dataset = LeRobotDataset(
        args.dataset_repo_id,
        root=args.dataset_root,
        delta_timestamps=resolve_delta_timestamps(policy.config, dataset.meta, rename),
    )
    pre, post = make_pre_post_processors(
        policy.config,
        args.checkpoint,
        preprocessor_overrides={"device_processor": {"device": args.device}},
    )
    names = [n.removesuffix(".pos") for n in dataset.meta.features["action"]["names"]]
    errors, active_errors, samples, rows = [], [], [], []
    for episode in [int(e) for e in args.episodes.split(",")]:
        meta = dataset.meta.episodes[episode]
        start, stop = int(meta["dataset_from_index"]), int(meta["dataset_to_index"])
        for index in range(start, stop, args.stride):
            sample = dataset[index]
            batch = pre(
                {
                    "task": sample["task"],
                    "observation.state": sample["observation.state"],
                    **{key: sample[key] for key in dataset.meta.camera_keys},
                    **{
                        key: value
                        for key, value in sample.items()
                        if key.endswith("_is_pad") and key.startswith("observation.")
                    },
                }
            )
            with torch.inference_mode():
                prediction = post(policy.predict_action_chunk(batch))[0].float().cpu().numpy()
            gt = sample["action"].float().numpy()
            error = np.abs(prediction - gt)
            valid = np.arange(policy.config.chunk_size) < stop - index
            error[~valid] = np.nan
            errors.append(error)
            if index - start < 360:
                active_errors.append(error)
            samples.append({"episode": episode, "frame": index - start})
            for offset in np.flatnonzero(valid):
                row = {"episode": episode, "frame": index - start, "offset": int(offset)}
                for j, name in enumerate(names):
                    row[f"gt_{name}"] = float(gt[offset, j])
                    row[f"pred_{name}"] = float(prediction[offset, j])
                rows.append(row)
        print(f"Episode {episode} done; chunks evaluated: {len(errors)}", flush=True)
    errors = np.asarray(errors)
    mean = np.nanmean(errors, axis=(0, 2))
    active_mean = np.nanmean(np.asarray(active_errors), axis=(0, 2))
    summary = {
        "checkpoint": args.checkpoint,
        "rtc_enabled": False,
        "stride": args.stride,
        "sampled_chunks": len(errors),
        "samples": samples,
        "units": "original dataset action units; each offset compared to GT at frame+offset; episode padding excluded",
        "mae_by_offset": mean.tolist(),
        "first_12s_mae_by_offset": active_mean.tolist(),
        "per_joint_mae_by_offset": dict(zip(names, np.nanmean(errors, axis=0).T.tolist(), strict=True)),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (output / "chunks.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(mean, label="All sampled frames")
    ax.plot(active_mean, label="Samples in first 12 seconds")
    ax.set(
        xlabel="Offset inside predicted chunk [frames]",
        ylabel="MAE [dataset units]",
        title="SmolVLA 016000: future-action error, RTC fully disabled",
    )
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "error_by_offset.png", dpi=140)
    plt.close(fig)
    print(json.dumps({i: float(mean[i]) for i in [0, 1, 5, 6, 10, 20, 30, 49]}), flush=True)


if __name__ == "__main__":
    main()
