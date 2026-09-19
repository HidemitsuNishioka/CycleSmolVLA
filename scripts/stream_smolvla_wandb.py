#!/usr/bin/env python3
"""Backfill and follow LeRobot console metrics without restarting training.

Only metrics printed to the log are available, at their printed precision.
This does not upload checkpoints or enable the in-process W&B logger.
"""

import argparse
import ast
import json
import math
import re
import time
from pathlib import Path


ALIASES = {
    "smpl": "samples",
    "ep": "episodes",
    "epch": "epochs",
    "grdn": "grad_norm",
    "smp/s": "samples_per_s",
}
NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"


def parse_metric(line):
    train = re.search(r"\bstep:(\d+[KMBTQ]?)\s+(.*)", line)
    if train:
        values = {}
        for key, value, suffix in re.findall(rf"([\w/]+):({NUMBER})([KMBTQ]?)(?=\s|$)", train[2]):
            number = float(value) * 1000 ** (" KMBTQ".index(suffix) if suffix else 0)
            if math.isfinite(number):
                values[f"train/{ALIASES.get(key, key)}"] = number
        if "train/loss" in values:
            progress = re.search(r"\b(\d+)/\d+\s+\[", line)
            if not progress and not train[1].isdigit():
                raise ValueError("Rounded step count needs an exact tqdm step in the log.")
            step = int(progress[1] if progress else train[1])
            return "train", step, values
    evaluation = re.search(rf"\bstep (\d+): eval_loss=({NUMBER})", line)
    if evaluation:
        return "eval", int(evaluation[1]), {"eval/eval_loss": float(evaluation[2])}
    return None


def read_config(log):
    text = log.read_text(errors="replace")
    start = text.index("{")
    end = text.index("\nINFO ", start)
    # The training config's enum reprs are not Python literals.
    config = ast.literal_eval(re.sub(r"<[^<>]+: ([^<>]+)>", r"\1", text[start:end]))
    config["logging_source"] = "console_log_forwarder"
    config["source_log"] = str(log)
    return config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--project", required=True)
    parser.add_argument("--entity", default=None)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--idle-timeout", type=float, default=1800)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config = read_config(args.log)
    if args.dry_run:
        metrics = [m for line in args.log.read_text().splitlines() if (m := parse_metric(line))]
        print(json.dumps({"dataset": config["dataset"]["root"], "records": len(metrics),
                          "first": metrics[0] if metrics else None,
                          "last": metrics[-1] if metrics else None}, indent=2))
        return

    import wandb

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with wandb.init(project=args.project, entity=args.entity, id=args.run_id,
                    name=args.name, dir=str(args.output_dir), config=config,
                    mode="online", resume="allow", job_type="train_log_forwarding") as run:
        for mode in ("train", "eval"):
            run.define_metric(f"{mode}/step")
            run.define_metric(f"{mode}/*", step_metric=f"{mode}/step")
        last = {mode: int(run.summary.get(f"forwarder/{mode}_step", -1))
                for mode in ("train", "eval")}
        print(f"W&B run: {run.url}", flush=True)
        (args.output_dir / "run.json").write_text(json.dumps({"url": run.url, "path": run.path}))
        activity = time.monotonic()
        with args.log.open(errors="replace") as source:
            while True:
                position = source.tell()
                line = source.readline()
                if not line or not line.endswith("\n"):
                    source.seek(position)
                    if time.monotonic() - activity > args.idle_timeout:
                        raise TimeoutError("Training log stopped updating; check training status.")
                    time.sleep(2)
                    continue
                activity = time.monotonic()
                metric = parse_metric(line)
                if metric:
                    mode, step, values = metric
                    if step > last[mode]:
                        run.log({**values, f"{mode}/step": step, f"forwarder/{mode}_step": step})
                        last[mode] = step
                        print(f"Forwarded {mode} step {step}", flush=True)
                if "End of training" in line:
                    return
                if "Traceback (most recent call last)" in line:
                    raise RuntimeError("Training failed; inspect the training log.")


if __name__ == "__main__":
    main()
