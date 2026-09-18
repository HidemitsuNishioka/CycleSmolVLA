"""Validate rollout arguments and checkpoint files without opening hardware."""

import json
from pathlib import Path

from safetensors import safe_open

from lerobot.configs import parser
from lerobot.policies.smolvla import modeling_smolvla
from lerobot.rollout import RolloutConfig
from lerobot.scripts import lerobot_rollout  # noqa: F401 (register robot/camera configurations)


@parser.wrap()
def check(cfg: RolloutConfig):
    if cfg.policy.type != "smolvla":
        raise ValueError("This launcher requires a SmolVLA checkpoint")
    checkpoint = Path(cfg.policy.pretrained_path)
    for name in ("policy_preprocessor.json", "policy_postprocessor.json"):
        pipeline = json.loads((checkpoint / name).read_text())
        for step in pipeline["steps"]:
            if step.get("state_file") and not (checkpoint / step["state_file"]).is_file():
                raise FileNotFoundError(checkpoint / step["state_file"])
    with safe_open(checkpoint / "model.safetensors", framework="pt", device="cpu") as weights:
        if cfg.policy.cycle_enabled:
            for key in (
                "model.cycle_history_encoder.input_proj.weight",
                "model.cycle_progress_head.weight",
            ):
                if key not in weights.keys():
                    raise ValueError(f"CycleManip weight missing: {key}")
    print(f"Source: {modeling_smolvla.__file__}")
    print(f"CycleManip: {cfg.policy.cycle_enabled}; device: {cfg.device}")
    print(f"Image offsets: {cfg.policy.image_observation_delta_indices}")
    print(f"State offsets: {cfg.policy.state_observation_delta_indices}")
    print(f"Robot port exists: {Path(cfg.robot.port).exists()} ({cfg.robot.port})")
    for name, camera in cfg.robot.cameras.items():
        device = camera.index_or_path
        if str(device).startswith("/dev/"):
            print(f"Camera {name} exists: {Path(device).exists()} ({device})")
    print("OK: rollout arguments, imports, and checkpoint files validated.")
    print("No hardware connection, model inference, or robot motion was performed.")


if __name__ == "__main__":
    check()
