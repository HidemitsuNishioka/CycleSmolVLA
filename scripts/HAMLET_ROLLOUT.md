# Local HAMLET rollout

Run from the SO101 project directory. The model uses `arm` (five joint angles
in degrees), `gripper` (0–100), and a `top` RGB camera. Match the training
camera pose, robot calibration and task (`Shake the cup.` for this checkpoint).

```bash
# Read live observations, print predicted/clipped targets; no motor writes.
HAMLET_MAX_STEPS=32 ./scripts/rollout_hamlet_so101.sh

# Send up to 32 motor commands after typing MOVE.
HAMLET_EXECUTE=1 HAMLET_MAX_STEPS=32 ./scripts/rollout_hamlet_so101.sh
```

`HAMLET_MAX_STEPS` counts individual commands, not inference calls or chunks.
The default remains one command. The action rate is 30 Hz; the model produces
16 actions and its memory advances every 16 frames (about 0.533 seconds).
Warm-up runs before confirmation; fresh observations and a reset history are
used after confirmation. Exiting an actuated run disables torque.

Inference is synchronous: the previous goal is held during inference. Targets
whose frame interval has already elapsed are skipped. A chunk that takes at
least 0.533 seconds to infer is rejected. This does not guarantee uninterrupted
30 Hz commands through inference; it preserves the observation/action timeline
without sending an expired backlog. Observation delays exceeding one frame
beyond the memory stride reset the history. Use the printed inference latency
and chunk indices to check how much of each chunk is actually executed.

`predicted` is the decoded absolute model target, `target` is clipped relative
to measured position, and `sent` is the target returned by the motor API.
The default clip remains ±5 degrees per arm joint and ±5 units for the gripper;
it changes the predicted trajectory and is not a motor velocity limit.

Hardware-free regression checks:

```bash
PYTHONPATH=HAMLET-Isaac-GR00T:.lerobot-src/src \
  HAMLET-Isaac-GR00T/.venv/bin/python scripts/test_hamlet_rollout.py
```

The open-loop OpenCV decoder now supplies RGB. Old results generated with BGR
input must be distinguished from results of the corrected evaluator.
