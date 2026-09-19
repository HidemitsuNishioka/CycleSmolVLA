# CycleSmolVLA

**English** | [日本語](README.ja.md)

**Bringing CycleManip's ideas to SmolVLA for learning and executing repetitive tasks on SO-101.**

Inspired by [CycleManip](https://isee-laboratory.github.io/CycleManip/), this project adds a history encoder and a progress classification head to LeRobot's SmolVLA. It combines sparse image history with dense state history and uses progress prediction as an auxiliary learning task. Docker scripts cover data collection, training, offline evaluation, and real-robot rollout on SO-101.

This is an unofficial attempt to adapt CycleManip to SmolVLA. It includes design choices specific to SmolVLA and SO-101 and is not intended as an exact reproduction of the paper's main architecture or experimental results.

## Demo and example training dataset

The implementation has been tested on a physical SO-101. A video of the robot running the policy is available in this post:

**[▶ Watch the real-robot demo on X](https://x.com/mocha18231562/status/2101141258152677762?s=20)**

An example training dataset covers shaking a cup three times.

| Item | Details |
| --- | --- |
| Dataset | [Hidemitsu-Nishioka/so101_shake_cup_3times on Hugging Face](https://huggingface.co/datasets/Hidemitsu-Nishioka/so101_shake_cup_3times) |
| Instruction | `Shake the cup three times.` |
| Format and size | LeRobotDataset v3; 10 episodes, 5,990 frames, 30 fps, based on local metadata |
| Observations | Top camera images and a 6-dimensional joint/gripper state |
| Training and validation | Episodes 0–7 for training; 8–9 for validation |
| Training configuration | [config/smolvla_shake_cup_3times_300.env](config/smolvla_shake_cup_3times_300.env) |

The video is a demonstration, not a benchmark reporting aggregate task success rates or cycle count errors.

## Relationship to CycleManip

This implementation adopts six image frames, dense state history, 10-bin progress classification, and a progress loss weight of 0.1. The table compares the main architecture in the [CycleManip paper, v2, Sections 3.2–3.4 and 5.2](https://arxiv.org/html/2512.01022v2) with this repository's `cycle300` configuration.

| Component | CycleManip main architecture | CycleSmolVLA (this implementation) |
| --- | --- | --- |
| State history representation | Differences in end-effector position and orientation | Normalized joint/gripper states, without conversion to end-effector poses or temporal differences |
| State history horizon | All history from the start of the episode to the current frame | The latest 300 frames, including the current frame (about 10 seconds at 30 fps); the base configuration uses 32 frames |
| Image sampling | Binary sampling over the episode history combined with exponential sampling near the current frame | Six frames at fixed offsets from the current frame: `[-299, -150, -75, -2, -1, 0]` |
| History encoding | Global features from a Transformer's CLS token plus MLP features from recent frames | Mean features and the last valid frame's features from a two-layer Transformer, fused into one conditioning token |
| Progress prediction input | Fused visual and state history features | State history features alone, passed to the progress head before fusion with images or language |
| Action generation | Diffusion model with DDIM; action horizon of 8 | SmolVLA flow matching; action chunks of 50 |

The [paper's π₀ integration experiment (Section 5.7)](https://arxiv.org/html/2512.01022v2#S5.SS7) uses only the current image and the full joint history. This implementation's use of joint history is also close to that approach, but the combination of six images, a fixed state history window, and SmolVLA is specific to this project.

Progress targets are calculated from each frame's relative position within its episode and discretized into 10 classes; they do not directly label the number of completed cycles. History older than 300 frames falls outside the input window, and the auxiliary progress loss does not directly supervise visual/state fusion. These design differences matter when considering cycle counting in longer tasks.

Real-robot rollout also uses RTC for asynchronous inference, separating model inference from motor control. This is the execution setup used in this environment.

Implementation: [history sampling and encoding](.lerobot-src/src/lerobot/policies/smolvla/cyclemanip.py), [SmolVLA integration and progress loss](.lerobot-src/src/lerobot/policies/smolvla/modeling_smolvla.py), and [RTC rollout](scripts/rollout_cyclemanip.sh).

## Current support

| Feature | Status |
| --- | --- |
| Device detection, setup, and calibration through LeRobot | Supported workflow |
| Teleoperation and data collection | Supported workflow |
| SmolVLA training, offline evaluation, and real-robot rollout | Currently tested workflow |

This repository focuses on the SmolVLA/CycleManip workflow for SO-101.

## Requirements

- Ubuntu
- Docker Compose v2
- NVIDIA Container Toolkit and a CUDA-capable GPU
- SO-101 Leader / Follower
- USB camera (the current configuration uses one)
- Git

The Docker image is `huggingface/lerobot-gpu:latest`. LeRobot does not need to be installed on the host.

## Clone

```bash
git clone <your-repository-url>
cd SO101
```

## Initial setup

### 1. Check Docker and the GPU

```bash
nvidia-smi
docker compose version
./scripts/bootstrap.sh
```

`bootstrap.sh` pulls the official LeRobot GPU image and checks that LeRobot and the GPU are accessible inside the container.

### 2. Create a local configuration

```bash
cp -n config/dataset.env config/dataset.local.env
```

`config/dataset.local.env` is excluded by `.gitignore`. Use it for machine-specific values such as dataset names, output paths, training steps, and W&B settings.

Keep your W&B API key only in `config/dataset.local.env`. If `WANDB_ENABLE=true` but the key is empty, the wrapper automatically disables online logging.

### 3. Configure devices

Connect the SO-101 and camera, then detect them:

```bash
./scripts/detect_devices.sh
```

Use the detected device paths to update [config/robot.env](config/robot.env).

```bash
FOLLOWER_PORT="/dev/serial/by-id/<follower-id>"
LEADER_PORT="/dev/serial/by-id/<leader-id>"
FOLLOWER_ID="so101_follower"
LEADER_ID="so101_leader"
CAMERA_TOP_DEVICE="/dev/video1"
CAMERA_WRIST_DEVICE=""
CAMERA_WIDTH="640"
CAMERA_HEIGHT="480"
CAMERA_FPS="30"
DISPLAY_DATA="false"
```

The current SmolVLA configuration renames the dataset's `observation.images.top` to `camera1`. If you change camera feature names, update `SMOLVLA_RENAME_MAP` accordingly.

## Prepare the SO-101

Run these commands in order. Keep the area around the arm clear when operating the motors.

```bash
# Initial motor setup (first use or when needed)
./scripts/setup_motors.sh

# Calibrate the Leader / Follower
./scripts/calibrate.sh

# Check that the Follower follows the Leader
./scripts/teleop.sh
```

## Record a dataset

Check these values in `config/dataset.local.env` before recording:

```bash
DATASET_REPO_ID="your-hf-user/so101_task"
DATASET_ROOT="data/so101_task"
DATASET_TASK="Describe the task here."
DATASET_NUM_EPISODES="10"
DATASET_FPS="30"
```

The dataset is automatically uploaded to Hugging Face after recording. Set a token with write access in `config/dataset.local.env`:

```bash
HF_TOKEN="hf_..."
```

If `HF_TOKEN` is not set, you can use existing Hugging Face credentials by signing in with `hf auth login` beforehand. Set `DATASET_PUSH_TO_HUB="false"` to disable automatic uploads. Datasets are private by default (`DATASET_PRIVATE="true"`).

```bash
./scripts/record.sh
```

Data is saved under `data/` in LeRobotDataset v3 format. The `data/` directory is not tracked by Git.

Inspect the recording:

```bash
./scripts/inspect_dataset.sh
./scripts/visualize_dataset.sh 0
```

### Upload an existing dataset

Upload a LeRobot dataset already stored in `data/` with the following command. By default, it uses `DATASET_ROOT` and `DATASET_REPO_ID` from `config/dataset.local.env`.

```bash
./scripts/upload_dataset.sh
```

To specify a different dataset:

```bash
./scripts/upload_dataset.sh data/another_dataset your-hf-user/another_dataset
```

Metadata, Parquet files, videos, and the dataset card are uploaded together. Set `HF_TOKEN` with Hugging Face write access in `config/dataset.local.env`, or run `hf auth login` beforehand.

## Train SmolVLA

To train SmolVLA with CycleManip-style history on the local `data/so101_5eps` dataset:

```bash
SO101_DATASET_CONFIG=config/smolvla_so101_5eps_cycle300.env ./scripts/train_smolvla.sh
```

Despite the directory name, this dataset contains 10 episodes and 4,499 frames for the task `Pick up the object and place it in the target area`. Episodes 0–7 are used for training and 8–9 for validation. Training runs for 20,000 steps with a state history of 300 frames, an image history of six frames, action chunks of 50, and a batch size of 4. Validation uses up to 100 samples every 2,000 steps. Checkpoints are saved every 5,000 steps to reduce disk usage. Outputs go to `outputs/train/smolvla_so101_5eps_cycle300`. Online W&B logging is enabled for the `lerobot-so101` project; set `WANDB_API_KEY` in `config/dataset.local.env` (the shared script disables W&B if the key is missing). Hub uploads are disabled. Do not run multiple jobs with the same output directory.

To train on `Hidemitsu-Nishioka/so101_shake_cup_3times` with 300 frames of joint history:

```bash
SO101_DATASET_CONFIG=config/smolvla_shake_cup_3times_300.env ./scripts/train_smolvla.sh
```

In this workspace, the dataset has been downloaded to `data/so101_shake_cup_3times`. The input contains 300 consecutive frames at 30 fps, including the current frame (offsets −299 to 0); missing history at the start of an episode is masked. Image history contains six frames, and action chunks contain 50 frames. Training uses episodes 0–7 for 20,000 steps, with validation loss calculated every 2,000 steps on up to 100 samples from episodes 8–9. Outputs go to `outputs/train/smolvla_shake_cup_3times_cycle300`. Avoid concurrent runs with the same output directory.

The standard training command has also been tested:

```bash
./scripts/train_smolvla.sh
```

CycleManip-inspired history processing is enabled by default. At each timestep, the base configuration loads six image frames and a 32-frame joint state history, and adds a 10-class progress loss with weight 0.1 to the action loss. To disable it, set the following in `config/dataset.local.env`:

```bash
SMOLVLA_CYCLE_ENABLED="false"
```

Sparse history for expensive visual observations, dense history for inexpensive state observations, and auxiliary progress classification are adapted to LeRobot's temporal data loader and SmolVLA's conditioning tokens. See “Relationship to CycleManip” above for the specific design differences.

Default settings:

- Base model: `lerobot/smolvla_base`
- Batch size: `4`
- Steps: `20000`
- GPU: `cuda`
- Output directory: `outputs/train/smolvla_so101_wrist_20k_v1/`

For a short smoke test, add or override these values in `config/dataset.local.env`:

```bash
SMOLVLA_STEPS="100"
SMOLVLA_OUTPUT_DIR="outputs/train/smolvla_smoke"
SMOLVLA_JOB_NAME="smolvla_smoke"
```

## Offline SmolVLA evaluation

Evaluation uses the final `DATASET_EVAL_SPLIT` fraction of the dataset (20% by default). No actions are sent to the robot. The evaluation launcher selects the same Python environment as real-robot rollout and prioritizes `.lerobot-src/src`.

```bash
./scripts/evaluate_smolvla_open_loop.sh
```

Outputs go to `outputs/eval/smolvla_so101_wrist_20k_open_loop/`. For a shorter test, limit the episodes or frame count:

```bash
SMOLVLA_OPEN_LOOP_EPISODES=8 \
SMOLVLA_OPEN_LOOP_MAX_FRAMES=100 \
./scripts/evaluate_smolvla_open_loop.sh
```

To evaluate all of episodes 8 and 9 using `cycle300/016000`, run the following. The default `fresh` mode predicts a new action chunk for every frame and compares its first action. It does not use the real-robot RTC queue.

```bash
SO101_DATASET_CONFIG=config/smolvla_shake_cup_3times_300.env \
SMOLVLA_OPEN_LOOP_EPISODES=8,9 \
SMOLVLA_OPEN_LOOP_OUTPUT_DIR=outputs/eval/smolvla_cycle300_016000_ep8_ep9/fresh \
./scripts/evaluate_smolvla_open_loop.sh
```

To evaluate the same RTC engine used on the robot, run the following in LeRobot's Python environment. Recorded observations are supplied at 30 Hz, and outputs are compared after passing through actual asynchronous inference and the action queue. Video processing before and after inference is excluded from the measurement interval. Because observations come from recorded data, this does not evaluate physical tracking or task success. Frames waiting for an output are excluded from error aggregation; waiting time, queue underruns, and errors before and after the ±10 limit are also saved.

```bash
python scripts/evaluate_smolvla_rtc_open_loop.py \
  --checkpoint outputs/train/smolvla_shake_cup_3times_cycle300/016000 \
  --dataset-root data/so101_shake_cup_3times \
  --episodes 8,9 --device cuda --max-relative-target 10 \
  --output-dir outputs/eval/smolvla_cycle300_016000_ep8_ep9/rtc
```

Add `--disable-guidance` to bypass RTC correction inside the model entirely while retaining the asynchronous queue and delay compensation. With `--inference-type sync --n-action-steps 1`, the script uses the real-robot synchronous inference engine, disables the RTC queue and delay compensation, and predicts again at each step. Setting `--n-action-steps 50` instead executes 50 actions sequentially without correction. If synchronous inference cannot keep up with 30 Hz, recorded frames are still processed without skipping, and the measured processing rate is written to `summary.json`.

To compare the accuracy of the first and subsequent actions in a chunk, run `scripts/diagnose_smolvla_action_chunks.py` with `--checkpoint`, `--dataset-root`, and `--output-dir`. By default, it samples episodes 8 and 9 every 30 frames and compares each prediction with ground truth at the corresponding future timestep. Padding at episode boundaries is excluded.

## Real-robot rollout

Launch SmolVLA with CycleManip-style history using `scripts/rollout_cyclemanip.sh`. Inside Docker, it uses the training environment's `/workspace/.venv/bin/python` (or `python` if unavailable) and prioritizes the modified code in `.lerobot-src/src`. RTC is enabled by default and predicts action chunks in the background. This separates inference latency from the robot control loop so it can maintain a 30 Hz command stream.

While the terminal running the rollout has focus, press **R** to clear observation history, the action queue, and actions being interpolated, then resume inference from the current pose. Enter is not required. Reset waits for any in-flight RTC inference to finish so stale predictions do not enter the resumed run. The duration limit also restarts from the time you resume with R. The physical arm and cup positions are unchanged, so use this when they are positioned to start another run. Stop with Ctrl+C as usual. The launcher enables this shortcut with `--strategy.keyboard_restart=true`; input is accepted only from the running terminal. It is disabled in the `--interactive=true` command input mode.

For a comparison that disables RTC completely and executes only the first action of each newly predicted chunk, use the following command. Setting `sync` alone retains the checkpoint's default 50-action queue, so the one-action setting is also required. In this mode, the control rate is limited by model inference speed.

```bash
SO101_DATASET_CONFIG=config/smolvla_shake_cup_3times_300.env \
CYCLEMANIP_INFERENCE_TYPE=sync CYCLEMANIP_N_ACTION_STEPS=1 \
./scripts/rollout_cyclemanip.sh
```

Connect the Follower and camera, then set `FOLLOWER_PORT`, `FOLLOWER_ID`, and `CAMERA_TOP_DEVICE` in `config/robot.env` to match your hardware. Use a Follower calibrated under that same ID, with the same camera placement and joint units used during training.

```bash
ls -l /dev/serial/by-id/ /dev/ttyACM* /dev/video*
```

Example using a CycleManip-style checkpoint saved in this workspace:

```bash
SO101_DATASET_CONFIG=config/smolvla_shake_cup_3times_300.env \
CYCLEMANIP_CHECKPOINT="outputs/train/smolvla_shake_cup_3times_cycle300/016000" \
CYCLEMANIP_DURATION="10" \
./scripts/rollout_cyclemanip.sh
```

Whether CycleManip-style processing is enabled, the history length, and the action chunk length are read from the checkpoint. `SMOLVLA_CYCLE_ENABLED` is a training setting, not a rollout switch. `last` follows the latest saved checkpoint; use a specific directory such as `006000` to select a fixed model.

The CycleManip-style model implementation matches the training code at `KainaJetson:~/SO101/`. The history Transformer uses four attention heads and PreNorm, and the fusion layers use SiLU. These choices are not reflected in weight shapes, so loading weights successfully into a different implementation does not ensure identical predictions.

History is recorded at every control cycle, not only when inference runs. With RTC and `cycle300`, the model receives 300 joint state frames at 30 Hz and six images at offsets `[-299, -150, -75, -2, -1, 0]`. Missing history at startup is filled with the first frame and invalidated using the same `_is_pad` masks as training. A rollout reset also clears history. RTC stores images on the CPU and transfers only the six selected frames to the GPU during inference. With synchronous one-action inference, observation acquisition is slower too, so 300 frames do not necessarily correspond to 10 seconds.

Evaluations produced before the implementation alignment on 2026-09-19 used a model structure and image history that differed from training. Do not use those results to judge checkpoint training quality. Diagnostics and evaluations after alignment are stored in `outputs/eval/learning_cause_20260919/`.

To replay recorded data in the real-robot input format and verify input parity with offline evaluation, run the following in LeRobot's Python environment. It does not connect to a robot or camera. `--compare-actions` also records action output differences using the same noise. CPU/GPU image conversion can introduce rounding differences, so this check does not guarantee exact action equality.

```bash
python scripts/check_smolvla_input_parity.py \
  --checkpoint outputs/train/smolvla_shake_cup_3times_cycle300/016000 \
  --dataset-root data/so101_shake_cup_3times \
  --device cuda --compare-actions
```

Results are saved to `outputs/eval/smolvla_input_parity.json`.

The following checks Docker imports, CLI configuration, checkpoint files, and CycleManip-specific weight keys without connecting to the robot or camera. Missing devices are reported without preventing the check from completing. This does not test full model inference or physical robot behavior.

```bash
./scripts/rollout_smolvla.sh \
  --check \
  --checkpoint outputs/train/smolvla_shake_cup_3times_cycle300/016000
```

Before moving the robot, clear the area around the arm and use an initial pose and object placement consistent with the training data. The following commands start autonomous motion once model loading and device connection finish.

```bash
./scripts/rollout_cyclemanip.sh

# Explicitly select a checkpoint and rollout duration
SO101_DATASET_CONFIG=config/smolvla_shake_cup_3times_300.env \
CYCLEMANIP_CHECKPOINT="outputs/train/smolvla_shake_cup_3times_cycle300/016000" \
CYCLEMANIP_DURATION="10" \
./scripts/rollout_cyclemanip.sh
```

Specify a local checkpoint directory inside the project. `CYCLEMANIP_DURATION` controls the duration of the control loop. Ctrl+C requests termination, but the standard rollout shutdown returns the arm to its starting pose before disconnecting; it is not an immediate stop.

Set `CYCLEMANIP_INFERENCE_TYPE=sync` when comparing synchronous inference with RTC disabled. Change the RTC execution horizon and guidance weight with `CYCLEMANIP_RTC_EXECUTION_HORIZON` and `CYCLEMANIP_RTC_GUIDANCE_WEIGHT`, respectively.

Validation status: the startup `--check` and history-processing regression tests have passed. Real-robot results are linked in the demo section above. Quantitative control-rate measurements and aggregate task success rates have not been published. RTC separates inference latency from the control loop. Check the command cadence in the logs' Cadence summary; inference taking longer than one chunk (50 steps, approximately 1.67 seconds at 30 Hz for this checkpoint) can cause queue underruns, so adjust resolution, inference settings, or FPS as needed.

## Directory structure

```text
SO101/
├── config/
│   ├── dataset.env          # Tracked base settings; no secrets
│   ├── dataset.local.env    # Local overrides (not tracked)
│   └── robot.env            # SO-101 / camera settings
├── scripts/                 # Docker execution wrappers
├── .lerobot-src/            # LeRobot source with CycleManip-style support
├── data/                    # Datasets (not tracked)
└── outputs/                 # Checkpoints and evaluation results (not tracked)
```

## Push changes to Git

`data/`, `outputs/`, caches, and local settings are excluded by `.gitignore`. Review configuration and source files before committing.

```bash
git status
git add .
git commit -m "Add SO-101 SMOLVLA workflow"
git push -u origin HEAD
```

If you make changes inside a submodule, commit and push them in the submodule first, then update the gitlink in the parent repository.

## Official documentation

- [CycleManip paper](https://arxiv.org/abs/2512.01022)
- [CycleManip project](https://isee-laboratory.github.io/CycleManip/)
- [CycleManip repository](https://github.com/iSEE-Laboratory/CycleManip)
- [LeRobot repository](https://github.com/huggingface/lerobot)
- [LeRobot installation](https://huggingface.co/docs/lerobot/main/en/installation)
- [SO-101](https://huggingface.co/docs/lerobot/main/en/so101)
- [SmolVLA](https://huggingface.co/docs/lerobot/main/en/smolvla)
