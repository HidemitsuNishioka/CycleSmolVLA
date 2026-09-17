# SO-101 × SMOLVLA

SO-101 の模倣学習を、Hugging Face 公式 LeRobot と SMOLVLA で実行するための Docker wrapper です。

## 現在の対応状況

| 機能 | 状態 |
| --- | --- |
| LeRobot によるデバイス検出・設定・キャリブレーション | 使用対象 |
| テレオペレーション・データ収録 | 使用対象 |
| SMOLVLA の学習・offline評価・実機rollout | 現在の動作確認済みルート |
| HAMLET / GR00T N1.6 | サブモジュールのみ。未検証・動作保証なし |

このリポジトリでは、SMOLVLA 以外の学習方式（Diffusion Policy、MolmoAct2）の wrapper は削除しています。HAMLET は調査用に残していますが、まだ動く前提で扱わないでください。

## 必要な環境

- Ubuntu
- Docker Compose v2
- NVIDIA Container Toolkit と CUDA 対応 GPU
- SO-101 Leader / Follower
- USB カメラ（現在の設定は1台）
- Git

Docker image は `huggingface/lerobot-gpu:latest` を使用します。LeRobot 本体をホスト側へインストールする必要はありません。

## Clone

サブモジュールを含めて clone します。

```bash
git clone --recurse-submodules <your-repository-url>
cd SO101
```

既に clone 済みの場合は、次でサブモジュールを取得できます。

```bash
git submodule update --init --recursive
```

## 初期設定

### 1. Docker と GPU の確認

```bash
nvidia-smi
docker compose version
./scripts/bootstrap.sh
```

`bootstrap.sh` は公式 LeRobot GPU image を取得し、LeRobot と GPU がコンテナから見えるか確認します。

### 2. ローカル設定を作成

```bash
cp -n config/dataset.env config/dataset.local.env
```

`config/dataset.local.env` は `.gitignore` 対象です。データセット名、保存先、学習ステップ数、W&B の設定など、PCごとに変わる値はこのファイルへ書きます。

W&B を使う場合も API key は `config/dataset.local.env` にだけ設定してください。キーが空のまま `WANDB_ENABLE=true` の場合は、wrapper が online logging を自動的に無効化します。

### 3. デバイス設定

まず、SO-101 とカメラを接続して検出します。

```bash
./scripts/detect_devices.sh
```

検出結果を [config/robot.env](config/robot.env) に設定します。

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

現在の SMOLVLA 設定は、データセットの `observation.images.top` を `camera1` へ rename します。カメラの feature 名を変更した場合は、`SMOLVLA_RENAME_MAP` も同じように変更してください。

## SO-101 の準備

次の順番で実行します。モーターが動く操作では、アームの周囲を空けてください。

```bash
# モーターの初期設定（初回または必要時）
./scripts/setup_motors.sh

# Leader / Follower のキャリブレーション
./scripts/calibrate.sh

# Leader で Follower が追従することを確認
./scripts/teleop.sh
```

## データセット収録

`config/dataset.local.env` の次の値を確認してから収録します。

```bash
DATASET_REPO_ID="your-hf-user/so101_task"
DATASET_ROOT="data/so101_task"
DATASET_TASK="Describe the task here."
DATASET_NUM_EPISODES="10"
DATASET_FPS="30"
```

```bash
./scripts/record.sh
```

データは `data/` に LeRobotDataset v3 として保存されます。 `data/` は Git 管理外です。

収録後の確認:

```bash
./scripts/inspect_dataset.sh
./scripts/visualize_dataset.sh 0
```

## SMOLVLA 学習

現時点で動作確認済みの学習コマンドです。

```bash
./scripts/train_smolvla.sh
```

デフォルトでは次の設定です。

- base model: `lerobot/smolvla_base`
- batch size: `4`
- steps: `20000`
- GPU: `cuda`
- 出力先: `outputs/train/smolvla_so101_wrist_20k_v1/`

短い smoke test を行う場合は、`config/dataset.local.env` に追加・上書きします。

```bash
SMOLVLA_STEPS="100"
SMOLVLA_OUTPUT_DIR="outputs/train/smolvla_smoke"
SMOLVLA_JOB_NAME="smolvla_smoke"
```

## SMOLVLA offline 評価

評価対象はデータセットの最後の `DATASET_EVAL_SPLIT`（デフォルト20%）です。ロボットには action を送信しません。

```bash
./scripts/evaluate_smolvla_open_loop.sh
```

出力先は `outputs/eval/smolvla_so101_wrist_20k_open_loop/` です。テスト時は次のように対象 episode やフレーム数を制限できます。

```bash
SMOLVLA_OPEN_LOOP_EPISODES=8 \
SMOLVLA_OPEN_LOOP_MAX_FRAMES=100 \
./scripts/evaluate_smolvla_open_loop.sh
```

## 実機 rollout

学習済み checkpoint のパスを `config/dataset.local.env` の `SMOLVLA_POLICY_PATH` に設定します。

```bash
SMOLVLA_POLICY_PATH="outputs/train/smolvla_smoke/checkpoints/last/pretrained_model"
SMOLVLA_ROLLOUT_DURATION="10"
```

最初はアームの周囲を完全に空け、短い duration で実行してください。

```bash
./scripts/rollout_smolvla.sh
```

## HAMLET / GR00T N1.6

`HAMLET-Isaac-GR00T/` は独立したサブモジュールです。現時点では動作確認できていないため、このルートの SMOLVLA 手順と混同しないでください。

```bash
git -C HAMLET-Isaac-GR00T status
```

HAMLET 側の手順や依存関係は、サブモジュール内の README とそのリポジトリの状態を確認してください。親リポジトリの `git` が参照するのはサブモジュールの commit だけで、サブモジュール内の未コミット変更は含まれません。

## ディレクトリ構成

```text
SO101/
├── config/
│   ├── dataset.env          # Git管理する基本設定・秘密情報なし
│   ├── dataset.local.env    # ローカル上書き（Git管理外）
│   └── robot.env            # SO-101 / カメラ設定
├── scripts/                 # Docker経由の実行 wrapper
├── .lerobot-src/            # 公式 LeRobot の submodule
├── HAMLET-Isaac-GR00T/      # 未検証の HAMLET submodule
├── data/                    # データセット（Git管理外）
└── outputs/                 # checkpoint・評価結果（Git管理外）
```

## Git へ push する場合

`data/`、`outputs/`、キャッシュ、ローカル設定は `.gitignore` で除外されています。設定とソースを確認してから commit してください。

```bash
git status
git add .
git commit -m "Add SO-101 SMOLVLA workflow"
git push -u origin HEAD
```

サブモジュール内に変更がある場合は、先にサブモジュール側で commit・push し、その後に親リポジトリで gitlink を更新します。

## 公式ドキュメント

- [LeRobot repository](https://github.com/huggingface/lerobot)
- [LeRobot installation](https://huggingface.co/docs/lerobot/main/en/installation)
- [SO-101](https://huggingface.co/docs/lerobot/main/en/so101)
- [SmolVLA](https://huggingface.co/docs/lerobot/main/en/smolvla)
