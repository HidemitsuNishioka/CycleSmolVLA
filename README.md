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

収録が完了すると、データセットは自動的にHugging Faceへアップロードされます。
`config/dataset.local.env` に、書き込み権限のあるトークンを設定してください。

```bash
HF_TOKEN="hf_..."
```

`HF_TOKEN` を設定しない場合は、事前に `hf auth login` でログインしておけば、既存のHugging Faceログイン情報も利用できます。
`DATASET_PUSH_TO_HUB="false"` に変更すると自動アップロードを無効化できます。
データセットはデフォルトで非公開（`DATASET_PRIVATE="true"`）です。

```bash
./scripts/record.sh
```

データは `data/` に LeRobotDataset v3 として保存されます。 `data/` は Git 管理外です。

収録後の確認:

```bash
./scripts/inspect_dataset.sh
./scripts/visualize_dataset.sh 0
```

### 既存データセットのアップロード

すでに `data/` に保存されているLeRobotデータセットは、次のコマンドでアップロードできます。デフォルトでは `config/dataset.local.env` の `DATASET_ROOT` と `DATASET_REPO_ID` を使用します。

```bash
./scripts/upload_dataset.sh
```

別のデータセットを指定する場合:

```bash
./scripts/upload_dataset.sh data/another_dataset your-hf-user/another_dataset
```

メタデータ、Parquet、動画、Dataset Cardをまとめてアップロードします。Hugging Faceの書き込み権限を持つ `HF_TOKEN` を `config/dataset.local.env` に設定するか、事前に `hf auth login` を実行してください。

## SMOLVLA 学習

ローカルの `data/so101_5eps` を CycleManip 対応 SmolVLA で学習する設定:

```bash
SO101_DATASET_CONFIG=config/smolvla_so101_5eps_cycle300.env ./scripts/train_smolvla.sh
```

このデータはディレクトリ名にかかわらず10エピソード・4,499フレームで、タスクは `Pick up the object and place it in the target area` です。episode 0〜7を学習、8〜9を検証に使います。状態履歴300フレーム、画像履歴6フレーム、行動チャンク50フレーム、batch size 4で20,000ステップ学習します。2,000ステップごとに最大100サンプルで検証し、ディスク使用量を抑えるためチェックポイントは5,000ステップごとに保存します。出力先は `outputs/train/smolvla_so101_5eps_cycle300` です。W&Bは `lerobot-so101` プロジェクトへのオンライン送信を有効にしています。`config/dataset.local.env` に `WANDB_API_KEY` を設定してください（未設定の場合、共通スクリプトがW&Bを無効化します）。Hubへのアップロードは無効です。同じ出力先で重複起動しないでください。

`Hidemitsu-Nishioka/so101_shake_cup_3times` を関節履歴300フレームで学習する設定:

```bash
SO101_DATASET_CONFIG=config/smolvla_shake_cup_3times_300.env ./scripts/train_smolvla.sh
```

データは `data/so101_shake_cup_3times` に取得済みです。30 fpsで現在を含む連続300フレーム（オフセット -299〜0）を入力し、エピソード冒頭の不足分はマスクします。画像履歴は6フレーム、行動チャンクは50フレームです。episode 0〜7で20,000ステップ学習し、8〜9から最大100サンプルで2,000ステップごとに検証損失を計算します。出力先は `outputs/train/smolvla_shake_cup_3times_cycle300` です。同じ出力先での重複起動は避けてください。

現時点で動作確認済みの学習コマンドです。

```bash
./scripts/train_smolvla.sh
```

既定では、CycleManip論文に基づく履歴認識を有効にしています。各時刻で画像を6フレーム、関節状態を過去32フレーム読み込み、行動損失に10段階の進捗分類損失（重み0.1）を加えます。無効にする場合は、`config/dataset.local.env`で次を設定します。

```bash
SMOLVLA_CYCLE_ENABLED="false"
```

論文の実装が公開されていないため、CycleManipの論文と公開READMEに記載された仕様（高コストな画像の疎な履歴、低コストな状態の密な履歴、進捗の補助分類）を、LeRobotの時系列データローダとSmolVLAの条件トークンへ移植しています。

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
評価ランチャーも実機rolloutと同じPython環境を選び、`.lerobot-src/src` を優先して読み込みます。

```bash
./scripts/evaluate_smolvla_open_loop.sh
```

出力先は `outputs/eval/smolvla_so101_wrist_20k_open_loop/` です。テスト時は次のように対象 episode やフレーム数を制限できます。

```bash
SMOLVLA_OPEN_LOOP_EPISODES=8 \
SMOLVLA_OPEN_LOOP_MAX_FRAMES=100 \
./scripts/evaluate_smolvla_open_loop.sh
```

`cycle300/016000` のep8・ep9全体を評価する例です。既定の `fresh` は毎フレーム新しい
行動チャンクを推論し、その先頭を比較します。実機のRTCキューは使用しません。

```bash
SO101_DATASET_CONFIG=config/smolvla_shake_cup_3times_300.env \
SMOLVLA_OPEN_LOOP_EPISODES=8,9 \
SMOLVLA_OPEN_LOOP_OUTPUT_DIR=outputs/eval/smolvla_cycle300_016000_ep8_ep9/fresh \
./scripts/evaluate_smolvla_open_loop.sh
```

実機と同じRTCエンジンの評価には、LeRobotのPython環境内で次を実行します。記録した観測を
30 Hzで供給し、実際の非同期推論・行動キューを通した出力を比較します。推論前後の動画処理は
計測区間から除外します。各時刻の観測は記録データなので、実機の追従やタスク成功の評価ではありません。
出力待ちフレームは誤差集計から除外し、待ち時間・キュー不足・±10の制限適用前後の誤差も保存します。

```bash
python scripts/evaluate_smolvla_rtc_open_loop.py \
  --checkpoint outputs/train/smolvla_shake_cup_3times_cycle300/016000 \
  --dataset-root data/so101_shake_cup_3times \
  --episodes 8,9 --device cuda --max-relative-target 10 \
  --output-dir outputs/eval/smolvla_cycle300_016000_ep8_ep9/rtc
```

同じ評価スクリプトで `--disable-guidance` を付けると、モデル内部のRTC補正を完全に迂回します
（非同期キュー・遅延補償は残ります）。`--inference-type sync --n-action-steps 1` では、
実機用の同期推論エンジンを使い、RTCのキュー・遅延補償も無効にして毎回予測し直します。
`--n-action-steps 50` なら補正なしで50手を順に使用する比較になります。同期推論が30 Hzに
間に合わない場合も記録フレームは省略せず、実測の処理Hzを `summary.json` に記録します。

チャンクの先頭と後続の精度を切り分けるには `scripts/diagnose_smolvla_action_chunks.py` に
`--checkpoint`、`--dataset-root`、`--output-dir` を指定します。ep8・ep9を既定で30フレーム
ごとにサンプルし、各予測を対応する未来時刻のGTと比較します。エピソード端のパディングは除外します。

## 実機 rollout

CycleManip対応SmolVLAは `scripts/rollout_cyclemanip.sh` から起動します。
Docker内で学習に使った `/workspace/.venv/bin/python`（存在しない場合は `python`）を使い、
`.lerobot-src/src` の修正済みコードを優先して読み込みます。
既定ではRTCを使い、行動チャンクの推論をバックグラウンドで実行します。これにより、
推論の待ち時間で実機制御ループが止まらず、30 Hzのコマンド周期を維持できます。

実行中のターミナルにフォーカスして **Rキー** を押すと、観測履歴・行動キュー・補間中の
行動をリセットし、現在の姿勢から推論を再開します。Enterは不要です。RTCの実行中の
推論が完了してからリセットするため、古い予測が再開後に混ざりません。実行時間の上限も
Rを押して再開した時点から数え直します。アームやコップの物理的な配置はそのままなので、
次の周回を開始できる配置で操作してください。終了は従来どおりCtrl+Cです。
このキー操作はランチャーが `--strategy.keyboard_restart=true` で有効にし、入力は実行中の
ターミナルだけから受け付けます。`--interactive=true` のコマンド入力モードでは無効です。

RTCを完全に外して、毎回チャンクの先頭1手だけを使う比較は次で起動できます。
`sync` にするだけではチェックポイント既定の50手キューが残るため、1手の指定も必要です。
同期の1手推論では制御周期がモデルの推論速度に制約されます。

```bash
SO101_DATASET_CONFIG=config/smolvla_shake_cup_3times_300.env \
CYCLEMANIP_INFERENCE_TYPE=sync CYCLEMANIP_N_ACTION_STEPS=1 \
./scripts/rollout_cyclemanip.sh
```

まずFollowerとカメラを接続し、`config/robot.env` の `FOLLOWER_PORT`、`FOLLOWER_ID`、
`CAMERA_TOP_DEVICE` を実際の機器に合わせてください。Followerは同じIDでキャリブレーション済みのものを使います。
学習時と同じカメラ配置・関節単位を使ってください。

```bash
ls -l /dev/serial/by-id/ /dev/ttyACM* /dev/video*
```

この環境で保存済みのCycleManipチェックポイントを使う例です。

```bash
SO101_DATASET_CONFIG=config/smolvla_shake_cup_3times_300.env \
CYCLEMANIP_CHECKPOINT="outputs/train/smolvla_shake_cup_3times_cycle300/016000" \
CYCLEMANIP_DURATION="10" \
./scripts/rollout_cyclemanip.sh
```

CycleManipの有効・無効、履歴長、行動チャンク長はチェックポイントから読み込みます。
`SMOLVLA_CYCLE_ENABLED` は学習用設定なので、rollout時の切り替えには使いません。
`last` は最新の保存先に追従します。固定したモデルを使う場合は `006000` などを指定してください。

CycleManipのモデル実装は `KainaJetson:~/SO101/` の学習コードに合わせています。
履歴Transformerは4ヘッド・PreNorm、融合層の活性化はSiLUです。これらは重みの形状に
現れないため、重みが読み込めても別の実装では同じ予測になりません。

CycleManipの履歴は推論回数ではなく制御周期ごとに保存します。RTCの `cycle300` では
30 Hzの関節状態300フレームと、`[-299, -150, -75, -2, -1, 0]` フレームの画像6枚を
モデルに渡します。開始時の不足履歴は最初のフレームで埋め、学習時と同じ `_is_pad` マスクで
無効にします。実行リセット時には履歴も消去します。
RTCではCPU上に画像を保存し、推論時に選んだ6枚だけをGPUへ転送します。
同期の1手推論では観測取得も遅くなるため、300フレームが必ず10秒に相当するわけではありません。

2026-09-19の実装照合より前に作成した評価は、学習側とは異なるモデル構造・画像履歴での
結果でした。チェックポイントの学習品質を判断する数値としては使用しないでください。
照合後の診断・評価は `outputs/eval/learning_cause_20260919/` に保存しています。

記録済みデータを実機の入力形式で再生し、オープンループ評価との入力一致を検証する場合は、
LeRobotのPython環境内で次を実行します。ロボットやカメラへの接続は行いません。
`--compare-actions` は同じノイズでの行動出力差も記録します。画像のCPU/GPU変換の丸め差が
あるため、行動出力の完全一致を保証する検査ではありません。

```bash
python scripts/check_smolvla_input_parity.py \
  --checkpoint outputs/train/smolvla_shake_cup_3times_cycle300/016000 \
  --dataset-root data/so101_shake_cup_3times \
  --device cuda --compare-actions
```

検証結果は `outputs/eval/smolvla_input_parity.json` に保存されます。

ロボット・カメラに接続せず、Docker内のimport、CLI設定、チェックポイントのファイルと
CycleManipの重みキーを検査できます。機器が存在しない場合も、その状態を表示して検査を完了します。
モデル全体の推論や実機動作のテストではありません。

```bash
./scripts/rollout_smolvla.sh \
  --check \
  --checkpoint outputs/train/smolvla_shake_cup_3times_cycle300/016000
```

実機を動かす場合は、アームの周囲を空け、学習データに対応する初期姿勢・物体配置にして実行します。
次のコマンドはモデル読み込みと機器接続が完了すると自律動作を開始します。

```bash
./scripts/rollout_cyclemanip.sh

# 指定チェックポイントと実行秒数を明示する場合
SO101_DATASET_CONFIG=config/smolvla_shake_cup_3times_300.env \
CYCLEMANIP_CHECKPOINT="outputs/train/smolvla_shake_cup_3times_cycle300/016000" \
CYCLEMANIP_DURATION="10" \
./scripts/rollout_cyclemanip.sh
```

チェックポイントはプロジェクト内のローカルディレクトリを指定してください。
`CYCLEMANIP_DURATION` は制御ループの実行時間です。`Ctrl+C` で終了要求を送れますが、
標準rolloutの終了処理では起動時の姿勢へ戻ってから切断するため、即時停止とは異なります。

RTCを無効化して同期推論を比較する場合だけ、`CYCLEMANIP_INFERENCE_TYPE=sync` を指定します。
RTCの実行ホライズンとガイダンス強度は、それぞれ `CYCLEMANIP_RTC_EXECUTION_HORIZON` と
`CYCLEMANIP_RTC_GUIDANCE_WEIGHT` で変更できます。

検証状況：起動設定の `--check` と履歴処理の回帰テストは確認済みです。
実機動作・制御周期・タスク成功率は、この環境では未確認です。
RTCでは推論遅延を制御ループから分離しています。ログのCadence summaryでコマンド周期を確認し、
推論が1チャンク（このチェックポイントでは50ステップ、30 Hzで約1.67秒）を超える場合は
キュー枯渇が起きるため、解像度・推論設定・FPSを調整してください。

## HAMLET / GR00T N1.6

`HAMLET-Isaac-GR00T/` は独立したサブモジュールです。現時点では動作確認できていないため、このルートの SMOLVLA 手順と混同しないでください。

```bash
git -C HAMLET-Isaac-GR00T status
```

HAMLET 側の手順や依存関係は、サブモジュール内の README とそのリポジトリの状態を確認してください。親リポジトリの `git` が参照するのはサブモジュールの commit だけで、サブモジュール内の未コミット変更は含まれません。

### SO-101 データでHAMLETを学習する場合

HAMLETサブモジュールは、標準ではRoboMME/RMBench向けのLeRobot v2系レイアウトを読み込みます。`data/so101_shake_cup` はLeRobot v3形式なので、付属スクリプトで変換してから学習します。元データは変更されず、変換先は `data/so101_shake_cup_gr00t` です。

```bash
# Jetson Thorでは、汎用PyPI版TorchではなくDockerを使う（推奨）
MAX_STEPS=1000 bash scripts/train_hamlet_so101_docker.sh

# Dockerを使わずホスト環境で実行する場合
cd HAMLET-Isaac-GR00T
~/.local/bin/uv sync
~/.local/bin/uv pip install -e .
cd ..

# まず短い動作確認
MAX_STEPS=1000 bash scripts/train_hamlet_so101.sh

# 本学習の例
MAX_STEPS=10000 SAVE_STEPS=1000 bash scripts/train_hamlet_so101.sh
```

`NUM_GPUS`、`GLOBAL_BATCH_SIZE`、`GRAD_ACCUM`、`OUTPUT_DIR` は環境変数で上書きできます。GR00T-N1.6のベースモデル取得にはHugging Faceへのアクセスが必要です。
Thorでは `flash-attn` をビルドできないため、学習スクリプトはPyTorch SDPAを既定で使用します。FlashAttentionを使える環境だけ `USE_FLASH_ATTENTION=true` を指定してください。

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
