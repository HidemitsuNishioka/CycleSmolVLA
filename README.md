# SO-ARM101 imitation learning with official LeRobot

## HAMLET / GR00T N1.6 adapter

指定された HAMLET 実装は `HAMLET-Isaac-GR00T/` に追加しました。既存の SO-101 LeRobot v3 データを GR00T 形式へ変換し、HAMLET の学習と安全な（アームを動かさない）推論を行えます。詳しい手順は [HAMLET-Isaac-GR00T/run_scripts/so101/README.md](HAMLET-Isaac-GR00T/run_scripts/so101/README.md) を参照してください。

このrepositoryは、Hugging Face公式LeRobotをDockerから呼び出すための薄いwrapperです。SO-101のdriver、`LeRobotDataset`、Diffusion Policy、training、rolloutはLeRobot公式実装を使用します。独自のrobot driver、Dataset class、policy実装は含めません。

現在の公式 `main` はPython 3.12以上を要求し、公式リポジトリでは `lerobot` 0.6.2 として管理されています。GPU環境では公式の夜間ビルドイメージ `huggingface/lerobot-gpu:latest` を使います。イメージの実際のバージョンは `./scripts/bootstrap.sh` の `lerobot-info` で確認してください。

公式参考資料:

- [LeRobot repository](https://github.com/huggingface/lerobot)
- [Installation](https://huggingface.co/docs/lerobot/main/en/installation)
- [SO-101](https://huggingface.co/docs/lerobot/main/en/so101)
- [Imitation learning workflow](https://huggingface.co/docs/lerobot/main/en/il_robots)
- [LeRobotDataset v3](https://huggingface.co/docs/lerobot/main/en/lerobot-dataset-v3)
- [Policy deployment / `lerobot-rollout`](https://huggingface.co/docs/lerobot/main/en/inference)

## 1. Install / Docker

前提は Ubuntu、Docker Compose v2、NVIDIA Container Toolkit、SO-101 arm、USB cameraです。GPU確認:

```bash
nvidia-smi
docker info | grep -i runtime
```

公式GPUイメージを取得し、LeRobotとCUDAを確認します。

```bash
./scripts/bootstrap.sh
```

このrepositoryの実験用データは `data/`、checkpointは `outputs/`、Hugging Face cacheは `~/.cache/huggingface` に保存されます。HF Hubへuploadする場合だけ、tokenをcontainer内で設定します。

wrapperはホストのUID/GIDをcontainerへ渡し、`HF_LEROBOT_HOME`をホストcacheへ対応付けます。そのためLeRobot公式のcalibration保存構造を維持したまま、`~/.cache/huggingface/lerobot/calibration/`へ書き込めます。

```bash
SO101_HOST_UID="$(id -u)" SO101_HOST_GID="$(id -g)" \
  docker compose run --rm --service-ports \
  --user "$(id -u):$(id -g)" lerobot hf auth login
```

認証なしのlocal dataset作成だけなら `DATASET_PUSH_TO_HUB=false` のままで構いません。

## 2. Device detection

USBを接続した状態で、公式のport/camera finderを実行します。

```bash
./scripts/detect_devices.sh
```

`lerobot-find-port` はarmを一台ずつ抜き差ししてLeader/Followerのportを特定します。`lerobot-find-cameras opencv` は対応するdevice、解像度、FPSを表示し、テスト画像を `outputs/captured_images/` に保存します。検出結果を `config/robot.env` に設定してください。

設定例（値は検出結果で置き換えること）:

```bash
FOLLOWER_PORT="/dev/serial/by-id/<follower-id>"
LEADER_PORT="/dev/serial/by-id/<leader-id>"
CAMERA_TOP_DEVICE="/dev/video1"
CAMERA_WRIST_DEVICE=""
CAMERA_WIDTH="640"
CAMERA_HEIGHT="480"
CAMERA_FPS="30"
DISPLAY_DATA="false"
```

`CAMERA_WRIST_DEVICE` を設定すると、wrapperは公式LeRobotへ `top` と `wrist` の2カメラ設定を渡します。カメラの名前はDatasetに保存されるため、学習時とrollout時で同じ名前・解像度・FPSにしてください。

手首カメラしかない場合は、`CAMERA_TOP_DEVICE`を空にして`CAMERA_WRIST_DEVICE`だけ設定できます。その場合Datasetの画像キーは`observation.images.wrist`になります。SmolVLA用のrename mapも合わせて変更します。

```bash
CAMERA_TOP_DEVICE=""
CAMERA_WRIST_DEVICE="/dev/videoX"
SMOLVLA_RENAME_MAP='{"observation.images.wrist":"observation.images.camera1"}'
```

## 3. Motor setup

armの電源を入れ、各motor busへ接続して公式motor setupを実行します。

```bash
./scripts/setup_motors.sh
```

## 4. Calibration

Leader/Followerをそれぞれ中間姿勢にしてから実行します。画面の公式手順に従って全jointを可動範囲へ動かします。

```bash
./scripts/calibrate.sh
```

LeRobotが公式のcalibration保存場所へ保存します。`FOLLOWER_ID`、`LEADER_ID` はteleoperation、recording、rolloutで同じ値を使ってください。calibration fileの保存方式は変更していません。

## 5. Teleoperation

まずLeaderを動かし、Followerが追従することを確認します。`DISPLAY_DATA=true` にすると公式Rerun visualizationを有効化できます。画面表示が不要なら `false` のままで構いません。

```bash
./scripts/teleop.sh
```

## 6. Dataset recording

`config/dataset.env` を必要に応じて編集します。現在の設定は1 camera、640x480、30 FPS、10 episode、15秒/episodeです。

```bash
./scripts/record.sh
```

recording中の公式keyboard操作:

- Right Arrow / `n`: episodeを保存して次へ
- Left Arrow / `r`: episodeを破棄して取り直し
- Escape / `q`: recordingを停止して保存

データは独自HDF5/pickle/npzへ変換せず、公式LeRobotDataset v3として `DATASET_ROOT` に保存されます。Hubへuploadする場合は先に `hf auth login` を行い、`config/dataset.env` または `config/dataset.local.env` で `DATASET_PUSH_TO_HUB=true` にします。

`DATASET_REPO_ID` は例えば `my-user/so101_5eps` のように設定します。現行LeRobotのrecordingは指定repo idへ時刻suffixを自動付与するため、このwrapperでは公式の `--dataset.no_stamp=true` を指定し、local学習時のrepo idを安定させています。

## 7. Dataset visualization / verification

公式LeRobotDataset APIでepisode数、frame数、FPS、state/action/image featureを確認します。

```bash
./scripts/inspect_dataset.sh
```

公式viewerでepisode 0を開きます。

```bash
./scripts/visualize_dataset.sh 0
```

Hubへuploadした場合は、公式visualize_dataset Spaceへdataset repo idを入力して確認することもできます。

## 8. MolmoAct2 training（TOP camera only）

TOPカメラ1台でMolmoAct2を使う場合は、公式の元checkpoint
`allenai/MolmoAct2-SO100_101`を`policy.checkpoint_path`で読み込み、現在のDatasetから
`observation.images.top`だけを入力にしてfine-tuneします。公式の変換済み
`lerobot/MolmoAct2-SO100_101-LeRobot`は`cam0`と`cam1`を想定するため、TOP 1台構成では直接使いません。

現行のSO-101 calibration conventionと元checkpointの差分を補正するため、wrapperには公式ドキュメントの
`joint_signs` / `joint_offsets`を設定済みです。12GB GPU向けに、初期設定はBF16、VLM freeze、batch 1、gradient checkpointingです。
最初は次のコマンドで短い動作確認を行い、OOMが出ないことを確認してから`MOLMOACT2_STEPS`を増やしてください。

```bash
./scripts/train_molmoact2.sh
```

学習済みcheckpointで実機rolloutします。最初はロボット周辺を空け、durationを短くしてください。

```bash
./scripts/rollout_molmoact2.sh
```

設定は`config/dataset.env`またはgit管理外の`config/dataset.local.env`で変更できます。
MolmoAct2は5Bモデルで、公式モデルカードでもSO-100/101用のfine-tuning/inference checkpointとして説明されています。
1カメラfine-tuningは2カメラの元学習条件と異なるため、性能を上げるには将来2台目のカメラを追加するのが推奨です。

## 9. SmolVLA training（wrist camera only）

手首カメラ1台だけなら、現時点ではSmolVLAを優先します。公式`lerobot/smolvla_base`は約450Mのbase modelで、SO-101の6軸state/actionとLeRobotDatasetをそのまま使えます。
base checkpointは3つの画像slotを持つため、wrapperは公式の`empty_cameras=2`で未接続の2 slotをmaskし、現在のDatasetの`observation.images.top`を`camera1`へrenameします。物理的に手首に付いたカメラでも、Dataset上のキー名は入力mappingで吸収できます。

```bash
./scripts/train_smolvla.sh
```

最初の動作確認だけ行う場合は、`config/dataset.local.env`に次を追加してください。

```bash
SMOLVLA_STEPS="100"
SMOLVLA_OUTPUT_DIR="outputs/train/smolvla_so101_wrist_smoke"
```

学習後の実機rollout:

```bash
./scripts/rollout_smolvla.sh
```

### SmolVLA open-loop evaluation video

学習済みcheckpointを記録済みLeRobotDatasetのheld-out episodeへ入力し、ロボットへactionを送らずにGT、予測action、絶対誤差を確認します。既定では最後の20%（現在の10 episodeならepisode 8, 9）を評価します。

```bash
./scripts/evaluate_smolvla_open_loop.sh
```

出力:

```text
outputs/eval/smolvla_so101_wrist_20k_open_loop/
├── episode_008_gt_pred_error.mp4
├── episode_009_gt_pred_error.mp4
├── episode_008_angle_graph.png
├── episode_009_angle_graph.png
├── action_comparison.csv
└── summary.json
```

動画は左に記録画像、右に6 jointそれぞれの「時間-角度」グラフを表示します。各グラフの上段は`GT`と`Pred`、下段は`|GT-Pred|`の絶対誤差です。PNGはepisode全体を静止画で確認するための同じグラフです。既定の`fresh` modeは各フレームで公式`predict_action_chunk`を呼び、chunk先頭actionを比較します。実機rolloutと同じaction queue挙動を確認する場合は次を使います。

```bash
SMOLVLA_OPEN_LOOP_ACTION_MODE=rollout ./scripts/evaluate_smolvla_open_loop.sh
```

episodeやフレーム数を限定したテスト:

```bash
SMOLVLA_OPEN_LOOP_EPISODES=8 \
SMOLVLA_OPEN_LOOP_MAX_FRAMES=100 \
./scripts/evaluate_smolvla_open_loop.sh
```

Hugging Face公式はSmolVLAで約50 episodeを開始点として推奨しています。現在の10 episodeはCLI・checkpoint生成確認用で、性能評価用には追加収録してください。

## 10. Diffusion Policy training

公式LeRobotの `--policy.type=diffusion` を使います。現在の設定は、SO-101データからの20,000-step学習、batch size 2、CUDAです。

```bash
./scripts/train_diffusion.sh
```

現在の出力先は次です。

```text
outputs/train/diffusion_so101_from_scratch_20k_v1/checkpoints/last/pretrained_model
```

学習前後のopen-loop評価は、最後の20%（2 episodes）をheld-outにして、公式LeRobotの`eval_loss`で表示します。

```bash
./scripts/evaluate_open_loop.sh  # 未学習ベースライン
./scripts/train_diffusion.sh     # 学習後のeval_lossも表示
```

Weights & Biasesは`WANDB_ENABLE=true`、projectは`lerobot-so101`です。キーは`config/dataset.local.env`（git管理外）へ置いてください。今回表示されたキーは漏えい扱いにして、W&Bでrevokeしてから新しいキーを設定してください。
キー未設定で`WANDB_ENABLE=true`の場合、wrapperはオンラインW&Bを自動的に無効化して学習を継続します。オンライン記録を使う場合だけ、新しいキーを`config/dataset.local.env`へ設定してください。

設定を変える場合は `config/dataset.local.env` を作って上書きします（このファイルはgit管理外です）。例えば:

```bash
cp config/dataset.env config/dataset.local.env
```

`config/dataset.local.env` で `DIFFUSION_STEPS=50000`、`DIFFUSION_BATCH_SIZE=4` などを設定し、`DIFFUSION_OUTPUT_DIR`を既存runと分けて再度 `./scripts/train_diffusion.sh` を実行します。

### Hubのpretrained policyについて

`DIFFUSION_PRETRAINED_PATH`には、LeRobot Diffusion Policyとして保存され、現在のDatasetと同じstate/action/camera featureを持つHub repo id（例: `org/compatible-diffusion-policy`）またはローカル`pretrained_model`を指定できます。`--policy.path`は公式CLIにそのまま渡されます。

ただし、`lerobot/diffusion_pusht`はPushTシミュレータ用でSO-101とは入出力が異なり、`lerobot/MolmoAct2-SO100_101-LeRobot`は大規模なSO-100/101向けモデルですがDiffusion Policyではありません。これらをDiffusionの初期重みとして指定しないでください。互換性のあるモデルが見つかった場合だけ、`config/dataset.local.env`で次のように指定します。

```bash
DIFFUSION_PRETRAINED_PATH="org/compatible-diffusion-policy"
```

Hubへpolicyをuploadする場合は、`POLICY_PUSH_TO_HUB=true` と `POLICY_REPO_ID="my-user/so101_diffusion"` を設定します。今回の学習はローカル保存のみです。

## 11. Robot rollout

`POLICY_PATH` を生成されたcheckpoint、またはHub上のpolicyへ設定します。まずはrobot周辺を空け、短いdurationで実行してください。

```bash
./scripts/rollout_diffusion.sh
```

このwrapperは公式 `lerobot-rollout --strategy.type=base --inference.type=sync` を呼び出します。rollout時のcamera設定はtraining datasetと同じでなければなりません。停止時はduration終了を待つか、LeRobotの公式終了操作を使ってください。

## Reproducibility and scope

- Docker imageは公式 `huggingface/lerobot-gpu:latest`。更新時は `./scripts/bootstrap.sh` の `lerobot-info` と `docker image inspect huggingface/lerobot-gpu:latest` で実際のimage digest/versionを記録してください。
- `docker-compose.yml` はGPU、USB serial、V4L2 camera、HF cache、dataset、outputsだけを実験用に接続します。
- SO-101制御、calibration、teleoperation、dataset、Diffusion Policy、rolloutは全てLeRobot公式CLIに委譲しています。
- ACTへ切り替える場合は、train wrapperの `--policy.type=diffusion` と出力先を公式 `--policy.type=act` 用に変更するだけで、Dataset形式は維持できます。
