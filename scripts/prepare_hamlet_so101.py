#!/usr/bin/env python3
"""Convert a LeRobot v3 SO-101 dataset to the LeRobot-v2-like layout used by HAMLET.

The HAMLET loader in ``HAMLET-Isaac-GR00T`` expects one parquet file and one video
file per episode, plus ``episodes.jsonl``, ``tasks.jsonl`` and ``modality.json``.
LeRobot v3 stores the low-dimensional data in one or more dataset-level parquet
files and stores episode metadata in parquet, so the raw dataset cannot be passed
to HAMLET directly.

The converter writes a new dataset tree and never edits the source tree.
Dependencies are already part of the HAMLET environment: pyarrow and av.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any


def _require_dependencies():
    try:
        import av  # noqa: F401
        import numpy  # noqa: F401
        import pyarrow  # noqa: F401
        import pyarrow.parquet  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "Missing conversion dependency. Run this script with the HAMLET "
            "environment; it requires pyarrow, numpy and av."
        ) from exc


def _read_json(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return json.load(f)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(value, f, indent=2)
        f.write("\n")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_data_paths(source: Path) -> list[Path]:
    paths = sorted((source / "data").glob("**/*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No parquet data files found below {source / 'data'}")
    return paths


def _source_video_paths(source: Path, video_key: str) -> list[Path]:
    paths = sorted((source / "videos" / video_key).glob("**/*.mp4"))
    if not paths:
        raise FileNotFoundError(
            f"No videos found for {video_key!r} below {source / 'videos' / video_key}"
        )
    return paths


def _load_source(source: Path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    info = _read_json(source / "meta" / "info.json")
    source_stats = _read_json(source / "meta" / "stats.json")

    data_tables = [pq.read_table(path) for path in _source_data_paths(source)]
    data_table = pa.concat_tables(data_tables, promote_options="default")
    required = {"action", "observation.state", "episode_index", "task_index", "frame_index"}
    missing = sorted(required - set(data_table.column_names))
    if missing:
        raise ValueError(f"Source data is missing required columns: {missing}")

    episode_paths = sorted((source / "meta" / "episodes").glob("**/*.parquet"))
    if not episode_paths:
        raise FileNotFoundError(f"No episode metadata files found below {source / 'meta/episodes'}")
    episode_table = pa.concat_tables(
        [pq.read_table(path) for path in episode_paths], promote_options="default"
    )
    episodes = sorted(episode_table.to_pylist(), key=lambda row: int(row["episode_index"]))

    task_table = pq.read_table(source / "meta" / "tasks.parquet")
    tasks = {
        int(row["task_index"]): str(row["task"])
        for row in task_table.to_pylist()
    }
    if not tasks:
        raise ValueError("Source task table is empty")

    video_features = [
        key for key, feature in info.get("features", {}).items() if feature.get("dtype") == "video"
    ]
    if not video_features:
        raise ValueError("Source dataset has no video feature")

    return info, source_stats, data_table, episodes, tasks, video_features


def _copy_and_split_data(
    *,
    data_table,
    episodes: list[dict[str, Any]],
    output: Path,
    chunk_size: int,
) -> None:
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    output_columns = [
        "action",
        "observation.state",
        "timestamp",
        "frame_index",
        "episode_index",
        "index",
        "task_index",
    ]
    available = set(data_table.column_names)
    missing = sorted(set(output_columns) - available)
    if missing:
        raise ValueError(f"Source data is missing columns required for conversion: {missing}")

    episode_values = data_table["episode_index"]
    for episode in episodes:
        episode_index = int(episode["episode_index"])
        expected_length = int(episode["length"])
        mask = pc.equal(episode_values, episode_index)
        ep_table = data_table.filter(mask).select(output_columns)
        if ep_table.num_rows != expected_length:
            raise ValueError(
                f"Episode {episode_index} has {ep_table.num_rows} rows, "
                f"metadata says {expected_length}"
            )
        # The loader has an optional demo-frame path. Explicitly mark these as
        # non-demo so the stats and shard code take the deterministic branch.
        ep_table = ep_table.append_column(
            "is_demo", pa.array([False] * expected_length, type=pa.bool_())
        )
        out_path = (
            output
            / "data"
            / f"chunk-{episode_index // chunk_size:03d}"
            / f"episode_{episode_index:06d}.parquet"
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(ep_table, out_path, compression="zstd")


class _EpisodeVideoWriter:
    def __init__(self, path: Path, fps: int, width: int, height: int):
        import av

        self.container = av.open(str(path), mode="w")
        self.stream = self.container.add_stream("h264", rate=fps)
        self.stream.width = width
        self.stream.height = height
        self.stream.pix_fmt = "yuv420p"
        self.stream.options = {"g": "2", "crf": "23"}

    def write(self, frame) -> None:
        for packet in self.stream.encode(frame.reformat(format="yuv420p")):
            self.container.mux(packet)

    def close(self) -> None:
        for packet in self.stream.encode():
            self.container.mux(packet)
        self.container.close()


def _split_video(
    *,
    source_paths: list[Path],
    episode_lengths: list[int],
    video_key: str,
    output: Path,
    chunk_size: int,
    fps: int,
) -> None:
    import av

    total_expected = sum(episode_lengths)
    total_seen = 0
    episode_idx = 0
    frame_in_episode = 0
    writer: _EpisodeVideoWriter | None = None

    def open_writer(first_frame) -> _EpisodeVideoWriter:
        path = (
            output
            / "videos"
            / video_key
            / f"chunk-{episode_idx // chunk_size:03d}"
            / f"episode_{episode_idx:06d}.mp4"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        return _EpisodeVideoWriter(path, fps, first_frame.width, first_frame.height)

    try:
        for source_path in source_paths:
            with av.open(str(source_path)) as container:
                for frame in container.decode(video=0):
                    if episode_idx >= len(episode_lengths):
                        raise ValueError(f"Video {video_key} has more frames than metadata")
                    if writer is None:
                        writer = open_writer(frame)
                    writer.write(frame)
                    total_seen += 1
                    frame_in_episode += 1
                    if frame_in_episode == episode_lengths[episode_idx]:
                        writer.close()
                        writer = None
                        episode_idx += 1
                        frame_in_episode = 0
    finally:
        if writer is not None:
            writer.close()

    if total_seen != total_expected or episode_idx != len(episode_lengths):
        raise ValueError(
            f"Video {video_key} has {total_seen} frames, expected {total_expected}"
        )


def _build_metadata(
    *,
    source: Path,
    output: Path,
    info: dict[str, Any],
    source_stats: dict[str, Any],
    episodes: list[dict[str, Any]],
    tasks: dict[int, str],
    video_features: list[str],
    chunk_size: int,
) -> None:
    total_frames = sum(int(row["length"]) for row in episodes)
    # v3 episode metadata stores task strings in `tasks`; the data rows carry
    # integer task_index values. Keep the task table authoritative below.
    task_rows = [{"task_index": idx, "task": task} for idx, task in sorted(tasks.items())]
    episode_rows = []
    for row in episodes:
        episode_index = int(row["episode_index"])
        task_names = [str(task) for task in (row.get("tasks") or [])]
        if not task_names:
            raise ValueError(f"Episode {episode_index} has no task annotation")
        episode_rows.append(
            {
                "episode_index": episode_index,
                "tasks": task_names,
                "length": int(row["length"]),
            }
        )

    out_info = dict(info)
    out_info.update(
        {
            "codebase_version": "gr00t-hamlet-so101-v1",
            "total_episodes": len(episode_rows),
            "total_frames": total_frames,
            "total_tasks": len(task_rows),
            "chunks_size": chunk_size,
            "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
            "video_path": "videos/{video_key}/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.mp4",
            "splits": {"train": f"0:{len(episode_rows)}"},
        }
    )
    out_info["features"] = dict(out_info.get("features", {}))
    out_info["features"]["is_demo"] = {"dtype": "bool", "shape": [1], "names": ["is_demo"]}
    for video_key in video_features:
        video_info = out_info["features"].get(video_key, {}).get("info")
        if video_info is not None:
            # The v3 source is AV1, but the converter emits H.264 for broad
            # compatibility with torchcodec/PyAV in the GR00T environment.
            video_info["video.codec"] = "h264"
            video_info["video.video_backend"] = "pyav"

    # The loader uses the source feature names for the numerical stats. Keep
    # all source stats; this also allows the standard stats utility to validate
    # and refresh them later.
    _write_json(output / "meta" / "info.json", out_info)
    _write_json(output / "meta" / "stats.json", source_stats)
    _write_jsonl(output / "meta" / "episodes.jsonl", episode_rows)
    _write_jsonl(output / "meta" / "tasks.jsonl", task_rows)
    _write_json(
        output / "meta" / "modality.json",
        {
            "state": {
                "arm": {
                    "start": 0,
                    "end": 5,
                    "original_key": "observation.state",
                },
                "gripper": {
                    "start": 5,
                    "end": 6,
                    "original_key": "observation.state",
                },
            },
            "action": {
                "arm": {
                    "start": 0,
                    "end": 5,
                    "absolute": True,
                    "original_key": "action",
                },
                "gripper": {
                    "start": 5,
                    "end": 6,
                    "absolute": True,
                    "original_key": "action",
                },
            },
            "video": {
                "top": {"original_key": "observation.images.top"}
            },
            "annotation": {
                "human.action.task_description": {"original_key": "task_index"}
            },
        },
    )
    _write_json(
        output / "conversion.json",
        {
            "source": str(source.resolve()),
            "source_info_sha256": _sha256_file(source / "meta" / "info.json"),
            "source_total_episodes": len(episodes),
            "source_total_frames": total_frames,
            "video_features": video_features,
            "converter": "scripts/prepare_hamlet_so101.py",
            "notes": [
                "Raw LeRobot v3 source remains unchanged.",
                "Action horizon is selected by the SO-101 modality config (16 steps).",
                "arm actions are trained as relative joint actions; gripper is absolute.",
            ],
        },
    )
    (output / "README.md").write_text(
        "# SO-101 HAMLET training dataset\n\n"
        "Converted from the LeRobot v3 dataset for the HAMLET loader.\n\n"
        f"Episodes: {len(episode_rows)}\n\nFrames: {total_frames}\n\n"
        f"Task: {task_rows[0]['task']}\n"
    )


def _validate_output(output: Path) -> None:
    required = [
        output / "meta" / "info.json",
        output / "meta" / "stats.json",
        output / "meta" / "episodes.jsonl",
        output / "meta" / "tasks.jsonl",
        output / "meta" / "modality.json",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise ValueError(f"Converted dataset is incomplete; missing: {missing}")
    info = _read_json(output / "meta" / "info.json")
    episodes = [json.loads(line) for line in (output / "meta" / "episodes.jsonl").read_text().splitlines()]
    if len(episodes) != int(info["total_episodes"]):
        raise ValueError("episodes.jsonl count does not match meta/info.json")
    for row in episodes:
        ep = int(row["episode_index"])
        path = output / info["data_path"].format(
            episode_chunk=ep // int(info["chunks_size"]), episode_index=ep
        )
        if not path.is_file():
            raise ValueError(f"Missing converted parquet for episode {ep}: {path}")
        video_path = output / info["video_path"].format(
            episode_chunk=ep // int(info["chunks_size"]),
            episode_index=ep,
            video_key="observation.images.top",
        )
        if not video_path.is_file() or video_path.stat().st_size == 0:
            raise ValueError(f"Missing converted video for episode {ep}: {video_path}")


def convert(source: Path, output: Path, force: bool = False) -> None:
    source = source.resolve()
    output = output.resolve()
    if source == output:
        raise ValueError("Source and output must be different paths")
    if not (source / "meta" / "info.json").is_file():
        raise FileNotFoundError(f"Not a LeRobot v3 dataset: {source}")

    marker = output / "conversion.json"
    if marker.is_file() and not force:
        _validate_output(output)
        print(f"Already prepared: {output}")
        return
    if output.exists() and any(output.iterdir()):
        if not force:
            raise FileExistsError(
                f"Output directory is non-empty: {output}. Use --force only to replace this generated output."
            )
        shutil.rmtree(output)

    info, source_stats, data_table, episodes, tasks, video_features = _load_source(source)
    chunk_size = 1000
    staging_parent = output.parent
    staging_parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=staging_parent))
    try:
        _copy_and_split_data(
            data_table=data_table,
            episodes=episodes,
            output=staging,
            chunk_size=chunk_size,
        )
        _build_metadata(
            source=source,
            output=staging,
            info=info,
            source_stats=source_stats,
            episodes=episodes,
            tasks=tasks,
            video_features=video_features,
            chunk_size=chunk_size,
        )
        fps = int(info.get("fps", 30))
        for video_key in video_features:
            _split_video(
                source_paths=_source_video_paths(source, video_key),
                episode_lengths=[int(row["length"]) for row in episodes],
                video_key=video_key,
                output=staging,
                chunk_size=chunk_size,
                fps=fps,
            )
        _validate_output(staging)
        staging.rename(output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(f"Prepared HAMLET dataset: {output}")
    print(f"Episodes: {len(episodes)}")
    print(f"Frames: {sum(int(row['length']) for row in episodes)}")


def main() -> int:
    _require_dependencies()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    try:
        convert(args.source, args.output, force=args.force)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
