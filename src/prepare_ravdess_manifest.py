from __future__ import annotations

import argparse
import csv
import random
import subprocess
from collections import defaultdict
from pathlib import Path

import cv2
import imageio_ffmpeg

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = PROJECT_ROOT / "data" / "raw" / "ravdess"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "labels_ravdess.csv"
DEFAULT_PROCESSED = PROJECT_ROOT / "data" / "processed" / "ravdess"

EMOTION_MAP = {
    "01": "neutral",
    "02": "neutral",
    "03": "happiness",
    "04": "sadness",
    "05": "anger",
    "06": "fear",
    "07": "disgust",
    "08": "surprise",
}


def parse_ravdess_filename(path: Path) -> dict[str, str] | None:
    parts = path.stem.split("-")
    if len(parts) != 7 or not all(part.isdigit() for part in parts):
        return None
    return {
        "modality": parts[0],
        "vocal_channel": parts[1],
        "emotion_code": parts[2],
        "intensity": parts[3],
        "statement": parts[4],
        "repetition": parts[5],
        "actor": parts[6],
    }


def split_for_index(index: int, total: int) -> str:
    train_cut = int(total * 0.70)
    val_cut = int(total * 0.85)
    if index < train_cut:
        return "train"
    if index < val_cut:
        return "val"
    return "test"


def path_for_manifest(path: Path) -> str:
    resolved = path.resolve()
    data_root = (PROJECT_ROOT / "data").resolve()
    try:
        return str(resolved.relative_to(data_root))
    except ValueError:
        return str(resolved)


def extract_middle_frame(video_path: Path, output_path: Path, overwrite: bool) -> None:
    if output_path.exists() and not overwrite:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video for frame extraction: {video_path}")

    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    start_index = max(frame_count // 2, 0)
    frame = None
    for frame_index in [start_index, 0, max(frame_count - 1, 0)]:
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        success, candidate = capture.read()
        if success and candidate is not None:
            frame = candidate
            break
    capture.release()
    if frame is None:
        raise RuntimeError(f"Could not read any frame from {video_path}")
    if not cv2.imwrite(str(output_path), frame):
        raise RuntimeError(f"Could not write frame to {output_path}")


def extract_audio(video_path: Path, output_path: Path, overwrite: bool) -> None:
    if output_path.exists() and not overwrite:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    command = [
        ffmpeg,
        "-y",
        "-loglevel",
        "error",
        "-i",
        str(video_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        str(output_path),
    ]
    subprocess.run(command, check=True)
    if not output_path.exists() or output_path.stat().st_size == 0:
        raise RuntimeError(f"Audio extraction produced an empty file: {output_path}")


def normalize_actor_id(value: str) -> str:
    text = value.strip().replace("Actor_", "").replace("actor_", "")
    return f"{int(text):02d}" if text.isdigit() else text


def parse_actor_list(value: str | None) -> set[str]:
    if not value:
        return set()
    return {normalize_actor_id(part) for part in value.replace(";", ",").split(",") if part.strip()}


def assign_label_splits(grouped: dict[str, list[dict[str, str]]], seed: int) -> list[dict[str, str]]:
    rng = random.Random(seed)
    rows: list[dict[str, str]] = []
    for label in sorted(grouped):
        label_rows = grouped[label]
        rng.shuffle(label_rows)
        for index, row in enumerate(label_rows):
            row["split"] = split_for_index(index, len(label_rows))
            rows.append(row)
    return rows


def assign_actor_splits(
    rows: list[dict[str, str]],
    seed: int,
    train_actors: set[str],
    val_actors: set[str],
    test_actors: set[str],
) -> list[dict[str, str]]:
    actors = sorted({row["actor"] for row in rows})
    actor_set = set(actors)
    explicit_actors = train_actors | val_actors | test_actors
    unknown_actors = explicit_actors - actor_set
    if unknown_actors:
        raise ValueError(f"Requested actor(s) were not found in the data: {sorted(unknown_actors)}")

    split_by_actor: dict[str, str] = {}
    if explicit_actors:
        duplicates = (train_actors & val_actors) | (train_actors & test_actors) | (val_actors & test_actors)
        if duplicates:
            raise ValueError(f"Actor(s) cannot appear in multiple splits: {sorted(duplicates)}")
        if not train_actors:
            train_actors = actor_set - val_actors - test_actors
        for actor in train_actors:
            split_by_actor[actor] = "train"
        for actor in val_actors:
            split_by_actor[actor] = "val"
        for actor in test_actors:
            split_by_actor[actor] = "test"
    else:
        if len(actors) < 3:
            raise ValueError("Actor-independent split requires at least 3 actors.")
        rng = random.Random(seed)
        shuffled = actors[:]
        rng.shuffle(shuffled)
        train_count = max(1, int(len(shuffled) * 0.70))
        val_count = max(1, int(len(shuffled) * 0.15))
        if train_count + val_count >= len(shuffled):
            val_count = max(1, len(shuffled) - train_count - 1)
        train_actors = set(shuffled[:train_count])
        val_actors = set(shuffled[train_count:train_count + val_count])
        test_actors = set(shuffled[train_count + val_count:])
        for actor in train_actors:
            split_by_actor[actor] = "train"
        for actor in val_actors:
            split_by_actor[actor] = "val"
        for actor in test_actors:
            split_by_actor[actor] = "test"

    split_counts = defaultdict(int)
    for row in rows:
        row["split"] = split_by_actor[row["actor"]]
        split_counts[row["split"]] += 1
    missing_splits = {"train", "val", "test"} - set(split_counts)
    if missing_splits:
        raise ValueError(f"Actor split is missing required split(s): {sorted(missing_splits)}")
    return rows


def build_rows(
    root: Path,
    processed_dir: Path,
    seed: int,
    overwrite: bool,
    include_video_only: bool,
    split_strategy: str,
    train_actors: set[str],
    val_actors: set[str],
    test_actors: set[str],
) -> list[dict[str, str]]:
    videos = sorted(root.rglob("*.mp4"))
    if not videos:
        raise FileNotFoundError(f"No RAVDESS .mp4 files found under {root}")

    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    rows: list[dict[str, str]] = []
    for video_path in videos:
        parsed = parse_ravdess_filename(video_path)
        if parsed is None:
            continue
        if parsed["vocal_channel"] != "01":
            continue
        if parsed["emotion_code"] not in EMOTION_MAP:
            continue

        is_full_audio_video = parsed["modality"] == "01"
        is_video_only = parsed["modality"] == "02"
        if not is_full_audio_video and not (include_video_only and is_video_only):
            continue

        label = EMOTION_MAP[parsed["emotion_code"]]
        sample_id = f"ravdess_{video_path.stem}"
        frame_path = processed_dir / "frames" / f"{video_path.stem}.jpg"
        audio_path = processed_dir / "audio" / f"{video_path.stem}.wav"

        extract_middle_frame(video_path, frame_path, overwrite=overwrite)
        audio_value = ""
        if is_full_audio_video:
            extract_audio(video_path, audio_path, overwrite=overwrite)
            audio_value = path_for_manifest(audio_path)

        row = {
            "sample_id": sample_id,
            "split": "",
            "label": label,
            "audio_path": audio_value,
            "image_path": path_for_manifest(frame_path),
            "source_video": str(video_path),
            "actor": parsed["actor"],
            "modality": parsed["modality"],
            "vocal_channel": parsed["vocal_channel"],
            "emotion_code": parsed["emotion_code"],
            "intensity": parsed["intensity"],
            "statement": parsed["statement"],
            "repetition": parsed["repetition"],
        }
        rows.append(row)
        grouped[label].append(row)

    if split_strategy == "actor":
        return assign_actor_splits(rows, seed, train_actors, val_actors, test_actors)
    return assign_label_splits(grouped, seed)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare RAVDESS video clips for the embedding pipeline.")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT, help="Extracted RAVDESS root folder")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output labels CSV")
    parser.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED, help="Output folder for extracted audio and frames")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true", help="Re-extract existing frames/audio")
    parser.add_argument(
        "--split-strategy",
        choices=["label", "actor"],
        default="label",
        help="Use label-balanced random splitting or actor-independent splitting.",
    )
    parser.add_argument("--train-actors", type=str, default="", help="Comma-separated actor IDs for the train split, for example 01,02,03")
    parser.add_argument("--val-actors", type=str, default="", help="Comma-separated actor IDs for the validation split")
    parser.add_argument("--test-actors", type=str, default="", help="Comma-separated actor IDs for the test split")
    parser.add_argument(
        "--full-av-only",
        action="store_true",
        help="Use only full audio-video clips. By default video-only clips are included as visual-only samples.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    rows = build_rows(
        args.root,
        args.processed_dir,
        seed=args.seed,
        overwrite=args.overwrite,
        include_video_only=not args.full_av_only,
        split_strategy=args.split_strategy,
        train_actors=parse_actor_list(args.train_actors),
        val_actors=parse_actor_list(args.val_actors),
        test_actors=parse_actor_list(args.test_actors),
    )
    if not rows:
        raise ValueError(f"No usable RAVDESS rows were built from {args.root}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "sample_id",
        "split",
        "label",
        "audio_path",
        "image_path",
        "source_video",
        "actor",
        "modality",
        "vocal_channel",
        "emotion_code",
        "intensity",
        "statement",
        "repetition",
    ]
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    counts: dict[str, int] = defaultdict(int)
    for row in rows:
        counts[row["label"]] += 1
    print(f"Wrote {len(rows)} rows to {args.output}")
    print(f"Label counts: {dict(sorted(counts.items()))}")
    print("Note: RAVDESS calm clips are mapped to neutral for this project's 7-class emotion schema.")


if __name__ == "__main__":
    main()
