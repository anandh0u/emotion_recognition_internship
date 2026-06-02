from __future__ import annotations

import argparse
import csv
import random
import re
from collections import defaultdict
from pathlib import Path

from dataset import CLASS_NAMES, normalize_label

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "labels_extra.csv"

AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".npy"}
TOKEN_TO_LABEL = {
    "ang": "anger",
    "anger": "anger",
    "angry": "anger",
    "dis": "disgust",
    "disgust": "disgust",
    "fear": "fear",
    "fea": "fear",
    "hap": "happiness",
    "happy": "happiness",
    "happiness": "happiness",
    "neu": "neutral",
    "neutral": "neutral",
    "sad": "sadness",
    "sadness": "sadness",
    "sur": "surprise",
    "surprise": "surprise",
    "surprised": "surprise",
}


def tokenize(path: Path) -> list[str]:
    text = " ".join(path.parts + (path.stem,))
    return [token.lower() for token in re.split(r"[^A-Za-z0-9]+", text) if token]


def detect_label(path: Path) -> str | None:
    for token in tokenize(path):
        if token in TOKEN_TO_LABEL:
            return TOKEN_TO_LABEL[token]
        try:
            return normalize_label(token)
        except ValueError:
            continue
    return None


def stable_key(path: Path) -> str:
    stem = path.stem.lower()
    stem = re.sub(r"_(audio|image|frame|face|video)$", "", stem)
    stem = re.sub(r"[^a-z0-9]+", "_", stem).strip("_")
    return stem or path.stem.lower()


def split_for_index(index: int, total: int) -> str:
    train_cut = int(total * 0.70)
    val_cut = int(total * 0.85)
    if index < train_cut:
        return "train"
    if index < val_cut:
        return "val"
    return "test"


def build_rows(root: Path, seed: int) -> list[dict[str, str]]:
    media_by_label_key: dict[tuple[str, str], dict[str, Path]] = defaultdict(dict)
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        if suffix not in AUDIO_EXTENSIONS and suffix not in IMAGE_EXTENSIONS:
            continue
        label = detect_label(path)
        if label not in CLASS_NAMES:
            continue
        key = stable_key(path)
        modality = "audio_path" if suffix in AUDIO_EXTENSIONS else "image_path"
        media_by_label_key[(label, key)][modality] = path

    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for (label, key), media in media_by_label_key.items():
        grouped[label].append(
            {
                "sample_id": f"{root.name}_{label}_{key}",
                "split": "",
                "label": label,
                "audio_path": str(media.get("audio_path", "")),
                "image_path": str(media.get("image_path", "")),
            }
        )

    rng = random.Random(seed)
    rows: list[dict[str, str]] = []
    for label in CLASS_NAMES:
        label_rows = grouped[label]
        rng.shuffle(label_rows)
        for index, row in enumerate(label_rows):
            row["split"] = split_for_index(index, len(label_rows))
            rows.append(row)
    return rows


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a labels CSV from local audio/image dataset folders.")
    parser.add_argument("--root", type=Path, required=True, help="Dataset root folder to scan")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output labels CSV")
    parser.add_argument("--seed", type=int, default=42, help="Split random seed")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    rows = build_rows(args.root, seed=args.seed)
    if not rows:
        raise ValueError(f"No labeled audio/image files were found under {args.root}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample_id", "split", "label", "audio_path", "image_path"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
