from __future__ import annotations

import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DEFAULT_ROOT = DEFAULT_RAW_DIR / "animated"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "labels_animated.csv"

LABEL_MAP = {
    "0": "not_optimized",
    "1": "optimized",
}


def split_for_index(index: int, total: int) -> str:
    train_cut = int(total * 0.70)
    val_cut = int(total * 0.85)
    if index < train_cut:
        return "train"
    if index < val_cut:
        return "val"
    return "test"


def relative_or_absolute(path: Path, raw_dir: Path) -> str:
    path = path.resolve()
    raw_dir = raw_dir.resolve()
    try:
        return str(path.relative_to(raw_dir))
    except ValueError:
        return str(path)


def sample_number(row_index: int, storyboard_id: str) -> int:
    text = str(storyboard_id).strip().upper()
    if text.startswith("SB_") and text[3:].isdigit():
        return int(text[3:])
    return row_index + 1


def build_rows(root: Path, raw_dir: Path, seed: int) -> list[dict[str, str]]:
    metadata_path = root / "animation_storyboard_dataset.csv"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Animated dataset metadata not found: {metadata_path}")

    rows_by_label: dict[str, list[dict[str, str]]] = defaultdict(list)
    with metadata_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"Storyboard_ID", "Scene_ID", "Script_Text", "RL_Optimized_Label"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{metadata_path} is missing columns: {sorted(missing)}")

        for row_index, row in enumerate(reader):
            label_value = str(row["RL_Optimized_Label"]).strip()
            if label_value not in LABEL_MAP:
                raise ValueError(f"Unsupported RL_Optimized_Label value: {label_value!r}")

            number = sample_number(row_index, row["Storyboard_ID"])
            audio_path = root / "audio" / f"audio_{number}.wav"
            image_path = root / "images" / f"img_{number}.npy"
            if not audio_path.exists() or not image_path.exists():
                raise FileNotFoundError(f"Missing media for storyboard {row['Storyboard_ID']}: {audio_path}, {image_path}")

            label = LABEL_MAP[label_value]
            rows_by_label[label].append(
                {
                    "sample_id": f"animated_{row['Storyboard_ID']}",
                    "split": "",
                    "label": label,
                    "audio_path": relative_or_absolute(audio_path, raw_dir),
                    "image_path": relative_or_absolute(image_path, raw_dir),
                    "script_text": row["Script_Text"],
                    "scene_id": row["Scene_ID"],
                    "rl_optimized_label": label_value,
                }
            )

    rng = random.Random(seed)
    output_rows: list[dict[str, str]] = []
    for label in LABEL_MAP.values():
        label_rows = rows_by_label[label]
        rng.shuffle(label_rows)
        for index, row in enumerate(label_rows):
            row["split"] = split_for_index(index, len(label_rows))
            output_rows.append(row)
    return output_rows


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a manifest for the animated multimodal content dataset.")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT, help="Extracted animated dataset folder")
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR, help="Raw data root used by precompute.py")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output CSV manifest")
    parser.add_argument("--seed", type=int, default=42, help="Split random seed")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    rows = build_rows(args.root.resolve(), raw_dir=args.raw_dir.resolve(), seed=args.seed)
    if not rows:
        raise ValueError(f"No animated rows were built from {args.root}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "sample_id",
        "split",
        "label",
        "audio_path",
        "image_path",
        "script_text",
        "scene_id",
        "rl_optimized_label",
    ]
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {args.output}")
    print("Classes: not_optimized, optimized")


if __name__ == "__main__":
    main()
