from __future__ import annotations

import argparse
import csv
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dataset import CLASS_NAMES, normalize_label_for_classes, normalize_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "labels_audio_multi.csv"

AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"}

RAVDESS_EMOTION_MAP = {
    "01": "neutral",
    "02": "neutral",
    "03": "happiness",
    "04": "sadness",
    "05": "anger",
    "06": "fear",
    "07": "disgust",
    "08": "surprise",
}
SAVEE_EMOTION_MAP = {
    "a": "anger",
    "d": "disgust",
    "f": "fear",
    "h": "happiness",
    "n": "neutral",
    "sa": "sadness",
    "su": "surprise",
}
CREMAD_EMOTION_MAP = {
    "ANG": "anger",
    "DIS": "disgust",
    "FEA": "fear",
    "HAP": "happiness",
    "NEU": "neutral",
    "SAD": "sadness",
}
TESS_TOKEN_MAP = {
    "angry": "anger",
    "anger": "anger",
    "disgust": "disgust",
    "fear": "fear",
    "happy": "happiness",
    "happiness": "happiness",
    "neutral": "neutral",
    "sad": "sadness",
    "sadness": "sadness",
    "ps": "surprise",
    "surprise": "surprise",
}
EMODB_EMOTION_MAP = {
    "W": "anger",
    "E": "disgust",
    "A": "fear",
    "F": "happiness",
    "N": "neutral",
    "T": "sadness",
}


@dataclass
class AudioRow:
    sample_id: str
    split: str
    label: str
    audio_path: str
    image_path: str
    dataset: str
    speaker: str

    def as_dict(self) -> dict[str, str]:
        return {
            "sample_id": self.sample_id,
            "split": self.split,
            "label": self.label,
            "audio_path": self.audio_path,
            "image_path": self.image_path,
            "dataset": self.dataset,
            "speaker": self.speaker,
        }


def clean_token(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def manifest_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(resolved)


def resolve_audio_path(value: Any, labels_dir: Path) -> Path:
    raw_value = str(value or "").strip()
    path = Path(raw_value)
    if path.is_absolute():
        return path
    candidates = [
        labels_dir / path,
        labels_dir / "raw" / path,
        PROJECT_ROOT / path,
        PROJECT_ROOT / "data" / "raw" / path,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def parse_ravdess(path: Path) -> tuple[str, str, str] | None:
    parts = path.stem.split("-")
    if len(parts) != 7 or not all(part.isdigit() for part in parts):
        return None
    label = RAVDESS_EMOTION_MAP.get(parts[2])
    if label is None:
        return None
    actor = f"actor_{parts[6]}"
    return "ravdess", actor, label


def parse_savee(path: Path) -> tuple[str, str, str] | None:
    match = re.match(r"^(?P<speaker>[A-Za-z]+)_(?P<emotion>sa|su|a|d|f|h|n)\d+$", path.stem)
    if not match:
        return None
    label = SAVEE_EMOTION_MAP.get(match.group("emotion").lower())
    if label is None:
        return None
    speaker = f"speaker_{match.group('speaker').upper()}"
    return "savee", speaker, label


def parse_cremad(path: Path) -> tuple[str, str, str] | None:
    match = re.match(r"^(?P<speaker>\d{4})_[A-Z]{3}_(?P<emotion>ANG|DIS|FEA|HAP|NEU|SAD)_", path.stem)
    if not match:
        return None
    label = CREMAD_EMOTION_MAP.get(match.group("emotion"))
    if label is None:
        return None
    return "crema_d", f"speaker_{match.group('speaker')}", label


def parse_tess(path: Path) -> tuple[str, str, str] | None:
    tokens = [token.lower() for token in re.split(r"[^A-Za-z0-9]+", path.stem) if token]
    label = next((TESS_TOKEN_MAP[token] for token in tokens if token in TESS_TOKEN_MAP), None)
    if label is None:
        return None
    speaker = f"speaker_{tokens[0].upper()}" if tokens else "speaker_unknown"
    return "tess", speaker, label


def parse_emodb(path: Path) -> tuple[str, str, str] | None:
    match = re.match(r"^(?P<speaker>\d{2})[a-z]\d{2}(?P<emotion>[A-Z])[a-z]?$", path.stem)
    if not match:
        return None
    label = EMODB_EMOTION_MAP.get(match.group("emotion"))
    if label is None:
        return None
    return "emodb", f"speaker_{match.group('speaker')}", label


def parse_generic(path: Path) -> tuple[str, str, str] | None:
    tokens = [token.lower() for token in re.split(r"[^A-Za-z0-9]+", " ".join(path.parts + (path.stem,))) if token]
    for token in tokens:
        if token in TESS_TOKEN_MAP:
            dataset = clean_token(path.parent.name) or "generic"
            return dataset, "speaker_unknown", TESS_TOKEN_MAP[token]
        try:
            label = normalize_label_for_classes(token, CLASS_NAMES)
        except ValueError:
            continue
        dataset = clean_token(path.parent.name) or "generic"
        return dataset, "speaker_unknown", label
    return None


def parse_audio_file(path: Path) -> tuple[str, str, str] | None:
    for parser in [parse_ravdess, parse_savee, parse_cremad, parse_tess, parse_emodb, parse_generic]:
        parsed = parser(path)
        if parsed is not None:
            return parsed
    return None


def read_existing_manifest(path: Path) -> list[AudioRow]:
    rows: list[AudioRow] = []
    labels_dir = path.parent
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"sample_id", "split", "label", "audio_path"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        for row in reader:
            if not str(row.get("audio_path") or "").strip():
                continue
            audio_path = resolve_audio_path(row.get("audio_path"), labels_dir)
            if not audio_path.exists():
                continue
            label = normalize_label_for_classes(row.get("label"), CLASS_NAMES)
            split = normalize_split(row.get("split"))
            parsed = parse_audio_file(audio_path)
            dataset, speaker, _parsed_label = parsed if parsed is not None else ("manifest", "speaker_unknown", label)
            dataset = clean_token(str(row.get("dataset") or dataset or path.stem))
            speaker = clean_token(str(row.get("actor") or row.get("speaker") or speaker or "speaker_unknown"))
            rows.append(
                AudioRow(
                    sample_id=str(row.get("sample_id") or audio_path.stem),
                    split=split,
                    label=label,
                    audio_path=manifest_path(audio_path),
                    image_path=str(row.get("image_path") or ""),
                    dataset=dataset,
                    speaker=speaker,
                )
            )
    return rows


def scan_audio_root(root: Path) -> list[AudioRow]:
    rows: list[AudioRow] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in AUDIO_EXTENSIONS:
            continue
        parsed = parse_audio_file(path)
        if parsed is None:
            continue
        dataset, speaker, label = parsed
        sample_id = f"{dataset}_{clean_token(speaker)}_{clean_token(path.stem)}"
        rows.append(
            AudioRow(
                sample_id=sample_id,
                split="",
                label=label,
                audio_path=manifest_path(path),
                image_path="",
                dataset=dataset,
                speaker=clean_token(speaker),
            )
        )
    return rows


def split_group_count(group_count: int, train_ratio: float, val_ratio: float) -> tuple[int, int]:
    if group_count <= 1:
        return group_count, 0
    if group_count == 2:
        return 1, 0
    train_count = max(1, round(group_count * train_ratio))
    val_count = max(1, round(group_count * val_ratio))
    if train_count + val_count >= group_count:
        train_count = max(1, group_count - 2)
        val_count = 1
    return train_count, val_count


def assign_splits(rows: list[AudioRow], seed: int, train_ratio: float, val_ratio: float, resplit: bool) -> None:
    by_dataset: dict[str, list[AudioRow]] = defaultdict(list)
    for row in rows:
        if resplit or row.split not in {"train", "val", "test"}:
            row.split = ""
            by_dataset[row.dataset].append(row)

    rng = random.Random(seed)
    for dataset_rows in by_dataset.values():
        by_group: dict[str, list[AudioRow]] = defaultdict(list)
        for row in dataset_rows:
            group = row.speaker or row.sample_id
            by_group[group].append(row)
        groups = list(by_group.items())
        rng.shuffle(groups)
        train_count, val_count = split_group_count(len(groups), train_ratio, val_ratio)
        for group_index, (_group, group_rows) in enumerate(groups):
            if group_index < train_count:
                split = "train"
            elif group_index < train_count + val_count:
                split = "val"
            else:
                split = "test"
            for row in group_rows:
                row.split = split


def deduplicate(rows: list[AudioRow]) -> list[AudioRow]:
    seen_paths: set[str] = set()
    seen_ids: Counter[str] = Counter()
    output: list[AudioRow] = []
    for row in rows:
        key = str(Path(row.audio_path).resolve()) if Path(row.audio_path).is_absolute() else row.audio_path
        if key in seen_paths:
            continue
        seen_paths.add(key)
        seen_ids[row.sample_id] += 1
        if seen_ids[row.sample_id] > 1:
            row.sample_id = f"{row.sample_id}_{seen_ids[row.sample_id]}"
        output.append(row)
    return output


def write_manifest(rows: list[AudioRow], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["sample_id", "split", "label", "audio_path", "image_path", "dataset", "speaker"],
        )
        writer.writeheader()
        writer.writerows(row.as_dict() for row in rows)


def write_report(rows: list[AudioRow], output: Path, report: Path) -> None:
    by_dataset = Counter(row.dataset for row in rows)
    by_split = Counter(row.split for row in rows)
    by_label = Counter(row.label for row in rows)
    lines = [
        "# Multi-Dataset Audio Manifest Report",
        "",
        f"Manifest: `{output}`",
        f"Total audio rows: {len(rows)}",
        "",
        "## Rows By Dataset",
        "",
    ]
    lines.extend(f"- {name}: {count}" for name, count in sorted(by_dataset.items()))
    lines.extend(["", "## Rows By Split", ""])
    lines.extend(f"- {name}: {count}" for name, count in sorted(by_split.items()))
    lines.extend(["", "## Rows By Label", ""])
    lines.extend(f"- {name}: {count}" for name, count in sorted(by_label.items()))
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a combined audio-emotion manifest from multiple datasets.")
    parser.add_argument("--manifest", type=Path, action="append", default=[], help="Existing labels CSV to include")
    parser.add_argument("--root", type=Path, action="append", default=[], help="Audio dataset root folder to scan")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output combined labels CSV")
    parser.add_argument("--report", type=Path, default=None, help="Optional markdown report path")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--resplit", action="store_true", help="Reassign train/val/test splits for all rows")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    rows: list[AudioRow] = []
    for manifest in args.manifest:
        rows.extend(read_existing_manifest(manifest))
    for root in args.root:
        rows.extend(scan_audio_root(root))
    rows = deduplicate(rows)
    if not rows:
        raise ValueError("No usable audio rows were found.")
    assign_splits(rows, seed=args.seed, train_ratio=args.train_ratio, val_ratio=args.val_ratio, resplit=args.resplit)
    rows.sort(key=lambda row: (row.dataset, row.split, row.label, row.sample_id))
    write_manifest(rows, args.output)
    if args.report is not None:
        write_report(rows, args.output, args.report)
    print(f"Wrote {len(rows)} audio rows to {args.output}")
    print("Datasets:", dict(sorted(Counter(row.dataset for row in rows).items())))
    print("Splits:", dict(sorted(Counter(row.split for row in rows).items())))


if __name__ == "__main__":
    main()
