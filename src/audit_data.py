from __future__ import annotations

import argparse
import csv
import json
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "results"

FER2013_LABELS = {
    "0": "anger",
    "1": "disgust",
    "2": "fear",
    "3": "happiness",
    "4": "sadness",
    "5": "surprise",
    "6": "neutral",
}


def count_extensions(paths: list[Path]) -> dict[str, int]:
    counts = Counter(path.suffix.lower() or "<none>" for path in paths)
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))


def inspect_zip(path: Path) -> dict[str, Any]:
    with zipfile.ZipFile(path) as archive:
        names = [name for name in archive.namelist() if not name.endswith("/")]
    return {
        "file_count": len(names),
        "extension_counts": dict(Counter(Path(name).suffix.lower() or "<none>" for name in names).most_common()),
        "first_files": names[:15],
    }


def resolve_media_path(value: str, raw_dir: Path, labels_dir: Path) -> Path | None:
    text = str(value or "").strip()
    if not text:
        return None
    if "#row=" in text.lower() or text.lower().startswith("fer2013:"):
        return None
    path = Path(text)
    if path.is_absolute():
        return path
    for candidate in [raw_dir / path, labels_dir / path, PROJECT_ROOT / path]:
        if candidate.exists():
            return candidate
    return raw_dir / path


def parse_fer_reference(value: str) -> int | None:
    text = str(value or "").strip()
    if not text:
        return None
    lowered = text.lower()
    if lowered.startswith("fer2013:"):
        row_text = text.split(":", 1)[1].strip()
        return int(row_text) if row_text.isdigit() else None
    marker = "#row="
    index = lowered.find(marker)
    if index == -1:
        return None
    row_text = text[index + len(marker):].strip()
    return int(row_text) if row_text.isdigit() else None


def inspect_fer2013(path: Path, requested_rows: set[int]) -> dict[str, Any]:
    if not path.exists():
        return {"exists": False}

    label_counts: Counter[str] = Counter()
    requested_labels: dict[int, str] = {}
    row_count = 0
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row_index, row in enumerate(reader):
            row_count += 1
            label = FER2013_LABELS.get(str(row.get("emotion", "")).strip(), str(row.get("emotion", "")).strip())
            label_counts[label] += 1
            if row_index in requested_rows:
                requested_labels[row_index] = label

    missing_requested = sorted(requested_rows - set(requested_labels))
    return {
        "exists": True,
        "row_count": row_count,
        "label_counts": dict(label_counts),
        "requested_rows": len(requested_rows),
        "missing_requested_rows": missing_requested[:20],
    }


def inspect_manifest(path: Path, raw_dir: Path) -> dict[str, Any]:
    if not path.exists():
        return {"exists": False}

    labels_dir = path.parent
    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)

    split_counts = Counter(row.get("split", "") for row in rows)
    label_counts = Counter(row.get("label", "") for row in rows)
    missing_audio: list[str] = []
    missing_image: list[str] = []
    fer_rows: set[int] = set()
    paired = audio_only = image_only = 0

    for row in rows:
        audio_value = row.get("audio_path", "")
        image_value = row.get("image_path", "")
        audio_path = resolve_media_path(audio_value, raw_dir, labels_dir)
        image_path = resolve_media_path(image_value, raw_dir, labels_dir)
        fer_row = parse_fer_reference(image_value)
        if fer_row is not None:
            fer_rows.add(fer_row)

        has_audio = bool(str(audio_value or "").strip())
        has_image = bool(str(image_value or "").strip())
        if has_audio and audio_path is not None and not audio_path.exists():
            missing_audio.append(row.get("sample_id", ""))
        if has_image and fer_row is None and image_path is not None and not image_path.exists():
            missing_image.append(row.get("sample_id", ""))

        if has_audio and has_image:
            paired += 1
        elif has_audio:
            audio_only += 1
        elif has_image:
            image_only += 1

    result = {
        "exists": True,
        "row_count": len(rows),
        "split_counts": dict(split_counts),
        "label_counts": dict(label_counts),
        "paired_rows": paired,
        "audio_only_rows": audio_only,
        "image_only_rows": image_only,
        "missing_audio_count": len(missing_audio),
        "missing_audio_examples": missing_audio[:20],
        "missing_image_count": len(missing_image),
        "missing_image_examples": missing_image[:20],
        "fer2013_references": len(fer_rows),
    }
    if fer_rows:
        result["fer2013"] = inspect_fer2013(raw_dir / "fer2013.csv", fer_rows)
    return result


def build_audit(raw_dir: Path) -> dict[str, Any]:
    files = [path for path in raw_dir.rglob("*") if path.is_file()]
    zips = sorted(raw_dir.glob("*.zip"))
    manifests = sorted((PROJECT_ROOT / "data").glob("labels*.csv"))
    return {
        "raw_dir": str(raw_dir),
        "raw_file_count": len(files),
        "raw_extension_counts": count_extensions(files),
        "zip_files": {zip_path.name: inspect_zip(zip_path) for zip_path in zips},
        "extracted_roots": {
            "ALL": (raw_dir / "ALL").exists(),
            "animated": (raw_dir / "animated").exists(),
            "fer2013.csv": (raw_dir / "fer2013.csv").exists(),
        },
        "manifests": {manifest.name: inspect_manifest(manifest, raw_dir) for manifest in manifests},
    }


def write_markdown(audit: dict[str, Any], output_path: Path) -> None:
    lines = [
        "# Data Audit",
        "",
        f"Raw directory: `{audit['raw_dir']}`",
        f"Raw file count: `{audit['raw_file_count']}`",
        "",
        "## Raw Extensions",
    ]
    for extension, count in audit["raw_extension_counts"].items():
        lines.append(f"- `{extension}`: {count}")

    lines.extend(["", "## Zip Files"])
    for name, info in audit["zip_files"].items():
        lines.append(f"- `{name}`: {info['file_count']} files, extensions {info['extension_counts']}")

    lines.extend(["", "## Manifests"])
    for name, info in audit["manifests"].items():
        if not info.get("exists"):
            lines.append(f"- `{name}`: missing")
            continue
        lines.append(
            f"- `{name}`: {info['row_count']} rows, labels {info['label_counts']}, "
            f"splits {info['split_counts']}, missing audio {info['missing_audio_count']}, "
            f"missing image {info['missing_image_count']}"
        )

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit raw datasets and labels manifests.")
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)
    audit = build_audit(args.raw_dir)
    json_path = args.results_dir / "data_audit.json"
    markdown_path = args.results_dir / "data_audit.md"
    json_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    write_markdown(audit, markdown_path)
    print(f"Wrote {json_path}")
    print(f"Wrote {markdown_path}")


if __name__ == "__main__":
    main()
