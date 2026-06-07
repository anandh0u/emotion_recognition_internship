from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from collections import Counter
from pathlib import Path
from typing import Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = Path(os.environ.get("EMOTION_DATA_ROOT", r"E:\emotion_recognition_data"))
DEFAULT_WORKSPACE_ROOT = DEFAULT_DATA_ROOT / "agents"
PROJECT_RESULTS = PROJECT_ROOT / "results"

AGENTS = ["audio", "vision", "text", "animation", "multimodal", "supervisor"]
SUBDIRS = ["manifests", "features", "models", "results", "logs"]


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def create_agent_dirs(workspace_root: Path) -> None:
    for agent in AGENTS:
        for subdir in SUBDIRS:
            (workspace_root / agent / subdir).mkdir(parents=True, exist_ok=True)


def copy_if_exists(source: Path, destination: Path) -> bool:
    if not source.exists():
        return False
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return True


def count_by(rows: Iterable[dict[str, str]], key: str) -> dict[str, int]:
    return dict(sorted(Counter(str(row.get(key, "") or "missing") for row in rows).items()))


def build_audio_manifest(workspace_root: Path, data_root: Path) -> tuple[Path, list[dict[str, str]]]:
    source = data_root / "labels_audio_multi.csv"
    destination = workspace_root / "audio" / "manifests" / "labels_audio_multi.csv"
    rows = read_csv(source)
    if rows:
        copy_if_exists(source, destination)
    return destination, rows


def build_vision_manifest(workspace_root: Path, data_root: Path) -> tuple[Path, list[dict[str, str]]]:
    rows: list[dict[str, str]] = []
    fer_rows = read_csv(PROJECT_ROOT / "data" / "labels.csv")
    for row in fer_rows:
        image_path = str(row.get("image_path", "")).strip()
        if not image_path:
            continue
        rows.append(
            {
                "sample_id": f"fer2013_{row.get('sample_id', '')}",
                "split": str(row.get("split", "")),
                "label": str(row.get("label", "")),
                "image_path": image_path,
                "dataset": "fer2013",
                "source_id": str(row.get("sample_id", "")),
            }
        )

    ravdess_rows = read_csv(data_root / "labels_ravdess_full.csv")
    for row in ravdess_rows:
        image_path = str(row.get("image_path", "")).strip()
        if not image_path:
            continue
        rows.append(
            {
                "sample_id": f"ravdess_frame_{row.get('sample_id', '')}",
                "split": str(row.get("split", "")),
                "label": str(row.get("label", "")),
                "image_path": image_path,
                "dataset": "ravdess_frames",
                "source_id": str(row.get("sample_id", "")),
            }
        )

    destination = workspace_root / "vision" / "manifests" / "labels_vision.csv"
    write_csv(destination, rows, ["sample_id", "split", "label", "image_path", "dataset", "source_id"])
    return destination, rows


def build_animation_manifest(workspace_root: Path) -> tuple[Path, list[dict[str, str]]]:
    source = PROJECT_ROOT / "data" / "labels_animated.csv"
    destination = workspace_root / "animation" / "manifests" / "labels_animation.csv"
    rows = read_csv(source)
    if rows:
        copy_if_exists(source, destination)
    return destination, rows


def build_multimodal_manifests(workspace_root: Path, data_root: Path) -> dict[str, Path]:
    outputs = {
        "ravdess_audio_video": workspace_root / "multimodal" / "manifests" / "labels_ravdess_audio_video.csv",
        "savee_fer_paired": workspace_root / "multimodal" / "manifests" / "labels_savee_fer_paired.csv",
    }
    copy_if_exists(data_root / "labels_ravdess_full.csv", outputs["ravdess_audio_video"])
    copy_if_exists(PROJECT_ROOT / "data" / "labels.csv", outputs["savee_fer_paired"])
    return outputs


def build_text_manifest(workspace_root: Path) -> tuple[Path, list[dict[str, str]]]:
    destination = workspace_root / "text" / "manifests" / "labels_text.csv"
    if not destination.exists():
        write_csv(destination, [], ["sample_id", "split", "label", "text", "dataset", "source_id"])
    return destination, read_csv(destination)


def write_registry(
    workspace_root: Path,
    audio_manifest: Path,
    vision_manifest: Path,
    text_manifest: Path,
    animation_manifest: Path,
    multimodal_manifests: dict[str, Path],
) -> Path:
    registry = {
        "audio_agent": {
            "manifest": str(audio_manifest),
            "models_dir": str(workspace_root / "audio" / "models"),
            "results_dir": str(workspace_root / "audio" / "results"),
        },
        "vision_agent": {
            "manifest": str(vision_manifest),
            "models_dir": str(workspace_root / "vision" / "models"),
            "results_dir": str(workspace_root / "vision" / "results"),
        },
        "text_agent": {
            "manifest": str(text_manifest),
            "models_dir": str(workspace_root / "text" / "models"),
            "results_dir": str(workspace_root / "text" / "results"),
        },
        "animation_agent": {
            "manifest": str(animation_manifest),
            "models_dir": str(workspace_root / "animation" / "models"),
            "results_dir": str(workspace_root / "animation" / "results"),
        },
        "fusion_agent": {
            "manifests": {name: str(path) for name, path in multimodal_manifests.items()},
            "models_dir": str(workspace_root / "multimodal" / "models"),
            "results_dir": str(workspace_root / "multimodal" / "results"),
        },
        "supervisor_agent": {
            "registry": str(workspace_root / "supervisor" / "agent_registry.json"),
            "results_dir": str(workspace_root / "supervisor" / "results"),
        },
    }
    output = workspace_root / "supervisor" / "agent_registry.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(registry, indent=2), encoding="utf-8")
    return output


def write_report(
    workspace_root: Path,
    audio_rows: list[dict[str, str]],
    vision_rows: list[dict[str, str]],
    text_rows: list[dict[str, str]],
    animation_rows: list[dict[str, str]],
    registry_path: Path,
) -> Path:
    report = PROJECT_RESULTS / "agent_workspace_report.md"
    lines = [
        "# Official Agent Workspace Report",
        "",
        f"Workspace root: `{workspace_root}`",
        f"Supervisor registry: `{registry_path}`",
        "",
        "## Audio Agent",
        "",
        f"- Rows: {len(audio_rows)}",
        f"- By dataset: {count_by(audio_rows, 'dataset')}",
        f"- By label: {count_by(audio_rows, 'label')}",
        "",
        "## Vision Agent",
        "",
        f"- Rows: {len(vision_rows)}",
        f"- By dataset: {count_by(vision_rows, 'dataset')}",
        f"- By label: {count_by(vision_rows, 'label')}",
        "",
        "## Text Agent",
        "",
        f"- Rows: {len(text_rows)}",
        "- Status: ready for GoEmotions, MELD text, or transcript-labeled rows.",
        "",
        "## Animation Agent",
        "",
        f"- Rows: {len(animation_rows)}",
        f"- By label: {count_by(animation_rows, 'label')}",
        "- Status: separate binary animated-content task, not mixed with 7-class emotion labels.",
        "",
        "## Multimodal/Fusion Agent",
        "",
        "- RAVDESS audio-video manifest is stored under the multimodal workspace.",
        "- SAVEE+FER paired manifest is stored for the original late-fusion experiment.",
        "",
        "## Supervisor Agent",
        "",
        "- Uses the registry to locate replaceable audio, vision, text, animation, and fusion agents.",
    ]
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create official separate workspaces for each emotion agent.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--workspace-root", type=Path, default=DEFAULT_WORKSPACE_ROOT)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    create_agent_dirs(args.workspace_root)
    audio_manifest, audio_rows = build_audio_manifest(args.workspace_root, args.data_root)
    vision_manifest, vision_rows = build_vision_manifest(args.workspace_root, args.data_root)
    text_manifest, text_rows = build_text_manifest(args.workspace_root)
    animation_manifest, animation_rows = build_animation_manifest(args.workspace_root)
    multimodal_manifests = build_multimodal_manifests(args.workspace_root, args.data_root)
    registry_path = write_registry(
        args.workspace_root,
        audio_manifest,
        vision_manifest,
        text_manifest,
        animation_manifest,
        multimodal_manifests,
    )
    report = write_report(args.workspace_root, audio_rows, vision_rows, text_rows, animation_rows, registry_path)
    print(f"Created agent workspaces under {args.workspace_root}")
    print(f"Wrote supervisor registry to {registry_path}")
    print(f"Wrote report to {report}")


if __name__ == "__main__":
    main()

