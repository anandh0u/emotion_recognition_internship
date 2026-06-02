from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F

from dataset import CLASS_NAMES, load_records, partition_records
from metrics import compute_metrics, save_confusion_matrix, set_seed

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FEATURES = PROJECT_ROOT / "features" / "all_embeddings.pt"
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "results"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Cosine-similarity baseline using class prototypes.")
    parser.add_argument("--cache", type=Path, default=DEFAULT_FEATURES, help="Path to features/all_embeddings.pt")
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR, help="Directory for baseline metrics")
    parser.add_argument("--support-split", type=str, default="train", choices=["train", "val", "test"], help="Split used to build class prototypes")
    parser.add_argument("--query-split", type=str, default="test", choices=["train", "val", "test"], help="Split to evaluate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser


def fused_embedding(record: dict) -> torch.Tensor:
    return torch.cat([record["audio_embedding"], record["visual_embedding"]], dim=0).float()


def build_prototypes(records: list[dict]) -> dict[int, torch.Tensor]:
    grouped: dict[int, list[torch.Tensor]] = defaultdict(list)
    for record in records:
        grouped[record["label_id"]].append(fused_embedding(record))
    prototypes: dict[int, torch.Tensor] = {}
    for class_index in range(len(CLASS_NAMES)):
        vectors = grouped.get(class_index, [])
        if not vectors:
            raise ValueError(f"No support examples found for class {CLASS_NAMES[class_index]!r}.")
        prototype = torch.stack(vectors, dim=0).mean(dim=0)
        prototypes[class_index] = F.normalize(prototype, dim=0)
    return prototypes


def predict(records: list[dict], prototypes: dict[int, torch.Tensor]) -> tuple[list[int], list[int]]:
    ordered_prototypes = torch.stack([prototypes[index] for index in range(len(CLASS_NAMES))], dim=0)
    y_true: list[int] = []
    y_pred: list[int] = []
    for record in records:
        sample_embedding = F.normalize(fused_embedding(record), dim=0)
        scores = ordered_prototypes @ sample_embedding
        y_true.append(record["label_id"])
        y_pred.append(int(scores.argmax().item()))
    return y_true, y_pred


def main() -> None:
    args = build_arg_parser().parse_args()
    set_seed(args.seed)
    if not args.cache.exists():
        raise FileNotFoundError(f"Feature cache not found: {args.cache}")

    records = load_records(args.cache)
    train_records, val_records, test_records = partition_records(records, seed=args.seed)
    split_map = {"train": train_records, "val": val_records, "test": test_records}
    support_records = split_map[args.support_split]
    query_records = split_map[args.query_split]
    if len(support_records) == 0 or len(query_records) == 0:
        raise ValueError("Support and query splits must both contain data.")

    prototypes = build_prototypes(support_records)
    y_true, y_pred = predict(query_records, prototypes)
    metrics = compute_metrics(y_true, y_pred, CLASS_NAMES)

    args.results_dir.mkdir(parents=True, exist_ok=True)
    save_confusion_matrix(y_true, y_pred, args.results_dir / "zero_shot_confusion_matrix.png", CLASS_NAMES, title="Zero-Shot Prototype Baseline")
    with (args.results_dir / "zero_shot_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump({"support_split": args.support_split, "query_split": args.query_split, **metrics}, handle, indent=2)

    print(json.dumps({"support_split": args.support_split, "query_split": args.query_split, **metrics}, indent=2))


if __name__ == "__main__":
    main()
