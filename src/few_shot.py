from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict, Counter
from pathlib import Path

import torch
import torch.nn.functional as F

from dataset import CLASS_NAMES, load_records, partition_records
from metrics import compute_metrics, save_confusion_matrix, set_seed

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FEATURES = PROJECT_ROOT / "features" / "all_embeddings.pt"
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "results"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Few-shot prototype baseline built on precomputed embeddings.")
    parser.add_argument("--cache", type=Path, default=DEFAULT_FEATURES, help="Path to features/all_embeddings.pt")
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR, help="Directory for baseline metrics")
    parser.add_argument("--k", type=int, default=5, help="Support examples per class")
    parser.add_argument("--episodes", type=int, default=20, help="Number of support-set resampling episodes")
    parser.add_argument("--support-split", type=str, default="train", choices=["train", "val", "test"], help="Split used as the support pool")
    parser.add_argument("--query-split", type=str, default="test", choices=["train", "val", "test"], help="Split evaluated as the query set")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser


def fused_embedding(record: dict) -> torch.Tensor:
    return torch.cat([record["audio_embedding"], record["visual_embedding"]], dim=0).float()


def sample_support_set(records: list[dict], k: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    grouped: dict[int, list[dict]] = defaultdict(list)
    for record in records:
        grouped[record["label_id"]].append(record)
    support: list[dict] = []
    for class_index in range(len(CLASS_NAMES)):
        candidates = grouped.get(class_index, [])
        if not candidates:
            raise ValueError(f"No support examples available for class {CLASS_NAMES[class_index]!r}.")
        if len(candidates) <= k:
            support.extend(candidates)
        else:
            support.extend(rng.sample(candidates, k))
    return support


def build_prototypes(records: list[dict]) -> dict[int, torch.Tensor]:
    grouped: dict[int, list[torch.Tensor]] = defaultdict(list)
    for record in records:
        grouped[record["label_id"]].append(fused_embedding(record))
    prototypes: dict[int, torch.Tensor] = {}
    for class_index in range(len(CLASS_NAMES)):
        vectors = grouped.get(class_index, [])
        if not vectors:
            raise ValueError(f"No support vectors found for class {CLASS_NAMES[class_index]!r}.")
        prototype = torch.stack(vectors, dim=0).mean(dim=0)
        prototypes[class_index] = F.normalize(prototype, dim=0)
    return prototypes


def predict(records: list[dict], prototypes: dict[int, torch.Tensor]) -> list[int]:
    ordered_prototypes = torch.stack([prototypes[index] for index in range(len(CLASS_NAMES))], dim=0)
    predictions: list[int] = []
    for record in records:
        query_embedding = F.normalize(fused_embedding(record), dim=0)
        scores = ordered_prototypes @ query_embedding
        predictions.append(int(scores.argmax().item()))
    return predictions


def main() -> None:
    args = build_arg_parser().parse_args()
    set_seed(args.seed)
    if not args.cache.exists():
        raise FileNotFoundError(f"Feature cache not found: {args.cache}")

    records = load_records(args.cache)
    train_records, val_records, test_records = partition_records(records, seed=args.seed)
    split_map = {"train": train_records, "val": val_records, "test": test_records}
    support_pool = split_map[args.support_split]
    query_records = split_map[args.query_split]
    if len(support_pool) == 0 or len(query_records) == 0:
        raise ValueError("Support and query splits must both contain data.")

    vote_counts = [Counter() for _ in range(len(query_records))]
    true_labels = [record["label_id"] for record in query_records]

    for episode in range(args.episodes):
        support_records = sample_support_set(support_pool, args.k, seed=args.seed + episode)
        prototypes = build_prototypes(support_records)
        episode_predictions = predict(query_records, prototypes)
        for query_index, prediction in enumerate(episode_predictions):
            vote_counts[query_index][prediction] += 1

    final_predictions = [votes.most_common(1)[0][0] for votes in vote_counts]
    metrics = compute_metrics(true_labels, final_predictions, CLASS_NAMES)

    args.results_dir.mkdir(parents=True, exist_ok=True)
    save_confusion_matrix(
        true_labels,
        final_predictions,
        args.results_dir / f"few_shot_k{args.k}_confusion_matrix.png",
        CLASS_NAMES,
        title=f"Few-Shot Prototype Baseline (K={args.k})",
    )
    with (args.results_dir / f"few_shot_k{args.k}_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "k": args.k,
                "episodes": args.episodes,
                "support_split": args.support_split,
                "query_split": args.query_split,
                **metrics,
            },
            handle,
            indent=2,
        )

    print(
        json.dumps(
            {
                "k": args.k,
                "episodes": args.episodes,
                "support_split": args.support_split,
                "query_split": args.query_split,
                **metrics,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
