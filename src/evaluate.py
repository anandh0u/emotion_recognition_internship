from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import build_datasets
from metrics import compute_metrics, save_confusion_matrix, set_seed
from model import MultimodalEmotionModel
from train import select_logits

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FEATURES = PROJECT_ROOT / "features" / "all_embeddings.pt"
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "results"
DEFAULT_CHECKPOINT = PROJECT_ROOT / "models" / "best_model.pt"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a trained late-fusion checkpoint.")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT, help="Path to a saved checkpoint")
    parser.add_argument("--cache", type=Path, default=DEFAULT_FEATURES, help="Path to features/all_embeddings.pt")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test", "all"], help="Dataset split to evaluate")
    parser.add_argument("--modality", type=str, default="auto", choices=["auto", "fusion", "audio", "visual"], help="Prediction head to evaluate")
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR, help="Directory for output plots and metrics")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    class_names: list[str],
    modality: str = "auto",
) -> tuple[dict[str, float], list[int], list[int]]:
    model.eval()
    total_loss = 0.0
    total_items = 0
    y_true: list[int] = []
    y_pred: list[int] = []
    with torch.no_grad():
        for audio_embeddings, visual_embeddings, labels, _sample_ids, audio_available, visual_available in loader:
            audio_embeddings = audio_embeddings.to(device)
            visual_embeddings = visual_embeddings.to(device)
            labels = labels.to(device)
            audio_available = audio_available.to(device)
            visual_available = visual_available.to(device)
            outputs = model(audio_embeddings, visual_embeddings, modality="all")
            logits = select_logits(outputs, audio_available, visual_available, preferred_modality=modality)
            loss = criterion(logits, labels)
            predictions = logits.argmax(dim=-1)
            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_items += batch_size
            y_true.extend(labels.cpu().tolist())
            y_pred.extend(predictions.cpu().tolist())
    metrics = compute_metrics(y_true, y_pred, class_names)
    metrics["loss"] = float(total_loss / max(total_items, 1))
    return metrics, y_true, y_pred


def main() -> None:
    args = build_arg_parser().parse_args()
    set_seed(args.seed)

    if not args.cache.exists():
        raise FileNotFoundError(f"Feature cache not found: {args.cache}")
    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    datasets = build_datasets(args.cache, seed=args.seed)
    split_map = {"train": datasets[0], "val": datasets[1], "test": datasets[2]}
    cache_class_names = datasets[0].class_names
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    class_names = list(checkpoint.get("class_names") or cache_class_names)
    if cache_class_names != class_names:
        raise ValueError(f"Checkpoint classes {class_names} do not match cache classes {cache_class_names}.")
    model = MultimodalEmotionModel(num_classes=len(class_names)).to(device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    criterion = nn.CrossEntropyLoss()
    args.results_dir.mkdir(parents=True, exist_ok=True)

    selected_splits = list(split_map) if args.split == "all" else [args.split]
    results: dict[str, dict[str, float | str]] = {}
    for split_name in selected_splits:
        dataset = split_map[split_name]
        if len(dataset) == 0:
            raise ValueError(f"Split {split_name!r} is empty.")
        loader = DataLoader(dataset, batch_size=args.batch, shuffle=False, num_workers=0, pin_memory=torch.cuda.is_available())
        metrics, y_true, y_pred = evaluate_model(model, loader, criterion, device, class_names, modality=args.modality)
        output_name = "confusion_matrix.png" if split_name == "test" else f"confusion_matrix_{split_name}.png"
        save_confusion_matrix(y_true, y_pred, args.results_dir / output_name, class_names, title=f"{split_name.title()} Confusion Matrix")
        results[split_name] = {"split": split_name, "modality": args.modality, **metrics}

    metrics_file = "evaluation_all_metrics.json" if args.split == "all" else "evaluation_metrics.json"
    output_payload: dict[str, object] = {"results": results} if args.split == "all" else next(iter(results.values()))
    with (args.results_dir / metrics_file).open("w", encoding="utf-8") as handle:
        json.dump(output_payload, handle, indent=2)

    print(json.dumps(output_payload, indent=2))


if __name__ == "__main__":
    main()
