from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path
from typing import Iterable

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import EmotionFusionDataset, build_datasets
from metrics import compute_metrics, save_confusion_matrix, set_seed
from model import MultimodalEmotionModel

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FEATURES = PROJECT_ROOT / "features" / "all_embeddings.pt"
DEFAULT_MODELS_DIR = PROJECT_ROOT / "models"
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "results"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the late-fusion emotion classifier.")
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES, help="Path to features/all_embeddings.pt")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--batch", type=int, default=16, dest="batch_size", help="Batch size")
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--weight-decay", type=float, default=1e-2, help="AdamW weight decay")
    parser.add_argument("--grad-clip", type=float, default=1.0, help="Gradient clipping norm")
    parser.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS_DIR, help="Directory for checkpoints")
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR, help="Directory for metrics and plots")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader worker count")
    parser.add_argument("--device", type=str, default=None, help="Device override, for example cpu or cuda")
    return parser


def make_loader(dataset: EmotionFusionDataset, batch_size: int, shuffle: bool, num_workers: int) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def select_logits(
    outputs: dict[str, torch.Tensor],
    audio_available: torch.Tensor,
    visual_available: torch.Tensor,
    preferred_modality: str = "auto",
) -> torch.Tensor:
    selected: list[torch.Tensor] = []
    for index in range(audio_available.size(0)):
        has_audio = bool(audio_available[index].item())
        has_visual = bool(visual_available[index].item())
        if preferred_modality == "audio" and has_audio:
            selected.append(outputs["audio"][index])
        elif preferred_modality == "visual" and has_visual:
            selected.append(outputs["visual"][index])
        elif preferred_modality == "fusion" and has_audio and has_visual:
            selected.append(outputs["fusion"][index])
        elif has_audio and has_visual:
            selected.append(outputs["fusion"][index])
        elif has_audio:
            selected.append(outputs["audio"][index])
        elif has_visual:
            selected.append(outputs["visual"][index])
        else:
            raise ValueError("A batch item has no available modality.")
    return torch.stack(selected, dim=0)


def compute_multimodal_loss(
    outputs: dict[str, torch.Tensor],
    labels: torch.Tensor,
    audio_available: torch.Tensor,
    visual_available: torch.Tensor,
    criterion: nn.Module,
) -> torch.Tensor:
    losses: list[torch.Tensor] = []
    both_available = audio_available & visual_available
    if both_available.any():
        losses.append(criterion(outputs["fusion"][both_available], labels[both_available]))
    if audio_available.any():
        losses.append(criterion(outputs["audio"][audio_available], labels[audio_available]))
    if visual_available.any():
        losses.append(criterion(outputs["visual"][visual_available], labels[visual_available]))
    if not losses:
        raise ValueError("Cannot compute loss for a batch with no available modalities.")
    return torch.stack(losses).mean()


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    class_names: list[str],
    preferred_modality: str = "auto",
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
            logits = select_logits(outputs, audio_available, visual_available, preferred_modality=preferred_modality)
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


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip: float,
    class_names: list[str],
    preferred_modality: str = "auto",
) -> tuple[dict[str, float], list[int], list[int]]:
    model.train()
    total_loss = 0.0
    total_items = 0
    y_true: list[int] = []
    y_pred: list[int] = []
    for audio_embeddings, visual_embeddings, labels, _sample_ids, audio_available, visual_available in tqdm(loader, desc="Training", leave=False):
        audio_embeddings = audio_embeddings.to(device)
        visual_embeddings = visual_embeddings.to(device)
        labels = labels.to(device)
        audio_available = audio_available.to(device)
        visual_available = visual_available.to(device)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(audio_embeddings, visual_embeddings, modality="all")
        loss = compute_multimodal_loss(outputs, labels, audio_available, visual_available, criterion)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        logits = select_logits(outputs, audio_available, visual_available, preferred_modality=preferred_modality)
        predictions = logits.argmax(dim=-1)
        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_items += batch_size
        y_true.extend(labels.cpu().tolist())
        y_pred.extend(predictions.cpu().tolist())
    metrics = compute_metrics(y_true, y_pred, class_names)
    metrics["loss"] = float(total_loss / max(total_items, 1))
    return metrics, y_true, y_pred


def maybe_init_wandb(args: argparse.Namespace) -> object | None:
    if not args.wandb:
        return None
    try:
        import wandb
    except Exception as exc:  # pragma: no cover - logging fallback
        print(f"W&B logging disabled: {exc}")
        return None
    run = wandb.init(project="multimodal-emotion-recognition", config=vars(args), reinit=True)
    return run


def serializable_config(args: argparse.Namespace) -> dict[str, object]:
    config: dict[str, object] = {}
    for key, value in vars(args).items():
        config[key] = str(value) if isinstance(value, Path) else value
    return config


def main() -> None:
    args = build_arg_parser().parse_args()
    set_seed(args.seed)

    if not args.features.exists():
        raise FileNotFoundError(f"Feature file not found: {args.features}")

    train_dataset, val_dataset, test_dataset = build_datasets(args.features, seed=args.seed)
    if len(train_dataset) == 0 or len(val_dataset) == 0 or len(test_dataset) == 0:
        raise ValueError("Training, validation, and test splits must each contain at least one sample.")
    class_names = train_dataset.class_names

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested with --device cuda, but torch.cuda.is_available() is False.")
    print(f"Using device: {device}")
    model = MultimodalEmotionModel(num_classes=len(class_names)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))

    train_loader = make_loader(train_dataset, args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = make_loader(val_dataset, args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = make_loader(test_dataset, args.batch_size, shuffle=False, num_workers=args.num_workers)

    args.models_dir.mkdir(parents=True, exist_ok=True)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.models_dir / "best_model.pt"
    last_checkpoint_path = args.models_dir / "last_model.pt"
    wandb_run = maybe_init_wandb(args)
    config = serializable_config(args)

    best_val_f1 = -1.0
    best_state = None
    history: list[dict[str, float]] = []

    for epoch in range(1, args.epochs + 1):
        train_metrics, _train_true, _train_pred = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            args.grad_clip,
            class_names,
        )
        val_metrics, _val_true, _val_pred = evaluate_model(model, val_loader, criterion, device, class_names)
        scheduler.step()
        row = {f"train_{key}": value for key, value in train_metrics.items()}
        row.update({f"val_{key}": value for key, value in val_metrics.items()})
        row["epoch"] = float(epoch)
        history.append(row)
        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train_loss={train_metrics['loss']:.4f} train_f1={train_metrics['weighted_f1']:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} val_f1={val_metrics['weighted_f1']:.4f}"
        )
        if wandb_run is not None:
            wandb_run.log(row)
        if val_metrics["weighted_f1"] >= best_val_f1:
            best_val_f1 = val_metrics["weighted_f1"]
            best_state = deepcopy(model.state_dict())
            torch.save(
                {
                    "model_state_dict": best_state,
                    "class_names": class_names,
                    "epoch": epoch,
                    "best_val_f1": best_val_f1,
                    "config": config,
                },
                checkpoint_path,
            )
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "class_names": class_names,
                "epoch": epoch,
                "config": config,
            },
            last_checkpoint_path,
        )

    if best_state is not None:
        model.load_state_dict(best_state)
    test_metrics, test_true, test_pred = evaluate_model(model, test_loader, criterion, device, class_names)
    save_confusion_matrix(test_true, test_pred, args.results_dir / "confusion_matrix.png", class_names, title="Test Confusion Matrix")

    summary = {
        "best_val_f1": best_val_f1,
        "test_metrics": test_metrics,
        "history": history,
    }
    with (args.results_dir / "training_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    if wandb_run is not None:
        wandb_run.log({f"test_{key}": value for key, value in test_metrics.items()})
        wandb_run.finish()

    print(f"Best checkpoint saved to {checkpoint_path}")
    print(f"Test metrics: {json.dumps(test_metrics, indent=2)}")


if __name__ == "__main__":
    main()
