from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import Counter
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2Processor, get_cosine_schedule_with_warmup

from dataset import CLASS_NAMES, normalize_label_for_classes, normalize_split
from metrics import compute_metrics, save_confusion_matrix, set_seed

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LABELS = PROJECT_ROOT / "data" / "labels_ravdess.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "models" / "wav2vec2_emotion"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fine-tune Wav2Vec2 directly on emotion audio files.")
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS, help="CSV manifest with sample_id, split, label, audio_path")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for checkpoint and metrics")
    parser.add_argument("--model-name", type=str, default="facebook/wav2vec2-base", help="Hugging Face Wav2Vec2 checkpoint")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--batch", type=int, default=4, dest="batch_size")
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1, help="Accumulate gradients across this many batches")
    parser.add_argument("--amp", action="store_true", help="Use CUDA mixed precision training when a CUDA device is selected")
    parser.add_argument("--max-duration", type=float, default=4.0, help="Audio crop/pad length in seconds")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--freeze-feature-encoder", action="store_true", help="Freeze convolutional feature extractor")
    parser.add_argument("--freeze-base", action="store_true", help="Freeze Wav2Vec2 base and train classifier/projector only")
    parser.add_argument("--unfreeze-last-n", type=int, default=2, help="When --freeze-base is set, unfreeze last N encoder layers")
    parser.add_argument("--limit-train", type=int, default=None, help="Debug limit for train rows")
    parser.add_argument("--limit-val", type=int, default=None, help="Debug limit for validation rows")
    parser.add_argument("--limit-test", type=int, default=None, help="Debug limit for test rows")
    parser.add_argument("--no-save-model", action="store_true", help="Run training/evaluation without writing Wav2Vec2 weights")
    return parser


def resolve_path(value: Any, labels_dir: Path) -> Path:
    path = Path(str(value).strip())
    if path.is_absolute():
        return path
    candidates = [labels_dir / path, PROJECT_ROOT / path]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def load_manifest(labels_csv: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    labels_dir = labels_csv.parent
    with labels_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"sample_id", "split", "label", "audio_path"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{labels_csv} is missing required columns: {sorted(missing)}")
        for row in reader:
            audio_text = str(row.get("audio_path", "")).strip()
            if not audio_text:
                continue
            audio_path = resolve_path(audio_text, labels_dir)
            if not audio_path.exists():
                continue
            split = normalize_split(row.get("split"))
            if split not in {"train", "val", "test"}:
                continue
            label = normalize_label_for_classes(row["label"], CLASS_NAMES)
            rows.append(
                {
                    "sample_id": str(row.get("sample_id") or audio_path.stem),
                    "split": split,
                    "label": label,
                    "label_id": CLASS_NAMES.index(label),
                    "audio_path": str(audio_path),
                    "actor": str(row.get("actor", "")),
                }
            )
    if not rows:
        raise ValueError(f"No usable audio rows found in {labels_csv}")
    return rows


def limit_rows(rows: list[dict[str, Any]], limit: int | None, seed: int) -> list[dict[str, Any]]:
    if limit is None or len(rows) <= limit:
        return rows
    rng = random.Random(seed)
    selected = rows[:]
    rng.shuffle(selected)
    return selected[:limit]


class AudioEmotionDataset(Dataset):
    def __init__(self, rows: list[dict[str, Any]], max_duration: float, sample_rate: int = 16000) -> None:
        self.rows = rows
        self.max_samples = int(max_duration * sample_rate)
        self.sample_rate = sample_rate

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        waveform, _ = librosa.load(row["audio_path"], sr=self.sample_rate, mono=True)
        if waveform.size == 0:
            waveform = np.zeros(1, dtype=np.float32)
        waveform = waveform.astype(np.float32, copy=False)
        if waveform.shape[0] > self.max_samples:
            waveform = waveform[: self.max_samples]
        return {
            "waveform": waveform,
            "label": int(row["label_id"]),
            "sample_id": row["sample_id"],
        }


def build_collate_fn(processor: Wav2Vec2Processor, sample_rate: int = 16000):
    def collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
        waveforms = [item["waveform"] for item in batch]
        labels = torch.tensor([item["label"] for item in batch], dtype=torch.long)
        inputs = processor(waveforms, sampling_rate=sample_rate, return_tensors="pt", padding=True)
        inputs["labels"] = labels
        inputs["sample_ids"] = [item["sample_id"] for item in batch]
        return inputs

    return collate


def configure_trainable_layers(
    model: Wav2Vec2ForSequenceClassification,
    freeze_feature_encoder: bool,
    freeze_base: bool,
    unfreeze_last_n: int,
) -> None:
    if freeze_feature_encoder and hasattr(model, "freeze_feature_encoder"):
        model.freeze_feature_encoder()

    if not freeze_base:
        return

    for parameter in model.wav2vec2.parameters():
        parameter.requires_grad = False

    layers = getattr(model.wav2vec2.encoder, "layers", [])
    if unfreeze_last_n > 0:
        for layer in layers[-unfreeze_last_n:]:
            for parameter in layer.parameters():
                parameter.requires_grad = True

    for module_name in ["projector", "classifier"]:
        module = getattr(model, module_name, None)
        if module is not None:
            for parameter in module.parameters():
                parameter.requires_grad = True


def class_weights(rows: list[dict[str, Any]], device: torch.device) -> torch.Tensor:
    counts = Counter(int(row["label_id"]) for row in rows)
    total = sum(counts.values())
    weights = []
    for index in range(len(CLASS_NAMES)):
        count = max(counts.get(index, 0), 1)
        weights.append(total / (len(CLASS_NAMES) * count))
    return torch.tensor(weights, dtype=torch.float32, device=device)


def evaluate(
    model: Wav2Vec2ForSequenceClassification,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    use_amp: bool = False,
) -> tuple[dict[str, float], list[int], list[int]]:
    model.eval()
    total_loss = 0.0
    total_items = 0
    y_true: list[int] = []
    y_pred: list[int] = []
    with torch.no_grad():
        for batch in loader:
            labels = batch.pop("labels").to(device)
            batch.pop("sample_ids", None)
            inputs = {key: value.to(device) for key, value in batch.items()}
            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = model(**inputs).logits
                loss = criterion(logits, labels)
            predictions = logits.argmax(dim=-1)
            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_items += batch_size
            y_true.extend(labels.cpu().tolist())
            y_pred.extend(predictions.cpu().tolist())
    metrics = compute_metrics(y_true, y_pred, CLASS_NAMES)
    metrics["loss"] = float(total_loss / max(total_items, 1))
    return metrics, y_true, y_pred


def train_one_epoch(
    model: Wav2Vec2ForSequenceClassification,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    device: torch.device,
    grad_clip: float,
    gradient_accumulation_steps: int,
    use_amp: bool,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    total_items = 0
    y_true: list[int] = []
    y_pred: list[int] = []
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    accumulation_steps = max(int(gradient_accumulation_steps), 1)
    optimizer.zero_grad(set_to_none=True)
    for step, batch in enumerate(tqdm(loader, desc="Fine-tuning", leave=False), start=1):
        labels = batch.pop("labels").to(device)
        batch.pop("sample_ids", None)
        inputs = {key: value.to(device) for key, value in batch.items()}
        with torch.cuda.amp.autocast(enabled=use_amp):
            logits = model(**inputs).logits
            loss = criterion(logits, labels)
        scaled_loss = loss / accumulation_steps
        scaler.scale(scaled_loss).backward()
        if step % accumulation_steps == 0 or step == len(loader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
        predictions = logits.argmax(dim=-1)
        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_items += batch_size
        y_true.extend(labels.detach().cpu().tolist())
        y_pred.extend(predictions.detach().cpu().tolist())
    metrics = compute_metrics(y_true, y_pred, CLASS_NAMES)
    metrics["loss"] = float(total_loss / max(total_items, 1))
    return metrics


def main() -> None:
    args = build_arg_parser().parse_args()
    set_seed(args.seed)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    rows = load_manifest(args.labels)
    train_rows = limit_rows([row for row in rows if row["split"] == "train"], args.limit_train, args.seed)
    val_rows = limit_rows([row for row in rows if row["split"] == "val"], args.limit_val, args.seed)
    test_rows = limit_rows([row for row in rows if row["split"] == "test"], args.limit_test, args.seed)
    if not train_rows or not val_rows or not test_rows:
        raise ValueError("Train, validation, and test splits must each contain usable audio rows.")

    if args.gradient_accumulation_steps < 1:
        raise ValueError("--gradient-accumulation-steps must be at least 1.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    processor = Wav2Vec2Processor.from_pretrained(args.model_name)
    model = Wav2Vec2ForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=len(CLASS_NAMES),
        label2id={label: index for index, label in enumerate(CLASS_NAMES)},
        id2label={index: label for index, label in enumerate(CLASS_NAMES)},
        problem_type="single_label_classification",
    ).to(device)
    configure_trainable_layers(model, args.freeze_feature_encoder, args.freeze_base, args.unfreeze_last_n)

    collate_fn = build_collate_fn(processor)
    train_loader = DataLoader(
        AudioEmotionDataset(train_rows, args.max_duration),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        AudioEmotionDataset(val_rows, args.max_duration),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        AudioEmotionDataset(test_rows, args.max_duration),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available(),
    )

    criterion = nn.CrossEntropyLoss(weight=class_weights(train_rows, device))
    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = AdamW(trainable_parameters, lr=args.lr, weight_decay=args.weight_decay)
    optimizer_steps_per_epoch = int(math.ceil(len(train_loader) / max(args.gradient_accumulation_steps, 1)))
    total_steps = max(optimizer_steps_per_epoch * args.epochs, 1)
    warmup_steps = int(math.ceil(total_steps * args.warmup_ratio))
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    use_amp = bool(args.amp and device.type == "cuda")

    config = {
        **{key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "device": str(device),
        "train_rows": len(train_rows),
        "val_rows": len(val_rows),
        "test_rows": len(test_rows),
        "class_names": CLASS_NAMES,
    }
    best_val_f1 = -1.0
    history: list[dict[str, float]] = []
    best_state: dict[str, torch.Tensor] | None = None
    best_dir = args.output_dir / "best_model"

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            scheduler,
            device,
            args.grad_clip,
            args.gradient_accumulation_steps,
            use_amp,
        )
        val_metrics, _val_true, _val_pred = evaluate(model, val_loader, criterion, device, use_amp=use_amp)
        row = {f"train_{key}": value for key, value in train_metrics.items()}
        row.update({f"val_{key}": value for key, value in val_metrics.items()})
        row["epoch"] = float(epoch)
        history.append(row)
        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train_f1={train_metrics['weighted_f1']:.4f} val_f1={val_metrics['weighted_f1']:.4f} "
            f"val_acc={val_metrics['accuracy']:.4f}"
        )
        if val_metrics["weighted_f1"] >= best_val_f1:
            best_val_f1 = val_metrics["weighted_f1"]
            if args.no_save_model:
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            else:
                model.save_pretrained(best_dir)
                processor.save_pretrained(best_dir)

    if args.no_save_model:
        if best_state is not None:
            model.load_state_dict(best_state)
        model.to(device)
    else:
        model = Wav2Vec2ForSequenceClassification.from_pretrained(best_dir).to(device)
    test_metrics, test_true, test_pred = evaluate(model, test_loader, criterion, device, use_amp=use_amp)
    save_confusion_matrix(test_true, test_pred, args.output_dir / "confusion_matrix.png", CLASS_NAMES, "Wav2Vec2 Audio Test Confusion Matrix")
    summary = {
        "config": config,
        "best_val_f1": best_val_f1,
        "test_metrics": test_metrics,
        "history": history,
    }
    with (args.output_dir / "training_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps({"best_val_f1": best_val_f1, "test_metrics": test_metrics}, indent=2))


if __name__ == "__main__":
    main()
