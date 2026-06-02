from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

CLASS_NAMES = [
    "anger",
    "disgust",
    "fear",
    "happiness",
    "neutral",
    "sadness",
    "surprise",
]

LABEL_TO_INDEX = {name: index for index, name in enumerate(CLASS_NAMES)}
NUMERIC_LABEL_MAP = {
    "0": "anger",
    "1": "disgust",
    "2": "fear",
    "3": "happiness",
    "4": "sadness",
    "5": "surprise",
    "6": "neutral",
}
LABEL_ALIASES = {
    "a": "anger",
    "ang": "anger",
    "angry": "anger",
    "anger": "anger",
    "d": "disgust",
    "dis": "disgust",
    "disgust": "disgust",
    "f": "fear",
    "fear": "fear",
    "h": "happiness",
    "hap": "happiness",
    "happy": "happiness",
    "happiness": "happiness",
    "n": "neutral",
    "neu": "neutral",
    "neutral": "neutral",
    "sa": "sadness",
    "sad": "sadness",
    "sadness": "sadness",
    "su": "surprise",
    "surp": "surprise",
    "surprise": "surprise",
}
SPLIT_ALIASES = {
    "train": "train",
    "training": "train",
    "val": "val",
    "valid": "val",
    "validation": "val",
    "dev": "val",
    "test": "test",
    "eval": "test",
    "evaluation": "test",
}


def normalize_label(value: Any) -> str:
    if value is None:
        raise ValueError("Label value is missing.")
    text = str(value).strip().lower()
    if text in NUMERIC_LABEL_MAP:
        return NUMERIC_LABEL_MAP[text]
    if text in LABEL_ALIASES:
        return LABEL_ALIASES[text]
    raise ValueError(f"Unsupported label value: {value!r}")


def canonical_label(value: Any) -> str:
    return str(value).strip().lower().replace(" ", "_").replace("-", "_")


def normalize_label_for_classes(value: Any, class_names: Sequence[str] | None = None) -> str:
    names = list(class_names or CLASS_NAMES)
    if names == CLASS_NAMES:
        return normalize_label(value)

    class_lookup = {canonical_label(name): name for name in names}
    text = canonical_label(value)
    if text in class_lookup:
        return class_lookup[text]

    if text.isdigit():
        label_index = int(text)
        if 0 <= label_index < len(names):
            return names[label_index]

    try:
        emotion_label = normalize_label(value)
    except ValueError:
        emotion_label = ""
    if emotion_label in names:
        return emotion_label

    raise ValueError(f"Unsupported label value {value!r} for classes: {names}")


def normalize_split(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    if not text:
        return None
    return SPLIT_ALIASES.get(text, text)


def load_feature_store(feature_file: str | Path) -> dict[str, Any]:
    payload = torch.load(Path(feature_file), map_location="cpu")
    if isinstance(payload, list):
        return {"class_names": CLASS_NAMES, "samples": payload}
    if not isinstance(payload, dict):
        raise TypeError(f"Unexpected feature store format: {type(payload)!r}")
    if "samples" not in payload:
        if "entries" in payload:
            payload = dict(payload)
            payload["samples"] = payload["entries"]
        else:
            raise KeyError("Feature store does not contain a 'samples' key.")
    return payload


def get_class_names(payload: dict[str, Any] | None = None) -> list[str]:
    if payload is None:
        return CLASS_NAMES[:]
    values = payload.get("class_names") or CLASS_NAMES
    return [canonical_label(value) for value in values]


def normalize_record(record: dict[str, Any], class_names: Sequence[str] | None = None) -> dict[str, Any]:
    names = list(class_names or CLASS_NAMES)
    label_to_index = {name: index for index, name in enumerate(names)}
    item = dict(record)
    item["label"] = normalize_label_for_classes(item["label"], names)
    item["label_id"] = label_to_index[item["label"]]
    item["split"] = normalize_split(item.get("split"))
    sample_id = item.get("sample_id", item.get("id", ""))
    item["sample_id"] = str(sample_id) if sample_id is not None else ""
    audio_embedding = item.get("audio_embedding")
    visual_embedding = item.get("visual_embedding")
    item["audio_available"] = bool(item.get("audio_available", audio_embedding is not None))
    item["visual_available"] = bool(item.get("visual_available", visual_embedding is not None))

    if audio_embedding is None:
        item["audio_embedding"] = torch.zeros(768, dtype=torch.float32)
        item["audio_available"] = False
    else:
        item["audio_embedding"] = torch.as_tensor(audio_embedding, dtype=torch.float32).reshape(-1).cpu()

    if visual_embedding is None:
        item["visual_embedding"] = torch.zeros(768, dtype=torch.float32)
        item["visual_available"] = False
    else:
        item["visual_embedding"] = torch.as_tensor(visual_embedding, dtype=torch.float32).reshape(-1).cpu()

    if not item["audio_available"] and not item["visual_available"]:
        raise ValueError(f"Sample {item['sample_id']!r} has no available modality embeddings.")
    return item


def load_records(feature_file: str | Path) -> list[dict[str, Any]]:
    payload = load_feature_store(feature_file)
    class_names = get_class_names(payload)
    samples = payload["samples"]
    if not isinstance(samples, list):
        raise TypeError("Feature store samples must be a list.")
    return [normalize_record(sample, class_names) for sample in samples]


def load_records_and_class_names(feature_file: str | Path) -> tuple[list[dict[str, Any]], list[str]]:
    payload = load_feature_store(feature_file)
    class_names = get_class_names(payload)
    samples = payload["samples"]
    if not isinstance(samples, list):
        raise TypeError("Feature store samples must be a list.")
    return [normalize_record(sample, class_names) for sample in samples], class_names


def has_complete_explicit_splits(records: list[dict[str, Any]]) -> bool:
    splits = {record.get("split") for record in records}
    return {"train", "val", "test"}.issubset(splits)


def partition_records(
    records: list[dict[str, Any]],
    seed: int = 42,
    class_names: Sequence[str] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    normalized = [normalize_record(record, class_names) for record in records]
    if has_complete_explicit_splits(normalized):
        train_records = [record for record in normalized if record.get("split") == "train"]
        val_records = [record for record in normalized if record.get("split") == "val"]
        test_records = [record for record in normalized if record.get("split") == "test"]
        if train_records and val_records and test_records:
            return train_records, val_records, test_records

    labels = [record["label"] for record in normalized]
    indices = list(range(len(normalized)))
    if len(normalized) < 3:
        if len(normalized) == 0:
            return [], [], []
        if len(normalized) == 1:
            return normalized[:], [], []
        return normalized[:1], normalized[1:], []

    try:
        train_indices, temp_indices = train_test_split(
            indices,
            test_size=0.3,
            random_state=seed,
            stratify=labels,
        )
        temp_labels = [labels[index] for index in temp_indices]
        val_indices, test_indices = train_test_split(
            temp_indices,
            test_size=0.5,
            random_state=seed,
            stratify=temp_labels,
        )
    except ValueError:
        rng = random.Random(seed)
        rng.shuffle(indices)
        train_cut = max(1, int(round(len(indices) * 0.7)))
        val_cut = max(1, int(round(len(indices) * 0.15)))
        if train_cut + val_cut >= len(indices):
            val_cut = max(0, len(indices) - train_cut - 1)
        train_indices = indices[:train_cut]
        val_indices = indices[train_cut:train_cut + val_cut]
        test_indices = indices[train_cut + val_cut:]

    train_records = [normalized[index] for index in train_indices]
    val_records = [normalized[index] for index in val_indices]
    test_records = [normalized[index] for index in test_indices]
    return train_records, val_records, test_records


def build_datasets(feature_file: str | Path, seed: int = 42) -> tuple["EmotionFusionDataset", "EmotionFusionDataset", "EmotionFusionDataset"]:
    records, class_names = load_records_and_class_names(feature_file)
    train_records, val_records, test_records = partition_records(records, seed=seed, class_names=class_names)
    return (
        EmotionFusionDataset(train_records, class_names=class_names),
        EmotionFusionDataset(val_records, class_names=class_names),
        EmotionFusionDataset(test_records, class_names=class_names),
    )


class EmotionFusionDataset(Dataset):
    def __init__(
        self,
        data_source: str | Path | list[dict[str, Any]],
        split: str | None = None,
        class_names: Sequence[str] | None = None,
    ) -> None:
        if isinstance(data_source, (str, Path)):
            records, loaded_class_names = load_records_and_class_names(data_source)
            self.class_names = loaded_class_names
        else:
            self.class_names = list(class_names or CLASS_NAMES)
            records = [normalize_record(record, self.class_names) for record in data_source]
        if split is not None:
            split_name = normalize_split(split)
            if split_name is None:
                raise ValueError("Split must be a non-empty string.")
            records = [record for record in records if record.get("split") == split_name]
        self.records = records
        self.label_to_index = {name: index for index, name in enumerate(self.class_names)}

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, str, torch.Tensor, torch.Tensor]:
        record = self.records[index]
        audio_embedding = record["audio_embedding"].float()
        visual_embedding = record["visual_embedding"].float()
        label = torch.tensor(record["label_id"], dtype=torch.long)
        sample_id = record.get("sample_id", str(index))
        audio_available = torch.tensor(record["audio_available"], dtype=torch.bool)
        visual_available = torch.tensor(record["visual_available"], dtype=torch.bool)
        return audio_embedding, visual_embedding, label, sample_id, audio_available, visual_available
