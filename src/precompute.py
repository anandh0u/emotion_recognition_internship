from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any, Sequence

import librosa
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoImageProcessor, ViTModel, Wav2Vec2Model, Wav2Vec2Processor

from dataset import CLASS_NAMES, canonical_label, normalize_label, normalize_label_for_classes, normalize_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LABELS = PROJECT_ROOT / "data" / "labels.csv"
DEFAULT_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DEFAULT_OUTPUT = PROJECT_ROOT / "features" / "all_embeddings.pt"
ImageSource = Path | tuple[Path, int]


def resolve_path(value: Any, raw_dir: Path, labels_dir: Path) -> Path:
    path = Path(str(value).strip())
    if path.is_absolute() and path.exists():
        return path
    candidates = [raw_dir / path, labels_dir / path, PROJECT_ROOT / path]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not resolve path {value!r} from {raw_dir} or {labels_dir}.")


def parse_fer2013_reference(value: Any, raw_dir: Path, labels_dir: Path) -> tuple[Path, int] | None:
    text = str(value or "").strip()
    if not text:
        return None

    lowered = text.lower()
    if lowered.startswith("fer2013:"):
        row_text = text.split(":", 1)[1].strip()
        if not row_text.isdigit():
            raise ValueError(f"Invalid FER2013 row reference: {value!r}")
        return resolve_path("fer2013.csv", raw_dir=raw_dir, labels_dir=labels_dir), int(row_text)

    marker = "#row="
    marker_index = lowered.find(marker)
    if marker_index == -1:
        return None

    file_part = text[:marker_index].strip() or "fer2013.csv"
    row_text = text[marker_index + len(marker):].strip()
    if not row_text.isdigit():
        raise ValueError(f"Invalid FER2013 row reference: {value!r}")
    csv_path = resolve_path(file_part, raw_dir=raw_dir, labels_dir=labels_dir)
    if csv_path.suffix.lower() != ".csv":
        raise ValueError(f"FER2013 references must point to a CSV file: {value!r}")
    return csv_path, int(row_text)


def resolve_image_source(value: Any, raw_dir: Path, labels_dir: Path) -> ImageSource:
    text = str(value or "").strip()
    if not text:
        raise ValueError("image_path is empty. Provide an image file or a FER2013 row reference.")
    fer_reference = parse_fer2013_reference(text, raw_dir=raw_dir, labels_dir=labels_dir)
    if fer_reference is not None:
        return fer_reference
    return resolve_path(text, raw_dir=raw_dir, labels_dir=labels_dir)


def image_source_key(source: ImageSource) -> str:
    if isinstance(source, tuple):
        csv_path, row_index = source
        return f"{csv_path.resolve()}#row={row_index}"
    return str(source.resolve())


def fer2013_pixels_to_image(pixels: str) -> Image.Image:
    values = np.fromstring(pixels, dtype=np.uint8, sep=" ")
    side = int(np.sqrt(values.size))
    if side * side != values.size:
        raise ValueError(f"FER2013 pixel row has {values.size} values; expected a square image.")
    array = values.reshape(side, side)
    return Image.fromarray(array).convert("RGB")


def load_fer2013_image_cache(rows: list[dict[str, Any]], raw_dir: Path, labels_dir: Path) -> dict[str, Image.Image]:
    requests: dict[Path, set[int]] = {}
    expected_labels: dict[str, str] = {}

    for row in rows:
        reference = parse_fer2013_reference(row.get("image_path", ""), raw_dir=raw_dir, labels_dir=labels_dir)
        if reference is None:
            continue
        csv_path, row_index = reference
        requests.setdefault(csv_path, set()).add(row_index)
        expected_labels[image_source_key(reference)] = normalize_label(row["label"])

    image_cache: dict[str, Image.Image] = {}
    for csv_path, requested_indices in requests.items():
        found_indices: set[int] = set()
        with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            fieldnames = set(reader.fieldnames or [])
            missing = {"emotion", "pixels"} - fieldnames
            if missing:
                raise ValueError(f"{csv_path} is missing FER2013 columns: {sorted(missing)}")
            for row_index, fer_row in enumerate(reader):
                if row_index not in requested_indices:
                    continue
                source = (csv_path, row_index)
                key = image_source_key(source)
                fer_label = normalize_label(fer_row["emotion"])
                expected_label = expected_labels.get(key)
                if expected_label is not None and fer_label != expected_label:
                    raise ValueError(
                        f"FER2013 row {row_index} label is {fer_label!r}, "
                        f"but labels.csv expects {expected_label!r}."
                    )
                image_cache[key] = fer2013_pixels_to_image(fer_row["pixels"])
                found_indices.add(row_index)
                if found_indices == requested_indices:
                    break
        missing_indices = requested_indices - found_indices
        if missing_indices:
            preview = sorted(missing_indices)[:5]
            raise IndexError(f"{csv_path} does not contain requested FER2013 row(s): {preview}")

    return image_cache


def load_manifest(labels_csv: Path) -> list[dict[str, Any]]:
    if not labels_csv.exists():
        raise FileNotFoundError(f"Labels file not found: {labels_csv}")
    with labels_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or [])
        required = {"sample_id", "label", "audio_path", "image_path"}
        missing = required - fieldnames
        if missing:
            raise ValueError(f"labels.csv is missing required columns: {sorted(missing)}")
        rows = list(reader)
    if not rows:
        raise ValueError("labels.csv does not contain any data rows.")
    return rows


def parse_class_names(value: str | None) -> list[str] | None:
    if value is None:
        return None
    names = [canonical_label(part) for part in value.split(",") if part.strip()]
    if not names:
        raise ValueError("--class-names was provided but no class names were found.")
    if len(set(names)) != len(names):
        raise ValueError(f"Class names must be unique: {names}")
    return names


def infer_class_names(rows: list[dict[str, Any]], explicit_class_names: Sequence[str] | None = None) -> list[str]:
    if explicit_class_names is not None:
        return [canonical_label(name) for name in explicit_class_names]

    inferred_default_labels: list[str] = []
    unknown_labels: list[str] = []
    for row in rows:
        raw_label = row["label"]
        try:
            inferred_default_labels.append(normalize_label(raw_label))
        except ValueError:
            unknown_labels.append(canonical_label(raw_label))

    if not unknown_labels and set(inferred_default_labels).issubset(set(CLASS_NAMES)):
        return CLASS_NAMES[:]

    seen: list[str] = []
    for row in rows:
        label = canonical_label(row["label"])
        if label not in seen:
            seen.append(label)
    if len(seen) < 2:
        raise ValueError(f"At least two classes are required, found: {seen}")
    return seen


def has_value(value: Any) -> bool:
    return bool(str(value or "").strip())


def extract_audio_embedding(audio_path: Path, processor: Wav2Vec2Processor, model: Wav2Vec2Model, device: torch.device) -> torch.Tensor:
    waveform, _ = librosa.load(audio_path, sr=16000, mono=True)
    if waveform.size == 0:
        raise ValueError(f"Audio file is empty: {audio_path}")
    inputs = processor(waveform, sampling_rate=16000, return_tensors="pt", padding=True)
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.inference_mode():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze(0).detach().cpu()


def extract_visual_embedding_from_image(image: Image.Image, processor: AutoImageProcessor, model: ViTModel, device: torch.device) -> torch.Tensor:
    inputs = processor(images=image, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.inference_mode():
        outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :].squeeze(0).detach().cpu()


def numpy_array_to_image(array: np.ndarray, path: Path) -> Image.Image:
    if array.ndim == 2:
        return Image.fromarray(array.astype(np.uint8)).convert("RGB")
    if array.ndim == 3 and array.shape[-1] in {1, 3, 4}:
        if np.issubdtype(array.dtype, np.floating):
            max_value = float(np.nanmax(array)) if array.size else 1.0
            if max_value <= 1.0:
                array = array * 255.0
        array = np.nan_to_num(array, nan=0.0, posinf=255.0, neginf=0.0)
        array = np.clip(array, 0, 255).astype(np.uint8)
        if array.shape[-1] == 1:
            array = array[..., 0]
        return Image.fromarray(array).convert("RGB")
    raise ValueError(f"Unsupported .npy image shape for {path}: {array.shape}")


def load_image_file(image_path: Path) -> Image.Image:
    if image_path.suffix.lower() == ".npy":
        array = np.load(image_path, allow_pickle=False)
        return numpy_array_to_image(array, image_path)
    return Image.open(image_path).convert("RGB")


def extract_visual_embedding(image_path: Path, processor: AutoImageProcessor, model: ViTModel, device: torch.device) -> torch.Tensor:
    image = load_image_file(image_path)
    return extract_visual_embedding_from_image(image, processor, model, device)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Precompute Wav2Vec2 and ViT embeddings once.")
    parser.add_argument("--labels", "--labels-csv", dest="labels", type=Path, default=DEFAULT_LABELS, help="Path to data/labels.csv")
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR, help="Root directory for raw media files")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output .pt file for cached embeddings")
    parser.add_argument("--audio-model", type=str, default="facebook/wav2vec2-base", help="HuggingFace audio backbone")
    parser.add_argument("--visual-model", type=str, default="google/vit-base-patch16-224", help="HuggingFace visual backbone")
    parser.add_argument(
        "--class-names",
        type=str,
        default=None,
        help="Optional comma-separated classes. Defaults to the 7 emotion classes when possible, otherwise inferred from labels.",
    )
    parser.add_argument("--device", type=str, default=None, help="Device override, for example cpu or cuda")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite an existing output file")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    output_path = args.output
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"Feature cache already exists: {output_path}. Use --overwrite to replace it.")

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    labels_csv = args.labels
    raw_dir = args.raw_dir
    labels_dir = labels_csv.parent
    rows = load_manifest(labels_csv)
    class_names = infer_class_names(rows, parse_class_names(args.class_names))
    fer2013_images = load_fer2013_image_cache(rows, raw_dir=raw_dir, labels_dir=labels_dir)

    audio_processor = Wav2Vec2Processor.from_pretrained(args.audio_model)
    audio_model = Wav2Vec2Model.from_pretrained(args.audio_model).to(device).eval()
    visual_processor = AutoImageProcessor.from_pretrained(args.visual_model)
    visual_model = ViTModel.from_pretrained(args.visual_model).to(device).eval()

    audio_cache: dict[str, torch.Tensor] = {}
    visual_cache: dict[str, torch.Tensor] = {}
    samples: list[dict[str, Any]] = []

    for index, row in enumerate(tqdm(rows, desc="Precomputing embeddings")):
        sample_id = str(row.get("sample_id") or f"sample_{index:06d}")
        label = normalize_label_for_classes(row["label"], class_names)
        split = normalize_split(row.get("split"))
        has_audio = has_value(row.get("audio_path"))
        has_image = has_value(row.get("image_path"))
        if not has_audio and not has_image:
            raise ValueError(f"Sample {sample_id!r} has neither audio_path nor image_path.")

        audio_key = ""
        image_key = ""
        audio_embedding = None
        visual_embedding = None

        if has_audio:
            audio_path = resolve_path(row["audio_path"], raw_dir=raw_dir, labels_dir=labels_dir)
            audio_key = str(audio_path.resolve())
            if audio_key not in audio_cache:
                audio_cache[audio_key] = extract_audio_embedding(audio_path, audio_processor, audio_model, device)
            audio_embedding = audio_cache[audio_key]

        if has_image:
            image_source = resolve_image_source(row["image_path"], raw_dir=raw_dir, labels_dir=labels_dir)
            image_key = image_source_key(image_source)
            if image_key not in visual_cache:
                if isinstance(image_source, tuple):
                    visual_cache[image_key] = extract_visual_embedding_from_image(
                        fer2013_images[image_key],
                        visual_processor,
                        visual_model,
                        device,
                    )
                else:
                    visual_cache[image_key] = extract_visual_embedding(image_source, visual_processor, visual_model, device)
            visual_embedding = visual_cache[image_key]

        samples.append(
            {
                "sample_id": sample_id,
                "split": split,
                "label": label,
                "label_id": class_names.index(label),
                "audio_path": audio_key,
                "image_path": image_key,
                "audio_available": has_audio,
                "visual_available": has_image,
                "audio_embedding": audio_embedding,
                "visual_embedding": visual_embedding,
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "class_names": class_names,
            "samples": samples,
            "metadata": {
                "created_at": __import__("datetime").datetime.utcnow().isoformat(timespec="seconds") + "Z",
                "audio_model": args.audio_model,
                "visual_model": args.visual_model,
                "num_samples": len(samples),
                "device": str(device),
            },
        },
        output_path,
    )
    print(f"Saved {len(samples)} paired embeddings to {output_path}")


if __name__ == "__main__":
    main()
