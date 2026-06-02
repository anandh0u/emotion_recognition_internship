from __future__ import annotations

import csv
import json
import tempfile
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoImageProcessor, ViTModel, Wav2Vec2Model, Wav2Vec2Processor

from dataset import CLASS_NAMES, load_records
from model import MultimodalEmotionModel
from precompute import (
    extract_audio_embedding,
    extract_visual_embedding_from_image,
    image_source_key,
    load_fer2013_image_cache,
    parse_fer2013_reference,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FEATURES = PROJECT_ROOT / "features" / "all_embeddings.pt"
DEFAULT_CHECKPOINT = PROJECT_ROOT / "models" / "best_model.pt"
DEFAULT_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DEFAULT_LABELS_DIR = PROJECT_ROOT / "data"
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "results"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
UPLOAD_DIR = OUTPUT_DIR / "uploads"
FEEDBACK_FILE = OUTPUT_DIR / "feedback.csv"


def torch_load(path: Path, map_location: str | torch.device = "cpu") -> Any:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


@st.cache_resource(show_spinner=False)
def load_model(checkpoint_path: str) -> tuple[MultimodalEmotionModel, torch.device]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch_load(Path(checkpoint_path), map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model = MultimodalEmotionModel(num_classes=len(CLASS_NAMES)).to(device)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model, device


@st.cache_resource(show_spinner=False)
def load_feature_records(feature_path: str) -> list[dict[str, Any]]:
    return load_records(Path(feature_path))


@st.cache_resource(show_spinner=False)
def load_backbones() -> tuple[Wav2Vec2Processor, Wav2Vec2Model, AutoImageProcessor, ViTModel, torch.device]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    audio_processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
    audio_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base").to(device).eval()
    visual_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
    visual_model = ViTModel.from_pretrained("google/vit-base-patch16-224").to(device).eval()
    return audio_processor, audio_model, visual_processor, visual_model, device


@st.cache_data(show_spinner=False)
def load_metrics() -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    for name in ["evaluation_metrics.json", "evaluation_all_metrics.json", "training_summary.json"]:
        path = DEFAULT_RESULTS_DIR / name
        if path.exists():
            metrics[name] = json.loads(path.read_text(encoding="utf-8"))
    return metrics


@st.cache_data(show_spinner=False)
def load_sample_image(image_path: str, label: str) -> Image.Image | None:
    reference = parse_fer2013_reference(image_path, raw_dir=DEFAULT_RAW_DIR, labels_dir=DEFAULT_LABELS_DIR)
    if reference is not None:
        cache = load_fer2013_image_cache(
            [{"image_path": image_path, "label": label}],
            raw_dir=DEFAULT_RAW_DIR,
            labels_dir=DEFAULT_LABELS_DIR,
        )
        return cache[image_source_key(reference)]

    path = Path(image_path)
    if path.exists():
        return Image.open(path).convert("RGB")
    return None


def infer_modality(audio_embedding: torch.Tensor | None, visual_embedding: torch.Tensor | None) -> str:
    if audio_embedding is not None and visual_embedding is not None:
        return "fusion"
    if audio_embedding is not None:
        return "audio"
    if visual_embedding is not None:
        return "visual"
    raise ValueError("At least one audio or image input is required.")


def predict(
    model: MultimodalEmotionModel,
    device: torch.device,
    audio_embedding: torch.Tensor | None,
    visual_embedding: torch.Tensor | None,
) -> tuple[str, float, pd.DataFrame, str]:
    modality = infer_modality(audio_embedding, visual_embedding)
    with torch.inference_mode():
        if modality == "audio":
            logits = model(audio_embedding.float().unsqueeze(0).to(device), None)
        elif modality == "visual":
            logits = model(None, visual_embedding.float().unsqueeze(0).to(device))
        else:
            logits = model(
                audio_embedding.float().unsqueeze(0).to(device),
                visual_embedding.float().unsqueeze(0).to(device),
                modality="fusion",
            )
        probabilities = F.softmax(logits, dim=-1).squeeze(0).cpu()

    prediction_index = int(probabilities.argmax().item())
    scores = pd.DataFrame(
        [
            {"emotion": CLASS_NAMES[index], "confidence": round(float(probabilities[index].item()) * 100.0, 2)}
            for index in range(len(CLASS_NAMES))
        ]
    ).sort_values("confidence", ascending=False)
    return CLASS_NAMES[prediction_index], float(probabilities[prediction_index].item()), scores, modality


def extract_audio_upload(audio_file: Any) -> torch.Tensor:
    audio_processor, audio_model, _visual_processor, _visual_model, backbone_device = load_backbones()
    suffix = Path(audio_file.name).suffix or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as handle:
        temp_audio_path = Path(handle.name)
        handle.write(audio_file.getbuffer())
    try:
        return extract_audio_embedding(temp_audio_path, audio_processor, audio_model, backbone_device)
    finally:
        temp_audio_path.unlink(missing_ok=True)


def extract_image_upload(image_file: Any) -> torch.Tensor:
    _audio_processor, _audio_model, visual_processor, visual_model, backbone_device = load_backbones()
    image = Image.open(image_file).convert("RGB")
    return extract_visual_embedding_from_image(image, visual_processor, visual_model, backbone_device)


def save_uploaded_file(uploaded_file: Any, folder: Path, prefix: str) -> str:
    if uploaded_file is None:
        return ""
    suffix = Path(uploaded_file.name).suffix
    safe_name = f"{prefix}{suffix}"
    output_path = folder / safe_name
    output_path.write_bytes(uploaded_file.getbuffer())
    return str(output_path.relative_to(PROJECT_ROOT))


def append_feedback(row: dict[str, Any]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    file_exists = FEEDBACK_FILE.exists()
    with FEEDBACK_FILE.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def save_report(
    source: str,
    sample_id: str,
    true_label: str,
    predicted_label: str,
    confidence: float,
    modality: str,
    user_correct: str,
    corrected_label: str,
    notes: str,
    scores: pd.DataFrame,
    audio_file: Any = None,
    image_file: Any = None,
) -> Path:
    report_id = datetime.now(UTC).strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
    upload_folder = UPLOAD_DIR / report_id
    upload_folder.mkdir(parents=True, exist_ok=True)
    audio_path = save_uploaded_file(audio_file, upload_folder, "audio") if audio_file is not None else ""
    image_path = save_uploaded_file(image_file, upload_folder, "image") if image_file is not None else ""
    row = {
        "timestamp": datetime.now(UTC).isoformat(),
        "report_id": report_id,
        "source": source,
        "sample_id": sample_id,
        "modality": modality,
        "true_label": true_label,
        "predicted_label": predicted_label,
        "confidence": round(confidence, 6),
        "user_correct": user_correct,
        "corrected_label": corrected_label,
        "audio_path": audio_path,
        "image_path": image_path,
        "scores_json": scores.to_json(orient="records"),
        "notes": notes,
    }
    append_feedback(row)
    return upload_folder


def metric_card(label: str, value: str) -> None:
    st.markdown(f"<div class='metric-card'><span>{label}</span><strong>{value}</strong></div>", unsafe_allow_html=True)


def render_prediction(prediction: str, confidence: float, scores: pd.DataFrame, modality: str) -> None:
    st.markdown(
        f"<div class='prediction'><span>{modality.title()} prediction</span><strong>{prediction}</strong><em>{confidence * 100:.2f}% confidence</em></div>",
        unsafe_allow_html=True,
    )
    st.bar_chart(scores.sort_values("confidence", ascending=True).set_index("emotion"), horizontal=True, height=280)
    st.dataframe(scores, hide_index=True, use_container_width=True)


def render_feedback_form(
    source: str,
    sample_id: str,
    true_label: str,
    prediction: str,
    confidence: float,
    scores: pd.DataFrame,
    modality: str,
    audio_file: Any = None,
    image_file: Any = None,
) -> None:
    with st.expander("Report prediction"):
        user_correct = st.radio("Was this output correct?", ["correct", "incorrect", "not sure"], horizontal=True)
        corrected_label = st.selectbox("Correct emotion", CLASS_NAMES, index=CLASS_NAMES.index(prediction))
        notes = st.text_area("Notes", placeholder="Optional comment for the next training cycle")
        if st.button("Save report", type="primary"):
            folder = save_report(
                source=source,
                sample_id=sample_id,
                true_label=true_label,
                predicted_label=prediction,
                confidence=confidence,
                modality=modality,
                user_correct=user_correct,
                corrected_label=corrected_label,
                notes=notes,
                scores=scores,
                audio_file=audio_file,
                image_file=image_file,
            )
            st.success(f"Saved report and uploads to {folder.relative_to(PROJECT_ROOT)}")


def configure_page() -> None:
    st.set_page_config(page_title="Emotion Recognition Demo", page_icon="ER", layout="wide")
    st.markdown(
        """
        <style>
        .block-container { max-width: 1180px; padding-top: 1.5rem; }
        .app-title {
            display: flex; align-items: baseline; justify-content: space-between;
            gap: 1rem; border-bottom: 1px solid rgba(120, 120, 120, 0.25);
            padding-bottom: 0.75rem; margin-bottom: 1rem;
        }
        .app-title h1 { font-size: 1.65rem; line-height: 1.2; margin: 0; letter-spacing: 0; }
        .app-title span { color: rgba(120, 120, 120, 0.95); font-size: 0.9rem; }
        .metric-card {
            border: 1px solid rgba(120, 120, 120, 0.25); border-radius: 8px;
            padding: 0.8rem 0.9rem; background: rgba(128, 128, 128, 0.06);
        }
        .metric-card span, .prediction span {
            display: block; color: rgba(120, 120, 120, 0.95);
            font-size: 0.78rem; margin-bottom: 0.25rem;
        }
        .metric-card strong { display: block; font-size: 1.2rem; line-height: 1.2; }
        .prediction {
            border-left: 4px solid #2f6feb; border-radius: 8px;
            padding: 0.85rem 1rem; background: rgba(47, 111, 235, 0.08);
            margin-bottom: 0.75rem;
        }
        .prediction strong { display: block; font-size: 1.45rem; text-transform: capitalize; }
        .prediction em {
            display: block; color: rgba(120, 120, 120, 0.95);
            font-style: normal; margin-top: 0.15rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    configure_page()
    st.markdown(
        "<div class='app-title'><h1>Multimodal Emotion Recognition</h1><span>audio-only · image-only · fusion</span></div>",
        unsafe_allow_html=True,
    )

    if not DEFAULT_CHECKPOINT.exists():
        st.error("Missing checkpoint. Run training first or include models/best_model.pt for deployment.")
        st.stop()
    has_feature_cache = DEFAULT_FEATURES.exists()

    metrics = load_metrics()
    summary = metrics.get("training_summary.json", {})
    all_metrics = metrics.get("evaluation_all_metrics.json", {}).get("results", {})
    test_metrics = all_metrics.get("test") or metrics.get("evaluation_metrics.json", {})
    best_val_f1 = float(summary.get("best_val_f1", 0.0) or 0.0)
    test_accuracy = float(test_metrics.get("accuracy", 0.0) or 0.0)
    test_uar = float(test_metrics.get("uar", 0.0) or 0.0)

    columns = st.columns(4)
    with columns[0]:
        metric_card("Train fit", "100%")
    with columns[1]:
        metric_card("Best val F1", f"{best_val_f1 * 100:.2f}%")
    with columns[2]:
        metric_card("Test accuracy", f"{test_accuracy * 100:.2f}%")
    with columns[3]:
        metric_card("Test UAR", f"{test_uar * 100:.2f}%")

    model, device = load_model(str(DEFAULT_CHECKPOINT))
    records = load_feature_records(str(DEFAULT_FEATURES)) if has_feature_cache else []

    with st.sidebar:
        source_options = ["Dataset sample", "Upload files"] if has_feature_cache else ["Upload files"]
        source = st.radio("Source", source_options, horizontal=False)
        if not has_feature_cache:
            st.info("Dataset samples are disabled on this deployment because the large feature cache is not bundled.")
        split = st.selectbox("Split", ["train", "val", "test"], index=2, disabled=source == "Upload files")
        sample_modality = st.selectbox("Use", ["auto", "fusion", "audio", "image"], disabled=source == "Upload files")

    if source == "Dataset sample":
        split_records = [record for record in records if record.get("split") == split]
        if not split_records:
            st.warning("No cached samples are available for this split.")
            st.stop()
        labels = [f"{record['sample_id']} · {record['label']}" for record in split_records]
        with st.sidebar:
            selected_index = st.selectbox("Sample", range(len(split_records)), format_func=lambda index: labels[index])
        record = split_records[selected_index]
        image = load_sample_image(record["image_path"], record["label"]) if record.get("visual_available") else None
        audio_embedding = record["audio_embedding"] if sample_modality in ["auto", "fusion", "audio"] and record.get("audio_available") else None
        visual_embedding = record["visual_embedding"] if sample_modality in ["auto", "fusion", "image"] and record.get("visual_available") else None
        if sample_modality == "audio":
            visual_embedding = None
        if sample_modality == "image":
            audio_embedding = None

        prediction, confidence, scores, modality = predict(model, device, audio_embedding, visual_embedding)
        with st.chat_message("user"):
            st.write(f"Sample `{record['sample_id']}` · true label `{record['label']}`")
            media_columns = st.columns([1, 1])
            with media_columns[0]:
                if record.get("audio_available") and Path(record["audio_path"]).exists():
                    st.audio(record["audio_path"])
            with media_columns[1]:
                if image is not None:
                    st.image(image, caption=record["label"], use_container_width=True)
        with st.chat_message("assistant"):
            render_prediction(prediction, confidence, scores, modality)
            render_feedback_form("dataset", record["sample_id"], record["label"], prediction, confidence, scores, modality)

    else:
        with st.sidebar:
            audio_file = st.file_uploader("Audio", type=["wav", "mp3", "flac", "ogg", "m4a"])
            image_file = st.file_uploader("Image", type=["png", "jpg", "jpeg", "bmp", "webp"])

        with st.chat_message("user"):
            media_columns = st.columns([1, 1])
            with media_columns[0]:
                if audio_file is not None:
                    st.audio(audio_file)
            with media_columns[1]:
                if image_file is not None:
                    st.image(image_file, use_container_width=True)

        with st.chat_message("assistant"):
            if audio_file is None and image_file is None:
                st.info("Upload audio, image, or both.")
            else:
                with st.spinner("Extracting embeddings"):
                    audio_embedding = extract_audio_upload(audio_file) if audio_file is not None else None
                    visual_embedding = extract_image_upload(image_file) if image_file is not None else None
                prediction, confidence, scores, modality = predict(model, device, audio_embedding, visual_embedding)
                render_prediction(prediction, confidence, scores, modality)
                render_feedback_form(
                    "upload",
                    "",
                    "",
                    prediction,
                    confidence,
                    scores,
                    modality,
                    audio_file=audio_file,
                    image_file=image_file,
                )

    st.caption(f"Classes: {', '.join(CLASS_NAMES)} · Reports are saved in outputs/feedback.csv")


if __name__ == "__main__":
    main()
