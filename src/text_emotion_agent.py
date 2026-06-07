from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from dataset import CLASS_NAMES

DEFAULT_TEXT_MODEL = "j-hartmann/emotion-english-distilroberta-base"

TEXT_LABEL_MAP = {
    "anger": "anger",
    "angry": "anger",
    "disgust": "disgust",
    "fear": "fear",
    "joy": "happiness",
    "happy": "happiness",
    "happiness": "happiness",
    "neutral": "neutral",
    "sad": "sadness",
    "sadness": "sadness",
    "surprise": "surprise",
    "surprised": "surprise",
}

LEXICON = {
    "anger": {
        "angry", "anger", "annoyed", "furious", "mad", "irritated", "rage", "hate", "upset",
        "frustrated", "offended",
    },
    "disgust": {
        "disgust", "disgusted", "gross", "nasty", "awful", "repulsive", "sick", "horrible",
    },
    "fear": {
        "fear", "afraid", "scared", "terrified", "anxious", "panic", "worried", "nervous",
        "unsafe",
    },
    "happiness": {
        "happy", "joy", "joyful", "glad", "excited", "great", "good", "love", "delighted",
        "pleased", "smile",
    },
    "neutral": {
        "okay", "fine", "normal", "regular", "average", "plain", "neutral",
    },
    "sadness": {
        "sad", "sadness", "cry", "crying", "depressed", "lonely", "hurt", "miserable",
        "unhappy", "grief",
    },
    "surprise": {
        "surprise", "surprised", "shocked", "wow", "unexpected", "amazed", "astonished",
    },
}


@dataclass
class TextPrediction:
    label: str
    confidence: float
    scores: list[dict[str, float | str]]
    source: str


def normalize_text_label(label: str) -> str | None:
    return TEXT_LABEL_MAP.get(label.strip().lower().replace("_", " "))


def scores_from_probabilities(class_names: Sequence[str], probabilities: dict[str, float]) -> list[dict[str, float | str]]:
    total = max(sum(probabilities.values()), 1e-12)
    return [
        {"label": label, "confidence": round(float(probabilities.get(label, 0.0) / total) * 100.0, 2)}
        for label in class_names
    ]


class TextEmotionClassifier:
    """Replaceable text emotion agent used for typed text or transcripts."""

    def __init__(self, model_name: str = DEFAULT_TEXT_MODEL, device: str | None = None) -> None:
        self.model_name = model_name
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.tokenizer = None
        self.model = None
        self.load_error: str | None = None

    def _ensure_model(self) -> bool:
        if self.model is not None and self.tokenizer is not None:
            return True
        if self.load_error is not None:
            return False
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name).to(self.device)
            self.model.eval()
            return True
        except Exception as exc:
            self.load_error = str(exc)
            self.tokenizer = None
            self.model = None
            return False

    def predict(self, text: str, class_names: Sequence[str] = CLASS_NAMES) -> TextPrediction:
        cleaned = text.strip()
        if not cleaned:
            raise ValueError("Text emotion agent needs non-empty text.")

        if self._ensure_model():
            return self._predict_transformer(cleaned, class_names)
        return self._predict_lexicon(cleaned, class_names)

    def _predict_transformer(self, text: str, class_names: Sequence[str]) -> TextPrediction:
        assert self.model is not None
        assert self.tokenizer is not None
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with torch.inference_mode():
            logits = self.model(**inputs).logits.squeeze(0)
            raw_probabilities = torch.softmax(logits, dim=-1).detach().cpu().numpy()

        probabilities = {label: 0.0 for label in class_names}
        id_to_label = getattr(self.model.config, "id2label", {})
        for index, raw_probability in enumerate(raw_probabilities):
            raw_label = str(id_to_label.get(index, index))
            mapped_label = normalize_text_label(raw_label)
            if mapped_label in probabilities:
                probabilities[mapped_label] += float(raw_probability)

        if sum(probabilities.values()) <= 0:
            return self._predict_lexicon(text, class_names)

        scores = scores_from_probabilities(class_names, probabilities)
        label = max(scores, key=lambda row: float(row["confidence"]))["label"]
        confidence = float(max(row["confidence"] for row in scores)) / 100.0
        return TextPrediction(str(label), confidence, scores, "text_transformer")

    def _predict_lexicon(self, text: str, class_names: Sequence[str]) -> TextPrediction:
        tokens = re.findall(r"[a-z']+", text.lower())
        counts = Counter(tokens)
        raw_scores = {label: 0.05 for label in class_names}
        for label, words in LEXICON.items():
            if label not in raw_scores:
                continue
            raw_scores[label] += sum(counts[word] for word in words)

        if all(score == 0.05 for score in raw_scores.values()):
            raw_scores["neutral"] = raw_scores.get("neutral", 0.05) + 1.0

        values = np.array([raw_scores[label] for label in class_names], dtype=np.float64)
        probabilities = values / max(float(values.sum()), 1e-12)
        scores = [
            {"label": label, "confidence": round(float(probabilities[index]) * 100.0, 2)}
            for index, label in enumerate(class_names)
        ]
        label = max(scores, key=lambda row: float(row["confidence"]))["label"]
        confidence = float(max(row["confidence"] for row in scores)) / 100.0
        return TextPrediction(str(label), confidence, scores, "text_lexicon")

