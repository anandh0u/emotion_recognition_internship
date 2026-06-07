from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ModalityPrediction:
    """Output from one modality-specific emotion agent."""

    agent_name: str
    modality: str
    label: str
    confidence: float
    scores: list[dict[str, float | str]] = field(default_factory=list)
    available: bool = True
    notes: str = ""


@dataclass(frozen=True)
class SupervisorDecision:
    """Final decision made after comparing agent predictions."""

    label: str
    confidence: float
    selected_agent: str
    selected_modality: str
    reason: str
    uncertain: bool
    candidates: list[ModalityPrediction]


class EmotionSupervisorAgent:
    """
    Paper-aligned supervisor for modular multimodal emotion recognition.

    The supervisor keeps modality agents replaceable: audio can be served by a
    fine-tuned Wav2Vec2 model or a classical classifier, visual can use ViT, and
    fusion can use the late-fusion neural head. Confidence values from different
    models are not perfectly calibrated, so small reliability biases are used
    only to break close calls.
    """

    def __init__(
        self,
        uncertainty_threshold: float = 0.45,
        reliability_bias: dict[str, float] | None = None,
    ) -> None:
        self.uncertainty_threshold = uncertainty_threshold
        self.reliability_bias = reliability_bias or {
            "audio_svm_agent": 0.12,
            "fusion_agent": 0.03,
            "audio_agent": 0.0,
            "visual_agent": -0.04,
        }

    def decide(self, candidates: list[ModalityPrediction]) -> SupervisorDecision:
        available = [candidate for candidate in candidates if candidate.available]
        if not available:
            raise ValueError("Supervisor received no available modality predictions.")

        def supervisor_score(candidate: ModalityPrediction) -> float:
            return candidate.confidence + self.reliability_bias.get(candidate.agent_name, 0.0)

        selected = max(available, key=supervisor_score)
        selected_score = supervisor_score(selected)
        top_raw = max(available, key=lambda candidate: candidate.confidence)

        if selected.agent_name == top_raw.agent_name:
            reason = "highest confidence among available modality agents"
        else:
            reason = (
                f"selected {selected.agent_name.replace('_', ' ')} after reliability adjustment "
                f"({selected_score:.2f} adjusted score)"
            )

        return SupervisorDecision(
            label=selected.label,
            confidence=selected.confidence,
            selected_agent=selected.agent_name,
            selected_modality=selected.modality,
            reason=reason,
            uncertain=selected.confidence < self.uncertainty_threshold,
            candidates=available,
        )

