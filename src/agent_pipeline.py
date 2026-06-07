from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Protocol


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


AgentPredictionFactory = Callable[[], tuple[str, float, list[dict[str, float | str]], str | None]]


class EmotionAgent(Protocol):
    """Replaceable modality agent interface used by the supervisor."""

    agent_name: str
    modality: str
    notes: str

    def predict(self) -> ModalityPrediction:
        """Return one candidate prediction for the supervisor."""


class CallableEmotionAgent:
    """Small adapter that lets each model branch act as an independent agent."""

    def __init__(
        self,
        agent_name: str,
        modality: str,
        prediction_factory: AgentPredictionFactory,
        notes: str = "",
    ) -> None:
        self.agent_name = agent_name
        self.modality = modality
        self.prediction_factory = prediction_factory
        self.notes = notes

    def predict(self) -> ModalityPrediction:
        label, confidence, scores, modality = self.prediction_factory()
        return ModalityPrediction(
            agent_name=self.agent_name,
            modality=modality or self.modality,
            label=label,
            confidence=confidence,
            scores=scores,
            notes=self.notes,
        )


class AudioEmotionAgent(CallableEmotionAgent):
    """Speech/audio emotion agent."""


class VisionEmotionAgent(CallableEmotionAgent):
    """Face/image/video-frame emotion agent."""


class TextEmotionAgent(CallableEmotionAgent):
    """Transcript or typed-text emotion agent."""


class FusionEmotionAgent(CallableEmotionAgent):
    """Late-fusion agent over already encoded modalities."""


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
            "text_agent": 0.02,
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


class AgenticEmotionPipeline:
    """Runs replaceable modality agents and asks the supervisor for a decision."""

    def __init__(self, supervisor: EmotionSupervisorAgent | None = None) -> None:
        self.supervisor = supervisor or EmotionSupervisorAgent()

    def predict(self, agents: list[EmotionAgent]) -> SupervisorDecision:
        candidates = [agent.predict() for agent in agents]
        return self.supervisor.decide(candidates)

    def decide(self, candidates: list[ModalityPrediction]) -> SupervisorDecision:
        return self.supervisor.decide(candidates)
