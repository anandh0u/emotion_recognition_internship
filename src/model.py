from __future__ import annotations

import torch
from torch import nn


class MultimodalEmotionModel(nn.Module):
    def __init__(
        self,
        audio_dim: int = 768,
        visual_dim: int = 768,
        projection_dim: int = 256,
        fused_hidden_dim: int = 512,
        num_classes: int = 7,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.audio_proj = nn.Sequential(
            nn.Linear(audio_dim, projection_dim),
            nn.LayerNorm(projection_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.visual_proj = nn.Sequential(
            nn.Linear(visual_dim, projection_dim),
            nn.LayerNorm(projection_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.audio_classifier = nn.Sequential(
            nn.Linear(projection_dim, projection_dim),
            nn.LayerNorm(projection_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(projection_dim, num_classes),
        )
        self.visual_classifier = nn.Sequential(
            nn.Linear(projection_dim, projection_dim),
            nn.LayerNorm(projection_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(projection_dim, num_classes),
        )
        self.fusion_classifier = nn.Sequential(
            nn.Linear(projection_dim * 2, fused_hidden_dim),
            nn.LayerNorm(fused_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fused_hidden_dim, num_classes),
        )

    def encode(self, audio_embedding: torch.Tensor, visual_embedding: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        audio_features = self.audio_proj(audio_embedding)
        visual_features = self.visual_proj(visual_embedding)
        return audio_features, visual_features

    def forward(
        self,
        audio_embedding: torch.Tensor | None = None,
        visual_embedding: torch.Tensor | None = None,
        modality: str = "fusion",
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        if audio_embedding is None and visual_embedding is None:
            raise ValueError("At least one modality embedding is required.")

        if audio_embedding is None:
            visual_features = self.visual_proj(visual_embedding)
            return self.visual_classifier(visual_features)
        if visual_embedding is None:
            audio_features = self.audio_proj(audio_embedding)
            return self.audio_classifier(audio_features)

        audio_features, visual_features = self.encode(audio_embedding, visual_embedding)
        outputs = {
            "audio": self.audio_classifier(audio_features),
            "visual": self.visual_classifier(visual_features),
            "fusion": self.fusion_classifier(torch.cat([audio_features, visual_features], dim=-1)),
        }
        if modality == "all":
            return outputs
        if modality not in outputs:
            raise ValueError(f"Unsupported modality: {modality!r}")
        return outputs[modality]


class LateFusionModel(nn.Module):
    def __init__(
        self,
        audio_dim: int = 768,
        visual_dim: int = 768,
        projection_dim: int = 256,
        fused_hidden_dim: int = 512,
        num_classes: int = 7,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.model = MultimodalEmotionModel(
            audio_dim=audio_dim,
            visual_dim=visual_dim,
            projection_dim=projection_dim,
            fused_hidden_dim=fused_hidden_dim,
            num_classes=num_classes,
            dropout=dropout,
        )

    def forward(self, audio_embedding: torch.Tensor, visual_embedding: torch.Tensor) -> torch.Tensor:
        return self.model(audio_embedding, visual_embedding, modality="fusion")
