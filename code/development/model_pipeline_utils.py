"""Exact six-backbone model factory used by training and evaluation.

The classifier layout and module names are part of the checkpoint contract.
Changing them makes the released ``best_model_fold_*.pth`` state dictionaries
incompatible.
"""

from __future__ import annotations

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


FINAL_BASELINE_BACKBONES = (
    "swin_tiny_patch4_window7_224",
    "efficientnet_v2_s",
    "vit_base_patch16_224",
    "resnet50",
    "densenet121",
    "convnext_small",
)


class UnifiedHeadModel(nn.Module):
    """timm feature backbone with the study's shared two-layer classifier."""

    def __init__(
        self,
        backbone_name: str,
        num_classes: int,
        pretrained: bool = True,
        input_size: int = 224,
        dropout_rate: float = 0.5,
    ) -> None:
        super().__init__()
        if backbone_name not in FINAL_BASELINE_BACKBONES:
            raise ValueError(
                f"Unsupported backbone: {backbone_name}. "
                f"Expected one of {FINAL_BASELINE_BACKBONES}."
            )
        self.backbone_name = backbone_name
        if backbone_name == "efficientnet_v2_s":
            weights = (
                models.EfficientNet_V2_S_Weights.IMAGENET1K_V1
                if pretrained else None
            )
            backbone = models.efficientnet_v2_s(weights=weights)
            in_features = backbone.classifier[1].in_features
            backbone.classifier = nn.Identity()
            self.backbone = backbone
        else:
            self.backbone = self._create_timm_backbone(
                backbone_name, pretrained, input_size
            )
            was_training = self.backbone.training
            self.backbone.eval()
            with torch.no_grad():
                sample = self.backbone(torch.zeros(2, 3, input_size, input_size))
                in_features = self._pool_features(sample).shape[1]
            self.backbone.train(was_training)

        self.classifier = nn.Sequential(
            nn.Linear(in_features, in_features // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(in_features // 2, num_classes),
        )

    @staticmethod
    def _create_timm_backbone(
        backbone_name: str, pretrained: bool, input_size: int
    ) -> nn.Module:
        try:
            return timm.create_model(
                backbone_name,
                pretrained=pretrained,
                num_classes=0,
                global_pool="",
                img_size=input_size,
            )
        except TypeError:
            return timm.create_model(
                backbone_name,
                pretrained=pretrained,
                num_classes=0,
                global_pool="",
            )


    def _pool_features(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim == 4:
            if "swin" in self.backbone_name.lower():
                features = features.permute(0, 3, 1, 2)
            return F.adaptive_avg_pool2d(features, (1, 1)).flatten(1)
        if features.ndim == 3:
            if "vit" in self.backbone_name.lower() or "deit" in self.backbone_name.lower():
                return features[:, 0]
            return features.mean(dim=1)
        if features.ndim == 2:
            return features
        raise ValueError(
            f"Unexpected {self.backbone_name} feature shape: {tuple(features.shape)}"
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.classifier(self._pool_features(self.backbone(inputs)))


def get_model(
    backbone_name: str,
    num_classes: int,
    pretrained: bool = True,
    input_size: int = 224,
    dropout_rate: float = 0.5,
) -> UnifiedHeadModel:
    """Create one checkpoint-compatible baseline model."""
    return UnifiedHeadModel(
        backbone_name=backbone_name,
        num_classes=num_classes,
        pretrained=pretrained,
        input_size=input_size,
        dropout_rate=dropout_rate,
    )
