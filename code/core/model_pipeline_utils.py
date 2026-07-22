"""Exact six-backbone model factory used by training and evaluation.

The classifier layout and module names are part of the checkpoint contract.
Changing them makes the released ``best_model_fold_*.pth`` state dictionaries
incompatible.
"""

from __future__ import annotations

import base64
import hashlib
import json
import math
import re
from collections.abc import Mapping
from pathlib import Path

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

DEFAULT_EXPERIMENT_CONFIG = {
    "batch_size": 16,
    "learning_rate": 1e-5,
    "weight_decay": 1e-4,
    "dropout_rate": 0.5,
    "pretrained": True,
    "epochs": 50,
    "patience": 20,
    "num_workers": 0,
}
RUN_TAG_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,23}$")
RUN_PREFIX_ALIASES = {
    "swin_tiny_patch4_window7_224": "swin",
    "efficientnet_v2_s": "effv2s",
    "vit_base_patch16_224": "vitb",
    "resnet50": "r50",
    "densenet121": "dense121",
    "convnext_small": "convnexts",
}


def validate_run_tag(run_tag: str | None) -> str | None:
    """Validate an optional filesystem-safe experiment tag."""
    if run_tag is None:
        return None
    if not RUN_TAG_PATTERN.fullmatch(run_tag):
        raise ValueError(
            "run_tag must be 1-24 characters, start with an alphanumeric "
            "character, and contain only letters, digits, '.', '_' or '-'."
        )
    return run_tag


def _exact_float_token(value: float, name: str) -> str:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value!r}")
    # repr(float) is the shortest round-trip representation, avoiding the
    # collisions caused by the historical ``:.0e`` display formatting.
    return repr(value)


def canonical_experiment_config(
    backbone_name: str,
    augmentation: str,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    *,
    dropout_rate: float = 0.5,
    pretrained: bool = True,
    epochs: int = 50,
    patience: int = 20,
    num_workers: int = 0,
    run_tag: str | None = None,
) -> dict[str, object]:
    """Return the canonical full configuration used to identify a run."""
    return {
        "augmentation": str(augmentation),
        "backbone": str(backbone_name),
        "batch_size": int(batch_size),
        "dropout_rate": _exact_float_token(dropout_rate, "dropout_rate"),
        "epochs": int(epochs),
        "learning_rate": _exact_float_token(learning_rate, "learning_rate"),
        "num_workers": int(num_workers),
        "patience": int(patience),
        "pretrained": bool(pretrained),
        "run_tag": validate_run_tag(run_tag),
        "weight_decay": _exact_float_token(weight_decay, "weight_decay"),
    }


def _canonical_mapping_sha256(config: Mapping[str, object]) -> str:
    payload = json.dumps(
        config,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def experiment_config_sha256(**config_kwargs: object) -> str:
    """Return the SHA-256 hex digest of a canonical experiment configuration."""
    config = canonical_experiment_config(**config_kwargs)
    return _canonical_mapping_sha256(config)


def derive_fold_seed(run_identity_sha256: str, fold_id: int, dataset: str) -> int:
    """Derive a stable NumPy/PyTorch-compatible seed for an isolated fold."""
    if not re.fullmatch(r"[0-9a-f]{64}", run_identity_sha256):
        raise ValueError("run_identity_sha256 must be a lowercase SHA-256 hex digest")
    if int(fold_id) <= 0:
        raise ValueError("fold_id must be positive")
    payload = (
        f"pressure-injury|{dataset}|{run_identity_sha256}|fold={int(fold_id)}"
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")


def sha256_file(path: str | Path) -> str:
    """Return a streaming SHA-256 digest without loading the full file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run_prefix(backbone_name: str, augmentation: str) -> str:
    backbone = RUN_PREFIX_ALIASES.get(backbone_name)
    if backbone is None:
        backbone = re.sub(r"[^A-Za-z0-9]+", "", backbone_name)[:8] or "model"
    augmentation_code = re.sub(
        r"[^A-Za-z0-9]+", "", augmentation.split("_", 1)[0]
    )[:8] or "aug"
    return f"{backbone}-{augmentation_code}"[:20].rstrip("-")


def experiment_run_name(
    backbone_name: str,
    augmentation: str,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    *,
    dropout_rate: float = 0.5,
    pretrained: bool = True,
    epochs: int = 50,
    patience: int = 20,
    num_workers: int = 0,
    run_tag: str | None = None,
) -> str:
    """Return a compact, collision-resistant run directory name.

    The exact historical directory name is retained for the public default
    configuration without a tag. Every other run uses a human-readable short
    prefix plus the full 256-bit SHA-256 digest encoded as a 43-character
    URL-safe token. The marker stores the canonical source configuration.
    """
    run_tag = validate_run_tag(run_tag)
    default_comparison = {
        "batch_size": int(batch_size),
        "learning_rate": float(learning_rate),
        "weight_decay": float(weight_decay),
        "dropout_rate": float(dropout_rate),
        "pretrained": bool(pretrained),
        "epochs": int(epochs),
        "patience": int(patience),
        "num_workers": int(num_workers),
    }
    legacy_base = (
        f"{backbone_name}_Baseline_{augmentation}_bs{batch_size}_"
        f"lr{learning_rate:.0e}_wd{weight_decay:.0e}"
    )
    if default_comparison == DEFAULT_EXPERIMENT_CONFIG and run_tag is None:
        name = legacy_base
    else:
        digest_hex = experiment_config_sha256(
            backbone_name=backbone_name,
            augmentation=augmentation,
            batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            dropout_rate=dropout_rate,
            pretrained=pretrained,
            epochs=epochs,
            patience=patience,
            num_workers=num_workers,
            run_tag=run_tag,
        )
        digest_token = base64.urlsafe_b64encode(
            bytes.fromhex(digest_hex)
        ).decode("ascii").rstrip("=")
        name = f"{_run_prefix(backbone_name, augmentation)}_cfg-{digest_token}"
    return name


def load_checkpoint_state_dict(path: str | Path) -> dict[str, torch.Tensor]:
    """Load a weights-only state dictionary on CPU across supported PyTorch versions."""
    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # PyTorch versions predating weights_only
        state = torch.load(path, map_location="cpu")
    if not isinstance(state, dict) or not state:
        raise ValueError(f"Checkpoint is not a non-empty state dictionary: {path}")
    return state


def completion_marker_path(checkpoint_path: str | Path, fold_id: int) -> Path:
    """Return the trainer's completion marker paired with a fold checkpoint."""
    checkpoint_path = Path(checkpoint_path)
    return checkpoint_path.parent.parent / f"fold_{fold_id}_metrics.json"


def validate_completed_checkpoint(
    checkpoint_path: str | Path,
    fold_id: int,
    expected_config: Mapping[str, object],
) -> tuple[bool, str]:
    """Check that a checkpoint belongs to a completed, matching training run.

    This deliberately validates the marker before an evaluator loads the
    weights. During training, best-so-far weights live under an ``.incomplete``
    name and the completion marker is written only after the final checkpoint
    has been promoted, so an interrupted run cannot be mistaken for a result.
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        return False, "checkpoint file is missing"

    marker_path = completion_marker_path(checkpoint_path, fold_id)
    if not marker_path.is_file():
        return False, f"completion marker is missing: {marker_path}"
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return False, f"completion marker is unreadable: {marker_path} ({exc})"
    if not isinstance(marker, dict):
        return False, f"completion marker root is not an object: {marker_path}"

    config = marker.get("config")
    if marker.get("status") != "complete" or not isinstance(config, dict):
        return False, f"completion marker is not complete: {marker_path}"
    mismatches = {
        key: {"expected": expected, "recorded": config.get(key)}
        for key, expected in expected_config.items()
        if config.get(key) != expected
    }
    if mismatches:
        return False, f"completion marker configuration mismatch: {mismatches}"

    run_identity = config.get("run_identity")
    recorded_digest = config.get("run_identity_sha256")
    if (
        not isinstance(run_identity, dict)
        or not isinstance(recorded_digest, str)
        or not re.fullmatch(r"[0-9a-f]{64}", recorded_digest)
    ):
        return False, f"completion marker run identity is missing: {marker_path}"
    try:
        computed_digest = _canonical_mapping_sha256(run_identity)
    except (TypeError, ValueError):
        return False, f"completion marker run identity is invalid: {marker_path}"
    if recorded_digest != computed_digest:
        return False, f"completion marker run-identity digest is invalid: {marker_path}"
    run_name = config.get("run_name")
    if run_name != checkpoint_path.parent.parent.name:
        return False, f"completion marker run directory does not match: {marker_path}"
    if "_cfg-" in str(run_name):
        recorded_token = str(run_name).rsplit("_cfg-", 1)[1]
        expected_token = base64.urlsafe_b64encode(
            bytes.fromhex(recorded_digest)
        ).decode("ascii").rstrip("=")
        if recorded_token != expected_token:
            return False, f"completion marker run-name digest is invalid: {marker_path}"

    checkpoint_digest = marker.get("checkpoint_sha256")
    if not isinstance(checkpoint_digest, str) or not re.fullmatch(
        r"[0-9a-f]{64}", checkpoint_digest
    ):
        return False, f"completion marker checkpoint digest is missing: {marker_path}"
    if sha256_file(checkpoint_path) != checkpoint_digest:
        return False, f"checkpoint SHA-256 does not match marker: {checkpoint_path}"

    completed_epochs = marker.get("completed_epochs")
    best_epoch = marker.get("best_epoch")
    best_val_f1 = marker.get("best_val_f1_macro")
    configured_epochs = config.get("epochs")
    metadata_valid = (
        isinstance(completed_epochs, int)
        and isinstance(configured_epochs, int)
        and 1 <= completed_epochs <= configured_epochs
        and isinstance(best_epoch, int)
        and 1 <= best_epoch <= completed_epochs
        and isinstance(best_val_f1, (int, float))
        and math.isfinite(best_val_f1)
    )
    if not metadata_valid:
        return False, f"completion marker metadata is invalid: {marker_path}"
    return True, "complete"


class UnifiedHeadModel(nn.Module):
    """Study backbone plus its shared two-layer classifier.

    Five final backbones use timm. EfficientNet-V2-S is intentionally fixed to
    torchvision because the study's timm lookup fell back to that exact model.
    """

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
        if input_size != 224:
            raise ValueError(
                "Released checkpoint compatibility requires input_size=224; "
                f"received {input_size}."
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
                # Preserve the study trainer's RNG consumption before the
                # classifier layers are initialised.
                sample = self.backbone(torch.randn(2, 3, input_size, input_size))
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
