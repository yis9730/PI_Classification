"""Train the HUMC baseline models with the private dataset kept local.

This script mirrors the PIID experiment: six backbones, 17 augmentation
settings, five patient-level folds, and random seed 40.  HUMC images and
patient identifiers are not distributed.  Place the locally authorized data
and split files in the documented paths before running this script.

Default experiment:

    - Dataset: HUMC train/validation folds from data/splits/humc
    - Backbones: Swin-Tiny, EfficientNetV2-S, ViT-B/16, ResNet-50,
      DenseNet-121, ConvNeXt-S
    - Augmentation settings: 17 settings, exp00_NoAug through
      exp15_F_R_CZI_ZO_B_C
    - Seed: 40
    - Batch size: 16
    - Optimizer: AdamW
    - Learning rate: 1e-5
    - Weight decay: 1e-4
    - Loss: weighted cross entropy
    - Scheduler: none
    - Max epochs: 50
    - Early stopping patience: 20
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import tempfile
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from albumentations.pytorch import ToTensorV2
from PIL import Image
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


CODE_ROOT = Path(__file__).resolve().parents[1]
CORE_DIR = CODE_ROOT / "core"
if str(CORE_DIR) not in sys.path:
    sys.path.insert(0, str(CORE_DIR))

from model_pipeline_utils import (  # noqa: E402
    DEFAULT_EXPERIMENT_CONFIG,
    canonical_experiment_config,
    derive_fold_seed,
    experiment_config_sha256,
    experiment_run_name,
    get_model,
    load_checkpoint_state_dict,
    sha256_file,
    validate_run_tag,
)
from path_config import (  # noqa: E402
    HUMC_CHECKPOINT_DIR,
    HUMC_SPLIT_DIR,
    resolve_project_paths,
)


RANDOM_SEED = 40
INPUT_SIZE = 224
NUM_CLASSES = 4
N_FOLDS = 5
CLASS_NAMES = ["1", "2", "3", "4"]
EXPECTED_TRAINVAL_STAGE_COUNTS = (203, 605, 475, 273)
EXPECTED_TRAINVAL_IMAGES = sum(EXPECTED_TRAINVAL_STAGE_COUNTS)
EXPECTED_TRAINVAL_PATIENTS = 425
EXPECTED_FOLD_SIZES = ((1251, 305), (1269, 287), (1235, 321), (1171, 385), (1298, 258))

BACKBONES = [
    "swin_tiny_patch4_window7_224",
    "efficientnet_v2_s",
    "vit_base_patch16_224",
    "resnet50",
    "densenet121",
    "convnext_small",
]

AUGMENTATION_CONFIGS = {
    "exp00_NoAug": {},
    "exp01_Flip": {"use_flip": True},
    "exp02_Rotate90": {"use_rotate": True},
    "exp03a_RandomZoomIn": {"use_zoomin": True},
    "exp03b_CenterZoomIn": {"use_center_zoomin": True},
    "exp04_ZoomOut": {"use_zoomout": True},
    "exp05_Brightness": {"use_brightness": True},
    "exp06_Contrast": {"use_contrast": True},
    "exp07_F_R": {"use_flip": True, "use_rotate": True},
    "exp08_F_R_ZI": {"use_flip": True, "use_rotate": True, "use_zoomin": True},
    "exp09_F_R_ZI_ZO": {
        "use_flip": True, "use_rotate": True, "use_zoomin": True, "use_zoomout": True,
    },
    "exp10_F_R_ZI_ZO_B": {
        "use_flip": True, "use_rotate": True, "use_zoomin": True,
        "use_zoomout": True, "use_brightness": True,
    },
    "exp11_F_R_ZI_ZO_B_C": {
        "use_flip": True, "use_rotate": True, "use_zoomin": True,
        "use_zoomout": True, "use_brightness": True, "use_contrast": True,
    },
    "exp12_F_R_CZI": {"use_flip": True, "use_rotate": True, "use_center_zoomin": True},
    "exp13_F_R_CZI_ZO": {
        "use_flip": True, "use_rotate": True, "use_center_zoomin": True, "use_zoomout": True,
    },
    "exp14_F_R_CZI_ZO_B": {
        "use_flip": True, "use_rotate": True, "use_center_zoomin": True,
        "use_zoomout": True, "use_brightness": True,
    },
    "exp15_F_R_CZI_ZO_B_C": {
        "use_flip": True, "use_rotate": True, "use_center_zoomin": True,
        "use_zoomout": True, "use_brightness": True, "use_contrast": True,
    },
}


class AlbImageDataset(Dataset):
    """Image path dataset with Albumentations transforms."""

    def __init__(self, image_paths: list[str], labels: list[int], transform=None):
        self.image_paths = list(image_paths)
        self.labels = list(labels)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        path = self.image_paths[idx]
        label = int(self.labels[idx])
        with Image.open(path) as image:
            image = image.convert("RGB")
            image = np.asarray(image)
        if self.transform is not None:
            image = self.transform(image=image)["image"]
        return image, label


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def validate_args(args: argparse.Namespace) -> None:
    """Fail early on ambiguous or invalid experiment selections."""
    for name, values in {
        "models": args.models,
        "augmentations": args.augmentations,
        "folds": args.folds,
    }.items():
        if not values:
            raise ValueError(f"{name} must not be empty")
        if len(values) != len(set(values)):
            raise ValueError(f"Duplicate {name} are not allowed: {values}")
    invalid_folds = [fold for fold in args.folds if fold not in range(1, N_FOLDS + 1)]
    if invalid_folds:
        raise ValueError(f"folds must be between 1 and {N_FOLDS}: {invalid_folds}")
    for name in ("epochs", "patience", "batch_size"):
        if getattr(args, name) <= 0:
            raise ValueError(f"{name} must be positive")
    if args.num_workers < 0:
        raise ValueError("num_workers must be non-negative")
    if not np.isfinite(args.lr) or args.lr <= 0:
        raise ValueError("lr must be finite and positive")
    if not np.isfinite(args.weight_decay) or args.weight_decay < 0:
        raise ValueError("weight_decay must be finite and non-negative")
    if not np.isfinite(args.dropout) or not 0 <= args.dropout < 1:
        raise ValueError("dropout must be finite and in [0, 1)")
    validate_run_tag(args.run_tag)
    canonical_selection = (
        args.models == BACKBONES
        and args.augmentations == list(AUGMENTATION_CONFIGS)
        and args.folds == list(range(1, N_FOLDS + 1))
    )
    if not canonical_selection and args.run_tag is None:
        raise ValueError(
            "Any model/augmentation/fold subset or reordering requires an "
            "explicit --run-tag. This prevents a smoke or partial run from "
            "writing to a canonical study path."
        )


def _uses_historical_rng(args: argparse.Namespace) -> bool:
    """Return True only for the exact full historical invocation contract."""
    return (
        args.models == BACKBONES
        and args.augmentations == list(AUGMENTATION_CONFIGS)
        and args.folds == list(range(1, N_FOLDS + 1))
        and args.batch_size == DEFAULT_EXPERIMENT_CONFIG["batch_size"]
        and args.lr == DEFAULT_EXPERIMENT_CONFIG["learning_rate"]
        and args.weight_decay == DEFAULT_EXPERIMENT_CONFIG["weight_decay"]
        and args.dropout == DEFAULT_EXPERIMENT_CONFIG["dropout_rate"]
        and args.pretrained == DEFAULT_EXPERIMENT_CONFIG["pretrained"]
        and args.epochs == DEFAULT_EXPERIMENT_CONFIG["epochs"]
        and args.patience == DEFAULT_EXPERIMENT_CONFIG["patience"]
        and args.num_workers == DEFAULT_EXPERIMENT_CONFIG["num_workers"]
        and args.run_tag is None
    )


def _rng_contract(
    backbone: str,
    aug_name: str,
    fold_id: int,
    args: argparse.Namespace,
) -> tuple[str, int | None]:
    if _uses_historical_rng(args):
        return "historical_global_seed_40_canonical_order", None
    run_digest = experiment_config_sha256(
        backbone_name=backbone,
        augmentation=aug_name,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        dropout_rate=args.dropout,
        pretrained=args.pretrained,
        epochs=args.epochs,
        patience=args.patience,
        num_workers=args.num_workers,
        run_tag=args.run_tag,
    )
    return (
        "per_fold_sha256_v1",
        derive_fold_seed(run_digest, fold_id, dataset="HUMC"),
    )


def _training_config(
    backbone: str,
    aug_name: str,
    aug_flags: dict,
    fold_id: int,
    n_train: int,
    n_val: int,
    run_name: str,
    input_file_sha256: dict[str, str],
    rng_strategy: str,
    fold_seed: int | None,
    args: argparse.Namespace,
) -> dict:
    run_identity = canonical_experiment_config(
        backbone,
        aug_name,
        args.batch_size,
        args.lr,
        args.weight_decay,
        dropout_rate=args.dropout,
        pretrained=args.pretrained,
        epochs=args.epochs,
        patience=args.patience,
        num_workers=args.num_workers,
        run_tag=args.run_tag,
    )
    run_identity_sha256 = experiment_config_sha256(
        backbone_name=backbone,
        augmentation=aug_name,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        dropout_rate=args.dropout,
        pretrained=args.pretrained,
        epochs=args.epochs,
        patience=args.patience,
        num_workers=args.num_workers,
        run_tag=args.run_tag,
    )
    return {
        "schema_version": 1,
        "dataset": "HUMC",
        "model": backbone,
        "augmentation": aug_name,
        "augmentation_flags": dict(aug_flags),
        "fold": fold_id,
        "random_seed": RANDOM_SEED,
        "input_size": INPUT_SIZE,
        "num_classes": NUM_CLASSES,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "weight_decay": args.weight_decay,
        "dropout_rate": args.dropout,
        "pretrained": args.pretrained,
        "epochs": args.epochs,
        "patience": args.patience,
        "num_workers": args.num_workers,
        "drop_last": True,
        "n_train": n_train,
        "n_val": n_val,
        "run_tag": args.run_tag,
        "run_name": run_name,
        "run_identity": run_identity,
        "run_identity_sha256": run_identity_sha256,
        "rng_strategy": rng_strategy,
        "fold_seed": fold_seed,
        "input_file_sha256": dict(input_file_sha256),
    }


def _atomic_save_state_dict(model: nn.Module, destination: Path) -> None:
    with tempfile.NamedTemporaryFile(
        dir=destination.parent,
        prefix=".tmp-",
        suffix=".pth",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
    try:
        torch.save(model.state_dict(), temporary)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_write_json(destination: Path, payload: dict) -> None:
    with tempfile.NamedTemporaryFile(
        dir=destination.parent,
        prefix=".tmp-",
        suffix=".json",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
    try:
        temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def _completed_run_or_none(
    weight_path: Path,
    metrics_path: Path,
    expected_config: dict,
    overwrite: bool,
) -> dict | None:
    """Return a valid completion marker, or require an explicit overwrite."""
    checkpoint_exists = weight_path.is_file()
    marker_exists = metrics_path.is_file()
    if overwrite:
        metrics_path.unlink(missing_ok=True)
        return None
    if not checkpoint_exists and not marker_exists:
        return None
    if checkpoint_exists != marker_exists:
        raise RuntimeError(
            f"Incomplete run state for {weight_path}: checkpoint and completion "
            "metrics marker must both exist. Re-run with --overwrite to restart."
        )
    try:
        marker = json.loads(metrics_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"Invalid completion metrics marker: {metrics_path}. "
            "Re-run with --overwrite to restart."
        ) from exc
    if not isinstance(marker, dict):
        raise RuntimeError(
            f"Invalid completion metrics marker root: {metrics_path}. "
            "Re-run with --overwrite to restart."
        )
    completed_epochs = marker.get("completed_epochs")
    best_val_f1 = marker.get("best_val_f1_macro")
    marker_valid = (
        marker.get("status") == "complete"
        and marker.get("config") == expected_config
        and isinstance(completed_epochs, int)
        and 1 <= completed_epochs <= expected_config["epochs"]
        and isinstance(marker.get("best_epoch"), int)
        and 1 <= marker["best_epoch"] <= completed_epochs
        and isinstance(best_val_f1, (int, float))
        and np.isfinite(best_val_f1)
    )
    if not marker_valid:
        raise RuntimeError(
            f"Completion marker does not match the requested configuration: "
            f"{metrics_path}. Re-run with --overwrite or choose --run-tag."
        )
    checkpoint_digest = marker.get("checkpoint_sha256")
    if (
        not isinstance(checkpoint_digest, str)
        or len(checkpoint_digest) != 64
        or sha256_file(weight_path) != checkpoint_digest
    ):
        raise RuntimeError(
            f"Checkpoint SHA-256 does not match its completion marker: "
            f"{weight_path}. Re-run with --overwrite."
        )
    state = load_checkpoint_state_dict(weight_path)
    del state
    return marker


def _preflight_requested_runs(
    df_trainval: pd.DataFrame,
    fold_indices: dict,
    input_file_sha256: dict[str, str],
    args: argparse.Namespace,
) -> None:
    """Reject partial resume before any requested training work starts.

    The study's historical RNG contract is one global seed plus a fixed loop
    order. Skipping only some completed folds would change the RNG state seen
    by later folds, so exact reproduction requires either an all-new run or an
    entirely completed selection. ``--overwrite`` explicitly starts the whole
    requested selection again in its original order.
    """
    if args.overwrite:
        # Invalidate the entire selection up front. If the overwrite run is
        # interrupted, untouched old checkpoints remain visibly incomplete
        # instead of being mixed with newly completed folds on the next run.
        for backbone in args.models:
            for aug_name in args.augmentations:
                run_name = experiment_run_name(
                    backbone,
                    aug_name,
                    args.batch_size,
                    args.lr,
                    args.weight_decay,
                    dropout_rate=args.dropout,
                    pretrained=args.pretrained,
                    epochs=args.epochs,
                    patience=args.patience,
                    num_workers=args.num_workers,
                    run_tag=args.run_tag,
                )
                run_dir = HUMC_CHECKPOINT_DIR / run_name
                for fold_id in args.folds:
                    (run_dir / f"fold_{fold_id}_metrics.json").unlink(missing_ok=True)
        return

    completed = []
    for backbone in args.models:
        for aug_name in args.augmentations:
            aug_flags = AUGMENTATION_CONFIGS[aug_name]
            run_name = experiment_run_name(
                backbone,
                aug_name,
                args.batch_size,
                args.lr,
                args.weight_decay,
                dropout_rate=args.dropout,
                pretrained=args.pretrained,
                epochs=args.epochs,
                patience=args.patience,
                num_workers=args.num_workers,
                run_tag=args.run_tag,
            )
            run_dir = HUMC_CHECKPOINT_DIR / run_name
            for fold_id in args.folds:
                fold = fold_indices[f"fold_{fold_id}"]
                rng_strategy, fold_seed = _rng_contract(
                    backbone, aug_name, fold_id, args
                )
                config = _training_config(
                    backbone,
                    aug_name,
                    aug_flags,
                    fold_id,
                    len(fold["train_idx"]),
                    len(fold["val_idx"]),
                    run_name,
                    input_file_sha256,
                    rng_strategy,
                    fold_seed,
                    args,
                )
                weight_path = (
                    run_dir
                    / "best_models_weights"
                    / f"best_model_fold_{fold_id}.pth"
                )
                metrics_path = run_dir / f"fold_{fold_id}_metrics.json"
                marker = _completed_run_or_none(
                    weight_path,
                    metrics_path,
                    config,
                    overwrite=False,
                )
                completed.append(marker is not None)

    if _uses_historical_rng(args) and any(completed) and not all(completed):
        raise RuntimeError(
            "Partial resume is not reproducible under the historical global-RNG "
            "training order. Some requested runs are complete while others are "
            "not. Re-run the entire requested selection with --overwrite, or "
            "choose a new --run-tag."
        )


def build_train_transform(mean: list[float], std: list[float], aug_flags: dict) -> A.Compose:
    transforms = [A.Resize(INPUT_SIZE, INPUT_SIZE)]
    if aug_flags.get("use_flip"):
        transforms.append(A.Flip(p=0.5))
    if aug_flags.get("use_rotate"):
        transforms.append(A.RandomRotate90(p=0.5))
    if aug_flags.get("use_zoomin"):
        transforms.append(A.RandomResizedCrop(
            size=(INPUT_SIZE, INPUT_SIZE),
            scale=(0.5, 0.99),
            ratio=(1.0, 1.0),
            p=0.5,
        ))
    if aug_flags.get("use_center_zoomin"):
        crop_size = int(INPUT_SIZE * (0.5 ** 0.5))
        transforms.extend([
            A.CenterCrop(height=crop_size, width=crop_size, p=0.5),
            A.Resize(INPUT_SIZE, INPUT_SIZE),
        ])
    if aug_flags.get("use_zoomout"):
        transforms.append(A.Affine(
            scale=(0.7, 0.99),
            mode=cv2.BORDER_REFLECT_101,
            fit_output=False,
            p=0.5,
        ))
    if aug_flags.get("use_brightness"):
        transforms.append(A.RandomBrightnessContrast(
            brightness_limit=0.2,
            contrast_limit=0.0,
            p=0.5,
        ))
    if aug_flags.get("use_contrast"):
        transforms.append(A.RandomBrightnessContrast(
            brightness_limit=0.0,
            contrast_limit=0.2,
            p=0.5,
        ))
    transforms.extend([A.Normalize(mean=mean, std=std), ToTensorV2()])
    return A.Compose(transforms)


def build_eval_transform(mean: list[float], std: list[float]) -> A.Compose:
    return A.Compose([
        A.Resize(INPUT_SIZE, INPUT_SIZE),
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ])


def _validate_trainval_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    required = {"image_path", "stage", "patient_id"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"HUMC trainval CSV is missing columns: {sorted(missing)}")
    if len(df) != EXPECTED_TRAINVAL_IMAGES:
        raise ValueError(
            f"HUMC trainval CSV must contain {EXPECTED_TRAINVAL_IMAGES} rows; "
            f"found {len(df)}"
        )
    if df["image_path"].isna().any() or (df["image_path"].astype(str).str.strip() == "").any():
        raise ValueError("HUMC trainval CSV contains an empty image_path")
    if df["patient_id"].isna().any() or (df["patient_id"].astype(str).str.strip() == "").any():
        raise ValueError("HUMC trainval CSV contains an empty patient_id")
    patient_ids = df["patient_id"].astype(str).str.strip()
    if patient_ids.nunique() != EXPECTED_TRAINVAL_PATIENTS:
        raise ValueError(
            f"HUMC trainval CSV must contain exactly {EXPECTED_TRAINVAL_PATIENTS} patients"
        )
    numeric_stage = pd.to_numeric(df["stage"], errors="coerce")
    if numeric_stage.isna().any() or not np.all(numeric_stage == np.floor(numeric_stage)):
        raise ValueError("HUMC trainval stages must be integer labels 1..4")
    stages = numeric_stage.astype(int)
    if not stages.isin(range(1, NUM_CLASSES + 1)).all():
        raise ValueError("HUMC trainval stages must be in 1..4")
    counts = tuple(int((stages == stage).sum()) for stage in range(1, NUM_CLASSES + 1))
    if counts != EXPECTED_TRAINVAL_STAGE_COUNTS:
        raise ValueError(
            "HUMC trainval stage counts do not match the authorized split: "
            f"expected {EXPECTED_TRAINVAL_STAGE_COUNTS}, found {counts}"
        )
    resolved = [
        str(Path(path).resolve(strict=False)).casefold()
        for path in resolve_project_paths(df["image_path"].astype(str).tolist())
    ]
    if len(resolved) != len(set(resolved)):
        raise ValueError("HUMC trainval CSV contains duplicate image paths")
    validated = df.copy()
    validated["stage"] = stages
    validated["patient_id"] = patient_ids
    return validated


def _validate_fold_indices(fold_indices: dict, df_trainval: pd.DataFrame) -> None:
    if not isinstance(fold_indices, dict):
        raise ValueError("Fold JSON root must be an object")
    n_rows = len(df_trainval)
    expected_keys = {f"fold_{fold}" for fold in range(1, N_FOLDS + 1)}
    if set(fold_indices) != expected_keys:
        raise ValueError(
            f"Fold JSON must contain exactly {sorted(expected_keys)}; "
            f"found {sorted(fold_indices)}"
        )
    all_indices = set(range(n_rows))
    validation_occurrences = np.zeros(n_rows, dtype=np.int64)
    for fold_id in range(1, N_FOLDS + 1):
        fold = fold_indices[f"fold_{fold_id}"]
        if not isinstance(fold, dict) or set(fold) != {"train_idx", "val_idx"}:
            raise ValueError(
                f"fold_{fold_id} must contain exactly train_idx and val_idx"
            )
        parsed = {}
        for split_name in ("train_idx", "val_idx"):
            values = fold[split_name]
            if not isinstance(values, list) or not values:
                raise ValueError(f"fold_{fold_id}.{split_name} must be a non-empty list")
            if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
                raise ValueError(f"fold_{fold_id}.{split_name} must contain integers")
            if len(values) != len(set(values)):
                raise ValueError(f"fold_{fold_id}.{split_name} contains duplicate indices")
            value_set = set(values)
            if not value_set <= all_indices:
                raise ValueError(f"fold_{fold_id}.{split_name} contains out-of-range indices")
            parsed[split_name] = value_set
        train_set = parsed["train_idx"]
        val_set = parsed["val_idx"]
        if (len(train_set), len(val_set)) != EXPECTED_FOLD_SIZES[fold_id - 1]:
            raise ValueError(
                f"fold_{fold_id} train/validation sizes must be "
                f"{EXPECTED_FOLD_SIZES[fold_id - 1]}; found "
                f"{(len(train_set), len(val_set))}"
            )
        if train_set & val_set:
            raise ValueError(f"fold_{fold_id} train and validation indices overlap")
        if train_set | val_set != all_indices:
            raise ValueError(f"fold_{fold_id} does not cover the full trainval CSV")
        train_patients = set(df_trainval.iloc[list(train_set)]["patient_id"].astype(str))
        val_patients = set(df_trainval.iloc[list(val_set)]["patient_id"].astype(str))
        if train_patients & val_patients:
            raise ValueError(f"fold_{fold_id} has patient overlap between train and validation")
        validation_occurrences[list(val_set)] += 1
    if not np.all(validation_occurrences == 1):
        raise ValueError("Validation folds must partition trainval rows exactly once")


def _validate_normalization_dataframe(
    norm_df: pd.DataFrame,
) -> dict[int, dict[str, list[float]]]:
    columns = ["fold", "mean_r", "mean_g", "mean_b", "std_r", "std_g", "std_b"]
    missing = set(columns) - set(norm_df.columns)
    if missing:
        raise ValueError(f"Normalization CSV is missing columns: {sorted(missing)}")
    if len(norm_df) != N_FOLDS:
        raise ValueError(f"Normalization CSV must contain exactly {N_FOLDS} rows")
    unexpected = set(norm_df.columns) - set(columns) - {"n_train_images"}
    if unexpected:
        raise ValueError(f"Normalization CSV has unexpected columns: {sorted(unexpected)}")
    numeric = norm_df[columns].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError("Normalization CSV must contain only finite numeric values")
    folds = numeric["fold"].astype(int)
    if not np.all(numeric["fold"] == folds) or set(folds) != set(range(1, N_FOLDS + 1)):
        raise ValueError("Normalization CSV must contain unique integer folds 1..5")
    means = numeric[["mean_r", "mean_g", "mean_b"]].to_numpy(dtype=float)
    stds = numeric[["std_r", "std_g", "std_b"]].to_numpy(dtype=float)
    if np.any((means < 0) | (means > 1)) or np.any(stds <= 0):
        raise ValueError("Normalization means must be in [0,1] and std values positive")
    if "n_train_images" in norm_df.columns:
        n_train = pd.to_numeric(norm_df["n_train_images"], errors="coerce")
        expected_n_train = tuple(size[0] for size in EXPECTED_FOLD_SIZES)
        if (
            n_train.isna().any()
            or not np.all(n_train == np.floor(n_train))
            or tuple(n_train.astype(int)) != expected_n_train
        ):
            raise ValueError(
                f"Normalization n_train_images must be {expected_n_train}"
            )
    return {
        int(row["fold"]): {
            "mean": [float(row["mean_r"]), float(row["mean_g"]), float(row["mean_b"])],
            "std": [float(row["std_r"]), float(row["std_g"]), float(row["std_b"])],
        }
        for _, row in numeric.iterrows()
    }


def load_split_files() -> tuple[pd.DataFrame, dict, dict, dict[str, str]]:
    trainval_csv = HUMC_SPLIT_DIR / "trainval_set.csv"
    fold_json = HUMC_SPLIT_DIR / "fold_indices.json"
    norm_csv = HUMC_SPLIT_DIR / "normalization_stats.csv"
    if not trainval_csv.exists() or not fold_json.exists() or not norm_csv.exists():
        raise FileNotFoundError(
            "Private HUMC split files are missing. Run "
            "code/pipeline/dataset_split_normalization_humc_patient_level.py "
            "inside the authorized environment first."
        )

    df_trainval = _validate_trainval_dataframe(pd.read_csv(
        trainval_csv,
        dtype={"image_path": "string", "file_stem": "string", "patient_id": "string"},
    ))
    with open(fold_json, "r", encoding="utf-8") as f:
        fold_indices = json.load(f)
    _validate_fold_indices(fold_indices, df_trainval)
    norm_stats = _validate_normalization_dataframe(pd.read_csv(norm_csv))
    input_file_sha256 = {
        "trainval_csv": sha256_file(trainval_csv),
        "fold_indices_json": sha256_file(fold_json),
        "normalization_stats_csv": sha256_file(norm_csv),
    }
    return df_trainval, fold_indices, norm_stats, input_file_sha256


def class_weights(labels: np.ndarray) -> torch.Tensor:
    counts = np.bincount(labels, minlength=NUM_CLASSES).astype(np.float32)
    weights = counts.sum() / (NUM_CLASSES * np.maximum(counts, 1.0))
    return torch.tensor(weights, dtype=torch.float32)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> dict:
    model.eval()
    losses, y_true, y_pred = [], [], []
    for inputs, labels in loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        losses.append(float(loss.item()))
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(outputs.argmax(1).cpu().numpy())

    return {
        "loss": float(np.mean(losses)),
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }


def train_one_fold(
    backbone: str,
    aug_name: str,
    aug_flags: dict,
    fold_id: int,
    df_trainval: pd.DataFrame,
    fold_indices: dict,
    norm_stats: dict,
    input_file_sha256: dict[str, str],
    args: argparse.Namespace,
    device: torch.device,
) -> dict:
    validate_args(args)
    fold = fold_indices[f"fold_{fold_id}"]
    train_df = df_trainval.iloc[fold["train_idx"]].reset_index(drop=True)
    val_df = df_trainval.iloc[fold["val_idx"]].reset_index(drop=True)

    train_paths = resolve_project_paths(train_df["image_path"].tolist())
    val_paths = resolve_project_paths(val_df["image_path"].tolist())
    train_labels = (train_df["stage"].values - 1).astype(int)
    val_labels = (val_df["stage"].values - 1).astype(int)

    run_name = experiment_run_name(
        backbone,
        aug_name,
        args.batch_size,
        args.lr,
        args.weight_decay,
        dropout_rate=args.dropout,
        pretrained=args.pretrained,
        epochs=args.epochs,
        patience=args.patience,
        num_workers=args.num_workers,
        run_tag=args.run_tag,
    )
    run_dir = HUMC_CHECKPOINT_DIR / run_name
    weight_dir = run_dir / "best_models_weights"
    weight_dir.mkdir(parents=True, exist_ok=True)
    weight_path = weight_dir / f"best_model_fold_{fold_id}.pth"
    metrics_path = run_dir / f"fold_{fold_id}_metrics.json"
    rng_strategy, fold_seed = _rng_contract(backbone, aug_name, fold_id, args)
    config = _training_config(
        backbone,
        aug_name,
        aug_flags,
        fold_id,
        len(train_df),
        len(val_df),
        run_name,
        input_file_sha256,
        rng_strategy,
        fold_seed,
        args,
    )
    completed_marker = _completed_run_or_none(
        weight_path, metrics_path, config, args.overwrite
    )
    if completed_marker is not None:
        print(f"[SKIP] Completed checkpoint: {weight_path}")
        return {
            "model": backbone,
            "augmentation": aug_name,
            "fold": fold_id,
            "best_epoch": completed_marker["best_epoch"],
            "best_val_f1_macro": completed_marker["best_val_f1_macro"],
            "completed_epochs": completed_marker["completed_epochs"],
            "checkpoint_path": str(weight_path),
            "checkpoint_sha256": completed_marker["checkpoint_sha256"],
            "rng_strategy": rng_strategy,
            "fold_seed": fold_seed,
            "status": "complete",
            "skipped": True,
        }

    incomplete_weight_path = weight_path.with_name(f".{weight_path.name}.incomplete")
    incomplete_weight_path.unlink(missing_ok=True)

    if fold_seed is not None:
        set_seed(fold_seed)

    stats = norm_stats[fold_id]
    train_transform = build_train_transform(stats["mean"], stats["std"], aug_flags)
    val_transform = build_eval_transform(stats["mean"], stats["std"])
    if fold_seed is None:
        generator=torch.Generator().manual_seed(RANDOM_SEED + fold_id)  # noqa: E225
    else:
        generator = torch.Generator().manual_seed(fold_seed)

    train_loader = DataLoader(
        AlbImageDataset(train_paths, train_labels.tolist(), train_transform),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        generator=generator,
    )
    val_loader = DataLoader(
        AlbImageDataset(val_paths, val_labels.tolist(), val_transform),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    if len(train_loader) == 0:
        raise ValueError(
            f"Training loader is empty for fold {fold_id}; batch_size={args.batch_size} "
            f"exceeds n_train={len(train_df)} while drop_last=True."
        )

    model = get_model(
        backbone,
        num_classes=NUM_CLASSES,
        pretrained=args.pretrained,
        input_size=INPUT_SIZE,
        dropout_rate=args.dropout,
    ).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights(train_labels).to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_f1 = -1.0
    best_epoch = 0
    stale_epochs = 0
    history = []
    early_stopped = False

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses, train_true, train_pred = [], [], []
        progress = tqdm(train_loader, desc=f"{run_name} fold {fold_id} epoch {epoch}", leave=False)
        for inputs, labels in progress:
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_losses.append(float(loss.item()))
            train_true.extend(labels.detach().cpu().numpy())
            train_pred.extend(outputs.argmax(1).detach().cpu().numpy())

        train_metrics = {
            "loss": float(np.mean(train_losses)),
            "accuracy": accuracy_score(train_true, train_pred),
            "balanced_accuracy": balanced_accuracy_score(train_true, train_pred),
            "f1_macro": f1_score(train_true, train_pred, average="macro", zero_division=0),
        }
        val_metrics = evaluate(model, val_loader, criterion, device)

        history_row = {
            "epoch": epoch,
            **{f"train_{k}": v for k, v in train_metrics.items()},
            **{f"val_{k}": v for k, v in val_metrics.items()},
        }
        history.append(history_row)
        print(
            f"[{run_name} fold {fold_id}] epoch {epoch:03d} "
            f"train_f1={train_metrics['f1_macro']:.4f} val_f1={val_metrics['f1_macro']:.4f}"
        )

        if val_metrics["f1_macro"] > best_f1:
            best_f1 = float(val_metrics["f1_macro"])
            best_epoch = epoch
            stale_epochs = 0
            _atomic_save_state_dict(model, incomplete_weight_path)
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                print(f"[EARLY STOP] fold {fold_id}, epoch {epoch}")
                early_stopped = True
                break

    if not incomplete_weight_path.is_file():
        raise RuntimeError(
            f"Training completed without a best-checkpoint candidate: "
            f"{incomplete_weight_path}"
        )
    history_df = pd.DataFrame(history)
    history_df.to_csv(run_dir / f"fold_{fold_id}_history.csv", index=False)

    # Only the final promotion uses the public checkpoint name. The marker is
    # written afterwards, so evaluators reject every interrupted state.
    incomplete_weight_path.replace(weight_path)
    checkpoint_sha256 = sha256_file(weight_path)

    result = {
        "status": "complete",
        "config": config,
        "model": backbone,
        "augmentation": aug_name,
        "fold": fold_id,
        "best_epoch": best_epoch,
        "best_val_f1_macro": best_f1,
        "completed_epochs": len(history),
        "early_stopped": early_stopped,
        "stop_reason": "early_stopping_patience" if early_stopped else "max_epochs",
        "checkpoint_path": str(weight_path),
        "checkpoint_sha256": checkpoint_sha256,
        "rng_strategy": rng_strategy,
        "fold_seed": fold_seed,
        "skipped": False,
    }
    _atomic_write_json(metrics_path, result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train HUMC baseline models.")
    parser.add_argument("--models", nargs="+", default=BACKBONES, choices=BACKBONES)
    parser.add_argument("--augmentations", nargs="+", default=list(AUGMENTATION_CONFIGS), choices=list(AUGMENTATION_CONFIGS))
    parser.add_argument("--folds", nargs="+", type=int, default=list(range(1, N_FOLDS + 1)))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--run-tag", default=None, help="Optional safe suffix for an isolated run.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validate_args(args)
    set_seed(RANDOM_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[DEVICE] {device}")

    df_trainval, fold_indices, norm_stats, input_file_sha256 = load_split_files()
    HUMC_CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    _preflight_requested_runs(
        df_trainval, fold_indices, input_file_sha256, args
    )

    all_results = []
    for backbone in args.models:
        for aug_name in args.augmentations:
            aug_flags = AUGMENTATION_CONFIGS[aug_name]
            for fold_id in args.folds:
                result = train_one_fold(
                    backbone=backbone,
                    aug_name=aug_name,
                    aug_flags=aug_flags,
                    fold_id=fold_id,
                    df_trainval=df_trainval,
                    fold_indices=fold_indices,
                    norm_stats=norm_stats,
                    input_file_sha256=input_file_sha256,
                    args=args,
                    device=device,
                )
                all_results.append(result)
                pd.DataFrame(all_results).to_csv(
                    HUMC_CHECKPOINT_DIR / "__training_summary.csv",
                    index=False,
                )

    print(f"[DONE] Training summary: {HUMC_CHECKPOINT_DIR / '__training_summary.csv'}")


if __name__ == "__main__":
    main()
