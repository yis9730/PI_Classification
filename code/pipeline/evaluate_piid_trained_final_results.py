"""Evaluate PIID-trained checkpoints internally and on external datasets.

HUMC images remain private. When an authorized local HUMC folder is present,
the same code evaluates it without any source-code changes.

    - PIID_Test: held-out 15% internal test set
    - HUMC: private external validation set (optional local input)
    - Kaggle: public external validation set

Outputs:

    data/results/predictions/piid/__ALL_foldwise_results.csv
    data/results/predictions/piid/__ALL_summary_results.csv
    data/results/predictions/piid/{model}_{augmentation}/predictions/
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import uuid
from pathlib import Path

import albumentations as A
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from albumentations.pytorch import ToTensorV2
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


CODE_ROOT = Path(__file__).resolve().parents[1]
CORE_DIR = CODE_ROOT / "core"
if str(CORE_DIR) not in sys.path:
    sys.path.insert(0, str(CORE_DIR))

from model_pipeline_utils import (  # noqa: E402
    DEFAULT_EXPERIMENT_CONFIG,
    canonical_experiment_config,
    completion_marker_path,
    derive_fold_seed,
    experiment_config_sha256,
    experiment_run_name,
    get_model,
    load_checkpoint_state_dict,
    sha256_file,
    validate_completed_checkpoint,
    validate_run_tag,
)
from path_config import (  # noqa: E402
    HUMC_DATA_DIR,
    KAGGLE_DATA_DIR,
    PIID_CHECKPOINT_DIR,
    PIID_INFERENCE_DIR,
    PIID_SPLIT_DIR,
    resolve_project_paths,
)


INPUT_SIZE = 224
NUM_CLASSES = 4
N_FOLDS = 5
RANDOM_SEED = 40
CLASS_NAMES = ["1", "2", "3", "4"]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
EXPECTED_PIID_TEST_IMAGES = 163
EXPECTED_KAGGLE_IMAGES = 141
EXPECTED_HUMC_IMAGES = 1844
EXPECTED_PIID_TRAINVAL_IMAGES = 918
EXPECTED_PIID_TEST_STAGE_COUNTS = (35, 47, 41, 40)
EXPECTED_KAGGLE_STAGE_COUNTS = (27, 46, 41, 27)
EXPECTED_HUMC_STAGE_COUNTS = (233, 709, 575, 327)
EXPECTED_PIID_TOTAL_STAGE_COUNTS = (229, 311, 273, 268)
EXPECTED_NORMALIZATION_TRAIN_IMAGES = (734, 734, 734, 735, 735)

BACKBONES = [
    "swin_tiny_patch4_window7_224",
    "efficientnet_v2_s",
    "vit_base_patch16_224",
    "resnet50",
    "densenet121",
    "convnext_small",
]

AUGMENTATIONS = [
    "exp00_NoAug",
    "exp01_Flip",
    "exp02_Rotate90",
    "exp03a_RandomZoomIn",
    "exp03b_CenterZoomIn",
    "exp04_ZoomOut",
    "exp05_Brightness",
    "exp06_Contrast",
    "exp07_F_R",
    "exp08_F_R_ZI",
    "exp09_F_R_ZI_ZO",
    "exp10_F_R_ZI_ZO_B",
    "exp11_F_R_ZI_ZO_B_C",
    "exp12_F_R_CZI",
    "exp13_F_R_CZI_ZO",
    "exp14_F_R_CZI_ZO_B",
    "exp15_F_R_CZI_ZO_B_C",
]


class AlbImageDataset(Dataset):
    """Image path dataset for deterministic evaluation."""

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
        return image, label, path


def build_eval_transform(mean: list[float], std: list[float]) -> A.Compose:
    return A.Compose([
        A.Resize(INPUT_SIZE, INPUT_SIZE),
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ])


def _validate_normalization_dataframe(
    df: pd.DataFrame,
) -> dict[int, dict[str, list[float]]]:
    columns = ["fold", "mean_r", "mean_g", "mean_b", "std_r", "std_g", "std_b"]
    missing = set(columns) - set(df.columns)
    if missing:
        raise ValueError(f"Normalization CSV is missing columns: {sorted(missing)}")
    unexpected = set(df.columns) - set(columns) - {"n_train_images"}
    if unexpected:
        raise ValueError(f"Normalization CSV has unexpected columns: {sorted(unexpected)}")
    if len(df) != N_FOLDS:
        raise ValueError(f"Normalization CSV must contain exactly {N_FOLDS} rows")
    numeric_columns = columns + (["n_train_images"] if "n_train_images" in df.columns else [])
    numeric = df[numeric_columns].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError("Normalization CSV must contain finite numeric values")
    folds = numeric["fold"].astype(int)
    if not np.all(numeric["fold"] == folds) or set(folds) != set(range(1, N_FOLDS + 1)):
        raise ValueError("Normalization CSV must contain unique integer folds 1..5")
    if "n_train_images" in numeric.columns:
        observed = {
            int(row["fold"]): int(row["n_train_images"])
            for _, row in numeric.iterrows()
            if float(row["n_train_images"]).is_integer()
        }
        expected = {
            fold_id: count
            for fold_id, count in enumerate(EXPECTED_NORMALIZATION_TRAIN_IMAGES, start=1)
        }
        if len(observed) != N_FOLDS or observed != expected:
            raise ValueError(
                "Normalization n_train_images does not match the released PIID folds"
            )
    means = numeric[["mean_r", "mean_g", "mean_b"]].to_numpy(dtype=float)
    stds = numeric[["std_r", "std_g", "std_b"]].to_numpy(dtype=float)
    if np.any((means < 0) | (means > 1)) or np.any(stds <= 0):
        raise ValueError("Normalization means must be in [0,1] and std values positive")
    return {
        int(row["fold"]): {
            "mean": [float(row["mean_r"]), float(row["mean_g"]), float(row["mean_b"])],
            "std": [float(row["std_r"]), float(row["std_g"]), float(row["std_b"])],
        }
        for _, row in numeric.iterrows()
    }


def training_input_sha256() -> dict[str, str]:
    paths = {
        "trainval_csv": PIID_SPLIT_DIR / "piid_trainval_set.csv",
        "fold_indices_json": PIID_SPLIT_DIR / "piid_fold_indices.json",
        "normalization_stats_csv": PIID_SPLIT_DIR / "normalization_stats.csv",
    }
    missing = [path for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"PIID training split artifacts are missing: {missing}")
    return {name: sha256_file(path) for name, path in paths.items()}


def load_norm_stats() -> dict[int, dict[str, list[float]]]:
    path = PIID_SPLIT_DIR / "normalization_stats.csv"
    if not path.exists():
        raise FileNotFoundError(f"PIID normalization file not found: {path}")
    return _validate_normalization_dataframe(pd.read_csv(path))


def _validate_piid_split_metadata() -> None:
    path = PIID_SPLIT_DIR / "piid_split_meta.json"
    if not path.is_file():
        return
    try:
        meta = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid PIID split metadata: {path}") from exc
    expected_scalars = {
        "dataset": "PIID",
        "random_seed": RANDOM_SEED,
        "n_folds": N_FOLDS,
        "input_size": INPUT_SIZE,
        "total_images": sum(EXPECTED_PIID_TOTAL_STAGE_COUNTS),
        "n_trainval_images": 918,
        "n_test_images": EXPECTED_PIID_TEST_IMAGES,
    }
    mismatches = {
        key: {"expected": value, "found": meta.get(key)}
        for key, value in expected_scalars.items()
        if meta.get(key) != value
    }
    stage_distribution = meta.get("stage_distribution", {})
    expected_distributions = {
        "total": EXPECTED_PIID_TOTAL_STAGE_COUNTS,
        "test": EXPECTED_PIID_TEST_STAGE_COUNTS,
    }
    for scope, expected in expected_distributions.items():
        found = stage_distribution.get(scope, {}) if isinstance(stage_distribution, dict) else {}
        found_tuple = tuple(found.get(str(stage)) for stage in range(1, NUM_CLASSES + 1))
        if found_tuple != expected:
            mismatches[f"stage_distribution.{scope}"] = {
                "expected": expected,
                "found": found_tuple,
            }
    if mismatches:
        raise ValueError(f"PIID split metadata does not match the study contract: {mismatches}")


def load_piid_test() -> tuple[list[str], list[int]]:
    path = PIID_SPLIT_DIR / "piid_test_set.csv"
    if not path.exists():
        raise FileNotFoundError(f"PIID test split file not found: {path}")
    _validate_piid_split_metadata()
    df = pd.read_csv(path)
    if not {"image_path", "stage"} <= set(df.columns):
        raise ValueError("PIID test CSV must contain image_path and stage columns")
    if df["image_path"].isna().any() or df["stage"].isna().any():
        raise ValueError("PIID test CSV contains missing image_path or stage values")
    if df["image_path"].astype(str).str.strip().eq("").any():
        raise ValueError("PIID test CSV contains an empty image_path")
    numeric_stage = pd.to_numeric(df["stage"], errors="coerce")
    if numeric_stage.isna().any() or not np.all(numeric_stage == np.floor(numeric_stage)):
        raise ValueError("PIID test stages must be integer labels 1..4")
    if not numeric_stage.astype(int).isin(range(1, NUM_CLASSES + 1)).all():
        raise ValueError("PIID test stages must be in 1..4")
    paths = resolve_project_paths(df["image_path"].tolist())
    trainval_path = PIID_SPLIT_DIR / "piid_trainval_set.csv"
    trainval = pd.read_csv(trainval_path, dtype={"image_path": "string"})
    if "image_path" not in trainval.columns or len(trainval) != EXPECTED_PIID_TRAINVAL_IMAGES:
        raise ValueError(
            f"PIID trainval CSV must contain image_path and exactly "
            f"{EXPECTED_PIID_TRAINVAL_IMAGES} rows"
        )
    if trainval["image_path"].isna().any() or trainval["image_path"].str.strip().eq("").any():
        raise ValueError("PIID trainval CSV contains an empty image_path")
    trainval_keys = {
        str(Path(item).resolve(strict=False)).casefold()
        for item in resolve_project_paths(trainval["image_path"].tolist())
    }
    if len(trainval_keys) != len(trainval):
        raise ValueError("PIID trainval CSV contains duplicate image paths")
    test_keys = {str(Path(item).resolve(strict=False)).casefold() for item in paths}
    if test_keys & trainval_keys:
        raise ValueError("PIID trainval and held-out test image paths overlap")
    labels = (numeric_stage.to_numpy(dtype=int) - 1).tolist()
    return paths, labels


def load_stage_folder_dataset(root: Path) -> tuple[list[str], list[int]]:
    paths, labels = [], []
    for label_idx, stage in enumerate(CLASS_NAMES):
        stage_dir = root / stage
        if not stage_dir.exists():
            raise FileNotFoundError(f"Stage folder not found: {stage_dir}")
        stage_paths = [
            path for path in sorted(stage_dir.iterdir())
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ]
        if not stage_paths:
            raise RuntimeError(f"No images found in required stage folder: {stage_dir}")
        for path in stage_paths:
            try:
                with Image.open(path) as image:
                    image.load()
                    width, height = image.size
            except OSError:
                raise RuntimeError(
                    f"Stage {stage} contains an unreadable curated image"
                ) from None
            if width != height:
                raise RuntimeError(
                    f"Stage {stage} contains a non-square curated image"
                )
        paths.extend(str(path) for path in stage_paths)
        labels.extend([label_idx] * len(stage_paths))
    if not paths:
        raise RuntimeError(f"No images found under: {root}")
    return paths, labels


def stage_folder_dataset_available(root: Path) -> bool:
    """Return True only when every required stage contains an image."""
    if not root.is_dir():
        return False
    for stage in CLASS_NAMES:
        stage_dir = root / stage
        if not stage_dir.is_dir():
            return False
        if not any(
            path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
            for path in stage_dir.iterdir()
        ):
            return False
    return True


def stage_folder_dataset_has_images(root: Path) -> bool:
    """Distinguish an empty tracked placeholder from optional local data."""
    if not root.is_dir():
        return False
    return any(
        path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        for stage in CLASS_NAMES
        for path in ((root / stage).iterdir() if (root / stage).is_dir() else [])
    )


def require_dataset_contract(
    name: str,
    dataset: tuple[list[str], list[int]],
    expected_stage_counts: tuple[int, int, int, int],
) -> None:
    paths, labels = dataset
    expected = sum(expected_stage_counts)
    if len(paths) != expected or len(labels) != expected:
        raise RuntimeError(
            f"{name} must contain exactly {expected} images; found {len(paths)}. "
            "Re-run the documented data-curation step."
        )
    normalized_paths = [str(Path(path).resolve(strict=False)).casefold() for path in paths]
    if len(normalized_paths) != len(set(normalized_paths)):
        raise RuntimeError(f"{name} contains duplicate image paths")
    if any(isinstance(label, bool) or int(label) != label for label in labels):
        raise RuntimeError(f"{name} labels must be integer class indices 0..3")
    counts = tuple(int(sum(int(label) == stage for label in labels)) for stage in range(NUM_CLASSES))
    if counts != expected_stage_counts:
        raise RuntimeError(
            f"{name} stage counts must be {expected_stage_counts}; found {counts}"
        )


def calculate_specificity(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    cm = confusion_matrix(y_true, y_pred, labels=list(range(NUM_CLASSES)))
    specificities = []
    for i in range(NUM_CLASSES):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        tn = cm.sum() - tp - fp - fn
        specificities.append(tn / (tn + fp) if (tn + fp) else np.nan)
    return np.asarray(specificities, dtype=float)


def calculate_macro_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Evaluate one four-class softmax model with class-wise macro ROC-AUC."""
    try:
        y_bin = label_binarize(y_true, classes=list(range(NUM_CLASSES)))
        # This is an evaluation metric, not a separate classifier.
        return float(roc_auc_score(y_bin, y_prob, average="macro"))
    except ValueError:
        return float("nan")


@torch.no_grad()
def evaluate_dataset(model, paths, labels, transform, batch_size, num_workers, device) -> dict:
    dataset = AlbImageDataset(paths, labels, transform)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    model.eval()
    all_true, all_pred, all_prob, all_paths = [], [], [], []
    for inputs, y, batch_paths in tqdm(loader, desc="evaluate", leave=False):
        inputs = inputs.to(device)
        outputs = model(inputs)
        probs = F.softmax(outputs, dim=1)
        all_true.extend(y.numpy())
        all_pred.extend(outputs.argmax(1).cpu().numpy())
        all_prob.extend(probs.cpu().numpy())
        all_paths.extend(batch_paths)

    y_true = np.asarray(all_true)
    y_pred = np.asarray(all_pred)
    y_prob = np.asarray(all_prob)
    specificity = calculate_specificity(y_true, y_pred)
    per_f1 = f1_score(y_true, y_pred, labels=list(range(NUM_CLASSES)), average=None, zero_division=0)
    per_precision = precision_score(y_true, y_pred, labels=list(range(NUM_CLASSES)), average=None, zero_division=0)
    per_recall = recall_score(y_true, y_pred, labels=list(range(NUM_CLASSES)), average=None, zero_division=0)

    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Balanced_Accuracy": balanced_accuracy_score(y_true, y_pred),
        "Precision_Macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "Recall_Macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "Specificity_Macro": float(np.nanmean(specificity)),
        "F1_Macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "AUC_Macro": calculate_macro_auc(y_true, y_prob),
        "Cohens_Kappa": cohen_kappa_score(y_true, y_pred),
    }
    for class_idx, class_name in enumerate(CLASS_NAMES):
        metrics[f"Class_{class_name}_Precision"] = per_precision[class_idx]
        metrics[f"Class_{class_name}_Recall"] = per_recall[class_idx]
        metrics[f"Class_{class_name}_F1"] = per_f1[class_idx]
        metrics[f"Class_{class_name}_Specificity"] = specificity[class_idx]

    return {
        "metrics": metrics,
        "y_true": y_true,
        "y_pred": y_pred,
        "y_prob": y_prob,
        "paths": all_paths,
    }


def save_predictions(
    prediction_dir: Path,
    dataset_name: str,
    fold_id: int,
    result: dict,
) -> Path:
    prediction_dir.mkdir(parents=True, exist_ok=True)
    n_rows = len(result["paths"])
    if (
        len(result["y_true"]) != n_rows
        or len(result["y_pred"]) != n_rows
        or np.asarray(result["y_prob"]).shape != (n_rows, NUM_CLASSES)
    ):
        raise RuntimeError("Prediction arrays have inconsistent lengths or class shape")
    df = pd.DataFrame({
        "image_path": result["paths"],
        "true_label": result["y_true"],
        "predicted_label": result["y_pred"],
    })
    for idx, class_name in enumerate(CLASS_NAMES):
        df[f"prob_{class_name}"] = result["y_prob"][:, idx]
    destination = prediction_dir / f"{dataset_name}_fold{fold_id}_predictions.csv"
    _atomic_write_csv(df, destination)
    return destination


def _atomic_write_csv(frame: pd.DataFrame, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=destination.parent,
        prefix=".tmp-",
        suffix=".csv",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
    try:
        frame.to_csv(temporary, index=False)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_write_json(destination: Path, payload: dict) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
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


def _promote_snapshot(staging_root: Path, final_root: Path) -> None:
    """Atomically make one complete evaluation tree authoritative."""
    final_root.parent.mkdir(parents=True, exist_ok=True)
    backup = final_root.with_name(
        f".{final_root.name}-backup-{uuid.uuid4().hex[:8]}"
    )
    had_previous = final_root.exists()
    if had_previous:
        final_root.replace(backup)
    try:
        staging_root.replace(final_root)
    except Exception:
        if had_previous and backup.exists() and not final_root.exists():
            backup.replace(final_root)
        raise
    if backup.exists():
        shutil.rmtree(backup, ignore_errors=True)


def checkpoint_path(
    backbone: str,
    aug_name: str,
    fold_id: int,
    args: argparse.Namespace,
) -> Path:
    run_name = experiment_run_name(
        backbone,
        aug_name,
        args.batch_size,
        args.lr,
        args.weight_decay,
        dropout_rate=args.dropout,
        pretrained=args.training_pretrained,
        epochs=args.epochs,
        patience=args.patience,
        num_workers=args.training_num_workers,
        run_tag=args.run_tag,
    )
    return PIID_CHECKPOINT_DIR / run_name / "best_models_weights" / f"best_model_fold_{fold_id}.pth"


def _checkpoint_rng_contract(
    backbone: str,
    aug_name: str,
    fold_id: int,
    args: argparse.Namespace,
) -> tuple[str, int | None]:
    default_run = (
        args.batch_size == DEFAULT_EXPERIMENT_CONFIG["batch_size"]
        and args.lr == DEFAULT_EXPERIMENT_CONFIG["learning_rate"]
        and args.weight_decay == DEFAULT_EXPERIMENT_CONFIG["weight_decay"]
        and args.dropout == DEFAULT_EXPERIMENT_CONFIG["dropout_rate"]
        and args.training_pretrained == DEFAULT_EXPERIMENT_CONFIG["pretrained"]
        and args.epochs == DEFAULT_EXPERIMENT_CONFIG["epochs"]
        and args.patience == DEFAULT_EXPERIMENT_CONFIG["patience"]
        and args.training_num_workers == DEFAULT_EXPERIMENT_CONFIG["num_workers"]
        and args.run_tag is None
    )
    if default_run:
        return "historical_global_seed_40_canonical_order", None
    run_digest = experiment_config_sha256(
        backbone_name=backbone,
        augmentation=aug_name,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        dropout_rate=args.dropout,
        pretrained=args.training_pretrained,
        epochs=args.epochs,
        patience=args.patience,
        num_workers=args.training_num_workers,
        run_tag=args.run_tag,
    )
    return "per_fold_sha256_v1", derive_fold_seed(
        run_digest, fold_id, dataset="PIID"
    )


def expected_training_config(
    backbone: str,
    aug_name: str,
    fold_id: int,
    input_file_sha256: dict[str, str],
    args: argparse.Namespace,
) -> dict:
    """Configuration fields an evaluator can verify without private split data."""
    run_identity = canonical_experiment_config(
        backbone,
        aug_name,
        args.batch_size,
        args.lr,
        args.weight_decay,
        dropout_rate=args.dropout,
        pretrained=args.training_pretrained,
        epochs=args.epochs,
        patience=args.patience,
        num_workers=args.training_num_workers,
        run_tag=args.run_tag,
    )
    rng_strategy, fold_seed = _checkpoint_rng_contract(
        backbone, aug_name, fold_id, args
    )
    return {
        "schema_version": 1,
        "dataset": "PIID",
        "model": backbone,
        "augmentation": aug_name,
        "fold": fold_id,
        "random_seed": RANDOM_SEED,
        "input_size": INPUT_SIZE,
        "num_classes": NUM_CLASSES,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "weight_decay": args.weight_decay,
        "dropout_rate": args.dropout,
        "pretrained": args.training_pretrained,
        "epochs": args.epochs,
        "patience": args.patience,
        "num_workers": args.training_num_workers,
        "drop_last": True,
        "run_tag": args.run_tag,
        "run_name": experiment_run_name(
            backbone,
            aug_name,
            args.batch_size,
            args.lr,
            args.weight_decay,
            dropout_rate=args.dropout,
            pretrained=args.training_pretrained,
            epochs=args.epochs,
            patience=args.patience,
            num_workers=args.training_num_workers,
            run_tag=args.run_tag,
        ),
        "run_identity": run_identity,
        "run_identity_sha256": experiment_config_sha256(
            backbone_name=backbone,
            augmentation=aug_name,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            weight_decay=args.weight_decay,
            dropout_rate=args.dropout,
            pretrained=args.training_pretrained,
            epochs=args.epochs,
            patience=args.patience,
            num_workers=args.training_num_workers,
            run_tag=args.run_tag,
        ),
        "rng_strategy": rng_strategy,
        "fold_seed": fold_seed,
        "input_file_sha256": dict(input_file_sha256),
    }


def validate_args(args: argparse.Namespace) -> None:
    for name, values in {
        "models": args.models,
        "augmentations": args.augmentations,
        "folds": args.folds,
    }.items():
        if len(values) != len(set(values)):
            raise ValueError(f"Duplicate {name} are not allowed: {values}")
    invalid_folds = [fold for fold in args.folds if fold not in range(1, N_FOLDS + 1)]
    if invalid_folds:
        raise ValueError(f"folds must be between 1 and {N_FOLDS}: {invalid_folds}")
    for name in ("epochs", "patience", "batch_size"):
        if getattr(args, name) <= 0:
            raise ValueError(f"{name} must be positive")
    if args.num_workers < 0 or args.training_num_workers < 0:
        raise ValueError("worker counts must be non-negative")
    if not np.isfinite(args.lr) or args.lr <= 0:
        raise ValueError("lr must be finite and positive")
    if not np.isfinite(args.weight_decay) or args.weight_decay < 0:
        raise ValueError("weight_decay must be finite and non-negative")
    if not np.isfinite(args.dropout) or not 0 <= args.dropout < 1:
        raise ValueError("dropout must be finite and in [0, 1)")
    validate_run_tag(args.run_tag)


def summarize_foldwise(df: pd.DataFrame) -> pd.DataFrame:
    summary_rows = []
    for (model, aug), group in df.groupby(["Model", "Augmentation"], sort=False):
        row = {"Model": model, "Augmentation": aug}
        for dataset in sorted(group["Dataset"].unique()):
            ds_group = group[group["Dataset"] == dataset]
            for metric in ["Accuracy", "Balanced_Accuracy", "F1_Macro", "AUC_Macro", "Cohens_Kappa"]:
                row[f"{dataset}_{metric}_mean"] = ds_group[metric].mean()
                row[f"{dataset}_{metric}_std"] = ds_group[metric].std(ddof=0)
        summary_rows.append(row)
    return pd.DataFrame(summary_rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate PIID-trained checkpoints.")
    parser.add_argument("--models", nargs="+", default=BACKBONES, choices=BACKBONES)
    parser.add_argument("--augmentations", nargs="+", default=AUGMENTATIONS, choices=AUGMENTATIONS)
    parser.add_argument("--folds", nargs="+", type=int, default=list(range(1, N_FOLDS + 1)))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=50, help="Epoch setting used to name training runs.")
    parser.add_argument("--patience", type=int, default=20, help="Patience setting used to name training runs.")
    parser.add_argument("--training-num-workers", type=int, default=0)
    parser.add_argument(
        "--training-pretrained",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Pretrained setting used by the checkpoint's training run.",
    )
    parser.add_argument("--run-tag", default=None, help="Optional training-run suffix.")
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Explicitly permit a partial evaluation when requested checkpoints are absent.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--allow-legacy-checkpoints",
        action="store_true",
        help=(
            "Explicitly allow archived .pth files that predate completion markers. "
            "Their completion status will be recorded as unverified."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validate_args(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    PIID_INFERENCE_DIR.parent.mkdir(parents=True, exist_ok=True)
    staging_context = tempfile.TemporaryDirectory(
        prefix=f".{PIID_INFERENCE_DIR.name}-staging-",
        dir=PIID_INFERENCE_DIR.parent,
    )
    output_root = Path(staging_context.name)
    meta_path = output_root / "__evaluation_meta.json"
    progress_path = output_root / "__evaluation_in_progress.csv"
    _atomic_write_json(meta_path, {"status": "running", "complete": False})

    norm_stats = load_norm_stats()
    input_file_sha256 = training_input_sha256()
    piid_test = load_piid_test()
    split_metadata_present = (PIID_SPLIT_DIR / "piid_split_meta.json").is_file()
    require_dataset_contract(
        "PIID_Test", piid_test, EXPECTED_PIID_TEST_STAGE_COUNTS
    )
    kaggle = load_stage_folder_dataset(KAGGLE_DATA_DIR)
    require_dataset_contract("Kaggle", kaggle, EXPECTED_KAGGLE_STAGE_COUNTS)
    eval_sets = {"PIID_Test": piid_test, "Kaggle": kaggle}
    if stage_folder_dataset_has_images(HUMC_DATA_DIR):
        humc = load_stage_folder_dataset(HUMC_DATA_DIR)
        require_dataset_contract("HUMC", humc, EXPECTED_HUMC_STAGE_COUNTS)
        eval_sets["HUMC"] = humc
    else:
        print(f"[SKIP] Optional HUMC dataset is not present: {HUMC_DATA_DIR}")
    print({name: len(paths) for name, (paths, _) in eval_sets.items()})

    requested = {
        (backbone, aug_name, fold_id): checkpoint_path(
            backbone, aug_name, fold_id, args
        )
        for backbone in args.models
        for aug_name in args.augmentations
        for fold_id in args.folds
    }
    missing = {}
    legacy_unverified = {}
    for key, path in requested.items():
        backbone, aug_name, fold_id = key
        valid, reason = validate_completed_checkpoint(
            path,
            fold_id,
            expected_training_config(
                backbone, aug_name, fold_id, input_file_sha256, args
            ),
        )
        marker_missing = (
            path.is_file()
            and not completion_marker_path(path, fold_id).is_file()
        )
        if valid:
            continue
        if marker_missing and args.allow_legacy_checkpoints:
            legacy_unverified[key] = path
            continue
        missing[key] = {"path": path, "reason": reason}
    if missing and not args.allow_missing:
        examples = "\n".join(
            f"  - {item['path']}: {item['reason']}"
            for item in list(missing.values())[:10]
        )
        raise FileNotFoundError(
            f"Unavailable or incomplete: {len(missing)} of {len(requested)} "
            "requested checkpoints. "
            "Use --allow-missing only for an intentional partial evaluation.\n"
            f"{examples}"
        )
    if missing:
        print(
            f"[PARTIAL] Explicitly allowing {len(missing)} missing checkpoints "
            f"out of {len(requested)} requested."
        )

    foldwise_rows = []
    prediction_files = set()
    evaluated_checkpoint_count = 0
    for backbone in args.models:
        for aug_name in args.augmentations:
            run_name = experiment_run_name(
                backbone,
                aug_name,
                args.batch_size,
                args.lr,
                args.weight_decay,
                dropout_rate=args.dropout,
                pretrained=args.training_pretrained,
                epochs=args.epochs,
                patience=args.patience,
                num_workers=args.training_num_workers,
                run_tag=args.run_tag,
            )
            legacy_name = experiment_run_name(
                backbone, aug_name, 16, 1e-5, 1e-4
            )
            output_name = f"{backbone}_{aug_name}" if run_name == legacy_name else run_name
            run_output = output_root / output_name
            for fold_id in args.folds:
                ckpt = requested[(backbone, aug_name, fold_id)]
                if (backbone, aug_name, fold_id) in missing:
                    print(f"[SKIP] Unavailable checkpoint allowed explicitly: {ckpt}")
                    continue

                if (backbone, aug_name, fold_id) not in legacy_unverified:
                    valid, reason = validate_completed_checkpoint(
                        ckpt,
                        fold_id,
                        expected_training_config(
                            backbone,
                            aug_name,
                            fold_id,
                            input_file_sha256,
                            args,
                        ),
                    )
                    if not valid:
                        raise RuntimeError(
                            f"Checkpoint completion state changed during evaluation: "
                            f"{ckpt}: {reason}"
                        )

                model = get_model(
                    backbone,
                    num_classes=NUM_CLASSES,
                    pretrained=False,
                    input_size=INPUT_SIZE,
                    dropout_rate=args.dropout,
                )
                state = load_checkpoint_state_dict(ckpt)
                model.load_state_dict(state, strict=True)
                del state
                model.to(device)

                stats = norm_stats[fold_id]
                transform = build_eval_transform(stats["mean"], stats["std"])

                for dataset_name, (paths, labels) in eval_sets.items():
                    result = evaluate_dataset(
                        model=model,
                        paths=paths,
                        labels=labels,
                        transform=transform,
                        batch_size=args.batch_size,
                        num_workers=args.num_workers,
                        device=device,
                    )
                    if len(result["paths"]) != len(paths):
                        raise RuntimeError(
                            f"{dataset_name} prediction row count mismatch: "
                            f"expected {len(paths)}, found {len(result['paths'])}"
                        )
                    prediction_path = save_predictions(
                        run_output / "predictions", dataset_name, fold_id, result
                    )
                    prediction_files.add(prediction_path.relative_to(output_root))

                    row = {
                        "Model": backbone,
                        "Augmentation": aug_name,
                        "Fold": fold_id,
                        "Dataset": dataset_name,
                        "Checkpoint_Path": str(ckpt),
                        "Normalization_Fold": fold_id,
                    }
                    row.update(result["metrics"])
                    foldwise_rows.append(row)

                evaluated_checkpoint_count += 1

                _atomic_write_csv(
                    pd.DataFrame(foldwise_rows),
                    progress_path,
                )

    foldwise_df = pd.DataFrame(foldwise_rows)
    if foldwise_df.empty:
        raise RuntimeError("No checkpoint was evaluated.")
    expected_evaluated = len(requested) - len(missing)
    if evaluated_checkpoint_count != expected_evaluated:
        raise RuntimeError(
            f"Checkpoint evaluation count mismatch: expected {expected_evaluated}, "
            f"evaluated {evaluated_checkpoint_count}."
        )
    expected_rows = expected_evaluated * len(eval_sets)
    if len(foldwise_df) != expected_rows:
        raise RuntimeError(
            f"Foldwise result count mismatch: expected {expected_rows}, "
            f"found {len(foldwise_df)}."
        )
    actual_prediction_files = {
        path.relative_to(output_root)
        for path in output_root.glob("*/predictions/*_fold*_predictions.csv")
    }
    if (
        len(prediction_files) != expected_rows
        or actual_prediction_files != prediction_files
    ):
        raise RuntimeError(
            "Prediction artifact manifest mismatch: "
            f"expected {expected_rows}, tracked {len(prediction_files)}, "
            f"found {len(actual_prediction_files)}"
        )

    summary_df = summarize_foldwise(foldwise_df)
    _atomic_write_csv(
        foldwise_df,
        output_root / "__ALL_foldwise_results.csv",
    )
    _atomic_write_csv(
        summary_df,
        output_root / "__ALL_summary_results.csv",
    )
    progress_path.unlink(missing_ok=True)

    requested_scope_complete = (
        not missing and evaluated_checkpoint_count == len(requested)
    )
    canonical_selection = (
        args.models == BACKBONES
        and args.augmentations == AUGMENTATIONS
        and args.folds == list(range(1, N_FOLDS + 1))
    )
    default_training_config = (
        args.batch_size == DEFAULT_EXPERIMENT_CONFIG["batch_size"]
        and args.lr == DEFAULT_EXPERIMENT_CONFIG["learning_rate"]
        and args.weight_decay == DEFAULT_EXPERIMENT_CONFIG["weight_decay"]
        and args.dropout == DEFAULT_EXPERIMENT_CONFIG["dropout_rate"]
        and args.training_pretrained == DEFAULT_EXPERIMENT_CONFIG["pretrained"]
        and args.epochs == DEFAULT_EXPERIMENT_CONFIG["epochs"]
        and args.patience == DEFAULT_EXPERIMENT_CONFIG["patience"]
        and args.training_num_workers == DEFAULT_EXPERIMENT_CONFIG["num_workers"]
        and args.run_tag is None
    )
    manuscript_complete = (
        requested_scope_complete
        and canonical_selection
        and default_training_config
        and split_metadata_present
        and set(eval_sets) == {"PIID_Test", "Kaggle", "HUMC"}
        and not legacy_unverified
        and not args.allow_missing
        and len(foldwise_df) == len(BACKBONES) * len(AUGMENTATIONS) * N_FOLDS * 3
        and len(prediction_files) == len(BACKBONES) * len(AUGMENTATIONS) * N_FOLDS * 3
        and len(summary_df) == len(BACKBONES) * len(AUGMENTATIONS)
    )

    _atomic_write_json(meta_path, {
            "status": (
                "partial"
                if missing
                else "complete_unverified_legacy"
                if legacy_unverified
                else "complete_manuscript"
                if manuscript_complete
                else "complete_noncanonical_scope"
            ),
            "complete": manuscript_complete,
            "manuscript_complete": manuscript_complete,
            "requested_scope_complete": requested_scope_complete,
            "split_metadata_present": split_metadata_present,
            "checkpoint_completion_verified": not legacy_unverified and not missing,
            "allow_missing": args.allow_missing,
            "allow_legacy_checkpoints": args.allow_legacy_checkpoints,
            "requested_checkpoint_count": len(requested),
            "found_checkpoint_count": len(requested) - len(missing),
            "verified_completed_checkpoint_count": (
                len(requested) - len(missing) - len(legacy_unverified)
            ),
            "evaluated_checkpoint_count": evaluated_checkpoint_count,
            "missing_checkpoint_count": len(missing),
            "missing_checkpoints": [str(item["path"]) for item in missing.values()],
            "unavailable_checkpoint_reasons": {
                str(item["path"]): item["reason"] for item in missing.values()
            },
            "legacy_unverified_checkpoint_count": len(legacy_unverified),
            "legacy_unverified_checkpoints": [
                str(path) for path in legacy_unverified.values()
            ],
            "legacy_checkpoint_note": (
                "Completion and full historical training configuration are "
                "unverified for markerless archived checkpoints."
                if legacy_unverified
                else None
            ),
            "foldwise_result_count": len(foldwise_df),
            "prediction_file_count": len(prediction_files),
            "summary_row_count": len(summary_df),
            "evaluated_datasets": list(eval_sets),
            "dataset_counts": {
                name: len(paths) for name, (paths, _) in eval_sets.items()
            },
            "requested_models": args.models,
            "requested_augmentations": args.augmentations,
            "requested_folds": args.folds,
            "training_run_config": {
                "batch_size": args.batch_size,
                "learning_rate": args.lr,
                "weight_decay": args.weight_decay,
                "dropout_rate": args.dropout,
                "pretrained": args.training_pretrained,
                "epochs": args.epochs,
                "patience": args.patience,
                "num_workers": args.training_num_workers,
                "run_tag": args.run_tag,
            },
            "normalization_source": str(PIID_SPLIT_DIR / "normalization_stats.csv"),
            "training_input_sha256": input_file_sha256,
            "checkpoint_root": str(PIID_CHECKPOINT_DIR),
        })

    _promote_snapshot(output_root, PIID_INFERENCE_DIR)
    staging_context.cleanup()

    print(f"[DONE] Foldwise results: {PIID_INFERENCE_DIR / '__ALL_foldwise_results.csv'}")
    print(f"[DONE] Summary results: {PIID_INFERENCE_DIR / '__ALL_summary_results.csv'}")


if __name__ == "__main__":
    main()
