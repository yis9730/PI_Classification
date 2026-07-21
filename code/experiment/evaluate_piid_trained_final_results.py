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
import sys
from pathlib import Path

import albumentations as A
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from albumentations.pytorch import ToTensorV2
from PIL import Image, ImageOps
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
DEVELOPMENT_DIR = CODE_ROOT / "development"
if str(DEVELOPMENT_DIR) not in sys.path:
    sys.path.insert(0, str(DEVELOPMENT_DIR))

from model_pipeline_utils import get_model  # noqa: E402
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
CLASS_NAMES = ["1", "2", "3", "4"]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

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
            image = ImageOps.exif_transpose(image).convert("RGB")
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


def load_norm_stats() -> dict[int, dict[str, list[float]]]:
    path = PIID_SPLIT_DIR / "normalization_stats.csv"
    if not path.exists():
        raise FileNotFoundError(f"PIID normalization file not found: {path}")
    df = pd.read_csv(path)
    return {
        int(row["fold"]): {
            "mean": [float(row["mean_r"]), float(row["mean_g"]), float(row["mean_b"])],
            "std": [float(row["std_r"]), float(row["std_g"]), float(row["std_b"])],
        }
        for _, row in df.iterrows()
    }


def load_piid_test() -> tuple[list[str], list[int]]:
    path = PIID_SPLIT_DIR / "piid_test_set.csv"
    if not path.exists():
        raise FileNotFoundError(f"PIID test split file not found: {path}")
    df = pd.read_csv(path)
    paths = resolve_project_paths(df["image_path"].tolist())
    labels = (df["stage"].values - 1).astype(int).tolist()
    return paths, labels


def load_stage_folder_dataset(root: Path) -> tuple[list[str], list[int]]:
    paths, labels = [], []
    for label_idx, stage in enumerate(CLASS_NAMES):
        stage_dir = root / stage
        if not stage_dir.exists():
            raise FileNotFoundError(f"Stage folder not found: {stage_dir}")
        for path in sorted(stage_dir.iterdir()):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                paths.append(str(path))
                labels.append(label_idx)
    if not paths:
        raise RuntimeError(f"No images found under: {root}")
    return paths, labels


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
    try:
        y_bin = label_binarize(y_true, classes=list(range(NUM_CLASSES)))
        return float(roc_auc_score(y_bin, y_prob, average="macro", multi_class="ovr"))
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


def save_predictions(prediction_dir: Path, dataset_name: str, fold_id: int, result: dict) -> None:
    prediction_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({
        "image_path": result["paths"],
        "true_label": result["y_true"],
        "predicted_label": result["y_pred"],
    })
    for idx, class_name in enumerate(CLASS_NAMES):
        df[f"prob_{class_name}"] = result["y_prob"][:, idx]
    df.to_csv(prediction_dir / f"{dataset_name}_fold{fold_id}_predictions.csv", index=False)


def checkpoint_path(backbone: str, aug_name: str, fold_id: int, batch_size: int, lr: float, weight_decay: float) -> Path:
    run_name = f"{backbone}_Baseline_{aug_name}_bs{batch_size}_lr{lr:.0e}_wd{weight_decay:.0e}"
    return PIID_CHECKPOINT_DIR / run_name / "best_models_weights" / f"best_model_fold_{fold_id}.pth"


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
    parser.add_argument("--strict", action="store_true", help="Fail when a checkpoint is missing.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    PIID_INFERENCE_DIR.mkdir(parents=True, exist_ok=True)

    norm_stats = load_norm_stats()
    eval_sets = {"PIID_Test": load_piid_test()}
    for dataset_name, dataset_root in {
        "HUMC": HUMC_DATA_DIR,
        "Kaggle": KAGGLE_DATA_DIR,
    }.items():
        if dataset_root.exists():
            eval_sets[dataset_name] = load_stage_folder_dataset(dataset_root)
        else:
            print(f"[SKIP] Optional dataset not found: {dataset_root}")
    print({name: len(paths) for name, (paths, _) in eval_sets.items()})

    foldwise_rows = []
    for backbone in args.models:
        for aug_name in args.augmentations:
            run_output = PIID_INFERENCE_DIR / f"{backbone}_{aug_name}"
            for fold_id in args.folds:
                ckpt = checkpoint_path(backbone, aug_name, fold_id, args.batch_size, args.lr, args.weight_decay)
                if not ckpt.exists():
                    message = f"Missing checkpoint: {ckpt}"
                    if args.strict:
                        raise FileNotFoundError(message)
                    print(f"[SKIP] {message}")
                    continue

                model = get_model(
                    backbone,
                    num_classes=NUM_CLASSES,
                    pretrained=False,
                    input_size=INPUT_SIZE,
                    dropout_rate=args.dropout,
                )
                state = torch.load(ckpt, map_location=device)
                model.load_state_dict(state)
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
                    save_predictions(run_output / "predictions", dataset_name, fold_id, result)

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

                pd.DataFrame(foldwise_rows).to_csv(
                    PIID_INFERENCE_DIR / "__ALL_foldwise_results.csv",
                    index=False,
                )

    foldwise_df = pd.DataFrame(foldwise_rows)
    if foldwise_df.empty:
        raise RuntimeError("No checkpoint was evaluated.")

    summary_df = summarize_foldwise(foldwise_df)
    foldwise_df.to_csv(PIID_INFERENCE_DIR / "__ALL_foldwise_results.csv", index=False)
    summary_df.to_csv(PIID_INFERENCE_DIR / "__ALL_summary_results.csv", index=False)

    with open(PIID_INFERENCE_DIR / "__evaluation_meta.json", "w", encoding="utf-8") as f:
        json.dump({
            "evaluated_datasets": list(eval_sets),
            "normalization_source": str(PIID_SPLIT_DIR / "normalization_stats.csv"),
            "checkpoint_root": str(PIID_CHECKPOINT_DIR),
        }, f, indent=2)

    print(f"[DONE] Foldwise results: {PIID_INFERENCE_DIR / '__ALL_foldwise_results.csv'}")
    print(f"[DONE] Summary results: {PIID_INFERENCE_DIR / '__ALL_summary_results.csv'}")


if __name__ == "__main__":
    main()
