"""Train PIID-main baseline models.

This script trains six backbones across 17 augmentation settings using the
released PIID split files.

Default experiment:

    - Dataset: PIID train/validation folds from data/splits/piid
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

from model_pipeline_utils import get_model  # noqa: E402
from path_config import (  # noqa: E402
    PIID_CHECKPOINT_DIR,
    PIID_SPLIT_DIR,
    resolve_project_paths,
)


RANDOM_SEED = 40
INPUT_SIZE = 224
NUM_CLASSES = 4
N_FOLDS = 5
CLASS_NAMES = ["1", "2", "3", "4"]

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


def build_train_transform(mean: list[float], std: list[float], aug_flags: dict) -> A.Compose:
    transforms = [A.Resize(INPUT_SIZE, INPUT_SIZE)]
    if aug_flags.get("use_flip"):
        transforms.append(A.HorizontalFlip(p=0.5))
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


def load_split_files() -> tuple[pd.DataFrame, dict, dict]:
    trainval_csv = PIID_SPLIT_DIR / "piid_trainval_set.csv"
    fold_json = PIID_SPLIT_DIR / "piid_fold_indices.json"
    norm_csv = PIID_SPLIT_DIR / "normalization_stats.csv"
    if not trainval_csv.exists() or not fold_json.exists() or not norm_csv.exists():
        raise FileNotFoundError(
            "PIID split files are missing. Run "
            "code/pipeline/dataset_split_normalization_piid_main.py first."
        )

    df_trainval = pd.read_csv(trainval_csv)
    with open(fold_json, "r", encoding="utf-8") as f:
        fold_indices = json.load(f)
    norm_df = pd.read_csv(norm_csv)
    norm_stats = {
        int(row["fold"]): {
            "mean": [float(row["mean_r"]), float(row["mean_g"]), float(row["mean_b"])],
            "std": [float(row["std_r"]), float(row["std_g"]), float(row["std_b"])],
        }
        for _, row in norm_df.iterrows()
    }
    return df_trainval, fold_indices, norm_stats


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
    args: argparse.Namespace,
    device: torch.device,
) -> dict:
    fold = fold_indices[f"fold_{fold_id}"]
    train_df = df_trainval.iloc[fold["train_idx"]].reset_index(drop=True)
    val_df = df_trainval.iloc[fold["val_idx"]].reset_index(drop=True)

    train_paths = resolve_project_paths(train_df["image_path"].tolist())
    val_paths = resolve_project_paths(val_df["image_path"].tolist())
    train_labels = (train_df["stage"].values - 1).astype(int)
    val_labels = (val_df["stage"].values - 1).astype(int)

    stats = norm_stats[fold_id]
    train_transform = build_train_transform(stats["mean"], stats["std"], aug_flags)
    val_transform = build_eval_transform(stats["mean"], stats["std"])

    train_loader = DataLoader(
        AlbImageDataset(train_paths, train_labels.tolist(), train_transform),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        generator=torch.Generator().manual_seed(RANDOM_SEED + fold_id),
    )
    val_loader = DataLoader(
        AlbImageDataset(val_paths, val_labels.tolist(), val_transform),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
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

    run_name = f"{backbone}_Baseline_{aug_name}_bs{args.batch_size}_lr{args.lr:.0e}_wd{args.weight_decay:.0e}"
    run_dir = PIID_CHECKPOINT_DIR / run_name
    weight_dir = run_dir / "best_models_weights"
    weight_dir.mkdir(parents=True, exist_ok=True)
    weight_path = weight_dir / f"best_model_fold_{fold_id}.pth"

    if weight_path.exists() and not args.overwrite:
        print(f"[SKIP] Existing checkpoint: {weight_path}")
        return {
            "model": backbone,
            "augmentation": aug_name,
            "fold": fold_id,
            "checkpoint_path": str(weight_path),
            "skipped": True,
        }

    best_f1 = -1.0
    best_epoch = 0
    stale_epochs = 0
    history = []

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
            best_f1 = val_metrics["f1_macro"]
            best_epoch = epoch
            stale_epochs = 0
            torch.save(model.state_dict(), weight_path)
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                print(f"[EARLY STOP] fold {fold_id}, epoch {epoch}")
                break

    history_df = pd.DataFrame(history)
    history_df.to_csv(run_dir / f"fold_{fold_id}_history.csv", index=False)

    result = {
        "model": backbone,
        "augmentation": aug_name,
        "fold": fold_id,
        "best_epoch": best_epoch,
        "best_val_f1_macro": best_f1,
        "checkpoint_path": str(weight_path),
        "skipped": False,
    }
    with open(run_dir / f"fold_{fold_id}_metrics.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PIID baseline models.")
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
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(RANDOM_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[DEVICE] {device}")

    df_trainval, fold_indices, norm_stats = load_split_files()
    PIID_CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

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
                    args=args,
                    device=device,
                )
                all_results.append(result)
                pd.DataFrame(all_results).to_csv(
                    PIID_CHECKPOINT_DIR / "__training_summary.csv",
                    index=False,
                )

    print(f"[DONE] Training summary: {PIID_CHECKPOINT_DIR / '__training_summary.csv'}")


if __name__ == "__main__":
    main()
