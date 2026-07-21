"""Create PIID-main split files and fold-specific normalization statistics.

PIID is the main development dataset in this public reproduction package.
Because PIID does not provide patient identifiers, the split unit is the image.

Outputs:

    data/splits/piid/piid_all_images.csv
    data/splits/piid/piid_trainval_set.csv
    data/splits/piid/piid_test_set.csv
    data/splits/piid/piid_fold_indices.json
    data/splits/piid/piid_split_meta.json
    data/splits/piid/normalization_stats.csv

Split rule:

    - Random seed: 40
    - Internal test: image-level stratified 15%
    - Cross-validation: image-level StratifiedKFold 5-fold on remaining 85%
    - Normalization: computed only from each fold's PIID train images
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageOps
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


CODE_ROOT = Path(__file__).resolve().parents[1]
CORE_DIR = CODE_ROOT / "core"
if str(CORE_DIR) not in sys.path:
    sys.path.insert(0, str(CORE_DIR))

from path_config import (  # noqa: E402
    PIID_DATA_DIR,
    PIID_SPLIT_DIR,
    project_relative_path,
    resolve_project_paths,
)


RANDOM_SEED = 40
TEST_RATIO = 0.15
N_FOLDS = 5
INPUT_SIZE = 224
CLASS_NAMES = ["1", "2", "3", "4"]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


class NormStatsDataset(Dataset):
    """Load images as RGB tensors in [0, 1] for mean/std calculation."""

    def __init__(self, image_paths: list[str], input_size: int = INPUT_SIZE):
        self.image_paths = list(image_paths)
        self.input_size = input_size

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        path = self.image_paths[idx]
        with Image.open(path) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
            image = image.resize((self.input_size, self.input_size), Image.Resampling.BILINEAR)
            arr = np.asarray(image, dtype=np.float32) / 255.0
        return torch.from_numpy(arr).permute(2, 0, 1)


def collect_piid_images(image_root: Path = PIID_DATA_DIR) -> pd.DataFrame:
    """Collect PIID image paths and 1-based stage labels."""
    records: list[dict[str, object]] = []
    for stage in CLASS_NAMES:
        stage_dir = image_root / stage
        if not stage_dir.exists():
            raise FileNotFoundError(f"PIID stage folder not found: {stage_dir}")
        for path in sorted(stage_dir.iterdir()):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                records.append({
                    "image_path": project_relative_path(path),
                    "stage": int(stage),
                })

    df = pd.DataFrame(records)
    if df.empty:
        raise RuntimeError(f"No PIID images found under: {image_root}")
    return df


def create_split(df_all: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Create image-level held-out test split and 5-fold train/val indices."""
    all_indices = np.arange(len(df_all))
    trainval_idx, test_idx = train_test_split(
        all_indices,
        test_size=TEST_RATIO,
        random_state=RANDOM_SEED,
        stratify=df_all["stage"].values,
    )

    df_trainval = df_all.iloc[trainval_idx].reset_index(drop=True)
    df_test = df_all.iloc[test_idx].reset_index(drop=True)

    labels_zero_based = (df_trainval["stage"].values - 1).astype(int)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)

    fold_indices = {}
    for fold_id, (train_idx, val_idx) in enumerate(
        skf.split(np.arange(len(df_trainval)), labels_zero_based),
        start=1,
    ):
        fold_indices[f"fold_{fold_id}"] = {
            "train_idx": train_idx.tolist(),
            "val_idx": val_idx.tolist(),
        }

    return df_trainval, df_test, fold_indices


def calculate_channel_stats(image_paths: list[str], batch_size: int = 64) -> tuple[list[float], list[float]]:
    """Calculate RGB mean/std from train-fold images only."""
    dataset = NormStatsDataset(image_paths)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    pixel_sum = torch.zeros(3, dtype=torch.float64)
    pixel_count = 0
    for batch in tqdm(loader, desc="mean", leave=False):
        pixel_sum += batch.sum(dim=(0, 2, 3)).double()
        pixel_count += batch.shape[0] * batch.shape[2] * batch.shape[3]
    mean = pixel_sum / pixel_count

    var_sum = torch.zeros(3, dtype=torch.float64)
    for batch in tqdm(loader, desc="std", leave=False):
        diff = batch.double() - mean.view(1, 3, 1, 1)
        var_sum += (diff ** 2).sum(dim=(0, 2, 3))
    std = torch.sqrt(var_sum / pixel_count)

    return mean.float().tolist(), std.float().tolist()


def save_outputs(df_all: pd.DataFrame, df_trainval: pd.DataFrame, df_test: pd.DataFrame, fold_indices: dict) -> None:
    """Write split CSV/JSON files."""
    PIID_SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    df_all.to_csv(PIID_SPLIT_DIR / "piid_all_images.csv", index=False)
    df_trainval.to_csv(PIID_SPLIT_DIR / "piid_trainval_set.csv", index=False)
    df_test.to_csv(PIID_SPLIT_DIR / "piid_test_set.csv", index=False)
    with open(PIID_SPLIT_DIR / "piid_fold_indices.json", "w", encoding="utf-8") as f:
        json.dump(fold_indices, f, indent=2)

    meta = {
        "dataset": "PIID",
        "split_unit": "image",
        "random_seed": RANDOM_SEED,
        "test_ratio": TEST_RATIO,
        "n_folds": N_FOLDS,
        "total_images": int(len(df_all)),
        "trainval_images": int(len(df_trainval)),
        "test_images": int(len(df_test)),
        "normalization_rule": "RGB mean/std calculated from each fold's train images only",
    }
    with open(PIID_SPLIT_DIR / "piid_split_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def save_normalization_stats(df_trainval: pd.DataFrame, fold_indices: dict) -> None:
    """Calculate and write fold-specific normalization_stats.csv."""
    rows = []
    for fold_id in range(1, N_FOLDS + 1):
        train_idx = fold_indices[f"fold_{fold_id}"]["train_idx"]
        train_paths = resolve_project_paths(df_trainval.iloc[train_idx]["image_path"].tolist())
        mean, std = calculate_channel_stats(train_paths)
        rows.append({
            "fold": fold_id,
            "mean_r": mean[0],
            "mean_g": mean[1],
            "mean_b": mean[2],
            "std_r": std[0],
            "std_g": std[1],
            "std_b": std[2],
            "n_train_images": len(train_paths),
        })

    pd.DataFrame(rows).to_csv(PIID_SPLIT_DIR / "normalization_stats.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create PIID split and normalization files.")
    parser.add_argument(
        "--use-existing",
        action="store_true",
        help="Use existing split CSV/JSON if present and only recompute normalization if needed.",
    )
    args = parser.parse_args()

    required = [
        PIID_SPLIT_DIR / "piid_all_images.csv",
        PIID_SPLIT_DIR / "piid_trainval_set.csv",
        PIID_SPLIT_DIR / "piid_test_set.csv",
        PIID_SPLIT_DIR / "piid_fold_indices.json",
    ]

    if args.use_existing and all(path.exists() for path in required):
        print(f"[LOAD] Existing split files: {PIID_SPLIT_DIR}")
        df_all = pd.read_csv(PIID_SPLIT_DIR / "piid_all_images.csv")
        df_trainval = pd.read_csv(PIID_SPLIT_DIR / "piid_trainval_set.csv")
        df_test = pd.read_csv(PIID_SPLIT_DIR / "piid_test_set.csv")
        with open(PIID_SPLIT_DIR / "piid_fold_indices.json", "r", encoding="utf-8") as f:
            fold_indices = json.load(f)
    else:
        print(f"[SCAN] PIID images: {PIID_DATA_DIR}")
        df_all = collect_piid_images()
        df_trainval, df_test, fold_indices = create_split(df_all)
        save_outputs(df_all, df_trainval, df_test, fold_indices)

    save_normalization_stats(df_trainval, fold_indices)

    print(f"[OK] PIID total: {len(df_all)}")
    print(f"[OK] Train/val: {len(df_trainval)}")
    print(f"[OK] Internal test: {len(df_test)}")
    print(f"[DONE] Saved split and normalization files to: {PIID_SPLIT_DIR}")


if __name__ == "__main__":
    main()
