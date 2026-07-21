"""Create private HUMC patient-level splits and fold-wise normalization.

The script is public; the HUMC images, label workbook, patient identifiers,
and image-level split tables are not.  It writes them only inside the local
repository paths excluded by ``.gitignore``.  The study used seed 40, a 15%
patient-level held-out test set, and five patient-grouped folds.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import cv2
import torch
from PIL import Image, ImageOps
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

CODE_ROOT = Path(__file__).resolve().parents[1]
CORE_DIR = CODE_ROOT / "core"
if str(CORE_DIR) not in sys.path:
    sys.path.insert(0, str(CORE_DIR))

from path_config import (  # noqa: E402
    HUMC_DATA_DIR,
    HUMC_LABELING_PATH,
    HUMC_SPLIT_DIR,
    project_relative_path,
    resolve_project_paths,
)

RANDOM_SEED = 40
N_FOLDS = 5
TEST_RATIO = 0.15
INPUT_SIZE = 224
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def set_seed(seed: int = RANDOM_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_labels(path: Path, number_col: str, patient_col: str, stage_col: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Private HUMC label workbook not found: {path}. "
            "See docs/HUMC_PRIVATE_DATA.md."
        )
    table = pd.read_excel(path)
    missing = {number_col, patient_col, stage_col} - set(table.columns)
    if missing:
        raise ValueError(f"Missing required HUMC label columns: {sorted(missing)}")
    return table


def collect_records(
    labels: pd.DataFrame,
    image_root: Path,
    number_col: str,
    patient_col: str,
) -> pd.DataFrame:
    number_to_patient = dict(
        zip(labels[number_col].astype(str), labels[patient_col].astype(str))
    )
    records: list[dict[str, object]] = []
    for stage in range(1, 5):
        stage_dir = image_root / str(stage)
        if not stage_dir.exists():
            continue
        for image_path in sorted(stage_dir.iterdir()):
            if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            patient_id = number_to_patient.get(image_path.stem)
            if patient_id is None:
                print(f"[WARN] No patient identifier for {image_path.name}; skipped")
                continue
            records.append(
                {
                    "image_path": project_relative_path(image_path),
                    "file_stem": image_path.stem,
                    "stage": stage,
                    "patient_id": patient_id,
                }
            )
    table = pd.DataFrame(records)
    if table.empty:
        raise RuntimeError(f"No HUMC images were matched under {image_root}")
    return table


def split_patients(records: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    patient_table = (
        records.groupby("patient_id")
        .agg(dominant_stage=("stage", lambda values: values.mode().iloc[0]))
        .reset_index()
    )
    trainval_ids, test_ids = train_test_split(
        patient_table["patient_id"].to_numpy(),
        test_size=TEST_RATIO,
        random_state=RANDOM_SEED,
        stratify=patient_table["dominant_stage"].to_numpy(),
    )
    trainval = records[records["patient_id"].isin(trainval_ids)].reset_index(drop=True)
    test = records[records["patient_id"].isin(test_ids)].reset_index(drop=True)
    if set(trainval["patient_id"]) & set(test["patient_id"]):
        raise RuntimeError("Patient leakage between HUMC trainval and test sets")
    return trainval, test


def create_folds(trainval: pd.DataFrame) -> dict[str, dict[str, list[int]]]:
    splitter = StratifiedGroupKFold(
        n_splits=N_FOLDS,
        shuffle=True,
        random_state=RANDOM_SEED,
    )
    folds: dict[str, dict[str, list[int]]] = {}
    labels = trainval["stage"].to_numpy()
    groups = trainval["patient_id"].to_numpy()
    for fold_id, (train_idx, val_idx) in enumerate(
        splitter.split(np.arange(len(trainval)), labels, groups), start=1
    ):
        if set(groups[train_idx]) & set(groups[val_idx]):
            raise RuntimeError(f"Patient leakage in fold {fold_id}")
        folds[f"fold_{fold_id}"] = {
            "train_idx": train_idx.tolist(),
            "val_idx": val_idx.tolist(),
        }
    return folds


class RGBDataset(Dataset):
    def __init__(self, paths: list[str]):
        self.paths = paths

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> torch.Tensor:
        with Image.open(self.paths[index]) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
            array = np.asarray(image)
            array = cv2.resize(
                array,
                (INPUT_SIZE, INPUT_SIZE),
                interpolation=cv2.INTER_LINEAR,
            ).astype(np.float32) / 255.0
        return torch.from_numpy(array).permute(2, 0, 1)


def channel_stats(paths: list[str], batch_size: int, num_workers: int) -> tuple[list[float], list[float]]:
    loader = DataLoader(
        RGBDataset(paths),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    channel_sum = torch.zeros(3, dtype=torch.float64)
    channel_sq_sum = torch.zeros(3, dtype=torch.float64)
    pixel_count = 0
    for batch in tqdm(loader, leave=False):
        batch = batch.double()
        channel_sum += batch.sum(dim=(0, 2, 3))
        channel_sq_sum += (batch * batch).sum(dim=(0, 2, 3))
        pixel_count += batch.shape[0] * batch.shape[2] * batch.shape[3]
    mean = channel_sum / pixel_count
    variance = channel_sq_sum / pixel_count - mean * mean
    std = torch.sqrt(torch.clamp(variance, min=0))
    return mean.tolist(), std.tolist()


def save_outputs(
    records: pd.DataFrame,
    trainval: pd.DataFrame,
    test: pd.DataFrame,
    folds: dict,
    batch_size: int,
    num_workers: int,
) -> None:
    HUMC_SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    records.to_csv(HUMC_SPLIT_DIR / "all_images.csv", index=False)
    trainval.to_csv(HUMC_SPLIT_DIR / "trainval_set.csv", index=False)
    test.to_csv(HUMC_SPLIT_DIR / "test_set.csv", index=False)
    (HUMC_SPLIT_DIR / "fold_indices.json").write_text(
        json.dumps(folds, indent=2), encoding="utf-8"
    )

    resolved_paths = np.asarray(resolve_project_paths(trainval["image_path"].tolist()))
    stat_rows = []
    for fold_id in range(1, N_FOLDS + 1):
        train_paths = resolved_paths[folds[f"fold_{fold_id}"]["train_idx"]].tolist()
        mean, std = channel_stats(train_paths, batch_size, num_workers)
        stat_rows.append(
            {
                "fold": fold_id,
                "mean_r": mean[0], "mean_g": mean[1], "mean_b": mean[2],
                "std_r": std[0], "std_g": std[1], "std_b": std[2],
            }
        )
    pd.DataFrame(stat_rows).to_csv(
        HUMC_SPLIT_DIR / "normalization_stats.csv", index=False, float_format="%.6f"
    )

    metadata = {
        "dataset": "HUMC",
        "data_release": "private; not included in this repository",
        "random_seed": RANDOM_SEED,
        "test_ratio": TEST_RATIO,
        "n_folds": N_FOLDS,
        "split_unit": "patient_id",
        "total_images": int(len(records)),
        "trainval_images": int(len(trainval)),
        "test_images": int(len(test)),
        "total_patients": int(records["patient_id"].nunique()),
        "trainval_patients": int(trainval["patient_id"].nunique()),
        "test_patients": int(test["patient_id"].nunique()),
    }
    (HUMC_SPLIT_DIR / "split_meta.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, default=HUMC_LABELING_PATH)
    parser.add_argument("--image-root", type=Path, default=HUMC_DATA_DIR)
    parser.add_argument("--number-col", default="number")
    parser.add_argument("--patient-id-col", default="등록번호")
    parser.add_argument("--stage-col", default="stage")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed()
    sentinel = HUMC_SPLIT_DIR / "fold_indices.json"
    if sentinel.exists() and not args.overwrite:
        raise FileExistsError(
            f"Stored HUMC split already exists: {sentinel}. "
            "Use --overwrite only when intentionally regenerating it."
        )
    labels = load_labels(args.labels, args.number_col, args.patient_id_col, args.stage_col)
    records = collect_records(labels, args.image_root, args.number_col, args.patient_id_col)
    trainval, test = split_patients(records)
    folds = create_folds(trainval)
    save_outputs(records, trainval, test, folds, args.batch_size, args.num_workers)
    print(f"[DONE] Private HUMC split files written under {HUMC_SPLIT_DIR}")


if __name__ == "__main__":
    main()
