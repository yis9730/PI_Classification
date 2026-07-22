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
import tempfile
from pathlib import Path

import albumentations as A
import numpy as np
import pandas as pd
import sklearn
import torch
from PIL import Image
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
# Final analytic HUMC cohort reported in the released manuscript Table 1 source:
# data/aggregates/table_1_cohort_counts.csv.
EXPECTED_FINAL_IMAGE_COUNT = 1844
EXPECTED_STAGE_COUNTS = {1: 233, 2: 709, 3: 575, 4: 327}
EXPECTED_PATIENT_COUNT = 500
EXPECTED_TRAINVAL_IMAGE_COUNT = 1556
EXPECTED_TEST_IMAGE_COUNT = 288
EXPECTED_TRAINVAL_PATIENT_COUNT = 425
EXPECTED_TEST_PATIENT_COUNT = 75
EXPECTED_TRAINVAL_STAGE_COUNTS = {1: 203, 2: 605, 3: 475, 4: 273}
EXPECTED_TEST_STAGE_COUNTS = {1: 30, 2: 104, 3: 100, 4: 54}


def set_seed(seed: int = RANDOM_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _row_summary(mask: pd.Series) -> str:
    """Describe invalid workbook rows without exposing their values."""
    rows = (np.flatnonzero(mask.to_numpy()) + 2).tolist()
    preview = ", ".join(str(row) for row in rows[:10])
    if len(rows) > 10:
        preview += ", ..."
    return f"{len(rows)} row(s) (Excel row(s): {preview})"


def _normalise_image_key(value: object) -> str:
    """Convert an Excel image key to the direct image filename stem."""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)) and np.isfinite(value):
        numeric = float(value)
        if numeric.is_integer():
            return str(int(numeric))
    return str(value).strip()


def _blank_mask(values: pd.Series) -> pd.Series:
    return values.isna() | values.map(
        lambda value: isinstance(value, str) and not value.strip()
    )


def load_labels(path: Path, number_col: str, patient_col: str, stage_col: str) -> pd.DataFrame:
    """Load and strictly validate the private image-to-patient label table."""
    if not path.is_file():
        raise FileNotFoundError(
            "Private HUMC label workbook not found at the --labels location. "
            "See docs/HUMC_PRIVATE_DATA.md."
        )
    table = pd.read_excel(path)
    missing = {number_col, patient_col, stage_col} - set(table.columns)
    if missing:
        raise ValueError(f"Missing required HUMC label columns: {sorted(missing)}")

    number_blank = _blank_mask(table[number_col])
    if number_blank.any():
        raise ValueError(
            "HUMC image keys must be non-null and non-blank: "
            f"{_row_summary(number_blank)}"
        )
    table[number_col] = table[number_col].map(_normalise_image_key)
    duplicate_numbers = table[number_col].duplicated(keep=False)
    if duplicate_numbers.any():
        raise ValueError(
            "HUMC image keys must be unique after filename-stem normalisation: "
            f"{_row_summary(duplicate_numbers)}"
        )

    patient_blank = _blank_mask(table[patient_col])
    if patient_blank.any():
        raise ValueError(
            "HUMC patient identifiers must be non-null and non-blank: "
            f"{_row_summary(patient_blank)}"
        )
    table[patient_col] = table[patient_col].map(_normalise_image_key)

    raw_stage = table[stage_col]
    numeric_stage = pd.to_numeric(raw_stage, errors="coerce")
    finite_stage = numeric_stage.notna() & np.isfinite(numeric_stage)
    integer_stage = finite_stage & (numeric_stage % 1 == 0)
    valid_stage = integer_stage & numeric_stage.between(1, 4)
    boolean_stage = raw_stage.map(lambda value: isinstance(value, (bool, np.bool_)))
    valid_stage &= ~boolean_stage
    if not valid_stage.all():
        raise ValueError(
            "HUMC declared stages must be integers from 1 through 4: "
            f"{_row_summary(~valid_stage)}"
        )
    table[stage_col] = numeric_stage.astype(int)

    if len(table) != EXPECTED_FINAL_IMAGE_COUNT:
        raise ValueError(
            "The final HUMC label workbook must contain exactly "
            f"{EXPECTED_FINAL_IMAGE_COUNT:,} rows as reported in Table 1; "
            f"found {len(table):,}."
        )
    if table[stage_col].value_counts().sort_index().to_dict() != EXPECTED_STAGE_COUNTS:
        raise ValueError("HUMC workbook stage counts differ from the published Table 1 cohort")
    return table


def collect_records(
    labels: pd.DataFrame,
    image_root: Path,
    number_col: str,
    patient_col: str,
    stage_col: str,
) -> pd.DataFrame:
    """Validate a one-to-one label/image match and create private split rows."""
    if not image_root.is_dir():
        raise FileNotFoundError(
            "Private HUMC image root not found at the --image-root location. "
            "See docs/HUMC_PRIVATE_DATA.md."
        )

    image_entries: list[tuple[Path, str, int]] = []
    missing_stage_dirs: list[int] = []
    empty_stage_dirs: list[int] = []
    for stage in range(1, 5):
        stage_dir = image_root / str(stage)
        if not stage_dir.is_dir():
            missing_stage_dirs.append(stage)
            continue
        stage_images = sorted(
            path
            for path in stage_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not stage_images:
            empty_stage_dirs.append(stage)
            continue
        for path in stage_images:
            try:
                with Image.open(path) as image:
                    image.load()
                    width, height = image.size
                    image_format = image.format
            except OSError:
                raise ValueError(
                    f"HUMC stage {stage} contains an unreadable source image"
                ) from None
            if width <= 0 or height <= 0 or image_format is None:
                raise ValueError(
                    f"HUMC stage {stage} contains an invalid curated image"
                )
            if width != height:
                raise ValueError(
                    f"HUMC stage {stage} contains a non-square curated image"
                )
        image_entries.extend((path, path.stem, stage) for path in stage_images)

    if missing_stage_dirs or empty_stage_dirs:
        details = []
        if missing_stage_dirs:
            details.append(f"missing stage folder(s): {missing_stage_dirs}")
        if empty_stage_dirs:
            details.append(
                "stage folder(s) without a supported image file: "
                f"{empty_stage_dirs}"
            )
        raise ValueError("Invalid HUMC image layout; " + "; ".join(details))

    image_keys = pd.Series([entry[1] for entry in image_entries], dtype="object")
    duplicate_images = image_keys.duplicated(keep=False)
    label_keys = set(labels[number_col])
    unique_image_keys = set(image_keys)
    images_without_rows = unique_image_keys - label_keys
    rows_without_images = label_keys - unique_image_keys
    unmatched_row_mask = labels[number_col].isin(rows_without_images)

    declared_stage = dict(zip(labels[number_col], labels[stage_col]))
    stage_mismatch_keys = {
        image_key
        for _, image_key, folder_stage in image_entries
        if image_key in declared_stage and declared_stage[image_key] != folder_stage
    }
    stage_mismatch_rows = labels[number_col].isin(stage_mismatch_keys)

    problems: list[str] = []
    if len(image_entries) != EXPECTED_FINAL_IMAGE_COUNT:
        problems.append(
            f"expected {EXPECTED_FINAL_IMAGE_COUNT:,} eligible images but found "
            f"{len(image_entries):,}"
        )
    image_stage_counts = pd.Series(
        [stage for _, _, stage in image_entries], dtype="int64"
    ).value_counts().sort_index().to_dict()
    if image_stage_counts != EXPECTED_STAGE_COUNTS:
        problems.append("image-folder stage counts differ from the published Table 1 cohort")
    if duplicate_images.any():
        problems.append(
            f"{int(duplicate_images.sum()):,} image file(s) have a filename stem "
            "that is not unique"
        )
    if images_without_rows:
        problems.append(
            f"{len(images_without_rows):,} image filename stem(s) have no workbook row"
        )
    if rows_without_images:
        problems.append(
            "workbook rows have no matching image: "
            f"{_row_summary(unmatched_row_mask)}"
        )
    if stage_mismatch_keys:
        problems.append(
            "workbook stage disagrees with the containing stage folder: "
            f"{_row_summary(stage_mismatch_rows)}"
        )
    if problems:
        raise ValueError("HUMC image/label validation failed; " + "; ".join(problems))

    number_to_patient = dict(zip(labels[number_col], labels[patient_col]))
    records: list[dict[str, object]] = []
    for image_path, image_key, stage in image_entries:
        records.append(
            {
                "image_path": project_relative_path(image_path),
                "file_stem": image_key,
                "stage": stage,
                "patient_id": number_to_patient[image_key],
            }
        )
    table = pd.DataFrame(records)
    if len(table) != EXPECTED_FINAL_IMAGE_COUNT:
        raise RuntimeError("Validated HUMC record count changed unexpectedly")
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
    observed = {
        "total_patients": int(records["patient_id"].nunique()),
        "trainval_images": len(trainval),
        "test_images": len(test),
        "trainval_patients": int(trainval["patient_id"].nunique()),
        "test_patients": int(test["patient_id"].nunique()),
    }
    expected = {
        "total_patients": EXPECTED_PATIENT_COUNT,
        "trainval_images": EXPECTED_TRAINVAL_IMAGE_COUNT,
        "test_images": EXPECTED_TEST_IMAGE_COUNT,
        "trainval_patients": EXPECTED_TRAINVAL_PATIENT_COUNT,
        "test_patients": EXPECTED_TEST_PATIENT_COUNT,
    }
    if observed != expected:
        raise RuntimeError(
            "HUMC patient-level split sizes differ from the stored manuscript split"
        )
    if trainval["stage"].value_counts().sort_index().to_dict() != EXPECTED_TRAINVAL_STAGE_COUNTS:
        raise RuntimeError("HUMC train/validation stage counts differ from the manuscript split")
    if test["stage"].value_counts().sort_index().to_dict() != EXPECTED_TEST_STAGE_COUNTS:
        raise RuntimeError("HUMC test stage counts differ from the manuscript split")
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
    validation_coverage = np.zeros(len(trainval), dtype=int)
    for fold in folds.values():
        train_idx = np.asarray(fold["train_idx"], dtype=int)
        val_idx = np.asarray(fold["val_idx"], dtype=int)
        if (
            np.intersect1d(train_idx, val_idx).size
            or len(np.union1d(train_idx, val_idx)) != len(trainval)
        ):
            raise RuntimeError("A HUMC fold is not a disjoint full partition")
        validation_coverage[val_idx] += 1
    if not np.all(validation_coverage == 1):
        raise RuntimeError("Every HUMC train/validation image must be validation exactly once")
    return folds


class RGBDataset(Dataset):
    def __init__(self, paths: list[str]):
        self.paths = paths
        self.transform = A.Compose([A.Resize(INPUT_SIZE, INPUT_SIZE)])

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> torch.Tensor:
        with Image.open(self.paths[index]) as image:
            image = image.convert("RGB")
            array = np.asarray(image)
        array = self.transform(image=array)["image"].astype(np.float32) / 255.0
        return torch.from_numpy(array).permute(2, 0, 1)


def channel_stats(paths: list[str], batch_size: int, num_workers: int) -> tuple[list[float], list[float]]:
    loader = DataLoader(
        RGBDataset(paths),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    channel_sum = torch.zeros(3, dtype=torch.float64)
    pixel_count = 0
    for batch in tqdm(loader, desc="mean", leave=False):
        channel_sum += batch.sum(dim=(0, 2, 3)).double()
        pixel_count += batch.shape[0] * batch.shape[2] * batch.shape[3]
    mean = channel_sum / pixel_count

    variance_sum = torch.zeros(3, dtype=torch.float64)
    for batch in tqdm(loader, desc="std", leave=False):
        difference = batch.double() - mean.view(1, 3, 1, 1)
        variance_sum += (difference ** 2).sum(dim=(0, 2, 3))
    std = torch.sqrt(variance_sum / pixel_count)
    return mean.float().tolist(), std.float().tolist()


def atomic_write_csv(frame: pd.DataFrame, destination: Path, **kwargs) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=destination.parent,
        prefix=f".{destination.stem}-",
        suffix=".csv",
        mode="w",
        encoding="utf-8",
        newline="",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        frame.to_csv(handle, index=False, lineterminator="\n", **kwargs)
    try:
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_write_json(payload: dict, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=destination.parent,
        prefix=f".{destination.stem}-",
        suffix=".json",
        mode="w",
        encoding="utf-8",
        newline="",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        json.dump(payload, handle, indent=2, ensure_ascii=True)
        handle.write("\n")
    try:
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def save_outputs(
    records: pd.DataFrame,
    trainval: pd.DataFrame,
    test: pd.DataFrame,
    folds: dict,
    batch_size: int,
    num_workers: int,
) -> None:
    HUMC_SPLIT_DIR.mkdir(parents=True, exist_ok=True)
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
                "n_train_images": len(train_paths),
            }
        )

    metadata = {
        "dataset": "HUMC",
        "data_release": "private; not included in this repository",
        "random_seed": RANDOM_SEED,
        "test_ratio": TEST_RATIO,
        "n_folds": N_FOLDS,
        "split_unit": "patient_id",
        "total_images": int(len(records)),
        "n_trainval_images": int(len(trainval)),
        "n_test_images": int(len(test)),
        "total_patients": int(records["patient_id"].nunique()),
        "n_trainval_patients": int(trainval["patient_id"].nunique()),
        "n_test_patients": int(test["patient_id"].nunique()),
        "stage_distribution": {
            "total": {str(k): v for k, v in EXPECTED_STAGE_COUNTS.items()},
            "trainval": {str(k): v for k, v in EXPECTED_TRAINVAL_STAGE_COUNTS.items()},
            "test": {str(k): v for k, v in EXPECTED_TEST_STAGE_COUNTS.items()},
        },
        "normalization": {
            "input_size": INPUT_SIZE,
            "resize_method": "albumentations.Resize (OpenCV INTER_LINEAR)",
            "pixel_range": "[0, 1]",
            "computed_from": "train indices only, separately for each fold",
        },
        "sklearn_version": sklearn.__version__,
        "exact_reproducibility_note": (
            "Use the controlled stored fold_indices.json for the submitted run; "
            "seed alone may not reproduce group folds across library versions."
        ),
    }
    # Compute every expensive artifact before publishing any replacement.
    atomic_write_csv(records, HUMC_SPLIT_DIR / "all_images.csv")
    atomic_write_csv(trainval, HUMC_SPLIT_DIR / "trainval_set.csv")
    atomic_write_csv(test, HUMC_SPLIT_DIR / "test_set.csv")
    atomic_write_json(folds, HUMC_SPLIT_DIR / "fold_indices.json")
    atomic_write_csv(
        pd.DataFrame(stat_rows),
        HUMC_SPLIT_DIR / "normalization_stats.csv",
        float_format="%.6f",
    )
    # Metadata is the completion sentinel and is published last.
    atomic_write_json(metadata, HUMC_SPLIT_DIR / "split_meta.json")


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
    if args.batch_size <= 0 or args.num_workers < 0:
        raise ValueError("--batch-size must be positive and --num-workers cannot be negative")
    image_root = args.image_root.expanduser().resolve()
    split_root = HUMC_SPLIT_DIR.resolve()
    if image_root == split_root or image_root in split_root.parents or split_root in image_root.parents:
        raise ValueError("HUMC image input and split output directories must not overlap")
    set_seed()
    output_names = (
        "all_images.csv",
        "trainval_set.csv",
        "test_set.csv",
        "fold_indices.json",
        "normalization_stats.csv",
        "split_meta.json",
    )
    existing_outputs = [
        HUMC_SPLIT_DIR / name
        for name in output_names
        if (HUMC_SPLIT_DIR / name).exists()
    ]
    if existing_outputs and not args.overwrite:
        raise FileExistsError(
            f"Stored HUMC split artifacts already exist ({len(existing_outputs)} files). "
            "Use --overwrite only when intentionally regenerating it."
        )
    labels = load_labels(args.labels, args.number_col, args.patient_id_col, args.stage_col)
    records = collect_records(
        labels,
        args.image_root,
        args.number_col,
        args.patient_id_col,
        args.stage_col,
    )
    trainval, test = split_patients(records)
    folds = create_folds(trainval)
    save_outputs(records, trainval, test, folds, args.batch_size, args.num_workers)
    print(f"[DONE] Private HUMC split files written under {HUMC_SPLIT_DIR}")


if __name__ == "__main__":
    main()
