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
import tempfile
from pathlib import Path

import albumentations as A
import numpy as np
import pandas as pd
import torch
from PIL import Image
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
EXPECTED_STAGE_COUNTS = {1: 229, 2: 311, 3: 273, 4: 268}
EXPECTED_TRAINVAL_STAGE_COUNTS = {1: 194, 2: 264, 3: 232, 4: 228}
EXPECTED_TEST_STAGE_COUNTS = {1: 35, 2: 47, 3: 41, 4: 40}
EXPECTED_TRAINVAL_IMAGES = 918
EXPECTED_TEST_IMAGES = 163


class NormStatsDataset(Dataset):
    """Load images as RGB tensors in [0, 1] for mean/std calculation."""

    def __init__(self, image_paths: list[str], input_size: int = INPUT_SIZE):
        self.image_paths = list(image_paths)
        self.input_size = input_size
        self.resize = A.Compose([A.Resize(input_size, input_size)])

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        path = self.image_paths[idx]
        with Image.open(path) as image:
            image = image.convert("RGB")
            arr = np.asarray(image)
        arr = self.resize(image=arr)["image"].astype(np.float32) / 255.0
        return torch.from_numpy(arr).permute(2, 0, 1)


def collect_piid_images(image_root: Path = PIID_DATA_DIR) -> pd.DataFrame:
    """Collect PIID image paths and 1-based stage labels."""
    records: list[dict[str, object]] = []
    for stage in CLASS_NAMES:
        stage_dir = image_root / stage
        if not stage_dir.is_dir():
            raise FileNotFoundError(f"PIID stage folder not found: {stage_dir}")
        for path in sorted(stage_dir.iterdir()):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                try:
                    with Image.open(path) as image:
                        image.load()
                        width, height = image.size
                        image_format = image.format
                except OSError:
                    raise ValueError(
                        f"PIID stage {stage} contains an unreadable curated image"
                    ) from None
                if width <= 0 or height <= 0 or image_format is None:
                    raise ValueError(
                        f"PIID stage {stage} contains an invalid curated image"
                    )
                if width != height:
                    raise ValueError(
                        f"PIID stage {stage} contains a non-square curated image"
                    )
                records.append({
                    "image_path": project_relative_path(path),
                    "stage": int(stage),
                })

    df = pd.DataFrame(records)
    if df.empty:
        raise RuntimeError(f"No PIID images found under: {image_root}")
    observed = df["stage"].value_counts().sort_index().to_dict()
    if observed != EXPECTED_STAGE_COUNTS:
        raise RuntimeError(
            f"PIID stage counts differ from the released analytic cohort: {observed}"
        )
    return df


def validate_table(table: pd.DataFrame, label: str) -> pd.DataFrame:
    """Validate a PIID split table without exposing image names in errors."""
    required = {"image_path", "stage"}
    missing = required - set(table.columns)
    if missing:
        raise ValueError(f"{label} is missing columns: {sorted(missing)}")
    if table.empty or table[list(required)].isna().any().any():
        raise ValueError(f"{label} is empty or contains blank values")
    paths = table["image_path"].astype(str)
    if paths.str.strip().eq("").any() or paths.duplicated().any():
        raise ValueError(f"{label} contains blank or duplicate image paths")
    numeric = pd.to_numeric(table["stage"], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(numeric).all() or not np.equal(numeric, np.floor(numeric)).all():
        raise ValueError(f"{label} stage values must be finite integers")
    stages = numeric.astype(int)
    if not set(np.unique(stages)).issubset({1, 2, 3, 4}):
        raise ValueError(f"{label} contains a stage outside 1-4")
    validated = table.copy()
    validated["image_path"] = paths
    validated["stage"] = stages
    return validated


def validate_existing_split(
    df_all: pd.DataFrame,
    df_trainval: pd.DataFrame,
    df_test: pd.DataFrame,
    fold_indices: dict,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Reject partial, mixed, or misaligned released split artifacts."""
    df_all = validate_table(df_all, "piid_all_images.csv")
    df_trainval = validate_table(df_trainval, "piid_trainval_set.csv")
    df_test = validate_table(df_test, "piid_test_set.csv")
    if len(df_all) != sum(EXPECTED_STAGE_COUNTS.values()):
        raise ValueError("PIID all-images table does not contain 1,081 rows")
    if df_all["stage"].value_counts().sort_index().to_dict() != EXPECTED_STAGE_COUNTS:
        raise ValueError("PIID all-images stage counts differ from the released cohort")
    if len(df_trainval) != EXPECTED_TRAINVAL_IMAGES or len(df_test) != EXPECTED_TEST_IMAGES:
        raise ValueError("PIID train/validation or test row count differs from the released split")
    if df_trainval["stage"].value_counts().sort_index().to_dict() != EXPECTED_TRAINVAL_STAGE_COUNTS:
        raise ValueError("PIID train/validation stage counts differ from the released split")
    if df_test["stage"].value_counts().sort_index().to_dict() != EXPECTED_TEST_STAGE_COUNTS:
        raise ValueError("PIID test stage counts differ from the released split")

    all_map = dict(zip(df_all["image_path"], df_all["stage"], strict=True))
    trainval_map = dict(zip(df_trainval["image_path"], df_trainval["stage"], strict=True))
    test_map = dict(zip(df_test["image_path"], df_test["stage"], strict=True))
    if set(trainval_map) & set(test_map) or set(trainval_map) | set(test_map) != set(all_map):
        raise ValueError("PIID train/validation and test paths are not a disjoint partition")
    if any(all_map[path] != stage for path, stage in {**trainval_map, **test_map}.items()):
        raise ValueError("PIID stage labels disagree across split tables")

    expected_keys = {f"fold_{fold}" for fold in range(1, N_FOLDS + 1)}
    if not isinstance(fold_indices, dict) or set(fold_indices) != expected_keys:
        raise ValueError("PIID fold JSON must contain exactly fold_1 through fold_5")
    validation_coverage = np.zeros(len(df_trainval), dtype=int)
    for fold in range(1, N_FOLDS + 1):
        record = fold_indices[f"fold_{fold}"]
        if not isinstance(record, dict) or set(record) != {"train_idx", "val_idx"}:
            raise ValueError(f"fold_{fold} must contain train_idx and val_idx only")
        train = np.asarray(record["train_idx"])
        validation = np.asarray(record["val_idx"])
        for values, name in ((train, "train_idx"), (validation, "val_idx")):
            if values.ndim != 1 or values.dtype.kind not in "iu":
                raise ValueError(f"fold_{fold} {name} must be a one-dimensional integer list")
            if len(np.unique(values)) != len(values):
                raise ValueError(f"fold_{fold} {name} contains duplicate indices")
            if len(values) == 0 or values.min() < 0 or values.max() >= len(df_trainval):
                raise ValueError(f"fold_{fold} {name} contains an out-of-range index")
        if np.intersect1d(train, validation).size or len(train) + len(validation) != len(df_trainval):
            raise ValueError(f"fold_{fold} is not a disjoint full train/validation partition")
        if len(np.union1d(train, validation)) != len(df_trainval):
            raise ValueError(f"fold_{fold} does not cover the complete train/validation set")
        validation_coverage[validation] += 1
    if not np.all(validation_coverage == 1):
        raise ValueError("Every PIID train/validation image must be validation exactly once")

    # The environment is pinned, so the released files must equal a fresh
    # seed-40 stratified reconstruction rather than merely having plausible
    # sizes and disjoint paths.
    expected_trainval, expected_test, expected_folds = create_split(df_all)
    columns = ["image_path", "stage"]
    if not expected_trainval[columns].equals(df_trainval[columns]):
        raise ValueError("PIID train/validation rows differ from the released seed-40 split")
    if not expected_test[columns].equals(df_test[columns]):
        raise ValueError("PIID test rows differ from the released seed-40 split")
    if expected_folds != fold_indices:
        raise ValueError("PIID fold indices differ from the released seed-40 folds")

    missing_files = sum(
        not Path(path).is_file() for path in resolve_project_paths(df_all["image_path"].tolist())
    )
    if missing_files:
        raise FileNotFoundError(f"PIID split references {missing_files} unavailable image files")
    return df_all, df_trainval, df_test, fold_indices


def validate_normalization_stats(path: Path, fold_indices: dict) -> None:
    table = pd.read_csv(path)
    value_columns = [
        "mean_r", "mean_g", "mean_b", "std_r", "std_g", "std_b", "n_train_images"
    ]
    required = {"fold", *value_columns}
    if required - set(table.columns) or len(table) != N_FOLDS:
        raise ValueError("PIID normalization table has an unexpected schema or row count")
    numeric = table[["fold", *value_columns]].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError("PIID normalization table contains a non-finite value")
    for column in ("fold", "n_train_images"):
        values = numeric[column].to_numpy(dtype=float)
        if not np.equal(values, np.floor(values)).all():
            raise ValueError(f"PIID normalization {column} values must be integers")
    if set(numeric["fold"].astype(int)) != set(range(1, N_FOLDS + 1)):
        raise ValueError("PIID normalization table must contain folds 1-5 exactly once")
    if not ((numeric[["mean_r", "mean_g", "mean_b"]] >= 0).all().all() and
            (numeric[["mean_r", "mean_g", "mean_b"]] <= 1).all().all()):
        raise ValueError("PIID normalization means must lie between 0 and 1")
    if not (numeric[["std_r", "std_g", "std_b"]] > 0).all().all():
        raise ValueError("PIID normalization standard deviations must be positive")
    for fold in range(1, N_FOLDS + 1):
        observed = int(numeric.loc[numeric["fold"].eq(fold), "n_train_images"].iloc[0])
        if observed != len(fold_indices[f"fold_{fold}"]["train_idx"]):
            raise ValueError(f"PIID normalization train count disagrees for fold {fold}")


def validate_split_meta(path: Path) -> None:
    try:
        meta = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        raise ValueError("PIID split metadata is unreadable") from None
    expected = {
        "dataset": "PIID",
        "split_unit": "image",
        "random_seed": RANDOM_SEED,
        "test_ratio": TEST_RATIO,
        "n_folds": N_FOLDS,
        "total_images": sum(EXPECTED_STAGE_COUNTS.values()),
    }
    for key, value in expected.items():
        if meta.get(key) != value:
            raise ValueError(f"PIID split metadata has an unexpected {key}")
    trainval_n = meta.get("n_trainval_images", meta.get("trainval_images"))
    test_n = meta.get("n_test_images", meta.get("test_images"))
    if trainval_n != EXPECTED_TRAINVAL_IMAGES or test_n != EXPECTED_TEST_IMAGES:
        raise ValueError("PIID split metadata contains unexpected partition sizes")
    expected_distributions = {
        "total": {str(k): v for k, v in EXPECTED_STAGE_COUNTS.items()},
        "trainval": {str(k): v for k, v in EXPECTED_TRAINVAL_STAGE_COUNTS.items()},
        "test": {str(k): v for k, v in EXPECTED_TEST_STAGE_COUNTS.items()},
    }
    if meta.get("stage_distribution") != expected_distributions:
        raise ValueError("PIID split metadata stage distributions are inconsistent")


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


def save_outputs(df_all: pd.DataFrame, df_trainval: pd.DataFrame, df_test: pd.DataFrame, fold_indices: dict) -> None:
    """Write split CSV/JSON files."""
    PIID_SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    atomic_write_csv(df_all, PIID_SPLIT_DIR / "piid_all_images.csv")
    atomic_write_csv(df_trainval, PIID_SPLIT_DIR / "piid_trainval_set.csv")
    atomic_write_csv(df_test, PIID_SPLIT_DIR / "piid_test_set.csv")
    atomic_write_json(fold_indices, PIID_SPLIT_DIR / "piid_fold_indices.json")

    meta = {
        "dataset": "PIID",
        "main_role": "main development dataset",
        "split_unit": "image",
        "stratification_unit": "stage",
        "random_seed": RANDOM_SEED,
        "test_ratio": TEST_RATIO,
        "n_folds": N_FOLDS,
        "input_size": INPUT_SIZE,
        "total_images": int(len(df_all)),
        "n_trainval_images": int(len(df_trainval)),
        "n_test_images": int(len(df_test)),
        "stage_distribution": {
            "total": {str(k): v for k, v in EXPECTED_STAGE_COUNTS.items()},
            "trainval": {str(k): v for k, v in EXPECTED_TRAINVAL_STAGE_COUNTS.items()},
            "test": {str(k): v for k, v in EXPECTED_TEST_STAGE_COUNTS.items()},
        },
        "split_files": {
            "all_images": "data/splits/piid/piid_all_images.csv",
            "trainval": "data/splits/piid/piid_trainval_set.csv",
            "test": "data/splits/piid/piid_test_set.csv",
            "fold_indices": "data/splits/piid/piid_fold_indices.json",
            "normalization_stats": "data/splits/piid/normalization_stats.csv",
        },
        "normalization_policy": {
            "computed_from": "PIID train indices only, separately for each fold",
            "excluded": [
                "PIID validation fold",
                "PIID internal test",
                "HUMC external",
                "Kaggle external",
            ],
        },
    }
    atomic_write_json(meta, PIID_SPLIT_DIR / "piid_split_meta.json")


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

    atomic_write_csv(
        pd.DataFrame(rows),
        PIID_SPLIT_DIR / "normalization_stats.csv",
        float_format="%.6f",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Create PIID split and normalization files.")
    parser.add_argument(
        "--use-existing",
        action="store_true",
        help="Use existing split CSV/JSON and normalization values when present.",
    )
    parser.add_argument(
        "--recompute-normalization",
        action="store_true",
        help="Recompute fold normalization even when an existing CSV is available.",
    )
    args = parser.parse_args()

    required = [
        PIID_SPLIT_DIR / "piid_all_images.csv",
        PIID_SPLIT_DIR / "piid_trainval_set.csv",
        PIID_SPLIT_DIR / "piid_test_set.csv",
        PIID_SPLIT_DIR / "piid_fold_indices.json",
        PIID_SPLIT_DIR / "piid_split_meta.json",
    ]

    if args.use_existing and not all(path.exists() for path in required):
        missing = sum(not path.exists() for path in required)
        raise FileNotFoundError(
            f"--use-existing requires the complete PIID split set; {missing} files are missing"
        )

    if args.use_existing:
        print(f"[LOAD] Existing split files: {PIID_SPLIT_DIR}")
        df_all = pd.read_csv(PIID_SPLIT_DIR / "piid_all_images.csv")
        df_trainval = pd.read_csv(PIID_SPLIT_DIR / "piid_trainval_set.csv")
        df_test = pd.read_csv(PIID_SPLIT_DIR / "piid_test_set.csv")
        with open(PIID_SPLIT_DIR / "piid_fold_indices.json", "r", encoding="utf-8") as f:
            fold_indices = json.load(f)
        df_all, df_trainval, df_test, fold_indices = validate_existing_split(
            df_all, df_trainval, df_test, fold_indices
        )
        validate_split_meta(PIID_SPLIT_DIR / "piid_split_meta.json")
    else:
        print(f"[SCAN] PIID images: {PIID_DATA_DIR}")
        df_all = collect_piid_images()
        df_trainval, df_test, fold_indices = create_split(df_all)
        save_outputs(df_all, df_trainval, df_test, fold_indices)

    normalization_path = PIID_SPLIT_DIR / "normalization_stats.csv"
    if args.use_existing and normalization_path.exists() and not args.recompute_normalization:
        validate_normalization_stats(normalization_path, fold_indices)
        print(f"[LOAD] Existing normalization values: {normalization_path}")
    else:
        save_normalization_stats(df_trainval, fold_indices)

    print(f"[OK] PIID total: {len(df_all)}")
    print(f"[OK] Train/val: {len(df_trainval)}")
    print(f"[OK] Internal test: {len(df_test)}")
    print(f"[DONE] Saved split and normalization files to: {PIID_SPLIT_DIR}")


if __name__ == "__main__":
    main()
