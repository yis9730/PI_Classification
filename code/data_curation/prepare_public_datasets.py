"""Build the public PIID/Kaggle analytic datasets.

This script does three things:

1. Reads the public raw PIID and Kaggle stage folders.
2. Excludes duplicate or near-duplicate files listed in the released manifests.
3. Center-crops each image to a square and resizes it to 224 x 224 by default.

It never deletes files from data/raw. The generated dataset is written to:

    data/processed/analytic_data/PIID/{1,2,3,4}
    data/processed/analytic_data/Kaggle/{1,2,3,4}

Expected final counts:

    PIID   : 1,081 images
    Kaggle :   141 images
"""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

import pandas as pd
from PIL import Image, ImageOps
from tqdm import tqdm


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

PIID_STAGE_FOLDERS = {"1": "1", "2": "2", "3": "3", "4": "4"}
KAGGLE_STAGE_FOLDERS = {
    "1": "Stage_I",
    "2": "Stage_II",
    "3": "Stage_III",
    "4": "Stage_IV",
}

EXPECTED_COUNTS = {
    "PIID": {"1": 229, "2": 311, "3": 273, "4": 268, "total": 1081},
    "Kaggle": {"1": 27, "2": 46, "3": 41, "4": 27, "total": 141},
}


def repo_root() -> Path:
    """Return the repository root based on this script location."""
    return Path(__file__).resolve().parents[2]


def read_exclusions(csv_path: Path, key_columns: tuple[str, ...]) -> set[tuple[str, ...]]:
    """Read duplicate-exclusion keys from a CSV manifest."""
    if not csv_path.exists():
        raise FileNotFoundError(f"Exclusion manifest not found: {csv_path}")

    rows = pd.read_csv(csv_path)
    missing = [col for col in key_columns if col not in rows.columns]
    if missing:
        raise ValueError(f"Missing columns in {csv_path}: {missing}")

    return {
        tuple(str(row[col]) for col in key_columns)
        for _, row in rows.iterrows()
    }


def iter_images(folder: Path) -> list[Path]:
    """Return image files directly under a folder, sorted by file name."""
    if not folder.exists():
        raise FileNotFoundError(f"Input folder not found: {folder}")
    return sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def center_crop_square(image: Image.Image) -> Image.Image:
    """Center-crop a PIL image to the largest possible square."""
    width, height = image.size
    side = min(width, height)
    left = (width - side) // 2
    top = (height - side) // 2
    return image.crop((left, top, left + side, top + side))


def save_square_image(src: Path, dst: Path, image_size: int) -> tuple[int, int]:
    """Load an image, square-crop, resize, and save it."""
    with Image.open(src) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        image = center_crop_square(image)
        if image_size:
            image = image.resize((image_size, image_size), Image.Resampling.LANCZOS)
        dst.parent.mkdir(parents=True, exist_ok=True)
        image.save(dst, quality=95)
        return image.size


def reset_output_dir(output_dir: Path, overwrite: bool) -> None:
    """Create a clean output directory only when overwrite is explicitly requested."""
    if output_dir.exists() and overwrite:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def build_piid(
    raw_root: Path,
    output_root: Path,
    exclusion_csv: Path,
    image_size: int,
) -> pd.DataFrame:
    """Build PIID analytic stage folders and return a manifest DataFrame."""
    exclusions = read_exclusions(exclusion_csv, ("stage", "filename"))
    records: list[dict[str, object]] = []
    piid_output = output_root / "PIID"

    for stage, folder_name in PIID_STAGE_FOLDERS.items():
        src_dir = raw_root / folder_name
        dst_dir = piid_output / stage
        images = iter_images(src_dir)

        for src in tqdm(images, desc=f"PIID stage {stage}"):
            excluded = (stage, src.name) in exclusions
            if excluded:
                records.append({
                    "dataset": "PIID",
                    "stage": int(stage),
                    "source_path": src.as_posix(),
                    "output_path": "",
                    "excluded": True,
                    "reason": "duplicate_or_near_duplicate",
                })
                continue

            dst = dst_dir / src.name
            width, height = save_square_image(src, dst, image_size)
            records.append({
                "dataset": "PIID",
                "stage": int(stage),
                "source_path": src.as_posix(),
                "output_path": dst.as_posix(),
                "excluded": False,
                "reason": "",
                "width": width,
                "height": height,
            })

    return pd.DataFrame(records)


def build_kaggle(
    raw_root: Path,
    output_root: Path,
    exclusion_csv: Path,
    image_size: int,
) -> pd.DataFrame:
    """Build Kaggle analytic stage folders and return a manifest DataFrame."""
    exclusions = read_exclusions(exclusion_csv, ("stage", "source_folder", "filename"))
    records: list[dict[str, object]] = []
    kaggle_output = output_root / "Kaggle"

    for stage, source_folder in KAGGLE_STAGE_FOLDERS.items():
        src_dir = raw_root / source_folder
        dst_dir = kaggle_output / stage
        images = iter_images(src_dir)

        for src in tqdm(images, desc=f"Kaggle stage {stage}"):
            excluded = (stage, source_folder, src.name) in exclusions
            if excluded:
                records.append({
                    "dataset": "Kaggle",
                    "stage": int(stage),
                    "source_folder": source_folder,
                    "source_path": src.as_posix(),
                    "output_path": "",
                    "excluded": True,
                    "reason": "duplicate_or_near_duplicate",
                })
                continue

            dst = dst_dir / src.name
            width, height = save_square_image(src, dst, image_size)
            records.append({
                "dataset": "Kaggle",
                "stage": int(stage),
                "source_folder": source_folder,
                "source_path": src.as_posix(),
                "output_path": dst.as_posix(),
                "excluded": False,
                "reason": "",
                "width": width,
                "height": height,
            })

    return pd.DataFrame(records)


def validate_counts(output_root: Path) -> None:
    """Validate final stage-wise counts against the published reproduction target."""
    for dataset, expected in EXPECTED_COUNTS.items():
        dataset_root = output_root / dataset
        total = 0
        for stage in ["1", "2", "3", "4"]:
            stage_dir = dataset_root / stage
            count = len(iter_images(stage_dir))
            total += count
            if count != expected[stage]:
                raise RuntimeError(
                    f"{dataset} stage {stage}: expected {expected[stage]}, got {count}"
                )
        if total != expected["total"]:
            raise RuntimeError(
                f"{dataset} total: expected {expected['total']}, got {total}"
            )
        print(f"[OK] {dataset}: {total} images")


def parse_args() -> argparse.Namespace:
    root = repo_root()
    return argparse.ArgumentParser(
        description="Prepare public PIID and Kaggle analytic datasets."
    ).parse_args()


def build_arg_parser() -> argparse.ArgumentParser:
    root = repo_root()
    parser = argparse.ArgumentParser(
        description="Prepare public PIID and Kaggle analytic datasets."
    )
    parser.add_argument(
        "--piid-raw",
        type=Path,
        default=root / "data" / "raw" / "PIID" / "original_images",
        help="Folder containing PIID stage folders 1/2/3/4.",
    )
    parser.add_argument(
        "--kaggle-raw",
        type=Path,
        default=root / "data" / "raw" / "Kaggle" / "original_images",
        help="Folder containing Kaggle Stage_I/Stage_II/Stage_III/Stage_IV folders.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=root / "data" / "processed" / "analytic_data",
        help="Output root for PIID and Kaggle analytic datasets.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=224,
        help="Square output size. Use 0 to keep square-cropped native size.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete existing output PIID/Kaggle folders before rebuilding.",
    )
    return parser


def main() -> None:
    root = repo_root()
    args = build_arg_parser().parse_args()

    output_root = args.output_root
    reset_output_dir(output_root / "PIID", args.overwrite)
    reset_output_dir(output_root / "Kaggle", args.overwrite)

    manifest_dir = root / "data" / "processed"
    manifest_dir.mkdir(parents=True, exist_ok=True)

    piid_manifest = build_piid(
        raw_root=args.piid_raw,
        output_root=output_root,
        exclusion_csv=root / "code" / "data_curation" / "piid_duplicate_exclusions.csv",
        image_size=args.image_size,
    )
    kaggle_manifest = build_kaggle(
        raw_root=args.kaggle_raw,
        output_root=output_root,
        exclusion_csv=root / "code" / "data_curation" / "kaggle_duplicate_exclusions.csv",
        image_size=args.image_size,
    )

    piid_manifest.to_csv(manifest_dir / "piid_curation_manifest.csv", index=False)
    kaggle_manifest.to_csv(manifest_dir / "kaggle_curation_manifest.csv", index=False)

    validate_counts(output_root)
    print("[DONE] Public analytic datasets prepared.")


if __name__ == "__main__":
    main()

