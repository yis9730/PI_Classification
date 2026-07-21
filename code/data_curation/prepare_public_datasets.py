"""Build the public PIID/Kaggle analytic datasets.

This script does three things:

1. Reads the public raw PIID and Kaggle stage folders.
2. Excludes duplicate or near-duplicate files listed in the released manifests.
3. Copies each retained source image without changing its pixels, dimensions,
   encoding, or EXIF metadata.

It never deletes source images. The generated datasets are written to:

    data/piid/{1,2,3,4}
    data/kaggle/{1,2,3,4}

Expected final counts:

    PIID   : 1,081 images
    Kaggle :   141 images
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pandas as pd
from PIL import Image
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


def copy_source_image(src: Path, dst: Path) -> tuple[int, int]:
    """Validate an image, then copy the retained source file byte-for-byte."""
    with Image.open(src) as image:
        width, height = image.size
        image.verify()
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return width, height


def paths_overlap(first: Path, second: Path) -> bool:
    """Return True when either resolved path contains the other."""
    first = first.expanduser().resolve()
    second = second.expanduser().resolve()
    return first == second or first in second.parents or second in first.parents


def validate_source_output_separation(
    piid_source: Path,
    kaggle_source: Path,
    output_root: Path,
) -> None:
    """Prevent output cleanup or writes from touching either source tree."""
    sources = {"PIID": piid_source, "Kaggle": kaggle_source}
    outputs = {
        "PIID output": output_root / "piid",
        "Kaggle output": output_root / "kaggle",
    }
    for source_label, source in sources.items():
        for output_label, output in outputs.items():
            if paths_overlap(source, output):
                raise ValueError(
                    f"Unsafe overlapping paths: {source_label} source {source.resolve()} "
                    f"and {output_label} {output.resolve()}"
                )


def reset_output_dir(output_dir: Path, overwrite: bool) -> None:
    """Create a clean output directory only when overwrite is explicitly requested."""
    if output_dir.exists():
        if not output_dir.is_dir():
            raise NotADirectoryError(f"Output path is not a directory: {output_dir}")
        if overwrite:
            shutil.rmtree(output_dir)
        elif any(output_dir.iterdir()):
            raise FileExistsError(
                f"Output directory is not empty: {output_dir}. "
                "Choose another output root or rerun with --overwrite."
            )
    output_dir.mkdir(parents=True, exist_ok=True)


def build_piid(
    source_root: Path,
    output_root: Path,
    exclusion_csv: Path,
) -> pd.DataFrame:
    """Build PIID analytic stage folders and return a manifest DataFrame."""
    exclusions = read_exclusions(exclusion_csv, ("stage", "filename"))
    matched_exclusions: set[tuple[str, ...]] = set()
    records: list[dict[str, object]] = []
    piid_output = output_root / "piid"

    for stage, folder_name in PIID_STAGE_FOLDERS.items():
        src_dir = source_root / folder_name
        dst_dir = piid_output / stage
        images = iter_images(src_dir)
        expected_raw = EXPECTED_COUNTS["PIID"][stage] + sum(
            key[0] == stage for key in exclusions
        )
        if len(images) != expected_raw:
            raise RuntimeError(
                f"PIID stage {stage}: expected {expected_raw} source images, got {len(images)}"
            )

        for src in tqdm(images, desc=f"PIID stage {stage}"):
            key = (stage, src.name)
            excluded = key in exclusions
            if excluded:
                matched_exclusions.add(key)
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
            width, height = copy_source_image(src, dst)
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

    unmatched = exclusions - matched_exclusions
    if unmatched:
        raise RuntimeError(f"PIID exclusion entries not found in source folders: {sorted(unmatched)}")
    return pd.DataFrame(records)


def build_kaggle(
    source_root: Path,
    output_root: Path,
    exclusion_csv: Path,
) -> pd.DataFrame:
    """Build Kaggle analytic stage folders and return a manifest DataFrame."""
    exclusions = read_exclusions(exclusion_csv, ("stage", "source_folder", "filename"))
    matched_exclusions: set[tuple[str, ...]] = set()
    records: list[dict[str, object]] = []
    kaggle_output = output_root / "kaggle"

    for stage, source_folder in KAGGLE_STAGE_FOLDERS.items():
        src_dir = source_root / source_folder
        dst_dir = kaggle_output / stage
        images = iter_images(src_dir)
        expected_raw = EXPECTED_COUNTS["Kaggle"][stage] + sum(
            key[0] == stage for key in exclusions
        )
        if len(images) != expected_raw:
            raise RuntimeError(
                f"Kaggle stage {stage}: expected {expected_raw} source images, got {len(images)}"
            )

        for src in tqdm(images, desc=f"Kaggle stage {stage}"):
            key = (stage, source_folder, src.name)
            excluded = key in exclusions
            if excluded:
                matched_exclusions.add(key)
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
            width, height = copy_source_image(src, dst)
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

    unmatched = exclusions - matched_exclusions
    if unmatched:
        raise RuntimeError(f"Kaggle exclusion entries not found in source folders: {sorted(unmatched)}")
    return pd.DataFrame(records)


def validate_counts(output_root: Path) -> None:
    """Validate final stage-wise counts against the published reproduction target."""
    for dataset, expected in EXPECTED_COUNTS.items():
        dataset_root = output_root / dataset.lower()
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


def build_arg_parser() -> argparse.ArgumentParser:
    root = repo_root()
    parser = argparse.ArgumentParser(
        description="Prepare public PIID and Kaggle analytic datasets."
    )
    parser.add_argument(
        "--piid-source",
        type=Path,
        required=True,
        help="Folder containing PIID stage folders 1/2/3/4.",
    )
    parser.add_argument(
        "--kaggle-source",
        type=Path,
        required=True,
        help="Folder containing Kaggle Stage_I/Stage_II/Stage_III/Stage_IV folders.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=root / "data",
        help="Output root for PIID and Kaggle analytic datasets.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete existing output piid/kaggle folders before rebuilding.",
    )
    return parser


def main() -> None:
    root = repo_root()
    args = build_arg_parser().parse_args()

    output_root = args.output_root
    validate_source_output_separation(args.piid_source, args.kaggle_source, output_root)
    reset_output_dir(output_root / "piid", args.overwrite)
    reset_output_dir(output_root / "kaggle", args.overwrite)

    manifest_dir = root / "data" / "results" / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)

    piid_manifest = build_piid(
        source_root=args.piid_source,
        output_root=output_root,
        exclusion_csv=root / "code" / "data_curation" / "piid_duplicate_exclusions.csv",
    )
    kaggle_manifest = build_kaggle(
        source_root=args.kaggle_source,
        output_root=output_root,
        exclusion_csv=root / "code" / "data_curation" / "kaggle_duplicate_exclusions.csv",
    )

    piid_manifest.to_csv(manifest_dir / "piid_curation_manifest.csv", index=False)
    kaggle_manifest.to_csv(manifest_dir / "kaggle_curation_manifest.csv", index=False)

    validate_counts(output_root)
    print("[DONE] Public analytic datasets prepared.")


if __name__ == "__main__":
    main()
