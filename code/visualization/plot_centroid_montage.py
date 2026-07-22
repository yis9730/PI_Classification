"""Render Main Figure 4 from declared mean-centroid representatives.

The input CSV is produced by ``analysis/feature_space_statistics.py`` from raw
512-D frozen ResNet-18 features. This script does not repeat representative
selection. It renders the declared three images per dataset and stage using the
layout of the archived manuscript figure.

The square crop in this file is display-only. It does not modify the analytic
datasets or the 224 x 224 preprocessing used for feature extraction.
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont


DATASETS = ["PIID", "HUMC", "Kaggle"]
STAGES = [1, 2, 3, 4]
REPRESENTATIVES_PER_CELL = 3
TILE_SIZE = 160
TILE_GAP = 4
DATASET_GAP = 15
ROW_LABEL_WIDTH = 70
COLUMN_LABEL_HEIGHT = 50
OUTER_PADDING = 4
OUTPUT_DPI = 600
REQUIRED_COLUMNS = {"dataset", "stage", "rank", "image_path"}

FONT_CANDIDATES = (
    Path("/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf"),
    Path("/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"),
    Path("C:/Windows/Fonts/arialbd.ttf"),
)


def resolve_image(path_value: str, project_root: Path) -> Path:
    path = Path(path_value).expanduser()
    return path if path.is_absolute() else project_root / path


def load_bold_font(size: int, font_path: Path | None) -> ImageFont.ImageFont:
    """Load an explicit font or a portable bold sans-serif fallback."""
    if font_path is not None:
        if not font_path.is_file():
            raise FileNotFoundError(f"Figure font not found: {font_path}")
        return ImageFont.truetype(str(font_path), size)
    for candidate in FONT_CANDIDATES:
        if candidate.is_file():
            return ImageFont.truetype(str(candidate), size)
    try:
        return ImageFont.truetype("DejaVuSans-Bold.ttf", size)
    except OSError:
        return ImageFont.load_default(size=size)


def display_tile(path: Path) -> Image.Image:
    """Create the manuscript's square 160-pixel display tile."""
    with Image.open(path) as image:
        image = image.convert("RGB")
        width, height = image.size
        side = min(width, height)
        left = (width - side) // 2
        top = (height - side) // 2
        image = image.crop((left, top, left + side, top + side))
        return image.resize((TILE_SIZE, TILE_SIZE), Image.Resampling.LANCZOS)


def contact_sheet(paths: list[Path]) -> Image.Image:
    width = TILE_SIZE * len(paths) + TILE_GAP * (len(paths) - 1)
    canvas = Image.new("RGB", (width, TILE_SIZE), "white")
    for index, path in enumerate(paths):
        canvas.paste(display_tile(path), (index * (TILE_SIZE + TILE_GAP), 0))
    return canvas


def ordered_datasets(table: pd.DataFrame) -> list[str]:
    """Keep manuscript order, while allowing a public-only partial montage."""
    present = table["dataset"].drop_duplicates().tolist()
    unexpected = sorted(set(present) - set(DATASETS))
    if unexpected:
        raise ValueError(f"Unexpected manuscript dataset values: {unexpected}")
    return [dataset for dataset in DATASETS if dataset in present]


def save_canvas_atomic(canvas: Image.Image, output: Path) -> None:
    """Replace the montage only after Pillow has written a complete image."""
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=output.parent,
        prefix=f".{output.stem}-",
        suffix=output.suffix,
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
    try:
        canvas.save(temporary, dpi=(OUTPUT_DPI, OUTPUT_DPI))
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)


def validate_representatives(table: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Validate and normalize the representative-table contract."""
    missing_columns = sorted(REQUIRED_COLUMNS - set(table.columns))
    if missing_columns:
        raise ValueError(f"Representative table is missing columns: {missing_columns}")
    if table.empty:
        raise ValueError("No representative rows were supplied")

    normalized = table.copy()
    if normalized[list(REQUIRED_COLUMNS)].isna().any().any():
        raise ValueError("Representative table contains blank required values")
    normalized["dataset"] = normalized["dataset"].astype(str).str.strip()
    normalized["image_path"] = normalized["image_path"].astype(str).str.strip()
    if normalized["dataset"].eq("").any() or normalized["image_path"].eq("").any():
        raise ValueError("Representative table contains blank dataset or image path values")

    for column in ("stage", "rank"):
        numeric = pd.to_numeric(normalized[column], errors="coerce")
        if numeric.isna().any() or (numeric % 1 != 0).any():
            raise ValueError(f"Representative {column} values must be integers")
        normalized[column] = numeric.astype(int)

    unexpected_stages = sorted(set(normalized["stage"]) - set(STAGES))
    if unexpected_stages:
        raise ValueError(f"Unexpected stage values: {unexpected_stages}")
    unexpected_ranks = sorted(
        set(normalized["rank"]) - set(range(1, REPRESENTATIVES_PER_CELL + 1))
    )
    if unexpected_ranks:
        raise ValueError(f"Unexpected representative ranks: {unexpected_ranks}")
    if normalized.duplicated(["dataset", "stage", "rank"]).any():
        raise ValueError("Representative table contains duplicate dataset-stage-rank rows")

    datasets = ordered_datasets(normalized)
    expected_ranks = list(range(1, REPRESENTATIVES_PER_CELL + 1))
    for dataset in datasets:
        for stage in STAGES:
            ranks = sorted(
                normalized.loc[
                    normalized["dataset"].eq(dataset) & normalized["stage"].eq(stage),
                    "rank",
                ].tolist()
            )
            if ranks != expected_ranks:
                raise ValueError(
                    f"Expected ranks {expected_ranks} for {dataset}, stage {stage}; got {ranks}"
                )
    return normalized, datasets


def build_figure(
    table: pd.DataFrame,
    project_root: Path,
    output: Path,
    font_path: Path | None = None,
) -> None:
    table, datasets = validate_representatives(table)
    stage_font = load_bold_font(28, font_path)
    dataset_font = load_bold_font(36, font_path)

    cell_width = TILE_SIZE * REPRESENTATIVES_PER_CELL + TILE_GAP * (
        REPRESENTATIVES_PER_CELL - 1
    )
    width = (
        ROW_LABEL_WIDTH
        + len(datasets) * cell_width
        + max(0, len(datasets) - 1) * DATASET_GAP
        + OUTER_PADDING * 2
    )
    height = (
        COLUMN_LABEL_HEIGHT
        + len(STAGES) * (TILE_SIZE + TILE_GAP)
        - TILE_GAP
        + OUTER_PADDING * 2
    )
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)

    for column, dataset in enumerate(datasets):
        x = ROW_LABEL_WIDTH + column * (cell_width + DATASET_GAP) + cell_width // 2
        draw.text((x, COLUMN_LABEL_HEIGHT // 2), dataset, fill="black", font=dataset_font, anchor="mm")

    for row, stage in enumerate(STAGES):
        y = COLUMN_LABEL_HEIGHT + row * (TILE_SIZE + TILE_GAP)
        label = Image.new("RGBA", (TILE_SIZE, ROW_LABEL_WIDTH), (255, 255, 255, 0))
        label_draw = ImageDraw.Draw(label)
        label_draw.text(
            (TILE_SIZE // 2, ROW_LABEL_WIDTH // 2),
            f"Stage {stage}",
            fill="black",
            font=stage_font,
            anchor="mm",
        )
        rotated = label.rotate(90, expand=True)
        canvas.paste(rotated, (0, y), rotated)

        for column, dataset in enumerate(datasets):
            x = ROW_LABEL_WIDTH + column * (cell_width + DATASET_GAP)
            rows = table[
                table["dataset"].eq(dataset) & table["stage"].eq(stage)
            ].sort_values("rank")
            paths = [resolve_image(value, project_root) for value in rows["image_path"]]
            missing_ranks = [
                int(rank)
                for rank, path in zip(rows["rank"], paths)
                if not path.is_file()
            ]
            if missing_ranks:
                raise FileNotFoundError(
                    f"Missing representative image(s) for {dataset}, stage {stage}, "
                    f"rank(s) {missing_ranks}"
                )
            try:
                sheet = contact_sheet(paths)
            except (OSError, ValueError):
                raise OSError(
                    f"Could not render representative image(s) for {dataset}, stage {stage}"
                ) from None
            canvas.paste(sheet, (x, y))

    save_canvas_atomic(canvas, output)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--representatives", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--font",
        type=Path,
        help="Optional bold TrueType font; Liberation Sans Bold matches the archived figure.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    table = pd.read_csv(args.representatives)
    build_figure(table, args.project_root.resolve(), args.output, args.font)
    print(f"[DONE] Figure 4 mean-centroid montage written to {args.output}")


if __name__ == "__main__":
    main()
