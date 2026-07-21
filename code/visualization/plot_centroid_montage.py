"""Generate main Figure 4 from centroid-nearest feature-space representatives.

The input CSV is produced by ``analysis/feature_space_statistics.py`` from raw
512-D frozen ResNet-18 features. No model is retrained and no nearest-neighbor
selection is repeated here; this script only renders the declared selections.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont


DATASETS = ["PIID", "HUMC", "Kaggle"]
STAGES = [1, 2, 3, 4]
TILE_SIZE = 112
LABEL_HEIGHT = 30
MARGIN = 14


def resolve_image(path_value: str, project_root: Path) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else project_root / path


def contact_sheet(paths: list[Path]) -> Image.Image:
    canvas = Image.new("RGB", (TILE_SIZE * len(paths), TILE_SIZE), "white")
    for index, path in enumerate(paths):
        with Image.open(path) as image:
            image = image.convert("RGB")
            image.thumbnail((TILE_SIZE, TILE_SIZE))
            tile = Image.new("RGB", (TILE_SIZE, TILE_SIZE), "white")
            tile.paste(image, ((TILE_SIZE - image.width) // 2, (TILE_SIZE - image.height) // 2))
            canvas.paste(tile, (index * TILE_SIZE, 0))
    return canvas


def ordered_datasets(table: pd.DataFrame) -> list[str]:
    """Keep manuscript order, while allowing a public-only partial montage."""
    present = table["dataset"].drop_duplicates().tolist()
    preferred = [dataset for dataset in DATASETS if dataset in present]
    return preferred + [dataset for dataset in present if dataset not in preferred]


def build_figure(table: pd.DataFrame, project_root: Path, output: Path) -> None:
    font = ImageFont.load_default()
    n_representatives = int(table["rank"].max())
    datasets = ordered_datasets(table)
    if not datasets:
        raise ValueError("No representative rows were supplied")
    cell_width = TILE_SIZE * n_representatives
    width = MARGIN + 58 + len(datasets) * (cell_width + MARGIN)
    height = LABEL_HEIGHT + len(STAGES) * (TILE_SIZE + LABEL_HEIGHT + MARGIN) + MARGIN
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    for column, dataset in enumerate(datasets):
        x = MARGIN + 58 + column * (cell_width + MARGIN)
        draw.text((x, 6), dataset, fill="black", font=font)
    for row, stage in enumerate(STAGES):
        y = LABEL_HEIGHT + row * (TILE_SIZE + LABEL_HEIGHT + MARGIN)
        draw.text((MARGIN, y + TILE_SIZE // 2), f"Stage {stage}", fill="black", font=font)
        for column, dataset in enumerate(datasets):
            x = MARGIN + 58 + column * (cell_width + MARGIN)
            rows = table[(table["dataset"] == dataset) & (table["stage"] == stage)].sort_values("rank")
            if len(rows) != n_representatives:
                raise ValueError(f"Expected {n_representatives} representatives for {dataset}, stage {stage}")
            paths = [resolve_image(value, project_root) for value in rows["image_path"]]
            missing = [str(path) for path in paths if not path.is_file()]
            if missing:
                raise FileNotFoundError("Missing representative image(s): " + "; ".join(missing))
            canvas.paste(contact_sheet(paths), (x, y))
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--representatives", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    table = pd.read_csv(args.representatives)
    build_figure(table, args.project_root.resolve(), args.output)
    print(f"[DONE] Figure 4 centroid montage written to {args.output}")


if __name__ == "__main__":
    main()
