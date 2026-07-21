"""Generate main Figure 3 from frozen ResNet-18 feature vectors.

Use the separate UMAP environment after
``code/analysis/extract_resnet18_features.py`` has written feature arrays and
metadata CSVs. This script reads those fixed inputs and writes Figure 3 plus
the underlying UMAP coordinates; it never re-extracts image features.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import umap


RANDOM_SEED = 40
PALETTE = {"PIID": "#2878B5", "HUMC": "#C83D3D", "Kaggle": "#3A9D5D"}


def load_feature_sets(root: Path, datasets: list[str]) -> tuple[np.ndarray, pd.DataFrame]:
    feature_blocks: list[np.ndarray] = []
    metadata_blocks: list[pd.DataFrame] = []
    for dataset in datasets:
        dataset_dir = root / dataset
        features = np.load(dataset_dir / "features.npy").astype(np.float32)
        metadata = pd.read_csv(dataset_dir / "metadata.csv")
        if len(features) != len(metadata):
            raise ValueError(f"Feature/metadata length mismatch for {dataset}")
        metadata["dataset"] = dataset
        feature_blocks.append(features)
        metadata_blocks.append(metadata)
    return np.vstack(feature_blocks), pd.concat(metadata_blocks, ignore_index=True)


def scatter_by_dataset(axis: plt.Axes, coordinates: np.ndarray, metadata: pd.DataFrame, stage: int | None) -> None:
    if stage is None:
        active = np.ones(len(metadata), dtype=bool)
        title = "All images"
    else:
        active = metadata["stage"].eq(stage).to_numpy()
        axis.scatter(coordinates[~active, 0], coordinates[~active, 1], s=4, c="#D9D9D9", alpha=0.35)
        title = f"Stage {stage}"
    for dataset in metadata["dataset"].drop_duplicates():
        mask = active & metadata["dataset"].eq(dataset).to_numpy()
        axis.scatter(
            coordinates[mask, 0],
            coordinates[mask, 1],
            s=7,
            alpha=0.7,
            label=dataset,
            color=PALETTE.get(dataset),
        )
    axis.set_title(title)
    axis.set_xlabel("UMAP 1")
    axis.set_ylabel("UMAP 2")
    axis.set_xticks([])
    axis.set_yticks([])


def build_figure(coordinates: np.ndarray, metadata: pd.DataFrame, output: Path) -> None:
    figure, axes = plt.subplots(1, 5, figsize=(19, 4.1), constrained_layout=True)
    scatter_by_dataset(axes[0], coordinates, metadata, stage=None)
    for stage, axis in zip(range(1, 5), axes[1:]):
        scatter_by_dataset(axis, coordinates, metadata, stage=stage)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=len(labels), frameon=False)
    figure.savefig(output, dpi=300, bbox_inches="tight")
    figure.savefig(output.with_suffix(".svg"), bbox_inches="tight")
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--datasets", nargs="+", default=["PIID", "HUMC", "Kaggle"])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--metric", choices=["euclidean", "cosine"], default="euclidean")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    features, metadata = load_feature_sets(args.feature_root, args.datasets)
    coordinates = umap.UMAP(n_components=2, metric=args.metric, random_state=RANDOM_SEED).fit_transform(features)
    table = metadata.copy()
    table["umap_1"] = coordinates[:, 0]
    table["umap_2"] = coordinates[:, 1]
    table.to_csv(args.output_dir / "umap_coordinates.csv", index=False)
    build_figure(coordinates, metadata, args.output_dir / "figure_3_umap.png")
    print(f"[DONE] UMAP coordinates and Figure 3 written to {args.output_dir}")


if __name__ == "__main__":
    main()
