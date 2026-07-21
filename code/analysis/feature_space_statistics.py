"""Calculate non-visual statistics from raw 512-D ResNet-18 feature vectors.

Outputs include silhouette coefficients, centroid distances, and the three
nearest images to each dataset-by-stage centroid. UMAP coordinate generation
and plotting intentionally live in ``code/visualization/plot_umap.py``.
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score


def load_feature_sets(root: Path, datasets: list[str]) -> tuple[np.ndarray, pd.DataFrame]:
    feature_blocks = []
    metadata_blocks = []
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


def pairwise_centroid_rows(features: np.ndarray, metadata: pd.DataFrame) -> pd.DataFrame:
    rows = []
    datasets = metadata["dataset"].drop_duplicates().tolist()
    for left, right in itertools.combinations(datasets, 2):
        left_centroid = features[metadata["dataset"].eq(left).to_numpy()].mean(axis=0)
        right_centroid = features[metadata["dataset"].eq(right).to_numpy()].mean(axis=0)
        rows.append(
            {
                "grouping": "dataset_all_stages",
                "stage": "all",
                "group_1": left,
                "group_2": right,
                "centroid_distance": float(np.linalg.norm(left_centroid - right_centroid)),
            }
        )
        for stage in range(1, 5):
            left_mask = metadata["dataset"].eq(left) & metadata["stage"].eq(stage)
            right_mask = metadata["dataset"].eq(right) & metadata["stage"].eq(stage)
            if not left_mask.any() or not right_mask.any():
                continue
            left_centroid = features[left_mask.to_numpy()].mean(axis=0)
            right_centroid = features[right_mask.to_numpy()].mean(axis=0)
            rows.append(
                {
                    "grouping": "dataset_within_stage",
                    "stage": stage,
                    "group_1": left,
                    "group_2": right,
                    "centroid_distance": float(np.linalg.norm(left_centroid - right_centroid)),
                }
            )
    for left, right in itertools.combinations(range(1, 5), 2):
        left_centroid = features[metadata["stage"].eq(left).to_numpy()].mean(axis=0)
        right_centroid = features[metadata["stage"].eq(right).to_numpy()].mean(axis=0)
        rows.append(
            {
                "grouping": "stage_all_datasets",
                "stage": f"{left}_vs_{right}",
                "group_1": f"Stage {left}",
                "group_2": f"Stage {right}",
                "centroid_distance": float(np.linalg.norm(left_centroid - right_centroid)),
            }
        )
    return pd.DataFrame(rows)


def representative_rows(features: np.ndarray, metadata: pd.DataFrame, n: int = 3) -> pd.DataFrame:
    rows = []
    for (dataset, stage), indices in metadata.groupby(["dataset", "stage"]).groups.items():
        indices = np.asarray(list(indices), dtype=int)
        group_features = features[indices]
        centroid = group_features.mean(axis=0)
        distances = np.linalg.norm(group_features - centroid, axis=1)
        for rank, local_index in enumerate(np.argsort(distances)[:n], start=1):
            row = metadata.iloc[indices[local_index]]
            rows.append(
                {
                    "dataset": dataset,
                    "stage": int(stage),
                    "rank": rank,
                    "image_id": row["image_id"],
                    "image_path": row["image_path"],
                    "distance_to_centroid": float(distances[local_index]),
                }
            )
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--datasets", nargs="+", default=["PIID", "HUMC", "Kaggle"])
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    features, metadata = load_feature_sets(args.feature_root, args.datasets)

    silhouette = pd.DataFrame(
        [
            {
                "label_type": "dataset_source",
                "silhouette_coefficient": silhouette_score(features, metadata["dataset"], metric="euclidean"),
            },
            {
                "label_type": "clinical_stage",
                "silhouette_coefficient": silhouette_score(features, metadata["stage"], metric="euclidean"),
            },
        ]
    )
    silhouette.to_csv(args.output_dir / "silhouette_coefficients.csv", index=False)
    pairwise_centroid_rows(features, metadata).to_csv(
        args.output_dir / "centroid_distances.csv", index=False
    )
    representative_rows(features, metadata).to_csv(
        args.output_dir / "centroid_representatives.csv", index=False
    )

    print(f"[DONE] Feature-space statistics written to {args.output_dir}")


if __name__ == "__main__":
    main()
