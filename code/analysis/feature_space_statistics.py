"""Calculate statistics from the study's frozen 512-D ResNet-18 features.

The script validates that every requested dataset was extracted with the same
public timm ResNet-18 A1 weight and preprocessing configuration before combining
feature blocks. Outputs include overall and image-level silhouette coefficients,
mean-centroid distances, the manuscript's mean within-group pairwise distance,
sample-to-centroid dispersion, HD95, and the three centroid-nearest images for
each dataset-by-stage group. UMAP remains in ``code/visualization/plot_umap.py``.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist, pdist
from sklearn.metrics import silhouette_samples


STAGES = (1, 2, 3, 4)
FEATURE_DIMENSION = 512
N_REPRESENTATIVES = 3
HD_PERCENTILE = 95.0
EXPECTED_WEIGHT_SHA256 = "D63EAFA07A6E32A39D328E364F8C9F89D671444ECC7F02AA0F7EB8882AF3DD29"
EXPECTED_WEIGHT_URL = "https://github.com/huggingface/pytorch-image-models/releases/download/v0.1-rsb-weights/resnet18_a1_0-d63eafa0.pth"
SAFE_DATASET_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
REQUIRED_METADATA_COLUMNS = ("dataset", "stage", "image_id", "image_path")
REQUIRED_PROVENANCE_FIELDS = (
    "encoder",
    "weights",
    "weights_url",
    "checkpoint_filename",
    "frozen",
    "pretraining",
    "checkpoint_sha256",
    "checkpoint_hash_verified",
    "features_sha256",
    "metadata_sha256",
    "feature_dimension",
    "input_geometry",
    "resize",
    "normalization",
    "l2_normalized",
    "random_seed",
    "device",
    "torch_version",
    "torchvision_version",
    "timm_version",
    "pillow_version",
    "numpy_version",
    "n_images",
)
SHARED_PROVENANCE_FIELDS = tuple(
    field
    for field in REQUIRED_PROVENANCE_FIELDS
    if field not in {"features_sha256", "metadata_sha256", "n_images"}
)
EXPECTED_EXTRACTION_CONFIG = {
    "encoder": "timm.create_model",
    "weights": "resnet18.a1_in1k",
    "weights_url": EXPECTED_WEIGHT_URL,
    "checkpoint_filename": "resnet18_a1_0-d63eafa0.pth",
    "frozen": True,
    "pretraining": "ImageNet-1K (A1 recipe)",
    "checkpoint_sha256": EXPECTED_WEIGHT_SHA256,
    "checkpoint_hash_verified": True,
    "feature_dimension": FEATURE_DIMENSION,
    "input_geometry": "curated native square",
    "resize": "direct 224 x 224",
    "normalization": "ImageNet mean/std",
    "l2_normalized": False,
    "random_seed": 40,
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def validate_dataset_names(datasets: list[str]) -> list[str]:
    if len(datasets) < 2:
        raise ValueError("At least two datasets are required for dataset-source silhouette analysis")
    validated: list[str] = []
    seen: set[str] = set()
    for raw_name in datasets:
        name = str(raw_name).strip()
        if not SAFE_DATASET_NAME.fullmatch(name) or name in {".", ".."}:
            raise ValueError(
                "Dataset names may contain only letters, digits, dots, underscores, "
                "and hyphens, and cannot be '.' or '..'"
            )
        folded = name.casefold()
        if folded in seen:
            raise ValueError(f"Duplicate dataset name: {name}")
        seen.add(folded)
        validated.append(name)
    return validated


def resolve_feature_root(root: Path) -> Path:
    try:
        resolved = root.expanduser().resolve(strict=True)
    except (OSError, RuntimeError):
        raise FileNotFoundError("Feature root is not an existing directory") from None
    if not resolved.is_dir():
        raise NotADirectoryError("Feature root is not an existing directory")
    return resolved


def resolve_output_dir(output: Path) -> Path:
    try:
        resolved = output.expanduser().resolve()
    except (OSError, RuntimeError):
        raise ValueError("Output directory cannot be resolved") from None
    if resolved.exists() and not resolved.is_dir():
        raise NotADirectoryError("Output path exists and is not a directory")
    return resolved


def contained_dataset_dir(root: Path, dataset: str) -> Path:
    try:
        dataset_dir = (root / dataset).resolve(strict=True)
        dataset_dir.relative_to(root)
    except (OSError, RuntimeError, ValueError):
        raise FileNotFoundError(f"Contained feature directory not found for dataset {dataset}") from None
    if not dataset_dir.is_dir():
        raise NotADirectoryError(f"Feature entry is not a directory for dataset {dataset}")
    return dataset_dir


def contained_input_file(dataset_dir: Path, filename: str, dataset: str) -> Path:
    try:
        path = (dataset_dir / filename).resolve(strict=True)
        path.relative_to(dataset_dir)
    except (OSError, RuntimeError, ValueError):
        raise FileNotFoundError(f"Required {filename} not found for dataset {dataset}") from None
    if not path.is_file():
        raise FileNotFoundError(f"Required {filename} not found for dataset {dataset}")
    return path


def read_provenance(path: Path, dataset: str) -> dict[str, object]:
    try:
        provenance = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        raise ValueError(f"Invalid extraction.json for dataset {dataset}") from None
    if not isinstance(provenance, dict):
        raise ValueError(f"Invalid extraction.json object for dataset {dataset}")
    missing = [field for field in REQUIRED_PROVENANCE_FIELDS if field not in provenance]
    if missing:
        raise ValueError(
            f"Missing extraction provenance fields for dataset {dataset}: {', '.join(missing)}"
        )
    for field, expected in EXPECTED_EXTRACTION_CONFIG.items():
        observed = provenance[field]
        if field == "checkpoint_sha256":
            observed = str(observed).upper()
        if observed != expected:
            raise ValueError(f"Unexpected extraction provenance field for dataset {dataset}: {field}")
    feature_hash = str(provenance["features_sha256"]).upper()
    if not re.fullmatch(r"[0-9A-F]{64}", feature_hash):
        raise ValueError(f"Invalid features SHA-256 metadata for dataset {dataset}")
    metadata_hash = str(provenance["metadata_sha256"]).upper()
    if not re.fullmatch(r"[0-9A-F]{64}", metadata_hash):
        raise ValueError(f"Invalid metadata SHA-256 metadata for dataset {dataset}")
    if not isinstance(provenance["n_images"], int) or provenance["n_images"] <= 0:
        raise ValueError(f"Invalid n_images provenance for dataset {dataset}")
    provenance["checkpoint_sha256"] = str(provenance["checkpoint_sha256"]).upper()
    provenance["features_sha256"] = feature_hash
    provenance["metadata_sha256"] = metadata_hash
    return provenance


def read_feature_array(path: Path, dataset: str) -> np.ndarray:
    try:
        features = np.load(path, allow_pickle=False)
    except (OSError, ValueError):
        raise ValueError(f"Unable to load features.npy for dataset {dataset}") from None
    if features.ndim != 2 or features.shape[1] != FEATURE_DIMENSION:
        raise ValueError(
            f"Expected an (N, {FEATURE_DIMENSION}) feature array for dataset {dataset}; "
            f"received shape {features.shape}"
        )
    if len(features) == 0:
        raise ValueError(f"Feature array is empty for dataset {dataset}")
    if (
        not np.issubdtype(features.dtype, np.number)
        or np.issubdtype(features.dtype, np.complexfloating)
    ):
        raise ValueError(f"Feature array is not numeric for dataset {dataset}")
    features = features.astype(np.float32, copy=False)
    if not np.isfinite(features).all():
        raise ValueError(f"Feature array contains non-finite values for dataset {dataset}")
    return features


def read_metadata(path: Path, dataset: str, n_features: int) -> pd.DataFrame:
    try:
        metadata = pd.read_csv(
            path,
            dtype={"dataset": "string", "image_id": "string", "image_path": "string"},
        )
    except (OSError, UnicodeError, pd.errors.ParserError, ValueError):
        raise ValueError(f"Unable to load metadata.csv for dataset {dataset}") from None
    missing = [column for column in REQUIRED_METADATA_COLUMNS if column not in metadata.columns]
    if missing:
        raise ValueError(f"Missing metadata columns for dataset {dataset}: {', '.join(missing)}")
    if len(metadata) != n_features:
        raise ValueError(f"Feature/metadata length mismatch for dataset {dataset}")
    if metadata["dataset"].isna().any() or not metadata["dataset"].eq(dataset).all():
        raise ValueError(f"Metadata dataset labels do not match requested dataset {dataset}")

    numeric_stage = pd.to_numeric(metadata["stage"], errors="coerce").to_numpy(dtype=float)
    if (
        not np.isfinite(numeric_stage).all()
        or not np.equal(numeric_stage, np.floor(numeric_stage)).all()
    ):
        raise ValueError(f"Metadata stages must be finite integers for dataset {dataset}")
    metadata["stage"] = numeric_stage.astype(np.int64)
    if set(metadata["stage"].unique().tolist()) != set(STAGES):
        raise ValueError(f"Metadata must contain exactly stages 1 through 4 for dataset {dataset}")

    for column in ("image_id", "image_path"):
        if metadata[column].isna().any():
            raise ValueError(f"Metadata {column} contains missing values for dataset {dataset}")
        metadata[column] = metadata[column].astype(str).str.strip()
        if metadata[column].eq("").any():
            raise ValueError(f"Metadata {column} contains blank values for dataset {dataset}")
        duplicate_n = int(metadata[column].duplicated(keep=False).sum())
        if duplicate_n:
            raise ValueError(
                f"Metadata {column} contains {duplicate_n} non-unique rows for dataset {dataset}"
            )

    duplicate_rows = int(metadata.duplicated(list(REQUIRED_METADATA_COLUMNS), keep=False).sum())
    if duplicate_rows:
        raise ValueError(f"Metadata contains {duplicate_rows} duplicate rows for dataset {dataset}")
    stage_counts = metadata.groupby("stage", observed=True).size()
    if any(int(stage_counts.get(stage, 0)) < N_REPRESENTATIVES for stage in STAGES):
        raise ValueError(
            f"Every stage requires at least {N_REPRESENTATIVES} images for dataset {dataset}"
        )
    return metadata.loc[:, list(metadata.columns)].reset_index(drop=True)


def load_feature_sets(root: Path, datasets: list[str]) -> tuple[np.ndarray, pd.DataFrame]:
    feature_root = resolve_feature_root(root)
    dataset_names = validate_dataset_names(datasets)
    feature_blocks: list[np.ndarray] = []
    metadata_blocks: list[pd.DataFrame] = []
    reference_provenance: dict[str, object] | None = None

    for dataset in dataset_names:
        dataset_dir = contained_dataset_dir(feature_root, dataset)
        feature_path = contained_input_file(dataset_dir, "features.npy", dataset)
        metadata_path = contained_input_file(dataset_dir, "metadata.csv", dataset)
        provenance_path = contained_input_file(dataset_dir, "extraction.json", dataset)
        provenance = read_provenance(provenance_path, dataset)
        try:
            observed_hash = file_sha256(feature_path)
        except OSError:
            raise ValueError(f"Unable to hash features.npy for dataset {dataset}") from None
        expected_hash = str(provenance["features_sha256"]).upper()
        if observed_hash != expected_hash:
            raise ValueError(f"features.npy SHA-256 mismatch for dataset {dataset}")
        if file_sha256(metadata_path) != str(provenance["metadata_sha256"]).upper():
            raise ValueError(f"metadata.csv SHA-256 mismatch for dataset {dataset}")

        features = read_feature_array(feature_path, dataset)
        metadata = read_metadata(metadata_path, dataset, len(features))
        if provenance["n_images"] != len(features):
            raise ValueError(f"n_images provenance mismatch for dataset {dataset}")

        shared = {field: provenance[field] for field in SHARED_PROVENANCE_FIELDS}
        if reference_provenance is None:
            reference_provenance = shared
        else:
            mismatched = [
                field for field in SHARED_PROVENANCE_FIELDS
                if shared[field] != reference_provenance[field]
            ]
            if mismatched:
                raise ValueError(
                    f"Extraction provenance mismatch for dataset {dataset}: {', '.join(mismatched)}"
                )
        feature_blocks.append(features)
        metadata_blocks.append(metadata)

    features = np.vstack(feature_blocks).astype(np.float32, copy=False)
    metadata = pd.concat(metadata_blocks, ignore_index=True)
    if len(features) != len(metadata):
        raise RuntimeError("Combined feature and metadata rows are not aligned")
    return features, metadata


def validate_aligned_inputs(features: np.ndarray, metadata: pd.DataFrame) -> None:
    if features.ndim != 2 or features.shape[1] != FEATURE_DIMENSION:
        raise ValueError(f"Expected aligned features with shape (N, {FEATURE_DIMENSION})")
    if len(features) != len(metadata) or len(features) == 0:
        raise ValueError("Feature and metadata rows must be non-empty and aligned")
    missing = [column for column in REQUIRED_METADATA_COLUMNS if column not in metadata.columns]
    if missing:
        raise ValueError(f"Missing required metadata columns: {', '.join(missing)}")
    if not np.isfinite(features).all():
        raise ValueError("Features contain non-finite values")


def ordered_datasets(metadata: pd.DataFrame) -> list[str]:
    return metadata["dataset"].drop_duplicates().astype(str).tolist()


def pairwise_centroid_rows(features: np.ndarray, metadata: pd.DataFrame) -> pd.DataFrame:
    validate_aligned_inputs(features, metadata)
    rows: list[dict[str, object]] = []
    datasets = ordered_datasets(metadata)
    for left, right in itertools.combinations(datasets, 2):
        left_features = features[metadata["dataset"].eq(left).to_numpy()]
        right_features = features[metadata["dataset"].eq(right).to_numpy()]
        rows.append(
            {
                "grouping": "dataset_all_stages",
                "stage": "all",
                "group_1": left,
                "group_2": right,
                "centroid_distance": float(
                    np.linalg.norm(left_features.mean(axis=0) - right_features.mean(axis=0))
                ),
            }
        )
        for stage in STAGES:
            left_mask = metadata["dataset"].eq(left) & metadata["stage"].eq(stage)
            right_mask = metadata["dataset"].eq(right) & metadata["stage"].eq(stage)
            if not left_mask.any() or not right_mask.any():
                raise ValueError("Every dataset must contain all four stages")
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
    for left, right in itertools.combinations(STAGES, 2):
        left_features = features[metadata["stage"].eq(left).to_numpy()]
        right_features = features[metadata["stage"].eq(right).to_numpy()]
        if len(left_features) == 0 or len(right_features) == 0:
            raise ValueError("Combined metadata must contain all four stages")
        rows.append(
            {
                "grouping": "stage_all_datasets",
                "stage": f"{left}_vs_{right}",
                "group_1": f"Stage {left}",
                "group_2": f"Stage {right}",
                "centroid_distance": float(
                    np.linalg.norm(left_features.mean(axis=0) - right_features.mean(axis=0))
                ),
            }
        )
    return pd.DataFrame(
        rows,
        columns=("grouping", "stage", "group_1", "group_2", "centroid_distance"),
    )


def analysis_groups(
    features: np.ndarray,
    metadata: pd.DataFrame,
) -> list[tuple[str, str, np.ndarray]]:
    groups: list[tuple[str, str, np.ndarray]] = []
    for dataset in ordered_datasets(metadata):
        mask = metadata["dataset"].eq(dataset).to_numpy()
        groups.append(("dataset_all_stages", dataset, features[mask]))
    for stage in STAGES:
        mask = metadata["stage"].eq(stage).to_numpy()
        if not mask.any():
            raise ValueError("Combined metadata must contain all four stages")
        groups.append(("stage_all_datasets", f"Stage {stage}", features[mask]))
    return groups


def manuscript_intra_cluster_rows(
    features: np.ndarray,
    metadata: pd.DataFrame,
) -> pd.DataFrame:
    """Return the Table 3 mean of all unique within-group sample-pair distances."""
    validate_aligned_inputs(features, metadata)
    rows: list[dict[str, object]] = []
    for grouping, group, group_features in analysis_groups(features, metadata):
        n = len(group_features)
        if n < 2:
            raise ValueError("Intra-cluster pairwise distance requires at least two samples")
        rows.append(
            {
                "grouping": grouping,
                "group": group,
                "n": n,
                "pair_count": n * (n - 1) // 2,
                "manuscript_intra_cluster_pairwise_mean": float(
                    np.mean(pdist(group_features, metric="euclidean"))
                ),
            }
        )
    return pd.DataFrame(
        rows,
        columns=(
            "grouping",
            "group",
            "n",
            "pair_count",
            "manuscript_intra_cluster_pairwise_mean",
        ),
    )


def centroid_dispersion_summary(
    grouping: str,
    group: str,
    group_features: np.ndarray,
) -> dict[str, object]:
    """Summarize sample-to-own-mean-centroid distances (not Table 3 pairwise distance)."""
    centroid = group_features.mean(axis=0)
    distances = np.linalg.norm(group_features - centroid, axis=1)
    q25, median, q75 = np.percentile(distances, [25, 50, 75])
    return {
        "grouping": grouping,
        "group": group,
        "n": int(len(distances)),
        "mean_distance_to_centroid": float(np.mean(distances)),
        "std_distance_to_centroid": float(np.std(distances, ddof=0)),
        "median_distance_to_centroid": float(median),
        "q25_distance_to_centroid": float(q25),
        "q75_distance_to_centroid": float(q75),
        "min_distance_to_centroid": float(np.min(distances)),
        "max_distance_to_centroid": float(np.max(distances)),
    }


def centroid_dispersion_rows(features: np.ndarray, metadata: pd.DataFrame) -> pd.DataFrame:
    validate_aligned_inputs(features, metadata)
    rows: list[dict[str, object]] = []
    for grouping, group, group_features in analysis_groups(features, metadata):
        rows.append(centroid_dispersion_summary(grouping, group, group_features))
    return pd.DataFrame(
        rows,
        columns=(
            "grouping",
            "group",
            "n",
            "mean_distance_to_centroid",
            "std_distance_to_centroid",
            "median_distance_to_centroid",
            "q25_distance_to_centroid",
            "q75_distance_to_centroid",
            "min_distance_to_centroid",
            "max_distance_to_centroid",
        ),
    )


def directional_hd95(left: np.ndarray, right: np.ndarray) -> tuple[float, float, float]:
    distances = cdist(left, right, metric="euclidean")
    left_to_right = float(np.percentile(np.min(distances, axis=1), HD_PERCENTILE))
    right_to_left = float(np.percentile(np.min(distances, axis=0), HD_PERCENTILE))
    return left_to_right, right_to_left, max(left_to_right, right_to_left)


def hausdorff95_rows(features: np.ndarray, metadata: pd.DataFrame) -> pd.DataFrame:
    validate_aligned_inputs(features, metadata)
    rows: list[dict[str, object]] = []
    datasets = ordered_datasets(metadata)
    for left, right in itertools.combinations(datasets, 2):
        left_features = features[metadata["dataset"].eq(left).to_numpy()]
        right_features = features[metadata["dataset"].eq(right).to_numpy()]
        forward, reverse, symmetric = directional_hd95(left_features, right_features)
        rows.append(
            {
                "grouping": "dataset_all_stages",
                "stage": "all",
                "group_1": left,
                "group_2": right,
                "group_1_n": len(left_features),
                "group_2_n": len(right_features),
                "directed_95_group_1_to_group_2": forward,
                "directed_95_group_2_to_group_1": reverse,
                "hausdorff95_distance": symmetric,
            }
        )
    for left, right in itertools.combinations(STAGES, 2):
        left_features = features[metadata["stage"].eq(left).to_numpy()]
        right_features = features[metadata["stage"].eq(right).to_numpy()]
        if len(left_features) == 0 or len(right_features) == 0:
            raise ValueError("Combined metadata must contain all four stages")
        forward, reverse, symmetric = directional_hd95(left_features, right_features)
        rows.append(
            {
                "grouping": "stage_all_datasets",
                "stage": f"{left}_vs_{right}",
                "group_1": f"Stage {left}",
                "group_2": f"Stage {right}",
                "group_1_n": len(left_features),
                "group_2_n": len(right_features),
                "directed_95_group_1_to_group_2": forward,
                "directed_95_group_2_to_group_1": reverse,
                "hausdorff95_distance": symmetric,
            }
        )
    return pd.DataFrame(
        rows,
        columns=(
            "grouping",
            "stage",
            "group_1",
            "group_2",
            "group_1_n",
            "group_2_n",
            "directed_95_group_1_to_group_2",
            "directed_95_group_2_to_group_1",
            "hausdorff95_distance",
        ),
    )


def representative_rows(
    features: np.ndarray,
    metadata: pd.DataFrame,
    n: int = N_REPRESENTATIVES,
) -> pd.DataFrame:
    validate_aligned_inputs(features, metadata)
    if not isinstance(n, int) or n <= 0:
        raise ValueError("The representative count must be a positive integer")
    rows: list[dict[str, object]] = []
    for dataset in ordered_datasets(metadata):
        for stage in STAGES:
            mask = (metadata["dataset"].eq(dataset) & metadata["stage"].eq(stage)).to_numpy()
            indices = np.flatnonzero(mask)
            if len(indices) < n:
                raise ValueError(f"Dataset-stage groups require at least {n} representative candidates")
            group_features = features[indices]
            distances = np.linalg.norm(group_features - group_features.mean(axis=0), axis=1)
            image_ids = metadata.iloc[indices]["image_id"].astype(str).to_numpy()
            image_paths = metadata.iloc[indices]["image_path"].astype(str).to_numpy()
            order = sorted(
                range(len(indices)),
                key=lambda local: (
                    float(distances[local]),
                    image_ids[local],
                    image_paths[local],
                ),
            )[:n]
            for rank, local_index in enumerate(order, start=1):
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
    expected_rows = len(ordered_datasets(metadata)) * len(STAGES) * n
    if len(rows) != expected_rows:
        raise RuntimeError("Representative selection did not produce the expected number of rows")
    return pd.DataFrame(
        rows,
        columns=(
            "dataset",
            "stage",
            "rank",
            "image_id",
            "image_path",
            "distance_to_centroid",
        ),
    )


def silhouette_tables(
    features: np.ndarray,
    metadata: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    validate_aligned_inputs(features, metadata)
    definitions = (
        (
            "dataset_source",
            metadata["dataset"].astype(str).to_numpy(),
            ordered_datasets(metadata),
            lambda value: str(value),
        ),
        (
            "clinical_stage",
            metadata["stage"].to_numpy(dtype=int),
            list(STAGES),
            lambda value: f"Stage {int(value)}",
        ),
    )
    coefficient_rows: list[dict[str, object]] = []
    sample_blocks: list[pd.DataFrame] = []
    summary_rows: list[dict[str, object]] = []
    for label_type, labels, group_order, display in definitions:
        n_labels = len(np.unique(labels))
        if n_labels < 2 or n_labels >= len(features):
            raise ValueError(f"Silhouette analysis requires 2 to N-1 groups for {label_type}")
        values = silhouette_samples(features, labels, metric="euclidean")
        coefficient_rows.append(
            {
                "label_type": label_type,
                "silhouette_coefficient": float(np.mean(values)),
            }
        )
        sample_blocks.append(
            pd.DataFrame(
                {
                    "sample_index": np.arange(len(metadata), dtype=int),
                    "label_type": label_type,
                    "group": [display(value) for value in labels],
                    "dataset": metadata["dataset"].to_numpy(),
                    "stage": metadata["stage"].to_numpy(dtype=int),
                    "image_id": metadata["image_id"].to_numpy(),
                    "image_path": metadata["image_path"].to_numpy(),
                    "silhouette_coefficient": values,
                }
            )
        )
        for group in group_order:
            mask = labels == group
            group_values = values[mask]
            negative_n = int(np.sum(group_values < 0))
            summary_rows.append(
                {
                    "label_type": label_type,
                    "group": display(group),
                    "n": int(len(group_values)),
                    "mean_silhouette": float(np.mean(group_values)),
                    "negative_n": negative_n,
                    "negative_percent": float(100.0 * negative_n / len(group_values)),
                }
            )
    coefficients = pd.DataFrame(
        coefficient_rows,
        columns=("label_type", "silhouette_coefficient"),
    )
    samples = pd.concat(sample_blocks, ignore_index=True).loc[
        :,
        [
            "sample_index",
            "label_type",
            "group",
            "dataset",
            "stage",
            "image_id",
            "image_path",
            "silhouette_coefficient",
        ],
    ]
    summaries = pd.DataFrame(
        summary_rows,
        columns=("label_type", "group", "n", "mean_silhouette", "negative_n", "negative_percent"),
    )
    return coefficients, samples, summaries


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--datasets", nargs="+", default=["PIID", "HUMC", "Kaggle"])
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    datasets = validate_dataset_names(args.datasets)
    output_dir = resolve_output_dir(args.output_dir)
    features, metadata = load_feature_sets(args.feature_root, datasets)

    silhouette, silhouette_sample_table, silhouette_summary = silhouette_tables(features, metadata)
    centroid_distances = pairwise_centroid_rows(features, metadata)
    intra_cluster = manuscript_intra_cluster_rows(features, metadata)
    centroid_dispersion = centroid_dispersion_rows(features, metadata)
    hausdorff95 = hausdorff95_rows(features, metadata)
    representatives = representative_rows(features, metadata, n=N_REPRESENTATIVES)

    output_dir.mkdir(parents=True, exist_ok=True)
    silhouette.to_csv(output_dir / "silhouette_coefficients.csv", index=False)
    silhouette_sample_table.to_csv(output_dir / "silhouette_samples.csv", index=False)
    silhouette_summary.to_csv(output_dir / "silhouette_group_summary.csv", index=False)
    centroid_distances.to_csv(output_dir / "centroid_distances.csv", index=False)
    intra_cluster.to_csv(output_dir / "intra_cluster_distances.csv", index=False)
    centroid_dispersion.to_csv(output_dir / "centroid_dispersion.csv", index=False)
    hausdorff95.to_csv(output_dir / "hausdorff95_distances.csv", index=False)
    representatives.to_csv(output_dir / "centroid_representatives.csv", index=False)
    print(f"[DONE] Feature-space statistics written to {output_dir}")


if __name__ == "__main__":
    main()
