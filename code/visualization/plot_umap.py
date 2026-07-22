"""Generate Main Figure 3 from validated frozen ResNet-18 feature vectors.

Run this script in the separate UMAP environment after
``code/analysis/extract_resnet18_features.py``.  It consumes the exact
``features.npy``, ``metadata.csv``, and ``extraction.json`` contract enforced
by ``feature_space_statistics.load_feature_sets``; it never re-extracts image
features.

The public run uses PIID and Kaggle.  Adding HUMC produces the complete
three-dataset manuscript figure.  Requested datasets are concatenated in the
study's canonical HUMC, PIID, Kaggle order so that command-line ordering cannot
alter the stochastic UMAP input order.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import platform
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import umap


SCRIPT_DIR = Path(__file__).resolve().parent
ANALYSIS_DIR = SCRIPT_DIR.parent / "analysis"
if str(ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_DIR))

from feature_space_statistics import (  # noqa: E402
    contained_dataset_dir,
    contained_input_file,
    file_sha256,
    load_feature_sets,
    read_provenance,
    resolve_feature_root,
    resolve_output_dir,
    validate_dataset_names,
)


CANONICAL_DATASET_ORDER = ("HUMC", "PIID", "Kaggle")
DISPLAY_DATASET_ORDER = ("PIID", "HUMC", "Kaggle")
PUBLIC_DATASETS = frozenset(("PIID", "Kaggle"))
FULL_DATASETS = frozenset(CANONICAL_DATASET_ORDER)
STAGES = (1, 2, 3, 4)

N_COMPONENTS = 2
N_NEIGHBORS = 15
MIN_DIST = 0.1
METRIC = "euclidean"
RANDOM_SEED = 40

# Colours sampled from the submitted Figure 3.
PALETTE = {"PIID": "#0072B2", "HUMC": "#D73027", "Kaggle": "#FFC000"}
BACKGROUND_COLOR = "#D9D9D9"


def canonicalize_datasets(raw_datasets: list[str]) -> list[str]:
    """Validate a public or complete Figure 3 dataset request."""
    validated = validate_dataset_names(raw_datasets)
    requested = frozenset(validated)
    if requested not in (PUBLIC_DATASETS, FULL_DATASETS):
        raise ValueError(
            "Figure 3 datasets must be exactly PIID Kaggle (public subset) or "
            "PIID HUMC Kaggle (complete manuscript figure)"
        )
    return [name for name in CANONICAL_DATASET_ORDER if name in requested]


def positive_neighbors(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError("n-neighbors must be an integer") from None
    if parsed < 2:
        raise argparse.ArgumentTypeError("n-neighbors must be at least 2")
    return parsed


def bounded_min_dist(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError("min-dist must be numeric") from None
    if not math.isfinite(parsed) or not 0.0 <= parsed <= 1.0:
        raise argparse.ArgumentTypeError("min-dist must be finite and between 0 and 1")
    return parsed


def validate_umap_inputs(
    features: np.ndarray,
    metadata: pd.DataFrame,
    n_neighbors: int,
) -> None:
    if features.ndim != 2 or features.shape[1] != 512:
        raise ValueError("Figure 3 requires an (N, 512) feature array")
    if len(features) != len(metadata) or len(features) == 0:
        raise ValueError("Feature and metadata rows must be non-empty and aligned")
    if n_neighbors >= len(features):
        raise ValueError(
            f"n-neighbors ({n_neighbors}) must be smaller than the image count ({len(features)})"
        )
    if not np.isfinite(features).all():
        raise ValueError("Feature vectors contain non-finite values")
    observed_stages = set(pd.to_numeric(metadata["stage"], errors="coerce").tolist())
    if observed_stages != set(STAGES):
        raise ValueError("Combined metadata must contain exactly stages 1 through 4")


def axis_limits(coordinates: np.ndarray, margin: float = 0.05) -> tuple[float, float, float, float]:
    if coordinates.shape != (len(coordinates), N_COMPONENTS) or len(coordinates) == 0:
        raise ValueError("UMAP coordinates must have shape (N, 2)")
    if not np.isfinite(coordinates).all():
        raise ValueError("UMAP returned non-finite coordinates")
    x_min, y_min = coordinates.min(axis=0)
    x_max, y_max = coordinates.max(axis=0)
    x_span = float(x_max - x_min)
    y_span = float(y_max - y_min)
    if x_span <= 0.0 or y_span <= 0.0:
        raise ValueError("UMAP coordinates have a degenerate axis")
    return (
        float(x_min - margin * x_span),
        float(x_max + margin * x_span),
        float(y_min - margin * y_span),
        float(y_max + margin * y_span),
    )


def format_axis(axis: plt.Axes, limits: tuple[float, float, float, float]) -> None:
    axis.set_xlim(limits[0], limits[1])
    axis.set_ylim(limits[2], limits[3])
    axis.set_xlabel("UMAP 1")
    axis.set_ylabel("UMAP 2")
    axis.grid(True, color="#BFBFBF", linewidth=0.45, alpha=0.35)
    axis.set_axisbelow(True)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_linewidth(0.8)
    axis.spines["bottom"].set_linewidth(0.8)
    axis.tick_params(width=0.7, length=3.0)


def scatter_panel(
    axis: plt.Axes,
    coordinates: np.ndarray,
    metadata: pd.DataFrame,
    datasets: list[str],
    title: str,
    limits: tuple[float, float, float, float],
    stage: int | None = None,
) -> None:
    if stage is None:
        active = np.ones(len(metadata), dtype=bool)
    else:
        active = metadata["stage"].eq(stage).to_numpy()
        if not active.any():
            raise ValueError(f"No images found for Stage {stage}")
        axis.scatter(
            coordinates[~active, 0],
            coordinates[~active, 1],
            s=7,
            color=BACKGROUND_COLOR,
            alpha=0.28,
            edgecolors="none",
            rasterized=False,
            zorder=1,
        )

    for dataset in datasets:
        mask = active & metadata["dataset"].eq(dataset).to_numpy()
        if not mask.any():
            raise ValueError(f"No active images found for dataset {dataset} in {title}")
        axis.scatter(
            coordinates[mask, 0],
            coordinates[mask, 1],
            s=8,
            color=PALETTE[dataset],
            alpha=0.82,
            edgecolors="none",
            rasterized=False,
            zorder=2,
        )

    axis.set_title(title, pad=5)
    format_axis(axis, limits)


def build_figure(
    coordinates: np.ndarray,
    metadata: pd.DataFrame,
    datasets: list[str],
    output_stem: Path,
) -> dict[str, str]:
    """Render the submitted 2-by-3 Figure 3 layout."""
    limits = axis_limits(coordinates)
    display_datasets = [name for name in DISPLAY_DATASET_ORDER if name in datasets]
    with plt.rc_context(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
            "font.size": 8.5,
            "axes.titlesize": 11.0,
            "axes.labelsize": 9.0,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "svg.fonttype": "none",
            "svg.hashsalt": "PI_Classification_Figure3",
        }
    ):
        figure = plt.figure(figsize=(11.5, 5.725))
        grid = figure.add_gridspec(
            2,
            3,
            left=0.055,
            right=0.992,
            bottom=0.09,
            top=0.965,
            wspace=0.12,
            hspace=0.28,
        )
        axes = {
            "overall": figure.add_subplot(grid[0, 0]),
            "stage_1": figure.add_subplot(grid[0, 1]),
            "stage_2": figure.add_subplot(grid[0, 2]),
            "legend": figure.add_subplot(grid[1, 0]),
            "stage_3": figure.add_subplot(grid[1, 1]),
            "stage_4": figure.add_subplot(grid[1, 2]),
        }

        scatter_panel(
            axes["overall"],
            coordinates,
            metadata,
            display_datasets,
            "(a) Overall dataset",
            limits,
        )
        for panel, stage, letter in (
            ("stage_1", 1, "b"),
            ("stage_2", 2, "c"),
            ("stage_3", 3, "d"),
            ("stage_4", 4, "e"),
        ):
            scatter_panel(
                axes[panel],
                coordinates,
                metadata,
                display_datasets,
                f"({letter}) Stage {stage}",
                limits,
                stage=stage,
            )

        legend_axis = axes["legend"]
        legend_axis.set_axis_off()
        handles = [
            Line2D(
                [],
                [],
                linestyle="none",
                marker="o",
                markersize=6.0,
                markerfacecolor=PALETTE[dataset],
                markeredgecolor="none",
                label=dataset,
            )
            for dataset in display_datasets
        ]
        legend_axis.legend(
            handles=handles,
            labels=display_datasets,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.78),
            ncol=len(display_datasets),
            columnspacing=1.7,
            handletextpad=0.5,
            borderaxespad=0.0,
            frameon=False,
            fontsize=10.5,
        )

        outputs = {
            "png": output_stem.with_suffix(".png"),
            "svg": output_stem.with_suffix(".svg"),
            "tif": output_stem.with_suffix(".tif"),
        }
        figure.savefig(outputs["png"], dpi=600, facecolor="white")
        figure.savefig(outputs["svg"], facecolor="white", metadata={"Date": None})
        figure.savefig(
            outputs["tif"],
            dpi=600,
            facecolor="white",
            pil_kwargs={"compression": "tiff_lzw"},
        )
        plt.close(figure)
    return {kind: path.name for kind, path in outputs.items()}


def package_version(distribution: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def input_provenance(feature_root: Path, datasets: list[str]) -> list[dict[str, object]]:
    """Collect only non-path provenance fields after strict input validation."""
    root = resolve_feature_root(feature_root)
    rows: list[dict[str, object]] = []
    for dataset in datasets:
        dataset_dir = contained_dataset_dir(root, dataset)
        provenance_path = contained_input_file(dataset_dir, "extraction.json", dataset)
        provenance = read_provenance(provenance_path, dataset)
        rows.append(
            {
                "dataset": dataset,
                "n_images": provenance["n_images"],
                "features_sha256": provenance["features_sha256"],
                "metadata_sha256": provenance["metadata_sha256"],
                "encoder": provenance["encoder"],
                "weights": provenance["weights"],
                "weights_url": provenance["weights_url"],
                "checkpoint_filename": provenance["checkpoint_filename"],
                "checkpoint_sha256": provenance["checkpoint_sha256"],
                "checkpoint_hash_verified": provenance["checkpoint_hash_verified"],
                "pretraining": provenance["pretraining"],
                "frozen": provenance["frozen"],
                "feature_dimension": provenance["feature_dimension"],
                "input_geometry": provenance["input_geometry"],
                "resize": provenance["resize"],
                "normalization": provenance["normalization"],
                "l2_normalized": provenance["l2_normalized"],
            }
        )
    return rows


def write_outputs(
    coordinates: np.ndarray,
    metadata: pd.DataFrame,
    feature_root: Path,
    datasets: list[str],
    output_dir: Path,
    n_neighbors: int,
    min_dist: float,
    metric: str,
) -> None:
    coordinate_array = np.asarray(coordinates, dtype=np.float32)
    coordinate_npy = output_dir / "umap_coordinates.npy"
    np.save(coordinate_npy, coordinate_array, allow_pickle=False)

    coordinate_table = metadata.copy()
    coordinate_table.insert(0, "umap_row", np.arange(len(coordinate_table), dtype=np.int64))
    coordinate_table["umap_1"] = coordinate_array[:, 0]
    coordinate_table["umap_2"] = coordinate_array[:, 1]
    coordinate_csv = output_dir / "umap_coordinates.csv"
    coordinate_table.to_csv(coordinate_csv, index=False, float_format="%.9g", lineterminator="\n")

    figure_files = build_figure(
        coordinate_array,
        metadata,
        datasets,
        output_dir / "figure_3_umap",
    )
    dataset_stage_counts = (
        metadata.groupby(["dataset", "stage"], sort=False, observed=True)
        .size()
        .rename("n")
        .reset_index()
    )
    run_metadata = {
        "schema_version": 1,
        "artifact": "Main Figure 3 UMAP",
        "manuscript_configuration": (
            datasets == list(CANONICAL_DATASET_ORDER)
            and n_neighbors == N_NEIGHBORS
            and min_dist == MIN_DIST
            and metric == METRIC
        ),
        "dataset_scope": "complete" if set(datasets) == FULL_DATASETS else "public_subset",
        "datasets_in_concatenation_order": datasets,
        "n_images": int(len(metadata)),
        "stage_counts": [
            {
                "dataset": str(row.dataset),
                "stage": int(row.stage),
                "n": int(row.n),
            }
            for row in dataset_stage_counts.itertuples(index=False)
        ],
        "parameters": {
            "n_neighbors": n_neighbors,
            "min_dist": min_dist,
            "n_components": N_COMPONENTS,
            "metric": metric,
            "random_state": RANDOM_SEED,
        },
        "feature_inputs": input_provenance(feature_root, datasets),
        "software_versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "matplotlib": matplotlib.__version__,
            "umap-learn": package_version("umap-learn"),
            "numba": package_version("numba"),
            "pynndescent": package_version("pynndescent"),
            "scikit-learn": package_version("scikit-learn"),
            "Pillow": package_version("Pillow"),
        },
        "outputs": {
            "coordinates_csv": coordinate_csv.name,
            "coordinates_csv_sha256": file_sha256(coordinate_csv),
            "coordinates_npy": coordinate_npy.name,
            "coordinates_npy_sha256": file_sha256(coordinate_npy),
            "figures": {
                kind: {
                    "filename": filename,
                    "sha256": file_sha256(output_dir / filename),
                }
                for kind, filename in figure_files.items()
            },
        },
    }
    (output_dir / "umap_run.json").write_text(
        json.dumps(run_metadata, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--feature-root",
        type=Path,
        required=True,
        help="Directory containing one validated feature subdirectory per dataset",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["PIID", "Kaggle"],
        help="PIID Kaggle for the public subset; add HUMC for the complete figure",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n-neighbors", type=positive_neighbors, default=N_NEIGHBORS)
    parser.add_argument("--min-dist", type=bounded_min_dist, default=MIN_DIST)
    parser.add_argument("--metric", choices=("euclidean", "cosine"), default=METRIC)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    datasets = canonicalize_datasets(args.datasets)
    output_dir = resolve_output_dir(args.output_dir)
    features, metadata = load_feature_sets(args.feature_root, datasets)
    validate_umap_inputs(features, metadata, args.n_neighbors)
    output_dir.mkdir(parents=True, exist_ok=True)

    reducer = umap.UMAP(
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
        n_components=N_COMPONENTS,
        metric=args.metric,
        random_state=RANDOM_SEED,
    )
    coordinates = reducer.fit_transform(features)
    if coordinates.shape != (len(features), N_COMPONENTS):
        raise RuntimeError(f"Unexpected UMAP output shape: {coordinates.shape}")
    if not np.isfinite(coordinates).all():
        raise RuntimeError("UMAP returned non-finite coordinates")

    write_outputs(
        coordinates,
        metadata,
        args.feature_root,
        datasets,
        output_dir,
        args.n_neighbors,
        args.min_dist,
        args.metric,
    )
    scope = "complete manuscript" if set(datasets) == FULL_DATASETS else "public subset"
    print(f"[DONE] {scope} UMAP coordinates, provenance, and Figure 3 written to {output_dir}")


if __name__ == "__main__":
    main()
