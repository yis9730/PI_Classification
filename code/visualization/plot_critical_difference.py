"""Plot critical-difference diagrams from fold-wise rank-analysis outputs.

The displayed omnibus p-value and the decision to show Nemenyi cliques both
use the Friedman chi-square result reported in the submitted figure.

For PIID-trained models, this script also combines the three average-rank
panels and the corresponding bootstrap confidence-interval panels into the
submitted Main Figure 1 layout.  The composite is generated only from a
complete set of the manuscript analysis tables.

The archived predictions reproduce the manuscript figure's two-decimal
interval labels.  In particular, the Kaggle ResNet-50 lower endpoint is
0.452827... and is therefore displayed as 0.45.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CODE_ROOT = Path(__file__).resolve().parents[1]
CORE_DIR = CODE_ROOT / "core"
if str(CORE_DIR) not in sys.path:
    sys.path.insert(0, str(CORE_DIR))

from path_config import FIGURE_ROOT, TABLE_ROOT  # noqa: E402


N_FOLDS = 5
N_ARCHITECTURES = 6
N_BOOTSTRAP = 1000
RANDOM_SEED = 40
PIID_DATASETS = ("PIID_Test", "HUMC", "Kaggle")
HUMC_DATASETS = ("HUMC_Test", "PIID", "Kaggle")
TRAINING_LABELS = {"piid": "PIID-trained", "humc": "HUMC-trained"}
TRAINING_DATASETS = {"piid": PIID_DATASETS, "humc": HUMC_DATASETS}
EXPECTED_IMAGE_COUNTS = {"PIID_Test": 163, "HUMC": 1844, "Kaggle": 141}
DATASET_TITLES = {
    "PIID_Test": "PIID internal",
    "HUMC": "HUMC external",
    "Kaggle": "Kaggle external",
}
MODEL_NAMES = (
    "ResNet-50",
    "DenseNet-121",
    "EfficientNetV2-S",
    "ViT-B/16",
    "Swin-T",
    "ConvNeXt-S",
)
MODEL_KEYS = {
    "resnet50": "ResNet-50",
    "densenet121": "DenseNet-121",
    "efficientnet_v2_s": "EfficientNetV2-S",
    "vit_base_patch16_224": "ViT-B/16",
    "swin_tiny_patch4_window7_224": "Swin-T",
    "convnext_small": "ConvNeXt-S",
}

SUMMARY_COLUMNS = {
    "training",
    "evaluation_dataset",
    "n_folds",
    "n_architectures",
    "p_friedman_chi2",
    "omnibus_significant_friedman",
    "nemenyi_critical_difference",
}
RANK_COLUMNS = {"model", "mean_macro_f1", "average_rank"}
BOOTSTRAP_COLUMNS = {
    "training",
    "dataset",
    "model",
    "model_display",
    "macro_f1_mean",
    "ci_lower",
    "ci_upper",
    "n_images",
    "n_folds",
    "n_bootstrap",
    "bootstrap_unit",
    "fold_prediction_combination",
    "random_seed",
    "rng_algorithm",
}


def parse_bool(value) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    raise ValueError(f"Expected a boolean value, received {value!r}")


def _require_columns(table: pd.DataFrame, required: set[str], source: Path) -> None:
    missing = required - set(table.columns)
    if missing:
        raise ValueError(f"Missing columns in {source}: {sorted(missing)}")


def _finite_numeric(table: pd.DataFrame, columns: tuple[str, ...], source: Path) -> None:
    for column in columns:
        values = pd.to_numeric(table[column], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"{source} contains a non-finite {column} value")


def _exact_values(values: pd.Series, expected: set, label: str, source: Path) -> None:
    observed = set(values.tolist())
    missing = expected - observed
    unexpected = observed - expected
    if missing or unexpected:
        raise ValueError(
            f"Incomplete {label} values in {source}; "
            f"missing={sorted(missing)}, unexpected={sorted(unexpected)}"
        )


def _load_rank_analysis_tables(
    analysis_dir: Path,
    training: str,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Load a complete three-dataset rank analysis and reject stale partial tables."""
    datasets = TRAINING_DATASETS[training]
    summary_path = analysis_dir / "friedman_summary.csv"
    summary = pd.read_csv(summary_path)
    _require_columns(summary, SUMMARY_COLUMNS, summary_path)
    if len(summary) != len(datasets) or summary["evaluation_dataset"].duplicated().any():
        raise ValueError(
            f"{summary_path} must contain exactly one row for each of the "
            f"{len(datasets)} {training.upper()}-trained evaluation datasets"
        )
    _exact_values(
        summary["evaluation_dataset"], set(datasets), "evaluation_dataset", summary_path
    )
    if set(summary["training"].astype(str)) != {TRAINING_LABELS[training]}:
        raise ValueError(f"Unexpected training label in {summary_path}")
    _finite_numeric(
        summary,
        ("n_folds", "n_architectures", "p_friedman_chi2", "nemenyi_critical_difference"),
        summary_path,
    )
    if not summary["n_folds"].eq(N_FOLDS).all():
        raise ValueError(f"{summary_path} must report exactly {N_FOLDS} folds")
    if not summary["n_architectures"].eq(N_ARCHITECTURES).all():
        raise ValueError(
            f"{summary_path} must report exactly {N_ARCHITECTURES} architectures"
        )
    if not summary["p_friedman_chi2"].between(0, 1, inclusive="both").all():
        raise ValueError(f"Invalid Friedman P value in {summary_path}")
    expected_cd = 2.850 * np.sqrt(
        N_ARCHITECTURES * (N_ARCHITECTURES + 1) / (6 * N_FOLDS)
    )
    if not np.allclose(
        summary["nemenyi_critical_difference"].to_numpy(dtype=float),
        expected_cd,
        rtol=0,
        atol=1e-6,
    ):
        raise ValueError(f"Unexpected Nemenyi critical difference in {summary_path}")
    reported_significance = summary["omnibus_significant_friedman"].map(parse_bool)
    calculated_significance = summary["p_friedman_chi2"].lt(0.05)
    if not reported_significance.equals(calculated_significance):
        raise ValueError(f"Friedman significance flag disagrees with its P value in {summary_path}")

    rank_tables: dict[str, pd.DataFrame] = {}
    for dataset in datasets:
        rank_path = analysis_dir / "average_ranks" / f"average_ranks_{dataset}.csv"
        rank_table = pd.read_csv(rank_path)
        _require_columns(rank_table, RANK_COLUMNS, rank_path)
        if len(rank_table) != N_ARCHITECTURES or rank_table["model"].duplicated().any():
            raise ValueError(
                f"{rank_path} must contain exactly one row for each of the "
                f"{N_ARCHITECTURES} architectures"
            )
        _exact_values(rank_table["model"], set(MODEL_NAMES), "model", rank_path)
        _finite_numeric(rank_table, ("mean_macro_f1", "average_rank"), rank_path)
        if not rank_table["mean_macro_f1"].between(0, 1, inclusive="both").all():
            raise ValueError(f"Macro-F1 is outside [0, 1] in {rank_path}")
        if not rank_table["average_rank"].between(
            1, N_ARCHITECTURES, inclusive="both"
        ).all():
            raise ValueError(
                f"Average rank is outside [1, {N_ARCHITECTURES}] in {rank_path}"
            )
        expected_rank_sum = N_ARCHITECTURES * (N_ARCHITECTURES + 1) / 2
        if not np.isclose(
            rank_table["average_rank"].sum(), expected_rank_sum, rtol=0, atol=1e-6
        ):
            raise ValueError(f"Average ranks do not sum to {expected_rank_sum:g} in {rank_path}")
        rank_tables[dataset] = rank_table.copy()
    return summary, rank_tables


def _load_piid_figure_tables(
    analysis_dir: Path,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], pd.DataFrame]:
    """Load and cross-check the complete set of Main Figure 1 inputs."""
    summary, rank_tables = _load_rank_analysis_tables(analysis_dir, "piid")

    bootstrap_path = (
        TABLE_ROOT
        / "statistical_tests"
        / "bootstrap_foldwise"
        / "piid_trained"
        / "bootstrap_summary.csv"
    )
    bootstrap = pd.read_csv(bootstrap_path)
    _require_columns(bootstrap, BOOTSTRAP_COLUMNS, bootstrap_path)
    expected_rows = len(PIID_DATASETS) * N_ARCHITECTURES
    if len(bootstrap) != expected_rows or bootstrap.duplicated(["dataset", "model_display"]).any():
        raise ValueError(
            f"{bootstrap_path} must contain exactly one row for each of the "
            f"{expected_rows} dataset-model combinations"
        )
    _exact_values(bootstrap["dataset"], set(PIID_DATASETS), "dataset", bootstrap_path)
    if set(bootstrap["training"].astype(str)) != {"PIID-trained"}:
        raise ValueError(f"Unexpected training label in {bootstrap_path}")
    _finite_numeric(
        bootstrap,
        (
            "macro_f1_mean",
            "ci_lower",
            "ci_upper",
            "n_images",
            "n_folds",
            "n_bootstrap",
            "random_seed",
        ),
        bootstrap_path,
    )
    if not bootstrap["n_folds"].eq(N_FOLDS).all():
        raise ValueError(f"{bootstrap_path} must report exactly {N_FOLDS} folds")
    if not bootstrap["n_bootstrap"].eq(N_BOOTSTRAP).all():
        raise ValueError(f"{bootstrap_path} must report exactly {N_BOOTSTRAP} bootstrap samples")
    if not bootstrap["random_seed"].eq(RANDOM_SEED).all():
        raise ValueError(f"{bootstrap_path} must report random seed {RANDOM_SEED}")
    if set(bootstrap["rng_algorithm"].astype(str)) != {
        "numpy.random.RandomState(MT19937)"
    }:
        raise ValueError(f"Unexpected bootstrap RNG algorithm in {bootstrap_path}")
    if set(bootstrap["bootstrap_unit"].astype(str)) != {"image"}:
        raise ValueError(f"Unexpected bootstrap unit in {bootstrap_path}")
    if set(bootstrap["fold_prediction_combination"].astype(str)) != {"none"}:
        raise ValueError(f"Fold predictions must not be combined in {bootstrap_path}")
    if bootstrap["n_images"].le(0).any():
        raise ValueError(f"Non-positive image count in {bootstrap_path}")
    if not (
        bootstrap["ci_lower"].between(0, 1, inclusive="both").all()
        and bootstrap["macro_f1_mean"].between(0, 1, inclusive="both").all()
        and bootstrap["ci_upper"].between(0, 1, inclusive="both").all()
    ):
        raise ValueError(f"Macro-F1 estimate or CI is outside [0, 1] in {bootstrap_path}")
    if not (
        bootstrap["ci_lower"].le(bootstrap["macro_f1_mean"]).all()
        and bootstrap["macro_f1_mean"].le(bootstrap["ci_upper"]).all()
    ):
        raise ValueError(f"A confidence interval does not contain its estimate in {bootstrap_path}")

    for dataset in PIID_DATASETS:
        dataset_bootstrap = bootstrap.loc[bootstrap["dataset"].eq(dataset)].copy()
        if not dataset_bootstrap["n_images"].eq(EXPECTED_IMAGE_COUNTS[dataset]).all():
            raise ValueError(
                f"Unexpected image count for {dataset} in {bootstrap_path}; "
                f"expected {EXPECTED_IMAGE_COUNTS[dataset]}"
            )
        _exact_values(
            dataset_bootstrap["model_display"], set(MODEL_NAMES), "model_display", bootstrap_path
        )
        if not dataset_bootstrap["model"].map(MODEL_KEYS).equals(
            dataset_bootstrap["model_display"]
        ):
            raise ValueError(f"Model key/display-name mismatch for {dataset} in {bootstrap_path}")
        rank_means = rank_tables[dataset].set_index("model")["mean_macro_f1"].sort_index()
        bootstrap_means = (
            dataset_bootstrap.set_index("model_display")["macro_f1_mean"].sort_index()
        )
        if not np.allclose(
            rank_means.to_numpy(dtype=float),
            bootstrap_means.to_numpy(dtype=float),
            rtol=0,
            atol=1e-6,
        ):
            raise ValueError(
                f"Mean macro-F1 values disagree between rank and bootstrap tables for {dataset}"
            )

    return summary, rank_tables, bootstrap


def nonsignificant_intervals(ranks: np.ndarray, cd: float) -> list[tuple[int, int]]:
    intervals = []
    for left in range(len(ranks)):
        for right in range(left + 1, len(ranks)):
            if ranks[right] - ranks[left] <= cd:
                intervals.append((left, right))
    maximal = []
    for candidate in intervals:
        if not any(
            other != candidate and other[0] <= candidate[0] and other[1] >= candidate[1]
            for other in intervals
        ):
            maximal.append(candidate)
    return maximal


def save_figure_atomic(figure: plt.Figure, output: Path, **kwargs: object) -> None:
    """Replace one figure only after Matplotlib has completed the new file."""
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=output.parent,
        prefix=f".{output.stem}-",
        suffix=output.suffix,
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
    try:
        figure.savefig(temporary, **kwargs)
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)


def draw_diagram(ranks_table: pd.DataFrame, summary: pd.Series, title: str, output: Path) -> None:
    ranks_table = ranks_table.sort_values("average_rank").reset_index(drop=True)
    names = ranks_table["model"].tolist()
    ranks = ranks_table["average_rank"].to_numpy(dtype=float)
    cd = float(summary["nemenyi_critical_difference"])
    show_cliques = parse_bool(summary["omnibus_significant_friedman"])
    p_value = float(summary["p_friedman_chi2"])

    fig, axis = plt.subplots(figsize=(10, 4.8))
    axis.set_xlim(0.5, 6.5)
    axis.set_ylim(0, 1)
    axis.axis("off")
    axis.plot([1, 6], [0.74, 0.74], color="black", linewidth=1.2)
    for rank in range(1, 7):
        axis.plot([rank, rank], [0.72, 0.76], color="black", linewidth=1)
        axis.text(rank, 0.80, str(rank), ha="center", va="bottom")

    left_indices = list(range((len(names) + 1) // 2))
    right_indices = list(range((len(names) + 1) // 2, len(names)))
    for order, index in enumerate(left_indices):
        y = 0.62 - order * 0.12
        axis.plot([ranks[index], 0.7], [0.74, y], color="black", linewidth=0.8)
        axis.text(0.65, y, f"{names[index]} ({ranks[index]:.2f})", ha="right", va="center")
    for order, index in enumerate(right_indices):
        y = 0.62 - order * 0.12
        axis.plot([ranks[index], 6.3], [0.74, y], color="black", linewidth=0.8)
        axis.text(6.35, y, f"({ranks[index]:.2f}) {names[index]}", ha="left", va="center")

    if show_cliques:
        for level, (left, right) in enumerate(nonsignificant_intervals(ranks, cd)):
            y = 0.68 - level * 0.035
            axis.plot([ranks[left], ranks[right]], [y, y], color="#b2182b", linewidth=4, solid_capstyle="round")

    p_text = "P < .001" if p_value < 0.001 else f"P = {p_value:.3f}".replace("0.", ".")
    axis.text(3.5, 0.96, title, ha="center", va="top", fontsize=13, fontweight="bold")
    axis.text(3.5, 0.89, f"Friedman {p_text}; CD = {cd:.3f}", ha="center", va="top")
    try:
        save_figure_atomic(fig, output, dpi=300, bbox_inches="tight")
    finally:
        plt.close(fig)


def _format_p_value(p_value: float) -> str:
    return "P < .001" if p_value < 0.001 else f"P = {p_value:.3f}".replace("0.", ".")


def _draw_rank_panel(
    axis: plt.Axes,
    rank_table: pd.DataFrame,
    summary: pd.Series,
    panel_letter: str,
    dataset_title: str,
) -> None:
    ordered = rank_table.sort_values("average_rank", kind="stable").reset_index(drop=True)
    ranks = ordered["average_rank"].to_numpy(dtype=float)
    cd = float(summary["nemenyi_critical_difference"])
    p_value = float(summary["p_friedman_chi2"])

    axis.set_xlim(-2.65, 6.35)
    axis.set_ylim(0, 1)
    axis.axis("off")

    rank_axis_y = 0.72
    axis.plot([1, N_ARCHITECTURES], [rank_axis_y, rank_axis_y], color="black", linewidth=0.9)
    for rank in range(1, N_ARCHITECTURES + 1):
        axis.plot(
            [rank, rank],
            [rank_axis_y - 0.018, rank_axis_y + 0.018],
            color="black",
            linewidth=0.8,
        )
        axis.text(rank, rank_axis_y + 0.035, str(rank), ha="center", va="bottom", fontsize=7)

    if parse_bool(summary["omnibus_significant_friedman"]):
        for level, (left, right) in enumerate(nonsignificant_intervals(ranks, cd)):
            y = rank_axis_y - 0.065 - level * 0.038
            axis.plot(
                [ranks[left], ranks[right]],
                [y, y],
                color="black",
                linewidth=2.0,
                solid_capstyle="round",
            )

    label_y = np.linspace(0.50, 0.03, N_ARCHITECTURES)
    for index, y in enumerate(label_y):
        axis.plot(
            [0.87, ranks[index], ranks[index]],
            [y, y, rank_axis_y - 0.015],
            color="#4d4d4d",
            linewidth=0.65,
        )
        axis.text(
            0.78,
            y,
            f"{ordered.loc[index, 'model']} ({ordered.loc[index, 'mean_macro_f1']:.2f})",
            ha="right",
            va="center",
            fontsize=7.2,
        )

    axis.text(
        0.5,
        1.12,
        f"({panel_letter}) {dataset_title}: average rank",
        transform=axis.transAxes,
        ha="center",
        va="bottom",
        fontsize=8.4,
        fontweight="bold",
    )
    axis.text(
        0.5,
        1.01,
        f"Friedman {_format_p_value(p_value)}",
        transform=axis.transAxes,
        ha="center",
        va="bottom",
        fontsize=7.5,
    )


def _draw_forest_panel(
    axis: plt.Axes,
    bootstrap_table: pd.DataFrame,
    panel_letter: str,
    dataset_title: str,
) -> None:
    ordered = bootstrap_table.sort_values(
        ["macro_f1_mean", "model_display"], ascending=[False, True], kind="stable"
    ).reset_index(drop=True)
    y = np.arange(len(ordered), dtype=float)
    means = ordered["macro_f1_mean"].to_numpy(dtype=float)
    lower = ordered["ci_lower"].to_numpy(dtype=float)
    upper = ordered["ci_upper"].to_numpy(dtype=float)
    errors = np.vstack([means - lower, upper - means])

    ci_span = float(upper.max() - lower.min())
    if ci_span <= 0:
        ci_span = 0.1
    x_left = max(0.0, float(lower.min() - 0.08 * ci_span))
    value_x = float(upper.max() + 0.06 * ci_span)
    x_right = min(1.0, float(upper.max() + 0.88 * ci_span))
    if x_right <= value_x:
        x_right = min(1.0, value_x + 0.1)

    axis.errorbar(
        means,
        y,
        xerr=errors,
        fmt="s",
        markersize=3.7,
        color="#4d4d4d",
        ecolor="#4d4d4d",
        elinewidth=1.0,
        capsize=2.2,
        capthick=0.9,
    )
    for index, row in ordered.iterrows():
        axis.text(
            value_x,
            float(index),
            f"{row['macro_f1_mean']:.2f} ({row['ci_lower']:.2f}, {row['ci_upper']:.2f})",
            ha="left",
            va="center",
            fontsize=7,
        )

    axis.set_xlim(x_left, x_right)
    axis.set_ylim(-0.65, len(ordered) - 0.35)
    axis.invert_yaxis()
    axis.set_yticks(y, ordered["model_display"], fontsize=7.2)
    axis.tick_params(axis="y", length=0)
    axis.tick_params(axis="x", labelsize=6.8)
    axis.set_xlabel("Macro-F1", fontsize=7.3, labelpad=2)
    axis.grid(axis="x", color="#bdbdbd", alpha=0.45, linewidth=0.55)
    axis.set_axisbelow(True)
    axis.spines[["top", "right", "left"]].set_visible(False)
    axis.spines["bottom"].set_linewidth(0.7)
    axis.set_title(
        f"({panel_letter}) {dataset_title}: macro-F1",
        fontsize=8.4,
        fontweight="bold",
        pad=5,
    )


def draw_piid_main_figure_1(
    summary: pd.DataFrame,
    rank_tables: dict[str, pd.DataFrame],
    bootstrap: pd.DataFrame,
    output_dir: Path,
) -> tuple[Path, Path]:
    """Draw the submitted three-column by two-row Main Figure 1 composite."""
    summary_by_dataset = summary.set_index("evaluation_dataset")
    figure, axes = plt.subplots(
        2,
        3,
        figsize=(11.5, 3.665),
        gridspec_kw={"height_ratios": [1.03, 1.0]},
    )
    figure.subplots_adjust(
        left=0.075,
        right=0.985,
        bottom=0.14,
        top=0.88,
        wspace=0.62,
        hspace=0.72,
    )

    for column, dataset in enumerate(PIID_DATASETS):
        _draw_rank_panel(
            axes[0, column],
            rank_tables[dataset],
            summary_by_dataset.loc[dataset],
            chr(ord("a") + column),
            DATASET_TITLES[dataset],
        )
        _draw_forest_panel(
            axes[1, column],
            bootstrap.loc[bootstrap["dataset"].eq(dataset)].copy(),
            chr(ord("d") + column),
            DATASET_TITLES[dataset],
        )

    png_output = output_dir / "figure_1_piid_trained.png"
    tif_output = output_dir / "figure_1_piid_trained.tif"
    try:
        save_figure_atomic(figure, png_output, dpi=600, facecolor="white")
        save_figure_atomic(
            figure,
            tif_output,
            dpi=600,
            facecolor="white",
            pil_kwargs={"compression": "tiff_lzw"},
        )
    finally:
        plt.close(figure)
    return png_output, tif_output


def run(training: str) -> None:
    analysis_dir = TABLE_ROOT / "statistical_tests" / "friedman_nemenyi" / f"{training}_trained"
    if training == "piid":
        summary, rank_tables, bootstrap = _load_piid_figure_tables(analysis_dir)
    else:
        summary, rank_tables = _load_rank_analysis_tables(analysis_dir, training)
        bootstrap = pd.DataFrame()
    output_dir = FIGURE_ROOT / "critical_difference" / f"{training}_trained"
    for _, row in summary.iterrows():
        dataset = row["evaluation_dataset"]
        ranks = rank_tables[dataset]
        draw_diagram(
            ranks,
            row,
            f"{training.upper()}-trained models on {dataset}",
            output_dir / f"cd_macro_f1_{dataset}.png",
        )
    if training == "piid":
        draw_piid_main_figure_1(summary, rank_tables, bootstrap, output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training", choices=["piid", "humc", "both"], default="both")
    args = parser.parse_args()
    for training in ["piid", "humc"] if args.training == "both" else [args.training]:
        run(training)
    print("[DONE] Critical-difference figures complete")


if __name__ == "__main__":
    main()
