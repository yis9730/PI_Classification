"""Plot fold-averaged confusion-matrix Sankey diagrams.

The submitted Main Figure 2 uses the element-wise mean of five fold-specific
ResNet-50 ``exp00_NoAug`` confusion matrices for each training direction. It
is a visualization of averaged counts, not a probability ensemble. Every
requested prediction file must be present and image-aligned. An optional
six-architecture mean is available for supplementary inspection only.
"""

from __future__ import annotations

import argparse
import re
import sys
import tempfile
from pathlib import Path

import matplotlib.patches as patches
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.path import Path as MplPath
from sklearn.metrics import confusion_matrix

CODE_ROOT = Path(__file__).resolve().parents[1]
CORE_DIR = CODE_ROOT / "core"
if str(CORE_DIR) not in sys.path:
    sys.path.insert(0, str(CORE_DIR))

from path_config import FIGURE_ROOT, HUMC_INFERENCE_DIR, PIID_INFERENCE_DIR  # noqa: E402

N_FOLDS = 5
LABELS = [0, 1, 2, 3]
CLASS_NAMES = ["Stage 1", "Stage 2", "Stage 3", "Stage 4"]
COLORS = ["#2980B9", "#16A085", "#8E44AD", "#C0392B"]
MODELS = [
    "resnet50",
    "densenet121",
    "efficientnet_v2_s",
    "vit_base_patch16_224",
    "swin_tiny_patch4_window7_224",
    "convnext_small",
]
CONFIGS = {
    "piid": (PIID_INFERENCE_DIR, ["PIID_Test", "HUMC", "Kaggle"]),
    "humc": (HUMC_INFERENCE_DIR, ["HUMC_Test", "PIID", "Kaggle"]),
}
SAFE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")
REQUIRED_COLUMNS = {"image_path", "true_label", "predicted_label"}
EXPECTED_STAGE_COUNTS = {
    "PIID_Test": [35, 47, 41, 40],
    "PIID": [229, 311, 273, 268],
    "HUMC_Test": [30, 104, 100, 54],
    "HUMC": [233, 709, 575, 327],
    "Kaggle": [27, 46, 41, 27],
}


def _integer_array(values: pd.Series, label: str) -> np.ndarray:
    numeric = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(numeric).all() or not np.equal(numeric, np.floor(numeric)).all():
        raise ValueError(f"{label} must contain finite integer labels")
    return numeric.astype(int)


def normalize_labels(
    true_values: pd.Series, predicted_values: pd.Series
) -> tuple[np.ndarray, np.ndarray]:
    true_raw = _integer_array(true_values, "true_label")
    predicted_raw = _integer_array(predicted_values, "predicted_label")
    observed = set(np.unique(true_raw).tolist())
    if 0 in observed and observed <= set(LABELS):
        offset = 0
    elif 4 in observed and observed <= {1, 2, 3, 4}:
        offset = 1
    else:
        raise ValueError("Ambiguous true_label encoding; expected all four study stages")
    true = true_raw - offset
    predicted = predicted_raw - offset
    if set(np.unique(true)) != set(LABELS):
        raise ValueError("Every evaluation table must contain all four true stages")
    if not set(np.unique(predicted)).issubset(set(LABELS)):
        raise ValueError("predicted_label uses a different or invalid encoding")
    return true, predicted


def load_prediction(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(f"Required prediction file is missing: {path}")
    table = pd.read_csv(path)
    missing = REQUIRED_COLUMNS - set(table.columns)
    if missing:
        raise ValueError(f"Prediction table is missing columns: {sorted(missing)}")
    if table.empty or table["image_path"].isna().any():
        raise ValueError("Prediction table is empty or contains a blank image_path")
    table["image_path"] = table["image_path"].astype(str)
    if table["image_path"].str.strip().eq("").any() or table["image_path"].duplicated().any():
        raise ValueError("Prediction table contains blank or duplicate image paths")
    table = table.sort_values("image_path", kind="stable").reset_index(drop=True)
    true, predicted = normalize_labels(table["true_label"], table["predicted_label"])
    signature = np.column_stack([table["image_path"].to_numpy(), true.astype(str)])
    return true, predicted, signature


def mean_confusion(root: Path, model: str, dataset: str) -> tuple[np.ndarray, np.ndarray]:
    prediction_dir = root / f"{model}_exp00_NoAug" / "predictions"
    matrices = []
    reference: np.ndarray | None = None
    for fold in range(1, N_FOLDS + 1):
        path = prediction_dir / f"{dataset}_fold{fold}_predictions.csv"
        true, predicted, signature = load_prediction(path)
        expected_counts = EXPECTED_STAGE_COUNTS.get(dataset)
        if expected_counts is not None:
            observed_counts = np.bincount(true, minlength=len(LABELS)).tolist()
            if observed_counts != expected_counts:
                raise ValueError(
                    f"Unexpected true-stage counts for {dataset}: {observed_counts}"
                )
        if reference is None:
            reference = signature
        elif not np.array_equal(reference, signature):
            raise ValueError(f"Fold alignment mismatch: {model}, {dataset}, fold {fold}")
        matrices.append(confusion_matrix(true, predicted, labels=LABELS).astype(float))
    if len(matrices) != N_FOLDS or reference is None:
        raise RuntimeError(f"Expected exactly {N_FOLDS} aligned confusion matrices")
    return np.mean(matrices, axis=0), reference


def mean_confusion_across_models(root: Path, dataset: str) -> np.ndarray:
    matrices = []
    reference: np.ndarray | None = None
    for model in MODELS:
        matrix, signature = mean_confusion(root, model, dataset)
        if reference is None:
            reference = signature
        elif not np.array_equal(reference, signature):
            raise ValueError(f"Architecture alignment mismatch: {model}, {dataset}")
        matrices.append(matrix)
    if len(matrices) != len(MODELS):
        raise RuntimeError(f"Expected exactly {len(MODELS)} final architectures")
    return np.mean(matrices, axis=0)


def row_preserving_round(matrix: np.ndarray) -> np.ndarray:
    displayed = np.floor(matrix).astype(int)
    fractions = matrix - displayed
    for row in range(matrix.shape[0]):
        target = int(np.rint(matrix[row].sum()))
        count = target - int(displayed[row].sum())
        if count > 0:
            order = np.argsort(-fractions[row], kind="stable")
            displayed[row, order[:count]] += 1
    return displayed


def band(axis, x0, x1, y0_bottom, y0_top, y1_bottom, y1_top, color, alpha):
    controls = [
        (x0, y0_top),
        ((x0 + x1) / 2, y0_top),
        ((x0 + x1) / 2, y1_top),
        (x1, y1_top),
        (x1, y1_bottom),
        ((x0 + x1) / 2, y1_bottom),
        ((x0 + x1) / 2, y0_bottom),
        (x0, y0_bottom),
        (x0, y0_top),
    ]
    codes = [
        MplPath.MOVETO,
        MplPath.CURVE4,
        MplPath.CURVE4,
        MplPath.CURVE4,
        MplPath.LINETO,
        MplPath.CURVE4,
        MplPath.CURVE4,
        MplPath.CURVE4,
        MplPath.CLOSEPOLY,
    ]
    axis.add_patch(
        patches.PathPatch(
            MplPath(controls, codes), facecolor=color, edgecolor="none", alpha=alpha
        )
    )


def save_figure_atomic(figure: plt.Figure, output: Path, **kwargs: object) -> None:
    """Replace one Sankey image only after Matplotlib completes the new file."""
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


def draw_on_axis(
    axis: plt.Axes,
    matrix: np.ndarray,
    include_correct: bool,
    label_min_count: int = 1,
    panel_label: str | None = None,
    font_scale: float = 1.0,
    label_offsets: dict[tuple[int, int], float] | None = None,
) -> None:
    active = matrix.astype(float, copy=True)
    if not include_correct:
        np.fill_diagonal(active, 0.0)
    total = float(active.sum())
    if total <= 0 or active.shape != (4, 4) or not np.isfinite(active).all():
        raise ValueError("Sankey input must be a finite, non-empty 4 x 4 matrix")
    displayed = row_preserving_round(active)
    left_totals = active.sum(axis=1)
    right_totals = active.sum(axis=0)
    x_left, x_right, bar_width = 0.12, 0.80, 0.055
    gap, available = 0.025, 1 - 3 * 0.025

    def positions(totals: np.ndarray) -> list[tuple[float, float]]:
        result: list[tuple[float, float]] = []
        top = 1.0
        for value in totals:
            height = value / total * available
            result.append((top - height, top))
            top -= height + gap
        return result

    left_positions = positions(left_totals)
    right_positions = positions(right_totals)
    left_cursor = [top for _, top in left_positions]
    right_cursor = [top for _, top in right_positions]

    for source in LABELS:
        for target in LABELS:
            value = float(active[source, target])
            if value <= 0:
                continue
            height = value / total * available
            left_top, right_top = left_cursor[source], right_cursor[target]
            left_bottom, right_bottom = left_top - height, right_top - height
            left_cursor[source], right_cursor[target] = left_bottom, right_bottom
            band(
                axis,
                x_left + bar_width,
                x_right,
                left_bottom,
                left_top,
                right_bottom,
                right_top,
                COLORS[source],
                0.16 if source == target else 0.58,
            )
            count = int(displayed[source, target])
            if source != target and count >= label_min_count:
                label_y = (left_bottom + left_top + right_bottom + right_top) / 4
                if label_offsets:
                    label_y += label_offsets.get((source, target), 0.0)
                axis.text(
                    (x_left + bar_width + x_right) / 2,
                    label_y,
                    str(count),
                    ha="center",
                    va="center",
                    fontsize=12 * font_scale,
                    color="black",
                    path_effects=[
                        path_effects.Stroke(linewidth=3 * font_scale, foreground="white"),
                        path_effects.Normal(),
                    ],
                )

    for index in LABELS:
        for x, (bottom, top), count, align, offset in [
            (x_left, left_positions[index], displayed[index].sum(), "right", -0.012),
            (
                x_right,
                right_positions[index],
                displayed[:, index].sum(),
                "left",
                bar_width + 0.012,
            ),
        ]:
            axis.add_patch(
                patches.Rectangle(
                    (x, bottom),
                    bar_width,
                    top - bottom,
                    facecolor=COLORS[index],
                    edgecolor="#333333",
                )
            )
            axis.text(
                x + offset,
                (bottom + top) / 2,
                f"{CLASS_NAMES[index]}\n({int(count)})",
                ha=align,
                va="center",
                fontsize=12 * font_scale,
            )
    axis.text(
        x_left, -0.065, "True stage", ha="left", va="top", fontsize=13 * font_scale
    )
    axis.text(
        x_right + bar_width,
        -0.065,
        "Predicted stage",
        ha="right",
        va="top",
        fontsize=13 * font_scale,
    )
    if panel_label:
        axis.text(
            0.5,
            1.035,
            panel_label,
            ha="center",
            va="bottom",
            fontsize=18 * font_scale,
        )
    axis.set_xlim(0, 1)
    axis.set_ylim(-0.10, 1.08)
    axis.axis("off")


def draw(
    matrix: np.ndarray,
    title: str,
    output: Path,
    include_correct: bool,
) -> None:
    figure, axis = plt.subplots(figsize=(10, 7))
    draw_on_axis(axis, matrix, include_correct=include_correct)
    axis.set_title(title, pad=24)
    try:
        save_figure_atomic(figure, output, dpi=300, bbox_inches="tight")
    finally:
        plt.close(figure)


def draw_main_figure() -> tuple[Path, Path]:
    piid_to_humc, _ = mean_confusion(PIID_INFERENCE_DIR, "resnet50", "HUMC")
    humc_to_piid, _ = mean_confusion(HUMC_INFERENCE_DIR, "resnet50", "PIID")
    figure, axes = plt.subplots(1, 2, figsize=(14.2, 8.2))
    draw_on_axis(
        axes[0],
        piid_to_humc,
        True,
        label_min_count=50,
        panel_label="(a)",
        font_scale=1.55,
        label_offsets={(2, 0): 0.055, (1, 2): 0.035, (2, 1): -0.015, (2, 3): -0.045, (3, 2): 0.025},
    )
    draw_on_axis(
        axes[1],
        humc_to_piid,
        True,
        label_min_count=50,
        panel_label="(b)",
        font_scale=1.55,
        label_offsets={(2, 3): 0.045, (3, 2): -0.045},
    )
    figure.subplots_adjust(left=0.02, right=0.98, bottom=0.08, top=0.96, wspace=0.18)
    output_dir = FIGURE_ROOT / "sankey"
    png = output_dir / "figure_2_cross_dataset_sankey.png"
    tif = output_dir / "figure_2_cross_dataset_sankey.tif"
    try:
        save_figure_atomic(figure, png, dpi=300, facecolor="white")
        save_figure_atomic(
            figure,
            tif,
            dpi=300,
            facecolor="white",
            pil_kwargs={"compression": "tiff_lzw"},
        )
    finally:
        plt.close(figure)
    return png, tif


def validate_dataset_names(datasets: list[str]) -> list[str]:
    if not datasets:
        raise ValueError("At least one evaluation dataset is required")
    for dataset in datasets:
        if not SAFE_NAME.fullmatch(dataset) or dataset in {".", ".."}:
            raise ValueError(f"Unsafe evaluation dataset name: {dataset!r}")
    if len({dataset.casefold() for dataset in datasets}) != len(datasets):
        raise ValueError("Evaluation dataset names must be unique")
    return datasets


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training", choices=["piid", "humc", "both"], default="both")
    parser.add_argument("--model", choices=[*MODELS, "all"], default="resnet50")
    parser.add_argument("--datasets", nargs="+")
    parser.add_argument("--misclassification-only", action="store_true")
    parser.add_argument(
        "--main-figure",
        action="store_true",
        help="Generate submitted Figure 2 from both ResNet-50 training directions.",
    )
    args = parser.parse_args()
    if args.main_figure:
        png, tif = draw_main_figure()
        print(f"[DONE] Main Figure 2: {png} and {tif}")
        return

    trainings = list(CONFIGS) if args.training == "both" else [args.training]
    for training in trainings:
        root, defaults = CONFIGS[training]
        datasets = validate_dataset_names(args.datasets or defaults)
        for dataset in datasets:
            if args.model == "all":
                matrix = mean_confusion_across_models(root, dataset)
                model_label = "six_models_mean"
            else:
                matrix, _ = mean_confusion(root, args.model, dataset)
                model_label = args.model
            suffix = "errors" if args.misclassification_only else "all"
            draw(
                matrix,
                f"{training.upper()}-trained {model_label} on {dataset}",
                FIGURE_ROOT
                / "sankey"
                / f"{training}_trained_{model_label}_{dataset}_{suffix}.png",
                include_correct=not args.misclassification_only,
            )
    print("[DONE] Fold-averaged Sankey figures complete")


if __name__ == "__main__":
    main()
