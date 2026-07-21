"""Plot fold-averaged confusion-matrix Sankey diagrams.

Flow widths are based on the unrounded element-wise mean of five fold-specific
confusion matrices. This is a visualization of fold-averaged counts, not an
ensemble prediction. All five aligned prediction files are required.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.path import Path as MplPath
from sklearn.metrics import confusion_matrix

CODE_ROOT = Path(__file__).resolve().parents[1]
DEVELOPMENT_DIR = CODE_ROOT / "development"
if str(DEVELOPMENT_DIR) not in sys.path:
    sys.path.insert(0, str(DEVELOPMENT_DIR))

from path_config import FIGURE_ROOT, HUMC_INFERENCE_DIR, PIID_INFERENCE_DIR  # noqa: E402

N_FOLDS = 5
LABELS = [0, 1, 2, 3]
CLASS_NAMES = ["Stage 1", "Stage 2", "Stage 3", "Stage 4"]
COLORS = ["#2980B9", "#16A085", "#8E44AD", "#C0392B"]
CONFIGS = {
    "piid": (PIID_INFERENCE_DIR, ["PIID_Test", "HUMC", "Kaggle"]),
    "humc": (HUMC_INFERENCE_DIR, ["HUMC_Test", "PIID", "Kaggle"]),
}


def normalize(values: pd.Series) -> np.ndarray:
    array = values.astype(int).to_numpy()
    unique = set(np.unique(array).tolist())
    if unique <= set(LABELS):
        return array
    if unique <= {1, 2, 3, 4}:
        return array - 1
    raise ValueError(f"Unexpected labels: {sorted(unique)}")


def mean_confusion(root: Path, model: str, dataset: str) -> np.ndarray:
    prediction_dir = root / f"{model}_exp00_NoAug" / "predictions"
    matrices = []
    reference = None
    for fold in range(1, N_FOLDS + 1):
        path = prediction_dir / f"{dataset}_fold{fold}_predictions.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        table = pd.read_csv(path)
        signature = table[["image_path", "true_label"]].astype(str).to_numpy()
        if reference is None:
            reference = signature
        elif not np.array_equal(reference, signature):
            raise ValueError(f"Fold alignment mismatch: {model}, {dataset}, fold {fold}")
        matrices.append(
            confusion_matrix(
                normalize(table["true_label"]),
                normalize(table["predicted_label"]),
                labels=LABELS,
            ).astype(float)
        )
    if len(matrices) != N_FOLDS:
        raise RuntimeError(f"Expected {N_FOLDS} confusion matrices, got {len(matrices)}")
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
        (x0, y0_top), ((x0 + x1) / 2, y0_top), ((x0 + x1) / 2, y1_top), (x1, y1_top),
        (x1, y1_bottom), ((x0 + x1) / 2, y1_bottom), ((x0 + x1) / 2, y0_bottom), (x0, y0_bottom),
        (x0, y0_top),
    ]
    codes = [MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4,
             MplPath.LINETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4, MplPath.CLOSEPOLY]
    axis.add_patch(patches.PathPatch(MplPath(controls, codes), facecolor=color, edgecolor="none", alpha=alpha))


def draw(matrix: np.ndarray, title: str, output: Path, include_correct: bool) -> None:
    total = matrix.sum()
    displayed = row_preserving_round(matrix)
    left_totals = matrix.sum(axis=1)
    right_totals = matrix.sum(axis=0)
    x_left, x_right, bar_width = 0.12, 0.80, 0.055
    gap, available = 0.025, 1 - 3 * 0.025

    def positions(totals):
        result, top = [], 1.0
        for value in totals:
            height = value / total * available
            result.append((top - height, top))
            top -= height + gap
        return result

    left_positions = positions(left_totals)
    right_positions = positions(right_totals)
    left_cursor = [top for _, top in left_positions]
    right_cursor = [top for _, top in right_positions]
    fig, axis = plt.subplots(figsize=(10, 7))

    for source in range(4):
        for target in range(4):
            value = float(matrix[source, target])
            if value <= 0 or (not include_correct and source == target):
                continue
            height = value / total * available
            ltop, rtop = left_cursor[source], right_cursor[target]
            lbottom, rbottom = ltop - height, rtop - height
            left_cursor[source], right_cursor[target] = lbottom, rbottom
            band(
                axis, x_left + bar_width, x_right,
                lbottom, ltop, rbottom, rtop,
                COLORS[source], 0.16 if source == target else 0.58,
            )
            count = int(displayed[source, target])
            if source != target and count > 0:
                axis.text(
                    (x_left + bar_width + x_right) / 2,
                    (lbottom + ltop + rbottom + rtop) / 4,
                    str(count), ha="center", va="center", fontsize=9,
                )

    for index in range(4):
        for x, (bottom, top), count, align, offset in [
            (x_left, left_positions[index], displayed[index].sum(), "right", -0.012),
            (x_right, right_positions[index], displayed[:, index].sum(), "left", bar_width + 0.012),
        ]:
            axis.add_patch(
                patches.Rectangle((x, bottom), bar_width, top - bottom, facecolor=COLORS[index], edgecolor="#333333")
            )
            axis.text(
                x + offset,
                (bottom + top) / 2,
                f"{CLASS_NAMES[index]}\n({int(count)})",
                ha=align, va="center", fontsize=10,
            )
    axis.text(x_left, 1.055, "True stage", ha="left", fontweight="bold")
    axis.text(x_right + bar_width, 1.055, "Predicted stage", ha="right", fontweight="bold")
    axis.set_title(title, pad=24)
    axis.set_xlim(0, 1)
    axis.set_ylim(-0.02, 1.08)
    axis.axis("off")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training", choices=["piid", "humc", "both"], default="both")
    parser.add_argument("--model", default="resnet50")
    parser.add_argument("--datasets", nargs="+")
    parser.add_argument("--misclassification-only", action="store_true")
    args = parser.parse_args()
    trainings = list(CONFIGS) if args.training == "both" else [args.training]
    for training in trainings:
        root, defaults = CONFIGS[training]
        datasets = args.datasets or defaults
        for dataset in datasets:
            matrix = mean_confusion(root, args.model, dataset)
            suffix = "errors" if args.misclassification_only else "all"
            draw(
                matrix,
                f"{training.upper()}-trained {args.model} on {dataset}",
                FIGURE_ROOT / "sankey" / f"{training}_trained_{args.model}_{dataset}_{suffix}.png",
                include_correct=not args.misclassification_only,
            )
    print("[DONE] Fold-averaged Sankey figures complete")


if __name__ == "__main__":
    main()
