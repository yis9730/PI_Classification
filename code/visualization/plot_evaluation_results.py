"""Create confusion-matrix, ROC, and 17-condition augmentation figures."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import auc, confusion_matrix, roc_curve
from sklearn.preprocessing import label_binarize

CODE_ROOT = Path(__file__).resolve().parents[1]
CORE_DIR = CODE_ROOT / "core"
if str(CORE_DIR) not in sys.path:
    sys.path.insert(0, str(CORE_DIR))

from path_config import FIGURE_ROOT, HUMC_INFERENCE_DIR, PIID_INFERENCE_DIR  # noqa: E402

CLASS_NAMES = ["Stage 1", "Stage 2", "Stage 3", "Stage 4"]
CONFIGS = {
    "piid": ("PIID-trained", PIID_INFERENCE_DIR),
    "humc": ("HUMC-trained", HUMC_INFERENCE_DIR),
}


def parse_prediction_path(path: Path) -> tuple[str, str, int]:
    run_name = path.parents[1].name
    dataset = path.name.split("_fold")[0]
    fold = int(path.stem.split("_fold")[-1].split("_")[0])
    return run_name, dataset, fold


def plot_confusion(table: pd.DataFrame, title: str, output: Path) -> None:
    matrix = confusion_matrix(table["true_label"], table["predicted_label"], labels=[0, 1, 2, 3])
    fig, axis = plt.subplots(figsize=(5.2, 4.6))
    sns.heatmap(
        matrix, annot=True, fmt="d", cmap="Blues", cbar=False,
        xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=axis,
    )
    axis.set(xlabel="Predicted", ylabel="True", title=title)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output, dpi=300)
    plt.close(fig)


def plot_roc(table: pd.DataFrame, title: str, output: Path) -> None:
    y_true = table["true_label"].astype(int).to_numpy()
    y_binary = label_binarize(y_true, classes=[0, 1, 2, 3])
    probabilities = table[["prob_1", "prob_2", "prob_3", "prob_4"]].to_numpy()
    fig, axis = plt.subplots(figsize=(5.2, 4.6))
    for class_index, class_name in enumerate(CLASS_NAMES):
        if y_binary[:, class_index].sum() == 0:
            continue
        fpr, tpr, _ = roc_curve(y_binary[:, class_index], probabilities[:, class_index])
        axis.plot(fpr, tpr, label=f"{class_name} AUC={auc(fpr, tpr):.3f}")
    axis.plot([0, 1], [0, 1], color="0.7", linestyle="--")
    axis.set(xlabel="False positive rate", ylabel="True positive rate", title=title)
    axis.legend(fontsize=8)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output, dpi=300)
    plt.close(fig)


def prediction_figures(training: str, inference_root: Path) -> None:
    groups: dict[tuple[str, str], list[pd.DataFrame]] = {}
    for path in sorted(inference_root.glob("*/predictions/*_fold*_predictions.csv")):
        run_name, dataset, _ = parse_prediction_path(path)
        groups.setdefault((run_name, dataset), []).append(pd.read_csv(path))
    for (run_name, dataset), fold_tables in groups.items():
        # Fold predictions are stacked for display only; they are not ensembled.
        combined = pd.concat(fold_tables, ignore_index=True)
        output_dir = FIGURE_ROOT / "evaluation" / training / run_name
        title = f"{run_name} on {dataset} (five fold models)"
        plot_confusion(combined, title, output_dir / f"{dataset}_confusion_matrix.png")
        plot_roc(combined, title, output_dir / f"{dataset}_roc_curve.png")


def augmentation_heatmaps(training_label: str, training: str, inference_root: Path) -> None:
    path = inference_root / "__ALL_summary_results.csv"
    if not path.exists():
        print(f"[SKIP] Missing summary: {path}")
        return
    table = pd.read_csv(path)
    metric_columns = [column for column in table if column.endswith("_F1_Macro_mean")]
    for column in metric_columns:
        dataset = column.removesuffix("_F1_Macro_mean")
        pivot = table.pivot(index="Model", columns="Augmentation", values=column)
        fig, axis = plt.subplots(figsize=(15, 5.2))
        sns.heatmap(pivot, annot=True, fmt=".3f", cmap="viridis", ax=axis)
        axis.set_title(f"{training_label} macro-F1 on {dataset}")
        axis.tick_params(axis="x", rotation=45)
        fig.tight_layout()
        output = FIGURE_ROOT / "augmentation_heatmaps" / f"{training}_trained_{dataset}.png"
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=300, bbox_inches="tight")
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training", choices=["piid", "humc", "both"], default="both")
    args = parser.parse_args()
    keys = list(CONFIGS) if args.training == "both" else [args.training]
    for key in keys:
        label, root = CONFIGS[key]
        prediction_figures(key, root)
        augmentation_heatmaps(label, key, root)
    print("[DONE] Evaluation figures complete")


if __name__ == "__main__":
    main()
