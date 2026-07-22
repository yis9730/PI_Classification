"""Create confusion-matrix, ROC, and 17-condition augmentation figures."""

from __future__ import annotations

import argparse
import re
import sys
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
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
MODELS = (
    "swin_tiny_patch4_window7_224",
    "efficientnet_v2_s",
    "vit_base_patch16_224",
    "resnet50",
    "densenet121",
    "convnext_small",
)
AUGMENTATIONS = (
    "exp00_NoAug",
    "exp01_Flip",
    "exp02_Rotate90",
    "exp03a_RandomZoomIn",
    "exp03b_CenterZoomIn",
    "exp04_ZoomOut",
    "exp05_Brightness",
    "exp06_Contrast",
    "exp07_F_R",
    "exp08_F_R_ZI",
    "exp09_F_R_ZI_ZO",
    "exp10_F_R_ZI_ZO_B",
    "exp11_F_R_ZI_ZO_B_C",
    "exp12_F_R_CZI",
    "exp13_F_R_CZI_ZO",
    "exp14_F_R_CZI_ZO_B",
    "exp15_F_R_CZI_ZO_B_C",
)
EXPECTED_DATASETS = {
    "piid": ("PIID_Test", "HUMC", "Kaggle"),
    "humc": ("HUMC_Test", "PIID", "Kaggle"),
}
PREDICTION_NAME = re.compile(
    r"^(?P<dataset>[A-Za-z0-9][A-Za-z0-9._-]*)_fold(?P<fold>[1-5])_predictions\.csv$"
)
PREDICTION_COLUMNS = {
    "image_path", "true_label", "predicted_label",
    "prob_1", "prob_2", "prob_3", "prob_4",
}
EXPECTED_STAGE_COUNTS = {
    "PIID_Test": [35, 47, 41, 40],
    "PIID": [229, 311, 273, 268],
    "HUMC_Test": [30, 104, 100, 54],
    "HUMC": [233, 709, 575, 327],
    "Kaggle": [27, 46, 41, 27],
}


def save_figure_atomic(figure: plt.Figure, output: Path, **kwargs: object) -> None:
    """Replace one plot only after Matplotlib has written the complete file."""
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


def parse_prediction_path(path: Path) -> tuple[str, str, int]:
    match = PREDICTION_NAME.fullmatch(path.name)
    if match is None or path.parent.name != "predictions":
        raise ValueError(f"Unexpected prediction path layout: {path}")
    run_name = path.parents[1].name
    if not run_name or run_name in {".", ".."}:
        raise ValueError(f"Invalid run directory in prediction path: {path}")
    return run_name, match.group("dataset"), int(match.group("fold"))


def _integer_labels(series: pd.Series, column: str, path: Path) -> np.ndarray:
    numeric = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(numeric).all() or not np.equal(numeric, np.floor(numeric)).all():
        raise ValueError(f"{path}: {column} must contain integer labels")
    labels = numeric.astype(int)
    if not np.isin(labels, [0, 1, 2, 3]).all():
        raise ValueError(f"{path}: {column} must use zero-based labels 0-3")
    return labels


def load_prediction_table(path: Path) -> pd.DataFrame:
    table = pd.read_csv(path)
    missing = PREDICTION_COLUMNS.difference(table.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    if table.empty:
        raise ValueError(f"{path}: prediction table is empty")

    image_paths = table["image_path"]
    if image_paths.isna().any() or image_paths.astype(str).str.strip().eq("").any():
        raise ValueError(f"{path}: image_path contains blank values")
    if image_paths.astype(str).duplicated().any():
        raise ValueError(f"{path}: image_path values must be unique within a fold")

    validated = table.copy()
    validated["image_path"] = image_paths.astype(str)
    validated["true_label"] = _integer_labels(table["true_label"], "true_label", path)
    validated["predicted_label"] = _integer_labels(
        table["predicted_label"], "predicted_label", path
    )
    probability_columns = ["prob_1", "prob_2", "prob_3", "prob_4"]
    probabilities = validated[probability_columns].apply(
        pd.to_numeric, errors="coerce"
    ).to_numpy(dtype=float)
    if not np.isfinite(probabilities).all():
        raise ValueError(f"{path}: probabilities contain non-finite values")
    if ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError(f"{path}: probabilities must lie in [0, 1]")
    # Archived study CSVs store four-decimal probabilities; permit only the
    # maximum accumulated rounding error from four such values.
    if not np.allclose(probabilities.sum(axis=1), 1.0, rtol=0.0, atol=2.1e-4):
        raise ValueError(f"{path}: probability rows must sum to 1")
    validated[probability_columns] = probabilities
    predicted = validated["predicted_label"].to_numpy(dtype=int)
    selected_probability = probabilities[np.arange(len(probabilities)), predicted]
    # Four-decimal archived probabilities can turn the top two values into a
    # displayed tie even though the unrounded argmax selected the latter.
    if np.any(selected_probability + 1.1e-4 < probabilities.max(axis=1)):
        raise ValueError(f"{path}: predicted_label disagrees with the probability argmax")
    return validated


def validate_fold_alignment(
    run_name: str,
    dataset: str,
    fold_tables: dict[int, tuple[Path, pd.DataFrame]],
) -> list[pd.DataFrame]:
    expected = set(range(1, 6))
    observed = set(fold_tables)
    if observed != expected:
        raise ValueError(
            f"{run_name}/{dataset}: expected folds 1-5, found {sorted(observed)}"
        )
    ordered = [fold_tables[fold][1] for fold in range(1, 6)]
    reference = ordered[0][["image_path", "true_label"]].reset_index(drop=True)
    expected_counts = EXPECTED_STAGE_COUNTS.get(dataset)
    for fold, table in zip(range(2, 6), ordered[1:], strict=True):
        candidate = table[["image_path", "true_label"]].reset_index(drop=True)
        if not candidate.equals(reference):
            raise ValueError(
                f"{run_name}/{dataset}: fold {fold} image/label order differs from fold 1"
            )
    if expected_counts is not None:
        observed_counts = np.bincount(
            reference["true_label"].to_numpy(dtype=int), minlength=4
        ).tolist()
        if observed_counts != expected_counts:
            raise ValueError(
                f"{run_name}/{dataset}: unexpected true-stage counts {observed_counts}"
            )
    return ordered


def plot_confusion(table: pd.DataFrame, title: str, output: Path) -> None:
    matrix = confusion_matrix(table["true_label"], table["predicted_label"], labels=[0, 1, 2, 3])
    fig, axis = plt.subplots(figsize=(5.2, 4.6))
    sns.heatmap(
        matrix, annot=True, fmt="d", cmap="Blues", cbar=False,
        xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=axis,
    )
    axis.set(xlabel="Predicted", ylabel="True", title=title)
    try:
        fig.tight_layout()
        save_figure_atomic(fig, output, dpi=300)
    finally:
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
    try:
        fig.tight_layout()
        save_figure_atomic(fig, output, dpi=300)
    finally:
        plt.close(fig)


def prediction_figures(training: str, inference_root: Path) -> None:
    if not inference_root.is_dir():
        raise FileNotFoundError(f"Prediction root not found: {inference_root}")
    groups: dict[tuple[str, str], dict[int, tuple[Path, pd.DataFrame]]] = {}
    for path in sorted(inference_root.glob("*/predictions/*_fold*_predictions.csv")):
        run_name, dataset, fold = parse_prediction_path(path)
        group = groups.setdefault((run_name, dataset), {})
        if fold in group:
            raise ValueError(f"Duplicate fold {fold} for {run_name}/{dataset}")
        group[fold] = (path, load_prediction_table(path))
    if not groups:
        raise FileNotFoundError(f"No fold prediction CSVs found under {inference_root}")
    validated_groups: list[tuple[str, str, pd.DataFrame]] = []
    for (run_name, dataset), indexed_tables in groups.items():
        fold_tables = validate_fold_alignment(run_name, dataset, indexed_tables)
        # Fold predictions are stacked for display only; they are not ensembled.
        combined = pd.concat(fold_tables, ignore_index=True)
        validated_groups.append((run_name, dataset, combined))
    for run_name, dataset, combined in validated_groups:
        output_dir = FIGURE_ROOT / "evaluation" / training / run_name
        title = f"{run_name} on {dataset} (five fold models)"
        plot_confusion(combined, title, output_dir / f"{dataset}_confusion_matrix.png")
        plot_roc(combined, title, output_dir / f"{dataset}_roc_curve.png")


def augmentation_heatmaps(training_label: str, training: str, inference_root: Path) -> None:
    path = inference_root / "__ALL_summary_results.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing evaluation summary: {path}")
    table = pd.read_csv(path)
    required = {"Model", "Augmentation"}
    missing = required.difference(table.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    if table.empty:
        raise ValueError(f"{path}: summary table is empty")
    if table[["Model", "Augmentation"]].isna().any().any():
        raise ValueError(f"{path}: model/augmentation identifiers cannot be blank")
    table["Model"] = table["Model"].astype(str).str.strip()
    table["Augmentation"] = table["Augmentation"].astype(str).str.strip()
    expected_pairs = {(model, augmentation) for model in MODELS for augmentation in AUGMENTATIONS}
    observed_pairs = set(zip(table["Model"], table["Augmentation"], strict=True))
    if observed_pairs != expected_pairs:
        missing_pairs = len(expected_pairs - observed_pairs)
        unexpected_pairs = len(observed_pairs - expected_pairs)
        raise ValueError(
            f"{path}: expected the complete six-model/17-condition grid; "
            f"missing combinations={missing_pairs}, unexpected combinations={unexpected_pairs}"
        )
    if table.duplicated(["Model", "Augmentation"]).any():
        raise ValueError(f"{path}: duplicate model/augmentation rows")
    metric_columns = [column for column in table if column.endswith("_F1_Macro_mean")]
    observed_datasets = {column.removesuffix("_F1_Macro_mean") for column in metric_columns}
    if observed_datasets != set(EXPECTED_DATASETS[training]):
        raise ValueError(
            f"{path}: expected macro-F1 columns for {list(EXPECTED_DATASETS[training])}; "
            f"found {sorted(observed_datasets)}"
        )
    validated_metrics: list[tuple[str, pd.DataFrame]] = []
    for column in metric_columns:
        values = pd.to_numeric(table[column], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
            raise ValueError(f"{path}: {column} must contain finite values in [0, 1]")
        numeric_table = table.assign(**{column: values})
        dataset = column.removesuffix("_F1_Macro_mean")
        pivot = numeric_table.pivot(index="Model", columns="Augmentation", values=column)
        pivot = pivot.reindex(index=MODELS, columns=AUGMENTATIONS)
        if pivot.isna().any().any():
            raise ValueError(f"{path}: incomplete model/augmentation grid for {dataset}")
        validated_metrics.append((dataset, pivot))
    for dataset, pivot in validated_metrics:
        fig, axis = plt.subplots(figsize=(15, 5.2))
        sns.heatmap(pivot, annot=True, fmt=".3f", cmap="viridis", ax=axis)
        axis.set_title(f"{training_label} macro-F1 on {dataset}")
        axis.tick_params(axis="x", rotation=45)
        fig.tight_layout()
        output = FIGURE_ROOT / "augmentation_heatmaps" / f"{training}_trained_{dataset}.png"
        try:
            save_figure_atomic(fig, output, dpi=300, bbox_inches="tight")
        finally:
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
