"""Reproduce fold-averaged under- and overstaging analyses.

Only the six final ``exp00_NoAug`` architectures are included. For each
architecture and evaluation dataset, a 4 x 4 confusion matrix is calculated
for every fold and normalized within true stage. The five matrices are then
averaged within architecture and the six architecture matrices are averaged
for the manuscript-level stage-macro summary.

Image-weighted proportions are also reported separately. They answer a
different question and must not be substituted for the manuscript's
stage-macro proportions.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

CODE_ROOT = Path(__file__).resolve().parents[1]
CORE_DIR = CODE_ROOT / "core"
if str(CORE_DIR) not in sys.path:
    sys.path.insert(0, str(CORE_DIR))

from path_config import HUMC_INFERENCE_DIR, PIID_INFERENCE_DIR, TABLE_ROOT  # noqa: E402

N_FOLDS = 5
LABELS = [0, 1, 2, 3]
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
REQUIRED_COLUMNS = {"image_path", "true_label", "predicted_label"}
EXPECTED_COHORTS = {
    "PIID_Test": (163, (35, 47, 41, 40)),
    "HUMC": (1844, (233, 709, 575, 327)),
    "Kaggle": (141, (27, 46, 41, 27)),
    "HUMC_Test": (288, (30, 104, 100, 54)),
    "PIID": (1081, (229, 311, 273, 268)),
}


def atomic_to_csv(table: pd.DataFrame, path: Path, **kwargs: object) -> None:
    """Replace one CSV only after its complete temporary file is written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        table.to_csv(temporary, **kwargs)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _integer_array(values: pd.Series, label: str) -> np.ndarray:
    numeric = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(numeric).all() or not np.equal(numeric, np.floor(numeric)).all():
        raise ValueError(f"{label} must contain finite integer labels")
    return numeric.astype(int)


def normalize_labels(true_values: pd.Series, predicted_values: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """Infer one unambiguous encoding from true labels and apply it to both."""
    true_raw = _integer_array(true_values, "true_label")
    predicted_raw = _integer_array(predicted_values, "predicted_label")
    observed = set(np.unique(true_raw).tolist())
    if 0 in observed and observed <= set(LABELS):
        offset = 0
    elif 4 in observed and observed <= {1, 2, 3, 4}:
        offset = 1
    else:
        raise ValueError(
            "true_label encoding is ambiguous; each evaluation file must expose "
            "0-based labels including 0 or 1-based labels including 4"
        )
    true = true_raw - offset
    predicted = predicted_raw - offset
    if not set(np.unique(true)).issubset(set(LABELS)):
        raise ValueError("true_label contains values outside the four study stages")
    if not set(np.unique(predicted)).issubset(set(LABELS)):
        raise ValueError("predicted_label uses a different or invalid stage encoding")
    if set(np.unique(true)) != set(LABELS):
        raise ValueError("Each evaluation dataset must contain all four true stages")
    return true, predicted


def load_prediction(
    path: Path, dataset: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(f"Required fold prediction is missing: {path}")
    table = pd.read_csv(path)
    missing = REQUIRED_COLUMNS - set(table.columns)
    if missing:
        raise ValueError(f"Prediction table is missing columns: {sorted(missing)}")
    if table.empty:
        raise ValueError("Prediction table is empty")
    paths = table["image_path"]
    if paths.isna().any() or paths.astype(str).str.strip().eq("").any():
        raise ValueError("Prediction table contains a blank image_path")
    if paths.astype(str).duplicated().any():
        raise ValueError("Prediction table contains duplicate image_path values")
    table = table.assign(image_path=paths.astype(str)).sort_values(
        "image_path", kind="stable"
    ).reset_index(drop=True)
    true, predicted = normalize_labels(table["true_label"], table["predicted_label"])
    expected_n, expected_stage_counts = EXPECTED_COHORTS[dataset]
    observed_stage_counts = tuple(
        np.bincount(true, minlength=len(LABELS)).astype(int).tolist()
    )
    if len(true) != expected_n or observed_stage_counts != expected_stage_counts:
        raise ValueError(
            f"Unexpected {dataset} cohort in {path}: expected n={expected_n}, "
            f"stages={expected_stage_counts}; observed n={len(true)}, "
            f"stages={observed_stage_counts}"
        )
    signature = np.column_stack([table["image_path"].to_numpy(), true.astype(str)])
    return true, predicted, signature


def row_normalize(matrix: np.ndarray) -> np.ndarray:
    totals = matrix.sum(axis=1, keepdims=True)
    if np.any(totals == 0):
        raise ValueError("Every true stage must contain at least one image")
    return matrix / totals


def direction_components(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    under = np.asarray([matrix[stage, :stage].sum() for stage in LABELS], dtype=float)
    correct = np.diag(matrix).astype(float)
    over = np.asarray([matrix[stage, stage + 1 :].sum() for stage in LABELS], dtype=float)
    return under, correct, over


def summarize_matrices(counts: np.ndarray, normalized: np.ndarray) -> dict[str, float]:
    under_stage, correct_stage, over_stage = direction_components(normalized)
    under_count, correct_count, over_count = direction_components(counts)
    total = float(counts.sum())
    return {
        "n_images": int(round(total)),
        "stage_macro_understaging": float(under_stage.mean()),
        "stage_macro_correct": float(correct_stage.mean()),
        "stage_macro_overstaging": float(over_stage.mean()),
        "image_weighted_understaging": float(under_count.sum() / total),
        "image_weighted_correct": float(correct_count.sum() / total),
        "image_weighted_overstaging": float(over_count.sum() / total),
    }


def analyze_configuration(
    training: str, root: Path, datasets: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    fold_rows: list[dict[str, object]] = []
    stage_rows: list[dict[str, object]] = []
    model_rows: list[dict[str, object]] = []
    dataset_rows: list[dict[str, object]] = []

    for dataset in datasets:
        dataset_signature: np.ndarray | None = None
        model_count_matrices = []
        model_normalized_matrices = []
        for model in MODELS:
            fold_count_matrices = []
            fold_normalized_matrices = []
            for fold in range(1, N_FOLDS + 1):
                prediction_path = (
                    root
                    / f"{model}_exp00_NoAug"
                    / "predictions"
                    / f"{dataset}_fold{fold}_predictions.csv"
                )
                true, predicted, signature = load_prediction(prediction_path, dataset)
                if dataset_signature is None:
                    dataset_signature = signature
                elif not np.array_equal(dataset_signature, signature):
                    raise ValueError(
                        f"Prediction alignment mismatch for {training}-trained {dataset}"
                    )
                counts = confusion_matrix(true, predicted, labels=LABELS).astype(float)
                normalized = row_normalize(counts)
                fold_count_matrices.append(counts)
                fold_normalized_matrices.append(normalized)
                under = int(np.sum(predicted < true))
                correct = int(np.sum(predicted == true))
                over = int(np.sum(predicted > true))
                n_images = len(true)
                fold_rows.append(
                    {
                        "training": f"{training.upper()}-trained",
                        "model": model,
                        "augmentation": "exp00_NoAug",
                        "dataset": dataset,
                        "fold": fold,
                        "n_images": n_images,
                        "understaging_n": under,
                        "correct_n": correct,
                        "overstaging_n": over,
                        "image_weighted_understaging": under / n_images,
                        "image_weighted_correct": correct / n_images,
                        "image_weighted_overstaging": over / n_images,
                    }
                )

            if len(fold_count_matrices) != N_FOLDS:
                raise RuntimeError(f"Expected exactly {N_FOLDS} folds")
            mean_counts = np.mean(fold_count_matrices, axis=0)
            mean_normalized = np.mean(fold_normalized_matrices, axis=0)
            model_count_matrices.append(mean_counts)
            model_normalized_matrices.append(mean_normalized)
            summary = summarize_matrices(mean_counts, mean_normalized)
            model_rows.append(
                {
                    "training": f"{training.upper()}-trained",
                    "model": model,
                    "augmentation": "exp00_NoAug",
                    "dataset": dataset,
                    "n_folds": N_FOLDS,
                    **summary,
                }
            )
            for true_stage in LABELS:
                row = mean_normalized[true_stage]
                stage_rows.append(
                    {
                        "training": f"{training.upper()}-trained",
                        "model": model,
                        "dataset": dataset,
                        "true_stage": true_stage + 1,
                        "true_stage_n_images": EXPECTED_COHORTS[dataset][1][true_stage],
                        **{
                            f"predicted_stage_{predicted_stage + 1}": float(row[predicted_stage])
                            for predicted_stage in LABELS
                        },
                        "understaging": float(row[:true_stage].sum()),
                        "correct": float(row[true_stage]),
                        "overstaging": float(row[true_stage + 1 :].sum()),
                    }
                )

        if len(model_count_matrices) != len(MODELS):
            raise RuntimeError(f"Expected exactly {len(MODELS)} final architectures")
        mean_counts = np.mean(model_count_matrices, axis=0)
        mean_normalized = np.mean(model_normalized_matrices, axis=0)
        dataset_rows.append(
            {
                "training": f"{training.upper()}-trained",
                "dataset": dataset,
                "n_models": len(MODELS),
                "n_folds_per_model": N_FOLDS,
                "aggregation": "row-normalize each fold, average folds within model, average six models, macro-average true stages",
                **summarize_matrices(mean_counts, mean_normalized),
            }
        )

    return (
        pd.DataFrame(fold_rows),
        pd.DataFrame(stage_rows),
        pd.DataFrame(model_rows),
        pd.DataFrame(dataset_rows),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training", choices=["piid", "humc", "both"], default="both")
    args = parser.parse_args()
    keys = list(CONFIGS) if args.training == "both" else [args.training]
    output = TABLE_ROOT / "model_error_analysis" / "staging_direction"
    # Compute every requested configuration before replacing any published
    # table. The default manuscript run therefore fails as a whole if one
    # required prediction set is absent or incomplete.
    results = {
        key: analyze_configuration(key, *CONFIGS[key]) for key in keys
    }
    for key in keys:
        foldwise, stagewise, model_summary, dataset_summary = results[key]
        atomic_to_csv(
            foldwise,
            output / f"{key}_trained_foldwise.csv",
            index=False,
            float_format="%.6f",
        )
        atomic_to_csv(
            stagewise,
            output / f"{key}_trained_stagewise.csv",
            index=False,
            float_format="%.6f",
        )
        atomic_to_csv(
            model_summary,
            output / f"{key}_trained_model_summary.csv",
            index=False,
            float_format="%.6f",
        )
        atomic_to_csv(
            dataset_summary,
            output / f"{key}_trained_dataset_summary.csv",
            index=False,
            float_format="%.6f",
        )
    print(f"[DONE] Staging-direction tables written to {output}")


if __name__ == "__main__":
    main()
