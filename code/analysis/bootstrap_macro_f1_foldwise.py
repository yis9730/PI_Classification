"""Image-level bootstrap CIs for mean fold-specific macro-F1 (no ensemble).

For every resample, macro-F1 is calculated separately for each of the five
fixed fold-specific models and then averaged. Predictions or probabilities are
never combined across folds. The same sampled image indices are used for every
fold and architecture within a dataset. The study used 1,000 resamples, the
legacy NumPy ``RandomState`` sampler, and random seed 40. Keeping that sampler
is necessary to reproduce the reported percentile endpoints from the archived
prediction tables.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

CODE_ROOT = Path(__file__).resolve().parents[1]
CORE_DIR = CODE_ROOT / "core"
if str(CORE_DIR) not in sys.path:
    sys.path.insert(0, str(CORE_DIR))

from path_config import HUMC_INFERENCE_DIR, PIID_INFERENCE_DIR, TABLE_ROOT, FIGURE_ROOT  # noqa: E402

RANDOM_SEED = 40
N_FOLDS = 5
N_BOOTSTRAP = 1000
LABELS = [0, 1, 2, 3]
MODELS = [
    "resnet50",
    "densenet121",
    "efficientnet_v2_s",
    "vit_base_patch16_224",
    "swin_tiny_patch4_window7_224",
    "convnext_small",
]
DISPLAY_NAMES = {
    "resnet50": "ResNet-50",
    "densenet121": "DenseNet-121",
    "efficientnet_v2_s": "EfficientNetV2-S",
    "vit_base_patch16_224": "ViT-B/16",
    "swin_tiny_patch4_window7_224": "Swin-T",
    "convnext_small": "ConvNeXt-S",
}
CONFIGS = {
    "piid": {
        "label": "PIID-trained",
        "root": PIID_INFERENCE_DIR,
        "datasets": ["PIID_Test", "HUMC", "Kaggle"],
    },
    "humc": {
        "label": "HUMC-trained",
        "root": HUMC_INFERENCE_DIR,
        "datasets": ["HUMC_Test", "PIID", "Kaggle"],
    },
}
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


def normalize_labels(
    true_values: pd.Series, predicted_values: pd.Series
) -> tuple[np.ndarray, np.ndarray]:
    """Infer one unambiguous label offset from truth and apply it to predictions."""
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
        raise ValueError("Each evaluation table must contain all four true stages")
    if not set(np.unique(predicted)).issubset(set(LABELS)):
        raise ValueError("predicted_label uses a different or invalid encoding")
    return true, predicted


def load_folds(
    root: Path, model: str, dataset: str
) -> tuple[list[dict[str, np.ndarray]], np.ndarray]:
    prediction_dir = root / f"{model}_exp00_NoAug" / "predictions"
    folds = []
    reference_paths = None
    reference_true = None
    for fold_id in range(1, N_FOLDS + 1):
        path = prediction_dir / f"{dataset}_fold{fold_id}_predictions.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        table = pd.read_csv(path)
        required = {"image_path", "true_label", "predicted_label"}
        if not required <= set(table.columns):
            raise ValueError(f"Missing columns in {path}: {sorted(required - set(table.columns))}")
        if table.empty or table["image_path"].isna().any():
            raise ValueError("Prediction table is empty or contains a blank image_path")
        table["image_path"] = table["image_path"].astype(str)
        if table["image_path"].str.strip().eq("").any() or table["image_path"].duplicated().any():
            raise ValueError("Prediction table contains blank or duplicate image paths")
        table = table.sort_values("image_path", kind="stable").reset_index(drop=True)
        image_paths = table["image_path"].to_numpy()
        y_true, y_pred = normalize_labels(table["true_label"], table["predicted_label"])
        expected_n, expected_stage_counts = EXPECTED_COHORTS[dataset]
        observed_stage_counts = tuple(
            np.bincount(y_true, minlength=len(LABELS)).astype(int).tolist()
        )
        if len(y_true) != expected_n or observed_stage_counts != expected_stage_counts:
            raise ValueError(
                f"Unexpected {dataset} cohort in {path}: expected n={expected_n}, "
                f"stages={expected_stage_counts}; observed n={len(y_true)}, "
                f"stages={observed_stage_counts}"
            )
        if reference_paths is None:
            reference_paths, reference_true = image_paths, y_true
        elif not np.array_equal(reference_paths, image_paths) or not np.array_equal(reference_true, y_true):
            raise ValueError(f"Fold alignment mismatch: {model}, {dataset}, fold {fold_id}")
        folds.append({"y_true": y_true, "y_pred": y_pred})
    if reference_paths is None or reference_true is None or len(folds) != N_FOLDS:
        raise RuntimeError(f"Expected exactly {N_FOLDS} aligned folds")
    signature = np.column_stack([reference_paths, reference_true.astype(str)])
    return folds, signature


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(
        f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0)
    )


def analyze_model(
    folds: list[dict[str, np.ndarray]], bootstrap_indices: np.ndarray
) -> tuple[dict[str, float], list[dict[str, float]]]:
    fold_scores = [macro_f1(fold["y_true"], fold["y_pred"]) for fold in folds]
    bootstrap_scores = np.empty(len(bootstrap_indices), dtype=np.float64)
    for index, sampled in enumerate(bootstrap_indices):
        bootstrap_scores[index] = np.mean(
            [macro_f1(fold["y_true"][sampled], fold["y_pred"][sampled]) for fold in folds]
        )
    summary = {
        "macro_f1_mean": float(np.mean(fold_scores)),
        "fold_sd": float(np.std(fold_scores, ddof=1)),
        "ci_lower": float(np.percentile(bootstrap_scores, 2.5)),
        "ci_upper": float(np.percentile(bootstrap_scores, 97.5)),
        "n_images": int(len(folds[0]["y_true"])),
        "n_folds": N_FOLDS,
        "n_bootstrap": int(len(bootstrap_indices)),
    }
    fold_rows = [
        {"fold": fold_id, "macro_f1": score}
        for fold_id, score in enumerate(fold_scores, start=1)
    ]
    return summary, fold_rows


def forest_plot(table: pd.DataFrame, title: str, output: Path) -> None:
    ordered = table.sort_values("macro_f1_mean")
    y = np.arange(len(ordered))
    x = ordered["macro_f1_mean"].to_numpy()
    errors = np.vstack([x - ordered["ci_lower"], ordered["ci_upper"] - x])
    fig, axis = plt.subplots(figsize=(7.2, 4.2))
    axis.errorbar(x, y, xerr=errors, fmt="s", color="#1f4e79", capsize=4)
    axis.set_yticks(y, ordered["model_display"])
    axis.set_xlabel("Macro-F1")
    axis.set_title(title)
    axis.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.stem}.{os.getpid()}.tmp{output.suffix}")
    try:
        fig.savefig(temporary, dpi=300, bbox_inches="tight")
        os.replace(temporary, output)
    finally:
        plt.close(fig)
        temporary.unlink(missing_ok=True)


def analyze_configuration(
    key: str, n_bootstrap: int, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    config = CONFIGS[key]
    if n_bootstrap <= 0:
        raise ValueError("--n-bootstrap must be positive")
    summary_rows = []
    fold_rows = []
    for dataset in config["datasets"]:
        first_folds, reference_signature = load_folds(config["root"], MODELS[0], dataset)
        n_images = len(first_folds[0]["y_true"])
        # The submitted analysis used NumPy's legacy MT19937 RandomState
        # sequence. ``default_rng(seed)`` produces different resamples and
        # therefore different displayed confidence-interval endpoints.
        rng = np.random.RandomState(seed)
        bootstrap_indices = rng.choice(
            n_images, size=(n_bootstrap, n_images), replace=True
        )
        for model_index, model in enumerate(MODELS):
            if model_index == 0:
                folds = first_folds
            else:
                folds, signature = load_folds(config["root"], model, dataset)
                if not np.array_equal(reference_signature, signature):
                    raise ValueError(
                        f"Architecture alignment mismatch: {model}, {dataset}"
                    )
            summary, per_fold = analyze_model(folds, bootstrap_indices)
            summary_rows.append(
                {
                    "training": config["label"],
                    "dataset": dataset,
                    "model": model,
                    "model_display": DISPLAY_NAMES[model],
                    **summary,
                    "bootstrap_unit": "image",
                    "fold_prediction_combination": "none",
                    "random_seed": seed,
                    "rng_algorithm": "numpy.random.RandomState(MT19937)",
                }
            )
            for row in per_fold:
                fold_rows.append(
                    {
                        "training": config["label"],
                        "dataset": dataset,
                        "model": model,
                        **row,
                    }
                )
    summary_table = pd.DataFrame(summary_rows)
    return summary_table, pd.DataFrame(fold_rows)


def write_configuration(
    key: str, summary_table: pd.DataFrame, fold_table: pd.DataFrame
) -> None:
    """Publish a fully computed configuration without partially written CSVs."""
    config = CONFIGS[key]
    output_dir = TABLE_ROOT / "statistical_tests" / "bootstrap_foldwise" / f"{key}_trained"
    atomic_to_csv(
        summary_table,
        output_dir / "bootstrap_summary.csv",
        index=False,
        float_format="%.6f",
    )
    atomic_to_csv(
        fold_table,
        output_dir / "fold_metrics.csv",
        index=False,
        float_format="%.6f",
    )
    for dataset, subset in summary_table.groupby("dataset"):
        forest_plot(
            subset,
            f"{config['label']} on {dataset}",
            FIGURE_ROOT / "bootstrap_foldwise" / f"{key}_trained_{dataset}.png",
        )


def run_configuration(key: str, n_bootstrap: int, seed: int) -> None:
    """Compute then publish one explicitly requested training configuration."""
    write_configuration(key, *analyze_configuration(key, n_bootstrap, seed))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training", choices=["piid", "humc", "both"], default="both")
    parser.add_argument("--n-bootstrap", type=int, default=N_BOOTSTRAP)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    keys = list(CONFIGS) if args.training == "both" else [args.training]
    # Compute every requested configuration first. In the manuscript default
    # (``both``), a missing HUMC or PIID input therefore cannot leave a newly
    # written half-run beside stale results from the other configuration.
    results = {
        key: analyze_configuration(key, args.n_bootstrap, args.seed) for key in keys
    }
    for key in keys:
        write_configuration(key, *results[key])
    print("[DONE] Fold-wise bootstrap analysis complete")


if __name__ == "__main__":
    main()
