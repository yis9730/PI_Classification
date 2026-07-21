"""Image-level bootstrap CIs for mean fold-specific macro-F1 (no ensemble).

For every resample, macro-F1 is calculated separately for each of the five
fixed fold-specific models and then averaged. Predictions or probabilities are
never combined across folds. The same sampled image indices are used for every
fold and architecture within a dataset. The study used 1,000 resamples and
random seed 40.
"""

from __future__ import annotations

import argparse
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


def normalize_labels(values: pd.Series) -> np.ndarray:
    labels = values.astype(int).to_numpy()
    unique = set(np.unique(labels).tolist())
    if unique <= {0, 1, 2, 3}:
        return labels
    if unique <= {1, 2, 3, 4}:
        return labels - 1
    raise ValueError(f"Expected four-class labels, observed {sorted(unique)}")


def load_folds(root: Path, model: str, dataset: str) -> list[dict[str, np.ndarray]]:
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
        image_paths = table["image_path"].astype(str).to_numpy()
        y_true = normalize_labels(table["true_label"])
        y_pred = normalize_labels(table["predicted_label"])
        if reference_paths is None:
            reference_paths, reference_true = image_paths, y_true
        elif not np.array_equal(reference_paths, image_paths) or not np.array_equal(reference_true, y_true):
            raise ValueError(f"Fold alignment mismatch: {model}, {dataset}, fold {fold_id}")
        folds.append({"y_true": y_true, "y_pred": y_pred})
    return folds


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
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def run_configuration(key: str, n_bootstrap: int, seed: int) -> None:
    config = CONFIGS[key]
    summary_rows = []
    fold_rows = []
    for dataset in config["datasets"]:
        first_folds = load_folds(config["root"], MODELS[0], dataset)
        n_images = len(first_folds[0]["y_true"])
        rng = np.random.default_rng(seed)
        bootstrap_indices = rng.integers(0, n_images, size=(n_bootstrap, n_images))
        for model_index, model in enumerate(MODELS):
            folds = first_folds if model_index == 0 else load_folds(config["root"], model, dataset)
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
    output_dir = TABLE_ROOT / "statistical_tests" / "bootstrap_foldwise" / f"{key}_trained"
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_table = pd.DataFrame(summary_rows)
    summary_table.to_csv(output_dir / "bootstrap_summary.csv", index=False, float_format="%.6f")
    pd.DataFrame(fold_rows).to_csv(output_dir / "fold_metrics.csv", index=False, float_format="%.6f")
    for dataset, subset in summary_table.groupby("dataset"):
        forest_plot(
            subset,
            f"{config['label']} on {dataset}",
            FIGURE_ROOT / "bootstrap_foldwise" / f"{key}_trained_{dataset}.png",
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training", choices=["piid", "humc", "both"], default="both")
    parser.add_argument("--n-bootstrap", type=int, default=N_BOOTSTRAP)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    keys = list(CONFIGS) if args.training == "both" else [args.training]
    for key in keys:
        run_configuration(key, args.n_bootstrap, args.seed)
    print("[DONE] Fold-wise bootstrap analysis complete")


if __name__ == "__main__":
    main()
