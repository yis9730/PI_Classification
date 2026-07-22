"""Dataset-specific fold-wise Friedman--Nemenyi architecture analysis.

Each evaluation dataset is analyzed separately. The five fold-specific
macro-F1 values are paired blocks and the six architectures are treatments.
The Friedman chi-square p-value is the manuscript omnibus result and controls
whether the Nemenyi post-hoc comparison is emitted. The Iman--Davenport value
is retained as a clearly labelled sensitivity statistic. With only five
overlapping folds, results are exploratory.
"""

from __future__ import annotations

import argparse
import itertools
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import friedmanchisquare, rankdata
from sklearn.metrics import f1_score

CODE_ROOT = Path(__file__).resolve().parents[1]
CORE_DIR = CODE_ROOT / "core"
if str(CORE_DIR) not in sys.path:
    sys.path.insert(0, str(CORE_DIR))

from path_config import HUMC_INFERENCE_DIR, PIID_INFERENCE_DIR, TABLE_ROOT  # noqa: E402

N_FOLDS = 5
ALPHA = 0.05
LABELS = [0, 1, 2, 3]
MODELS = [
    "resnet50", "densenet121", "efficientnet_v2_s",
    "vit_base_patch16_224", "swin_tiny_patch4_window7_224", "convnext_small",
]
DISPLAY = {
    "resnet50": "ResNet-50", "densenet121": "DenseNet-121",
    "efficientnet_v2_s": "EfficientNetV2-S", "vit_base_patch16_224": "ViT-B/16",
    "swin_tiny_patch4_window7_224": "Swin-T", "convnext_small": "ConvNeXt-S",
}
CONFIGS = {
    "piid": ("PIID-trained", PIID_INFERENCE_DIR, ["PIID_Test", "HUMC", "Kaggle"]),
    "humc": ("HUMC-trained", HUMC_INFERENCE_DIR, ["HUMC_Test", "PIID", "Kaggle"]),
}
Q_ALPHA_005 = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164}
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


def labels(true_values: pd.Series, predicted_values: pd.Series) -> tuple[np.ndarray, np.ndarray]:
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


def scores_for_dataset(root: Path, dataset: str) -> pd.DataFrame:
    score_columns = {}
    reference = None
    for model in MODELS:
        values = []
        for fold in range(1, N_FOLDS + 1):
            path = root / f"{model}_exp00_NoAug" / "predictions" / f"{dataset}_fold{fold}_predictions.csv"
            if not path.exists():
                raise FileNotFoundError(path)
            table = pd.read_csv(path)
            required = {"image_path", "true_label", "predicted_label"}
            missing = required - set(table.columns)
            if missing:
                raise ValueError(f"Missing prediction columns: {sorted(missing)}")
            if table.empty or table["image_path"].isna().any():
                raise ValueError("Prediction table is empty or contains a blank image_path")
            table["image_path"] = table["image_path"].astype(str)
            if table["image_path"].str.strip().eq("").any() or table["image_path"].duplicated().any():
                raise ValueError("Prediction table contains blank or duplicate image paths")
            table = table.sort_values("image_path", kind="stable").reset_index(drop=True)
            y_true, y_pred = labels(table["true_label"], table["predicted_label"])
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
            signature = np.column_stack([table["image_path"].to_numpy(), y_true.astype(str)])
            if reference is None:
                reference = signature
            elif not np.array_equal(reference, signature):
                raise ValueError(f"Prediction alignment mismatch: {model}, {dataset}, fold {fold}")
            values.append(
                f1_score(
                    y_true, y_pred,
                    labels=LABELS, average="macro", zero_division=0,
                )
            )
        score_columns[DISPLAY[model]] = values
    output = pd.DataFrame(score_columns, index=range(1, N_FOLDS + 1))
    output.index.name = "fold"
    return output


def analyze(scores: pd.DataFrame) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame]:
    n_blocks, n_models = scores.shape
    values = scores.to_numpy(dtype=float)
    if values.shape != (N_FOLDS, len(MODELS)) or not np.isfinite(values).all():
        raise ValueError(
            f"Expected a finite {N_FOLDS} x {len(MODELS)} fold-score matrix"
        )
    if np.all(np.ptp(values, axis=1) == 0):
        chi2, p_chi2 = 0.0, 1.0
    else:
        chi2, p_chi2 = friedmanchisquare(
            *[scores[column].to_numpy(dtype=float) for column in scores.columns]
        )
        if not np.isfinite(chi2) or not np.isfinite(p_chi2):
            raise RuntimeError("Friedman test returned a non-finite statistic")
    denominator = n_blocks * (n_models - 1) - chi2
    f_stat = ((n_blocks - 1) * chi2) / denominator if denominator > 0 else np.inf
    p_iman = float(stats.f.sf(f_stat, n_models - 1, (n_models - 1) * (n_blocks - 1)))
    ranks = scores.apply(lambda row: rankdata(-row, method="average"), axis=1, result_type="expand")
    ranks.columns = scores.columns
    rank_table = pd.DataFrame(
        {
            "model": scores.columns,
            "mean_macro_f1": scores.mean().to_numpy(),
            "sd_macro_f1": scores.std(ddof=1).to_numpy(),
            "average_rank": ranks.mean().to_numpy(),
        }
    ).sort_values("average_rank")
    cd = Q_ALPHA_005[n_models] * np.sqrt(n_models * (n_models + 1) / (6 * n_blocks))
    pair_rows = []
    for left, right in itertools.combinations(rank_table["model"], 2):
        left_rank = float(rank_table.loc[rank_table["model"].eq(left), "average_rank"].iloc[0])
        right_rank = float(rank_table.loc[rank_table["model"].eq(right), "average_rank"].iloc[0])
        pair_rows.append(
            {
                "model_1": left, "model_2": right,
                "rank_difference": abs(left_rank - right_rank),
                "critical_difference": cd,
                "significant_nemenyi": abs(left_rank - right_rank) > cd,
            }
        )
    summary = {
        "n_folds": n_blocks,
        "n_architectures": n_models,
        "friedman_chi2": float(chi2),
        "p_friedman_chi2": float(p_chi2),
        "omnibus_significant_friedman": float(p_chi2) < ALPHA,
        "iman_davenport_f": float(f_stat),
        "p_iman_davenport": p_iman,
        "omnibus_significant_iman_davenport": p_iman < ALPHA,
        "nemenyi_critical_difference": float(cd),
    }
    return summary, rank_table, pd.DataFrame(pair_rows)


def analyze_configuration(
    key: str,
) -> list[tuple[str, pd.DataFrame, dict[str, object], pd.DataFrame, pd.DataFrame]]:
    training_label, root, datasets = CONFIGS[key]
    results = []
    for dataset in datasets:
        scores = scores_for_dataset(root, dataset)
        summary, ranks, pairs = analyze(scores)
        results.append((dataset, scores, summary, ranks, pairs))
    return results


def write_configuration(
    key: str,
    results: list[
        tuple[str, pd.DataFrame, dict[str, object], pd.DataFrame, pd.DataFrame]
    ],
) -> None:
    training_label = CONFIGS[key][0]
    output = TABLE_ROOT / "statistical_tests" / "friedman_nemenyi" / f"{key}_trained"
    summaries = []
    for dataset, scores, summary, ranks, pairs in results:
        summaries.append(
            {"training": training_label, "evaluation_dataset": dataset, **summary}
        )
        atomic_to_csv(
            scores,
            output / "fold_scores" / f"macro_f1_by_fold_{dataset}.csv",
            float_format="%.6f",
        )
        atomic_to_csv(
            ranks,
            output / "average_ranks" / f"average_ranks_{dataset}.csv",
            index=False,
            float_format="%.6f",
        )
        posthoc_path = output / "posthoc" / f"nemenyi_{dataset}.csv"
        if summary["omnibus_significant_friedman"]:
            atomic_to_csv(
                pairs,
                posthoc_path,
                index=False,
                float_format="%.6f",
            )
        else:
            # A post-hoc table from an older significant run would otherwise
            # falsely look current after the omnibus result becomes null.
            posthoc_path.unlink(missing_ok=True)
    atomic_to_csv(
        pd.DataFrame(summaries),
        output / "friedman_summary.csv",
        index=False,
        float_format="%.6f",
    )


def run(key: str) -> None:
    """Compute then publish one explicitly requested training configuration."""
    write_configuration(key, analyze_configuration(key))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training", choices=["piid", "humc", "both"], default="both")
    args = parser.parse_args()
    keys = list(CONFIGS) if args.training == "both" else [args.training]
    # Validate and compute the complete requested manuscript run before any
    # existing result is replaced.
    results = {key: analyze_configuration(key) for key in keys}
    for key in keys:
        write_configuration(key, results[key])
    print("[DONE] Friedman--Nemenyi analysis complete")


if __name__ == "__main__":
    main()
