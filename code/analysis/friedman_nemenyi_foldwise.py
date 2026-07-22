"""Dataset-specific fold-wise Friedman--Nemenyi architecture analysis.

Each evaluation dataset is analyzed separately. The five fold-specific
macro-F1 values are paired blocks and the six architectures are treatments.
The Iman--Davenport p-value is the omnibus decision used before the Nemenyi
post-hoc comparison. With only five overlapping folds, results are exploratory.
"""

from __future__ import annotations

import argparse
import itertools
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


def labels(values: pd.Series) -> np.ndarray:
    array = values.astype(int).to_numpy()
    unique = set(np.unique(array).tolist())
    if unique <= set(LABELS):
        return array
    if unique <= {1, 2, 3, 4}:
        return array - 1
    raise ValueError(f"Unexpected labels: {sorted(unique)}")


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
            signature = table[["image_path", "true_label"]].astype(str).to_numpy()
            if reference is None:
                reference = signature
            elif not np.array_equal(reference, signature):
                raise ValueError(f"Prediction alignment mismatch: {model}, {dataset}, fold {fold}")
            values.append(
                f1_score(
                    labels(table["true_label"]), labels(table["predicted_label"]),
                    labels=LABELS, average="macro", zero_division=0,
                )
            )
        score_columns[DISPLAY[model]] = values
    output = pd.DataFrame(score_columns, index=range(1, N_FOLDS + 1))
    output.index.name = "fold"
    return output


def analyze(scores: pd.DataFrame) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame]:
    n_blocks, n_models = scores.shape
    chi2, p_chi2 = friedmanchisquare(*[scores[column] for column in scores.columns])
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
        "iman_davenport_f": float(f_stat),
        "p_iman_davenport": p_iman,
        "omnibus_significant_iman_davenport": p_iman < ALPHA,
        "nemenyi_critical_difference": float(cd),
    }
    return summary, rank_table, pd.DataFrame(pair_rows)


def run(key: str) -> None:
    training_label, root, datasets = CONFIGS[key]
    output = TABLE_ROOT / "statistical_tests" / "friedman_nemenyi" / f"{key}_trained"
    (output / "fold_scores").mkdir(parents=True, exist_ok=True)
    (output / "average_ranks").mkdir(parents=True, exist_ok=True)
    (output / "posthoc").mkdir(parents=True, exist_ok=True)
    summaries = []
    for dataset in datasets:
        scores = scores_for_dataset(root, dataset)
        summary, ranks, pairs = analyze(scores)
        summaries.append({"training": training_label, "evaluation_dataset": dataset, **summary})
        scores.to_csv(output / "fold_scores" / f"macro_f1_by_fold_{dataset}.csv", float_format="%.6f")
        ranks.to_csv(output / "average_ranks" / f"average_ranks_{dataset}.csv", index=False, float_format="%.6f")
        if summary["omnibus_significant_iman_davenport"]:
            pairs.to_csv(output / "posthoc" / f"nemenyi_{dataset}.csv", index=False, float_format="%.6f")
    pd.DataFrame(summaries).to_csv(output / "friedman_summary.csv", index=False, float_format="%.6f")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training", choices=["piid", "humc", "both"], default="both")
    args = parser.parse_args()
    for key in CONFIGS if args.training == "both" else [args.training]:
        run(key)
    print("[DONE] Friedman--Nemenyi analysis complete")


if __name__ == "__main__":
    main()
