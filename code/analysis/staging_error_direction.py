"""Summarize under-staging, correct, and over-staging counts by fold/model."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

CODE_ROOT = Path(__file__).resolve().parents[1]
CORE_DIR = CODE_ROOT / "core"
if str(CORE_DIR) not in sys.path:
    sys.path.insert(0, str(CORE_DIR))

from path_config import HUMC_INFERENCE_DIR, PIID_INFERENCE_DIR, TABLE_ROOT  # noqa: E402

CONFIGS = {"piid": PIID_INFERENCE_DIR, "humc": HUMC_INFERENCE_DIR}


def summarize(root: Path, training: str) -> pd.DataFrame:
    rows = []
    for path in sorted(root.glob("*/predictions/*_fold*_predictions.csv")):
        run_name = path.parents[1].name
        dataset = path.name.split("_fold")[0]
        fold = int(path.stem.split("_fold")[-1].split("_")[0])
        table = pd.read_csv(path)
        true = table["true_label"].astype(int).to_numpy()
        predicted = table["predicted_label"].astype(int).to_numpy()
        if min(true.min(), predicted.min()) == 1:
            true, predicted = true - 1, predicted - 1
        under = int(np.sum(predicted < true))
        correct = int(np.sum(predicted == true))
        over = int(np.sum(predicted > true))
        errors = under + over
        rows.append(
            {
                "training": f"{training.upper()}-trained",
                "run_name": run_name,
                "dataset": dataset,
                "fold": fold,
                "n": len(table),
                "understaging_n": under,
                "correct_n": correct,
                "overstaging_n": over,
                "understaging_share_of_errors": under / errors if errors else 0.0,
                "overstaging_share_of_errors": over / errors if errors else 0.0,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training", choices=["piid", "humc", "both"], default="both")
    args = parser.parse_args()
    keys = list(CONFIGS) if args.training == "both" else [args.training]
    output = TABLE_ROOT / "model_error_analysis" / "staging_direction"
    output.mkdir(parents=True, exist_ok=True)
    for key in keys:
        table = summarize(CONFIGS[key], key)
        table.to_csv(output / f"{key}_trained_foldwise.csv", index=False)
    print(f"[DONE] Staging-direction tables written to {output}")


if __name__ == "__main__":
    main()
