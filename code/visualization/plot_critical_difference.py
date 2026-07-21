"""Plot critical-difference diagrams from fold-wise rank-analysis outputs.

The displayed omnibus p-value and the decision to show Nemenyi cliques both
use the Iman--Davenport result, avoiding mismatched decision and display tests.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CODE_ROOT = Path(__file__).resolve().parents[1]
CORE_DIR = CODE_ROOT / "core"
if str(CORE_DIR) not in sys.path:
    sys.path.insert(0, str(CORE_DIR))

from path_config import FIGURE_ROOT, TABLE_ROOT  # noqa: E402


def parse_bool(value) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return str(value).strip().lower() in {"true", "1", "yes"}


def nonsignificant_intervals(ranks: np.ndarray, cd: float) -> list[tuple[int, int]]:
    intervals = []
    for left in range(len(ranks)):
        for right in range(left + 1, len(ranks)):
            if ranks[right] - ranks[left] <= cd:
                intervals.append((left, right))
    maximal = []
    for candidate in intervals:
        if not any(
            other != candidate and other[0] <= candidate[0] and other[1] >= candidate[1]
            for other in intervals
        ):
            maximal.append(candidate)
    return maximal


def draw_diagram(ranks_table: pd.DataFrame, summary: pd.Series, title: str, output: Path) -> None:
    ranks_table = ranks_table.sort_values("average_rank").reset_index(drop=True)
    names = ranks_table["model"].tolist()
    ranks = ranks_table["average_rank"].to_numpy(dtype=float)
    cd = float(summary["nemenyi_critical_difference"])
    show_cliques = parse_bool(summary["omnibus_significant_iman_davenport"])
    p_value = float(summary["p_iman_davenport"])

    fig, axis = plt.subplots(figsize=(10, 4.8))
    axis.set_xlim(0.5, 6.5)
    axis.set_ylim(0, 1)
    axis.axis("off")
    axis.plot([1, 6], [0.74, 0.74], color="black", linewidth=1.2)
    for rank in range(1, 7):
        axis.plot([rank, rank], [0.72, 0.76], color="black", linewidth=1)
        axis.text(rank, 0.80, str(rank), ha="center", va="bottom")

    left_indices = list(range((len(names) + 1) // 2))
    right_indices = list(range((len(names) + 1) // 2, len(names)))
    for order, index in enumerate(left_indices):
        y = 0.62 - order * 0.12
        axis.plot([ranks[index], 0.7], [0.74, y], color="black", linewidth=0.8)
        axis.text(0.65, y, f"{names[index]} ({ranks[index]:.2f})", ha="right", va="center")
    for order, index in enumerate(right_indices):
        y = 0.62 - order * 0.12
        axis.plot([ranks[index], 6.3], [0.74, y], color="black", linewidth=0.8)
        axis.text(6.35, y, f"({ranks[index]:.2f}) {names[index]}", ha="left", va="center")

    if show_cliques:
        for level, (left, right) in enumerate(nonsignificant_intervals(ranks, cd)):
            y = 0.68 - level * 0.035
            axis.plot([ranks[left], ranks[right]], [y, y], color="#b2182b", linewidth=4, solid_capstyle="round")

    p_text = "P < .001" if p_value < 0.001 else f"P = {p_value:.3f}".replace("0.", ".")
    axis.text(3.5, 0.96, title, ha="center", va="top", fontsize=13, fontweight="bold")
    axis.text(3.5, 0.89, f"Iman--Davenport {p_text}; CD = {cd:.3f}", ha="center", va="top")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def run(training: str) -> None:
    analysis_dir = TABLE_ROOT / "statistical_tests" / "friedman_nemenyi" / f"{training}_trained"
    summary = pd.read_csv(analysis_dir / "friedman_summary.csv")
    output_dir = FIGURE_ROOT / "critical_difference" / f"{training}_trained"
    for _, row in summary.iterrows():
        dataset = row["evaluation_dataset"]
        ranks = pd.read_csv(analysis_dir / "average_ranks" / f"average_ranks_{dataset}.csv")
        draw_diagram(
            ranks,
            row,
            f"{training.upper()}-trained models on {dataset}",
            output_dir / f"cd_macro_f1_{dataset}.png",
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training", choices=["piid", "humc", "both"], default="both")
    args = parser.parse_args()
    for training in ["piid", "humc"] if args.training == "both" else [args.training]:
        run(training)
    print("[DONE] Critical-difference figures complete")


if __name__ == "__main__":
    main()
