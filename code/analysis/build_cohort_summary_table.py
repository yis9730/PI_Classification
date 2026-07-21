"""Build a Table 1-ready cohort and split summary without exposing private rows.

By default, the script uses only public PIID split metadata and locally prepared
public Kaggle folders. A project owner with authorised HUMC access may provide a
local aggregate-only HUMC JSON with ``--humc-meta``; no HUMC metadata is
distributed in this package.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


STAGES = ("1", "2", "3", "4")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def read_json(path: Path) -> dict[str, object]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def stage_counts_from_folder(root: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    for stage in STAGES:
        folder = root / stage
        if not folder.is_dir():
            raise FileNotFoundError(f"Prepared stage folder not found: {folder}")
        counts[stage] = sum(
            path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
            for path in folder.iterdir()
        )
    return counts


def metadata_rows(metadata: dict[str, object], source_access: str) -> list[dict[str, object]]:
    distribution = metadata["stage_distribution"]
    rows = []
    for split_set, key in (("All", "total"), ("Train-validation", "trainval"), ("Held-out test", "test")):
        stage_distribution = distribution[key]
        rows.append(
            {
                "dataset": metadata["dataset"],
                "access": source_access,
                "analysis_set": split_set,
                "split_unit": metadata["split_unit"],
                "n_images": sum(int(stage_distribution[stage]) for stage in STAGES),
                "n_patients": metadata.get(f"{key}_patients", ""),
                **{f"stage_{stage}": int(stage_distribution[stage]) for stage in STAGES},
            }
        )
    return rows


def parse_args() -> argparse.Namespace:
    root = repo_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--piid-meta",
        type=Path,
        default=root / "data" / "splits" / "piid" / "piid_split_meta.json",
    )
    parser.add_argument(
        "--humc-meta",
        type=Path,
        help="Optional local aggregate-only HUMC summary JSON; never included in the public release.",
    )
    parser.add_argument(
        "--kaggle-root",
        type=Path,
        default=root / "data" / "kaggle",
        help="Prepared Kaggle folder containing stage directories 1/2/3/4.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "data" / "results" / "tables" / "main_artifacts" / "table_1_cohort_summary.csv",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    piid = read_json(args.piid_meta)
    kaggle_counts = stage_counts_from_folder(args.kaggle_root)

    rows = metadata_rows(piid, "public")
    if args.humc_meta is not None:
        rows.extend(metadata_rows(read_json(args.humc_meta), "controlled; local summary"))
    rows.append(
        {
            "dataset": "Kaggle",
            "access": "public",
            "analysis_set": "External validation",
            "split_unit": "not split by this study",
            "n_images": sum(kaggle_counts.values()),
            "n_patients": "",
            **{f"stage_{stage}": kaggle_counts[stage] for stage in STAGES},
        }
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[DONE] Table 1 cohort summary written to {args.output}")


if __name__ == "__main__":
    main()
