"""Build manuscript Table 1 from its approved, aggregate-only source values.

The released source contains one row per dataset and only the image counts and
stage percentages already reported in Table 1. It contains no patient row,
identifier, image path, split membership, or controlled clinical attribute.
The HUMC row therefore reconstructs the published table; it does not rederive
the values from private HUMC records.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


DATASETS = ("PIID", "HUMC", "Kaggle")
STAGES = ("1", "2", "3", "4")
SOURCE_FIELDS = (
    "dataset",
    "initial_images",
    "excluded_images",
    "final_images",
    *(field for stage in STAGES for field in (f"stage_{stage}_count", f"stage_{stage}_percent")),
)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def read_aggregate_source(path: Path) -> dict[str, dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != SOURCE_FIELDS:
            raise ValueError(
                f"Unexpected Table 1 source columns in {path}: {reader.fieldnames}"
            )
        source_rows = list(reader)

    if len(source_rows) != len(DATASETS):
        raise ValueError(
            f"Expected exactly one source row per dataset; found {len(source_rows)} rows"
        )
    rows: dict[str, dict[str, str]] = {}
    for row in source_rows:
        dataset = row["dataset"]
        if dataset in rows:
            raise ValueError(f"Duplicate Table 1 source row: {dataset}")
        rows[dataset] = row

    if set(rows) != set(DATASETS):
        raise ValueError(f"Expected exactly {DATASETS}; found {tuple(rows)}")

    for dataset, row in rows.items():
        initial = int(row["initial_images"])
        excluded = int(row["excluded_images"])
        final = int(row["final_images"])
        stage_total = sum(int(row[f"stage_{stage}_count"]) for stage in STAGES)
        if initial - excluded != final:
            raise ValueError(f"{dataset}: initial - excluded does not equal final")
        if stage_total != final:
            raise ValueError(f"{dataset}: stage counts do not sum to final_images")
        for stage in STAGES:
            count = int(row[f"stage_{stage}_count"])
            reported = float(row[f"stage_{stage}_percent"])
            calculated = 100.0 * count / final
            if abs(reported - calculated) > 0.11:
                raise ValueError(
                    f"{dataset} stage {stage}: reported percentage is inconsistent"
                )
    return rows


def build_table(rows: dict[str, dict[str, str]]) -> list[dict[str, str]]:
    output = []
    for label, field in (
        ("Initial image set, count", "initial_images"),
        ("Images excluded during curation, count", "excluded_images"),
        ("Final image set, count", "final_images"),
    ):
        output.append(
            {"Characteristic": label, **{dataset: rows[dataset][field] for dataset in DATASETS}}
        )
    output.append(
        {
            "Characteristic": "Pressure injury stage distribution, count (%)",
            **{dataset: "" for dataset in DATASETS},
        }
    )
    for stage in STAGES:
        output.append(
            {
                "Characteristic": f"Stage {stage}",
                **{
                    dataset: (
                        f'{rows[dataset][f"stage_{stage}_count"]} '
                        f'({rows[dataset][f"stage_{stage}_percent"]})'
                    )
                    for dataset in DATASETS
                },
            }
        )
    return output


def parse_args() -> argparse.Namespace:
    root = repo_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=root / "data" / "aggregates" / "table_1_cohort_counts.csv",
        help="Approved aggregate-only source matching manuscript Table 1.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "data" / "results" / "tables" / "main_artifacts" / "table_1_cohort_summary.csv",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    table_rows = build_table(read_aggregate_source(args.source))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("Characteristic", *DATASETS))
        writer.writeheader()
        writer.writerows(table_rows)
    print(f"[DONE] Table 1 cohort summary written to {args.output}")


if __name__ == "__main__":
    main()
