"""Validate the repository before creating a public Git commit."""

from __future__ import annotations

import ast
import csv
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TEXT_SUFFIXES = {".py", ".md", ".txt", ".csv", ".json", ".yml", ".yaml"}
REQUIRED = [
    "requirements.txt",
    "README.md",
    "CHECKSUMS.sha256",
    "code/data_curation/duplicate_pairs.csv",
    "code/data_curation/piid_duplicate_exclusions.csv",
    "code/data_curation/kaggle_duplicate_exclusions.csv",
    "code/experiment/train_piid_6models_17augmentations.py",
    "code/experiment/train_humc_6models_17augmentations.py",
    "code/experiment/evaluate_piid_trained_final_results.py",
    "code/experiment/evaluate_humc_trained_final_results.py",
    "code/check_checkpoint_compatibility.py",
    "code/generate_release_checksums.py",
    "code/analysis/bootstrap_macro_f1_foldwise.py",
    "code/analysis/friedman_nemenyi_foldwise.py",
    "code/visualization/plot_critical_difference.py",
    "code/visualization/plot_sankey_fold_averaged.py",
    "docs/HUMC_PRIVATE_DATA.md",
    "docs/RESNET18_FEATURES.md",
]


def text_files() -> list[Path]:
    return [
        path for path in ROOT.rglob("*")
        if path.is_file() and path.suffix.lower() in TEXT_SUFFIXES
    ]


def augmentation_count(path: Path) -> int | None:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if isinstance(node, ast.Assign):
            names = [target.id for target in node.targets if isinstance(target, ast.Name)]
            if "AUGMENTATION_CONFIGS" in names and isinstance(node.value, ast.Dict):
                return len(node.value.keys)
    return None


def validate_training_seed_contract(path: Path) -> list[str]:
    """Check the study-wide seed and fold-local loader shuffle contract."""
    content = path.read_text(encoding="utf-8")
    failures = []
    if "RANDOM_SEED = 40" not in content:
        failures.append(f"{path.name}: RANDOM_SEED is not fixed at 40")
    if "generator=torch.Generator().manual_seed(RANDOM_SEED + fold_id)" not in content:
        failures.append(f"{path.name}: fold-local DataLoader generator is missing")
    if "set_seed(RANDOM_SEED + fold_id)" in content:
        failures.append(f"{path.name}: full RNG state must not be reset to 40 + fold_id")
    return failures


def csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    failures: list[str] = []
    for relative in REQUIRED:
        if not (ROOT / relative).is_file():
            failures.append(f"missing required file: {relative}")

    requirements = (ROOT / "requirements.txt").read_text(encoding="utf-8")
    if "torch==2.9.0" not in requirements or "torchvision==0.24.0" not in requirements:
        failures.append("PyTorch/TorchVision release versions are not locked as expected")

    for relative in (
        "code/experiment/train_piid_6models_17augmentations.py",
        "code/experiment/train_humc_6models_17augmentations.py",
    ):
        count = augmentation_count(ROOT / relative)
        if count != 17:
            failures.append(f"{relative}: expected 17 augmentation configurations, found {count}")
        failures.extend(validate_training_seed_contract(ROOT / relative))

    pairs = csv_rows(ROOT / "code/data_curation/duplicate_pairs.csv")
    piid = csv_rows(ROOT / "code/data_curation/piid_duplicate_exclusions.csv")
    kaggle = csv_rows(ROOT / "code/data_curation/kaggle_duplicate_exclusions.csv")
    if len(pairs) != 20:
        failures.append(f"expected 20 duplicate pairs, found {len(pairs)}")
    if len(piid) != 10 or len(kaggle) != 18:
        failures.append(
            f"expected 10 PIID and 18 Kaggle exclusions, found {len(piid)} and {len(kaggle)}"
        )

    prohibited_fragments = [
        "C:" + chr(92) + "Users" + chr(92),
        "/storage" + "01/",
        "pny" + "235711",
        "yis" + "9730",
        "mc" + "nemar",
    ]
    for path in text_files():
        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            failures.append(f"non-UTF-8 text file: {path.relative_to(ROOT)}")
            continue
        folded = content.casefold()
        for fragment in prohibited_fragments:
            if fragment.casefold() in folded:
                failures.append(
                    f"prohibited legacy/private fragment in {path.relative_to(ROOT)}: {fragment}"
                )

    for path in ROOT.rglob("*.py"):
        try:
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (SyntaxError, UnicodeDecodeError) as exc:
            failures.append(f"Python parse failure in {path.relative_to(ROOT)}: {exc}")

    suspicious_private = [
        ROOT / "data/splits/humc_patient_level/all_images.csv",
        ROOT / "data/splits/humc_patient_level/trainval_set.csv",
        ROOT / "data/splits/humc_patient_level/test_set.csv",
        ROOT / "data/splits/humc_patient_level/fold_indices.json",
        ROOT / "data/splits/humc_patient_level/split_meta.json",
    ]
    for path in suspicious_private:
        if path.exists():
            failures.append(f"private HUMC split material present: {path.relative_to(ROOT)}")

    if failures:
        print("[FAIL] Public release validation")
        for failure in failures:
            print(f" - {failure}")
        raise SystemExit(1)

    print("[PASS] Public release validation")
    print(f" - parsed {len(list(ROOT.rglob('*.py')))} Python files")
    print(" - 17 training conditions in each training entry point")
    print(" - 20 duplicate pairs; 10 PIID and 18 Kaggle exclusions")
    print(" - PyTorch 2.9.0 / TorchVision 0.24.0 locked")
    print(" - no prohibited personal/server paths or legacy test references")


if __name__ == "__main__":
    main()
