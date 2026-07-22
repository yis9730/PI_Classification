"""Validate the repository before creating a public Git commit."""

from __future__ import annotations

import ast
import csv
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TEXT_SUFFIXES = {".py", ".md", ".txt", ".csv", ".json", ".yml", ".yaml"}
REQUIRED = [
    "requirements.txt",
    "requirements_train_eval.txt",
    "requirements_umap_analysis.txt",
    "README.md",
    "code/data_curation/duplicate_pairs.csv",
    "code/data_curation/piid_duplicate_exclusions.csv",
    "code/data_curation/kaggle_duplicate_exclusions.csv",
    "code/core/model_pipeline_utils.py",
    "code/pipeline/train_piid_6models_17augmentations.py",
    "code/pipeline/train_humc_6models_17augmentations.py",
    "code/pipeline/evaluate_piid_trained_final_results.py",
    "code/pipeline/evaluate_humc_trained_final_results.py",
    "code/check_checkpoint_compatibility.py",
    "code/analysis/bootstrap_macro_f1_foldwise.py",
    "code/analysis/build_cohort_summary_table.py",
    "code/analysis/feature_space_statistics.py",
    "code/analysis/friedman_nemenyi_foldwise.py",
    "code/visualization/plot_critical_difference.py",
    "code/visualization/plot_centroid_montage.py",
    "code/visualization/plot_sankey_fold_averaged.py",
    "code/visualization/plot_umap.py",
    "docs/ENVIRONMENTS.md",
    "docs/HUMC_PRIVATE_DATA.md",
    "docs/MAIN_ARTIFACTS.md",
    "docs/REPOSITORY_ARCHITECTURE.md",
    "docs/REPRODUCTION_WORKFLOW.md",
    "docs/RESNET18_FEATURES.md",
]

GENERATED_RUNTIME_PREFIXES = (
    ("data", "results", "checkpoints"),
    ("data", "results", "figures"),
    ("data", "results", "manifests"),
    ("data", "results", "predictions"),
    ("data", "results", "tables"),
)


def text_files() -> list[Path]:
    return [
        path for path in ROOT.rglob("*")
        if path.is_file() and path.suffix.lower() in TEXT_SUFFIXES
        and not is_generated_runtime_file(path)
    ]


def is_generated_runtime_file(path: Path) -> bool:
    relative_parts = path.relative_to(ROOT).parts
    return any(
        relative_parts[:len(prefix)] == prefix
        for prefix in GENERATED_RUNTIME_PREFIXES
    )


def augmentation_count(path: Path) -> int | None:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if isinstance(node, ast.Assign):
            names = [target.id for target in node.targets if isinstance(target, ast.Name)]
            if "AUGMENTATION_CONFIGS" in names and isinstance(node.value, ast.Dict):
                return len(node.value.keys)
    return None


def assigned_integer(tree: ast.Module, name: str) -> int | None:
    """Return a module-level integer assignment, if present."""
    for node in tree.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        value = node.value
        if any(isinstance(target, ast.Name) and target.id == name for target in targets):
            if isinstance(value, ast.Constant) and isinstance(value.value, int):
                return value.value
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

    train_requirements = (ROOT / "requirements_train_eval.txt").read_text(encoding="utf-8")
    umap_requirements = (ROOT / "requirements_umap_analysis.txt").read_text(encoding="utf-8")
    if "torch==2.9.0" not in train_requirements or "torchvision==0.24.0" not in train_requirements:
        failures.append("PyTorch/TorchVision release versions are not locked as expected")
    if "numpy==2.2.6" not in train_requirements or "opencv-python==4.12.0.88" not in train_requirements:
        failures.append("training/evaluation NumPy/OpenCV versions are not locked as expected")
    if "numpy==1.26.4" not in umap_requirements or "umap-learn==0.5.6" not in umap_requirements:
        failures.append("UMAP environment is not locked as expected")

    feature_source = (ROOT / "code/analysis/extract_resnet18_features.py").read_text(encoding="utf-8")
    duplicate_source = (ROOT / "code/data_curation/screen_duplicate_candidates.py").read_text(encoding="utf-8")
    curation_source = (ROOT / "code/data_curation/prepare_public_datasets.py").read_text(encoding="utf-8")
    for label, source in (
        ("manuscript feature extraction", feature_source),
        ("duplicate screening", duplicate_source),
    ):
        if "transforms.Resize((224, 224))" not in source:
            failures.append(f"{label} must resize model inputs directly to 224 x 224")
        if "transforms.Resize(256)" in source or "transforms.CenterCrop" in source:
            failures.append(f"{label} must not use a 256 resize or center crop")

    if "shutil.copy2(src, dst)" not in curation_source:
        failures.append("PIID curation must copy retained source files unchanged")
    for snippet in (
        "crop_kaggle_native_square",
        "side = min(width, height)",
        "image.crop((left, top, left + side, top + side))",
    ):
        if snippet not in curation_source:
            failures.append(f"Kaggle native centre-square curation is missing: {snippet}")
    for forbidden in (
        "ImageOps.exif_transpose",
        "image.resize(",
        "Resize(256)",
    ):
        if forbidden in curation_source:
            failures.append(f"public-data curation contains an unreported transform: {forbidden}")
    for required_guard in (
        "validate_source_output_separation",
        "paths_overlap",
        "matched_exclusions",
        "expected_raw",
    ):
        if required_guard not in curation_source:
            failures.append(f"public-data curation safety guard is missing: {required_guard}")

    classification_paths = (
        "code/pipeline/dataset_split_normalization_piid_main.py",
        "code/pipeline/dataset_split_normalization_humc_patient_level.py",
        "code/pipeline/train_piid_6models_17augmentations.py",
        "code/pipeline/train_humc_6models_17augmentations.py",
        "code/pipeline/evaluate_piid_trained_final_results.py",
        "code/pipeline/evaluate_humc_trained_final_results.py",
    )
    classification_sources = {}
    for relative in classification_paths:
        path = ROOT / relative
        source = path.read_text(encoding="utf-8")
        classification_sources[relative] = source
        tree = ast.parse(source, filename=str(path))
        if assigned_integer(tree, "INPUT_SIZE") != 224:
            failures.append(f"{relative}: INPUT_SIZE must be the integer 224")
        if "ImageOps.exif_transpose" in source:
            failures.append(f"{relative}: unreported EXIF-orientation transform is present")

    required_resize_snippets = {
        "code/pipeline/dataset_split_normalization_piid_main.py":
            "self.resize = A.Compose([A.Resize(input_size, input_size)])",
        "code/pipeline/dataset_split_normalization_humc_patient_level.py":
            "self.transform = A.Compose([A.Resize(INPUT_SIZE, INPUT_SIZE)])",
        "code/pipeline/evaluate_piid_trained_final_results.py":
            "A.Resize(INPUT_SIZE, INPUT_SIZE)",
        "code/pipeline/evaluate_humc_trained_final_results.py":
            "A.Resize(INPUT_SIZE, INPUT_SIZE)",
    }
    for relative, snippet in required_resize_snippets.items():
        source = classification_sources[relative]
        if snippet not in source or source.count("A.Resize(") != 1:
            failures.append(f"{relative}: expected one direct Albumentations resize")
        if "CenterCrop" in source:
            failures.append(f"{relative}: evaluation/normalisation must not center-crop images")

    for relative in (
        "code/pipeline/train_piid_6models_17augmentations.py",
        "code/pipeline/train_humc_6models_17augmentations.py",
    ):
        source = classification_sources[relative]
        if "transforms = [A.Resize(INPUT_SIZE, INPUT_SIZE)]" not in source:
            failures.append(f"{relative}: training must begin with direct 224 x 224 resize")
        if source.count("A.Resize(INPUT_SIZE, INPUT_SIZE)") != 3:
            failures.append(f"{relative}: train, centre-zoom, and eval resize contract changed")
        if source.count("A.CenterCrop(") != 1 or "use_center_zoomin" not in source:
            failures.append(f"{relative}: CenterCrop must occur only in centre zoom-in augmentation")
        if "transforms.append(A.Flip(p=0.5))" not in source:
            failures.append(f"{relative}: study Flip augmentation contract changed")
        if "drop_last=True" not in source:
            failures.append(f"{relative}: training DataLoader must drop the final incomplete batch")

    model_source = (ROOT / "code/core/model_pipeline_utils.py").read_text(encoding="utf-8")
    if "torch.randn(2, 3, input_size, input_size)" not in model_source:
        failures.append("model feature-probe RNG contract changed")

    for relative in (
        "code/pipeline/evaluate_piid_trained_final_results.py",
        "code/pipeline/evaluate_humc_trained_final_results.py",
    ):
        source = classification_sources[relative]
        if "stage_folder_dataset_available" not in source:
            failures.append(f"{relative}: incomplete optional datasets must be skipped")
        if "model.load_state_dict(state, strict=True)" not in source:
            failures.append(f"{relative}: checkpoint loading must be explicitly strict")

    for relative in (
        "code/pipeline/train_piid_6models_17augmentations.py",
        "code/pipeline/train_humc_6models_17augmentations.py",
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
        if is_generated_runtime_file(path):
            continue
        try:
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (SyntaxError, UnicodeDecodeError) as exc:
            failures.append(f"Python parse failure in {path.relative_to(ROOT)}: {exc}")

    suspicious_private = [
        ROOT / "data/splits/humc/all_images.csv",
        ROOT / "data/splits/humc/trainval_set.csv",
        ROOT / "data/splits/humc/test_set.csv",
        ROOT / "data/splits/humc/fold_indices.json",
        ROOT / "data/splits/humc/split_meta.json",
        ROOT / "data/splits/humc/split_meta_public.json",
        ROOT / "data/splits/humc/normalization_stats.csv",
        ROOT / "data/templates/humc_label_template.csv",
    ]
    for path in suspicious_private:
        if path.exists():
            failures.append(f"private HUMC split material present: {path.relative_to(ROOT)}")

    controlled_roots = {
        ROOT / "data/humc": {".gitkeep"},
        ROOT / "data/splits/humc": set(),
        ROOT / "data/results": {"README.md"},
    }
    for folder, allowed_files in controlled_roots.items():
        if not folder.exists():
            continue
        for path in folder.rglob("*"):
            if path.is_file() and path.relative_to(folder).as_posix() not in allowed_files:
                failures.append(f"generated or controlled file present: {path.relative_to(ROOT)}")

    if failures:
        print("[FAIL] Public release validation")
        for failure in failures:
            print(f" - {failure}")
        raise SystemExit(1)

    print("[PASS] Public release validation")
    print(f" - parsed {len(list(ROOT.rglob('*.py')))} Python files")
    print(" - 17 training conditions in each training entry point")
    print(" - 20 duplicate pairs; 10 PIID and 18 Kaggle exclusions")
    print(" - PyTorch 2.9.0 / TorchVision 0.24.0 and two environment contracts locked")
    print(" - PIID is copied unchanged; Kaggle uses the native centre-square analytic crop")
    print(" - direct 224 x 224 model-input resize contracts verified")
    print(" - model-pipeline CenterCrop is confined to the centre zoom-in augmentation")
    print(" - no prohibited personal/server paths in the current scanned files")
    print(" - Git history and commit metadata require a separate release-owner review")


if __name__ == "__main__":
    main()
