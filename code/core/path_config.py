"""Portable paths for the public reproduction package.

All paths are resolved relative to the repository root. Set the optional
``PI_PROJECT_ROOT`` environment variable only when scripts are launched from
outside the repository.
"""

from __future__ import annotations

import os
from pathlib import Path


def find_project_root() -> Path:
    """Return the nearest ancestor containing both code/ and data/."""
    candidates: list[Path] = []
    env_root = os.environ.get("PI_PROJECT_ROOT")
    if env_root:
        candidates.append(Path(env_root).expanduser())
    candidates.extend([Path(__file__).resolve(), Path.cwd().resolve()])
    for start in candidates:
        base = start if start.is_dir() else start.parent
        for candidate in [base, *base.parents]:
            if (candidate / "code").exists() and (candidate / "data").exists():
                return candidate
    raise RuntimeError(
        "Project root not found. Run from inside the repository or set "
        "PI_PROJECT_ROOT to the folder containing code/ and data/."
    )


PROJECT_ROOT = find_project_root()
CODE_ROOT = PROJECT_ROOT / "code"
DATA_ROOT = PROJECT_ROOT / "data"

PIID_DATA_DIR = DATA_ROOT / "piid"
KAGGLE_DATA_DIR = DATA_ROOT / "kaggle"
HUMC_DATA_DIR = DATA_ROOT / "humc"
HUMC_LABELING_PATH = HUMC_DATA_DIR / "labels.xlsx"

SPLIT_ROOT = DATA_ROOT / "splits"
PIID_SPLIT_DIR = SPLIT_ROOT / "piid"
HUMC_SPLIT_DIR = SPLIT_ROOT / "humc"

RESULTS_ROOT = DATA_ROOT / "results"
CHECKPOINT_ROOT = RESULTS_ROOT / "checkpoints"
PIID_CHECKPOINT_DIR = CHECKPOINT_ROOT / "piid_trained"
HUMC_CHECKPOINT_DIR = CHECKPOINT_ROOT / "humc_trained"
FEATURE_EXTRACTOR_CHECKPOINT_DIR = CHECKPOINT_ROOT / "feature_extractors"
RESNET18_FEATURE_WEIGHT = FEATURE_EXTRACTOR_CHECKPOINT_DIR / "resnet18.pth"

PREDICTION_ROOT = RESULTS_ROOT / "predictions"
PIID_INFERENCE_DIR = PREDICTION_ROOT / "piid"
HUMC_INFERENCE_DIR = PREDICTION_ROOT / "humc"
TABLE_ROOT = RESULTS_ROOT / "tables"
FIGURE_ROOT = RESULTS_ROOT / "figures"


def project_path(relative_path: str | Path) -> Path:
    """Resolve a repository-relative path unless it is already absolute."""
    path_obj = Path(relative_path)
    return path_obj if path_obj.is_absolute() else PROJECT_ROOT / path_obj


def project_relative_path(path: str | Path) -> str:
    """Return a stable repository-relative path string when possible."""
    resolved = Path(path).expanduser()
    if not resolved.is_absolute():
        return resolved.as_posix()
    try:
        return resolved.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(resolved)


def resolve_project_path(path: str | Path) -> Path:
    """Resolve an image path stored in a split CSV."""
    return project_path(path)


def resolve_project_paths(paths) -> list[str]:
    """Resolve image paths into strings accepted by image loaders."""
    return [str(resolve_project_path(path)) for path in paths]
