"""Check checkpoint layout and strict state-dictionary compatibility.

This performs no training and no inference. By default it checks fold 1 of the
no-augmentation run for all six architectures in both training archives.
"""

from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

import torch

CODE_ROOT = Path(__file__).resolve().parent
DEVELOPMENT_DIR = CODE_ROOT / "development"
if str(DEVELOPMENT_DIR) not in sys.path:
    sys.path.insert(0, str(DEVELOPMENT_DIR))

from model_pipeline_utils import FINAL_BASELINE_BACKBONES, get_model  # noqa: E402
from path_config import CHECKPOINT_ROOT  # noqa: E402

AUGMENTATIONS = [
    "exp00_NoAug", "exp01_Flip", "exp02_Rotate90", "exp03a_RandomZoomIn",
    "exp03b_CenterZoomIn", "exp04_ZoomOut", "exp05_Brightness", "exp06_Contrast",
    "exp07_F_R", "exp08_F_R_ZI", "exp09_F_R_ZI_ZO", "exp10_F_R_ZI_ZO_B",
    "exp11_F_R_ZI_ZO_B_C", "exp12_F_R_CZI", "exp13_F_R_CZI_ZO",
    "exp14_F_R_CZI_ZO_B", "exp15_F_R_CZI_ZO_B_C",
]


def checkpoint_path(root: Path, training: str, model: str, augmentation: str, fold: int) -> Path:
    run = f"{model}_Baseline_{augmentation}_bs16_lr1e-05_wd1e-04"
    return root / f"{training}_trained" / run / "best_models_weights" / f"best_model_fold_{fold}.pth"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-root", type=Path, default=CHECKPOINT_ROOT)
    parser.add_argument("--training", nargs="+", choices=["piid", "humc"], default=["piid", "humc"])
    parser.add_argument("--models", nargs="+", choices=FINAL_BASELINE_BACKBONES, default=list(FINAL_BASELINE_BACKBONES))
    parser.add_argument("--augmentations", nargs="+", choices=AUGMENTATIONS, default=["exp00_NoAug"])
    parser.add_argument("--folds", nargs="+", type=int, default=[1])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checked = 0
    for model_name in args.models:
        model = get_model(model_name, num_classes=4, pretrained=False, input_size=224, dropout_rate=0.5)
        for training in args.training:
            for augmentation in args.augmentations:
                for fold in args.folds:
                    path = checkpoint_path(
                        args.checkpoint_root, training, model_name, augmentation, fold
                    )
                    if not path.is_file():
                        raise FileNotFoundError(path)
                    state = torch.load(path, map_location="cpu", weights_only=True)
                    model.load_state_dict(state, strict=True)
                    checked += 1
                    print(f"[OK] {path.relative_to(args.checkpoint_root)}")
                    del state
        del model
        gc.collect()
    print(f"[PASS] Strictly loaded {checked} checkpoints; no training was run")


if __name__ == "__main__":
    main()
