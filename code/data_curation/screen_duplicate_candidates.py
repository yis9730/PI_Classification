"""Supplementary screen for within- and cross-dataset duplicate candidates.

Images are encoded by a frozen ResNet-18, L2-normalized, and screened at cosine
similarity >= 0.85. Each feature candidate is also compared after resizing RGB
pixels to 128 x 128; normalized MAE <= 0.15 is reported as a corroborating
pixel-screen flag. This utility does not make exclusion decisions. The released
pair decisions and exclusion manifests are the authoritative curation record.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import timm
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

CODE_ROOT = Path(__file__).resolve().parents[1]
CORE_DIR = CODE_ROOT / "core"
if str(CORE_DIR) not in sys.path:
    sys.path.insert(0, str(CORE_DIR))

from path_config import RESNET18_FEATURE_WEIGHT  # noqa: E402

EXPECTED_WEIGHT_SHA256 = "69E2B9D2711F7CFB70B67091D16027EFAA781BCE78E1084A780AC1D1839B82F9"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def collect_images(dataset_specs: list[str]) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for spec in dataset_specs:
        if "=" not in spec:
            raise ValueError(f"Expected NAME=PATH, received: {spec}")
        dataset, raw_path = spec.split("=", 1)
        root = Path(raw_path).expanduser().resolve()
        for stage in range(1, 5):
            folder = root / str(stage)
            if not folder.exists():
                continue
            for path in sorted(folder.iterdir()):
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                    records.append(
                        {
                            "dataset": dataset,
                            "stage": stage,
                            "image_path": str(path),
                            "image_id": path.stem,
                        }
                    )
    table = pd.DataFrame(records)
    if table.empty:
        raise RuntimeError("No images found in the supplied dataset folders")
    return table


class FeatureDataset(Dataset):
    def __init__(self, paths: list[str]):
        self.paths = paths
        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> torch.Tensor:
        with Image.open(self.paths[index]) as image:
            image = image.convert("RGB")
            return self.transform(image)


def unwrap_state_dict(state):
    if isinstance(state, dict):
        for key in ("model", "state_dict", "model_state_dict"):
            if key in state and isinstance(state[key], dict):
                state = state[key]
                break
    return {str(key).removeprefix("module."): value for key, value in state.items()}


def load_encoder(weights: Path, device: torch.device, verify_hash: bool) -> torch.nn.Module:
    if not weights.exists():
        raise FileNotFoundError(
            f"ResNet-18 checkpoint not found: {weights}. See docs/RESNET18_FEATURES.md."
        )
    observed_hash = sha256(weights)
    if verify_hash and observed_hash != EXPECTED_WEIGHT_SHA256:
        raise ValueError(
            f"Unexpected ResNet-18 checkpoint SHA-256: {observed_hash}; "
            f"expected {EXPECTED_WEIGHT_SHA256}"
        )
    model = timm.create_model("resnet18", pretrained=False, num_classes=0)
    model.load_state_dict(unwrap_state_dict(torch.load(weights, map_location="cpu")), strict=False)
    model.eval().to(device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


@torch.no_grad()
def extract_features(
    model: torch.nn.Module,
    paths: list[str],
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> np.ndarray:
    loader = DataLoader(
        FeatureDataset(paths),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    batches = []
    for images in tqdm(loader, desc="ResNet-18 features"):
        features = model(images.to(device))
        features = torch.nn.functional.normalize(features, p=2, dim=1)
        batches.append(features.cpu().numpy())
    return np.vstack(batches).astype(np.float32)


def pixel_array(path: str, size: int) -> np.ndarray:
    with Image.open(path) as image:
        image = image.convert("RGB")
        image = image.resize((size, size), Image.Resampling.LANCZOS)
        return np.asarray(image, dtype=np.float32) / 255.0


def candidate_pairs(
    table: pd.DataFrame,
    features: np.ndarray,
    cosine_threshold: float,
    mae_threshold: float,
    pixel_size: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    similarity = features @ features.T
    for left in tqdm(range(len(table)), desc="Candidate pairs"):
        candidates = np.where(similarity[left, left + 1 :] >= cosine_threshold)[0]
        if not len(candidates):
            continue
        left_pixels = pixel_array(table.iloc[left]["image_path"], pixel_size)
        for offset in candidates:
            right = left + 1 + int(offset)
            right_pixels = pixel_array(table.iloc[right]["image_path"], pixel_size)
            mae = float(np.mean(np.abs(left_pixels - right_pixels)))
            pixel_screen_passed = mae <= mae_threshold
            a, b = table.iloc[left], table.iloc[right]
            rows.append(
                {
                    "dataset_1": a["dataset"],
                    "stage_1": int(a["stage"]),
                    "image_1": a["image_id"],
                    "path_1": a["image_path"],
                    "dataset_2": b["dataset"],
                    "stage_2": int(b["stage"]),
                    "image_2": b["image_id"],
                    "path_2": b["image_path"],
                    "cosine_similarity": float(similarity[left, right]),
                    "normalized_mae": mae,
                    "pixel_similarity": 1.0 - mae,
                    "pixel_screen_passed": pixel_screen_passed,
                    "candidate_basis": (
                        "resnet18_cosine_and_pixel"
                        if pixel_screen_passed
                        else "resnet18_cosine_only"
                    ),
                    "requires_expert_review": True,
                }
            )
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        action="append",
        required=True,
        metavar="NAME=PATH",
        help="Analytic stage-folder dataset; repeat for PIID, Kaggle, and authorized HUMC.",
    )
    parser.add_argument("--weights", type=Path, default=RESNET18_FEATURE_WEIGHT)
    parser.add_argument("--output", type=Path, default=Path("duplicate_candidates.csv"))
    parser.add_argument("--cosine-threshold", type=float, default=0.85)
    parser.add_argument("--mae-threshold", type=float, default=0.15)
    parser.add_argument("--pixel-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--skip-weight-hash-check", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    table = collect_images(args.dataset)
    model = load_encoder(args.weights, device, not args.skip_weight_hash_check)
    features = extract_features(
        model, table["image_path"].tolist(), device, args.batch_size, args.num_workers
    )
    pairs = candidate_pairs(
        table, features, args.cosine_threshold, args.mae_threshold, args.pixel_size
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pairs.to_csv(args.output, index=False)
    print(f"[DONE] {len(pairs)} candidate pairs written to {args.output}")


if __name__ == "__main__":
    main()
