"""Extract the frozen 512-D ResNet-18 features used in feature-space analyses.

Unlike duplicate screening, this analysis preserves the raw pooled 512-D
vectors (no L2 normalization) and directly resizes images to 224 x 224. These
vectors are shared by the UMAP, silhouette, centroid-distance, and
representative-image analyses.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import timm
import torch
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

CODE_ROOT = Path(__file__).resolve().parents[1]
CORE_DIR = CODE_ROOT / "core"
if str(CORE_DIR) not in sys.path:
    sys.path.insert(0, str(CORE_DIR))

from path_config import RESNET18_FEATURE_WEIGHT, TABLE_ROOT  # noqa: E402

RANDOM_SEED = 40
EXPECTED_WEIGHT_SHA256 = "69E2B9D2711F7CFB70B67091D16027EFAA781BCE78E1084A780AC1D1839B82F9"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def unwrap_state_dict(state):
    if isinstance(state, dict):
        for key in ("model", "state_dict", "model_state_dict"):
            if key in state and isinstance(state[key], dict):
                state = state[key]
                break
    return {str(key).removeprefix("module."): value for key, value in state.items()}


def load_model(weights: Path, device: torch.device, verify_hash: bool) -> torch.nn.Module:
    if not weights.exists():
        raise FileNotFoundError(f"ResNet-18 checkpoint not found: {weights}")
    observed = file_sha256(weights)
    if verify_hash and observed != EXPECTED_WEIGHT_SHA256:
        raise ValueError(f"Checkpoint SHA-256 {observed} != expected {EXPECTED_WEIGHT_SHA256}")
    model = timm.create_model("resnet18", pretrained=False, num_classes=0)
    model.load_state_dict(unwrap_state_dict(torch.load(weights, map_location="cpu")), strict=False)
    model.eval().to(device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if int(model.num_features) != 512:
        raise RuntimeError(f"Expected 512 features, got {model.num_features}")
    return model


class ImageDataset(Dataset):
    def __init__(self, paths: list[str]):
        self.paths = paths
        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> torch.Tensor:
        with Image.open(self.paths[index]) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
            return self.transform(image)


def collect(root: Path, dataset: str) -> pd.DataFrame:
    rows = []
    for stage in range(1, 5):
        folder = root / str(stage)
        if not folder.exists():
            continue
        for path in sorted(folder.iterdir()):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                rows.append(
                    {
                        "dataset": dataset,
                        "stage": stage,
                        "image_id": path.stem,
                        "image_path": str(path.resolve()),
                    }
                )
    table = pd.DataFrame(rows)
    if table.empty:
        raise RuntimeError(f"No images found for {dataset}: {root}")
    return table


@torch.no_grad()
def extract(
    model: torch.nn.Module,
    paths: list[str],
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> np.ndarray:
    loader = DataLoader(
        ImageDataset(paths),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    output = []
    for images in tqdm(loader, desc="Raw 512-D features"):
        output.append(model(images.to(device)).cpu().numpy())
    return np.vstack(output).astype(np.float32)


def parse_dataset_specs(specs: list[str]) -> list[tuple[str, Path]]:
    parsed = []
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"Expected NAME=PATH, received: {spec}")
        name, path = spec.split("=", 1)
        parsed.append((name, Path(path).expanduser().resolve()))
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", action="append", required=True, metavar="NAME=PATH")
    parser.add_argument("--weights", type=Path, default=RESNET18_FEATURE_WEIGHT)
    parser.add_argument("--output-dir", type=Path, default=TABLE_ROOT / "feature_space" / "features")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--skip-weight-hash-check", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.weights, device, not args.skip_weight_hash_check)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for dataset, root in parse_dataset_specs(args.dataset):
        metadata = collect(root, dataset)
        features = extract(
            model,
            metadata["image_path"].tolist(),
            device,
            args.batch_size,
            args.num_workers,
        )
        dataset_dir = args.output_dir / dataset
        dataset_dir.mkdir(parents=True, exist_ok=True)
        np.save(dataset_dir / "features.npy", features)
        metadata.to_csv(dataset_dir / "metadata.csv", index=False)
        (dataset_dir / "extraction.json").write_text(
            json.dumps(
                {
                    "encoder": "timm resnet18",
                    "frozen": True,
                    "pretraining": "ImageNet",
                    "checkpoint_sha256": file_sha256(args.weights),
                    "feature_dimension": 512,
                    "resize": "direct 224 x 224",
                    "normalization": "ImageNet mean/std",
                    "l2_normalized": False,
                    "random_seed": RANDOM_SEED,
                    "n_images": len(metadata),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"[DONE] {dataset}: {features.shape} -> {dataset_dir}")


if __name__ == "__main__":
    main()
