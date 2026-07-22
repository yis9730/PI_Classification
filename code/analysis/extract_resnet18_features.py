"""Extract the frozen 512-D ResNet-18 features used in feature-space analyses.

The encoder is the official timm ``resnet18.a1_in1k`` model used by the study.
The complete public classifier checkpoint is downloaded and hash-verified, its
two classifier tensors are removed, and the remaining 120 encoder tensors are
loaded strictly. This analysis preserves the raw pooled 512-D vectors (no L2
normalization) and directly resizes images to 224 x 224. These vectors are
shared by the UMAP, silhouette, centroid-distance, and representative-image
analyses.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import tempfile
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd
import PIL
import timm
import torch
import torchvision
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

CODE_ROOT = Path(__file__).resolve().parents[1]
CORE_DIR = CODE_ROOT / "core"
if str(CORE_DIR) not in sys.path:
    sys.path.insert(0, str(CORE_DIR))

from path_config import TABLE_ROOT  # noqa: E402

RANDOM_SEED = 40
FEATURE_MODEL_NAME = "resnet18.a1_in1k"
OFFICIAL_WEIGHT_URL = "https://github.com/huggingface/pytorch-image-models/releases/download/v0.1-rsb-weights/resnet18_a1_0-d63eafa0.pth"
OFFICIAL_WEIGHT_FILENAME = "resnet18_a1_0-d63eafa0.pth"
OFFICIAL_WEIGHT_SHA256 = "D63EAFA07A6E32A39D328E364F8C9F89D671444ECC7F02AA0F7EB8882AF3DD29"
CLASSIFIER_KEYS = {"fc.weight", "fc.bias"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
DATASET_NAME_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")
EXPECTED_STAGE_COUNTS = {
    "PIID": {1: 229, 2: 311, 3: 273, 4: 268},
    "HUMC": {1: 233, 2: 709, 3: 575, 4: 327},
    "Kaggle": {1: 27, 2: 46, 3: 41, 4: 27},
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def official_weight_path(
    cache_dir: Path | None = None,
    *,
    progress: bool = True,
) -> Path:
    """Return a fully verified cached copy of the official timm A1 weight."""
    if cache_dir is None:
        cache_dir = Path(torch.hub.get_dir()) / "checkpoints"
    cache_dir = cache_dir.expanduser().resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    cached = cache_dir / OFFICIAL_WEIGHT_FILENAME

    if cached.is_file():
        observed = file_sha256(cached)
        if observed != OFFICIAL_WEIGHT_SHA256:
            raise ValueError(
                f"Cached official ResNet-18 weight has SHA-256 {observed}, expected "
                f"{OFFICIAL_WEIGHT_SHA256}. Remove the corrupted file and rerun: {cached}"
            )
        return cached

    with tempfile.NamedTemporaryFile(
        dir=cache_dir,
        prefix=".resnet18-a1-download-",
        suffix=".pth",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
    try:
        torch.hub.download_url_to_file(
            OFFICIAL_WEIGHT_URL,
            str(temporary),
            progress=progress,
        )
        observed = file_sha256(temporary)
        if observed != OFFICIAL_WEIGHT_SHA256:
            raise ValueError(
                f"Downloaded official ResNet-18 weight has SHA-256 {observed}, expected "
                f"{OFFICIAL_WEIGHT_SHA256}"
            )
        temporary.replace(cached)
    finally:
        temporary.unlink(missing_ok=True)
    return cached


def unwrap_state_dict(state: object) -> dict[str, object]:
    if isinstance(state, Mapping):
        for key in ("model", "state_dict", "model_state_dict"):
            if key in state and isinstance(state[key], Mapping):
                state = state[key]
                break
    if not isinstance(state, Mapping) or not state:
        raise ValueError("Checkpoint does not contain a non-empty state dictionary")
    return {str(key).removeprefix("module."): value for key, value in state.items()}


def load_model(device: torch.device) -> tuple[torch.nn.Module, str]:
    """Load the verified public timm A1 checkpoint as a headless encoder."""
    weights = official_weight_path()
    checkpoint_sha256 = file_sha256(weights)
    if checkpoint_sha256 != OFFICIAL_WEIGHT_SHA256:
        raise ValueError(
            f"Unexpected official ResNet-18 weight SHA-256: {checkpoint_sha256}; "
            f"expected {OFFICIAL_WEIGHT_SHA256}"
        )
    state = unwrap_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
    classifier_keys = {key for key in state if key.startswith("fc.")}
    if classifier_keys != CLASSIFIER_KEYS:
        raise ValueError(
            "Official ResNet-18 checkpoint classifier keys changed: "
            f"found {sorted(classifier_keys)}, expected {sorted(CLASSIFIER_KEYS)}"
        )
    encoder_state = {key: value for key, value in state.items() if key not in CLASSIFIER_KEYS}
    model = timm.create_model(FEATURE_MODEL_NAME, pretrained=False, num_classes=0)
    model.load_state_dict(encoder_state, strict=True)
    feature_dimension = int(model.num_features)
    model.eval().to(device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if feature_dimension != 512:
        raise RuntimeError(f"Expected 512 pooled features, got {feature_dimension}")
    return model, checkpoint_sha256


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
            image = image.convert("RGB")
            return self.transform(image)


def collect(root: Path, dataset: str) -> pd.DataFrame:
    rows = []
    for stage in range(1, 5):
        folder = root / str(stage)
        if not folder.is_dir():
            raise FileNotFoundError(f"Stage folder not found for {dataset}: {folder}")
        stage_count = 0
        for path in sorted(folder.iterdir()):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                try:
                    with Image.open(path) as image:
                        width, height = image.size
                        image.load()
                except (OSError, ValueError):
                    raise ValueError(
                        f"{dataset} stage {stage}: unreadable image encountered"
                    ) from None
                if width <= 0 or height <= 0:
                    raise ValueError(
                        f"{dataset} stage {stage}: invalid image dimensions encountered"
                    )
                if width != height:
                    raise ValueError(
                        f"{dataset} stage {stage}: non-square image encountered; "
                        "run the documented curation workflow before feature extraction"
                    )
                stage_count += 1
                rows.append(
                    {
                        "dataset": dataset,
                        "stage": stage,
                        "image_id": path.stem,
                        "image_path": str(path.resolve()),
                    }
                )
        if stage_count == 0:
            raise RuntimeError(f"No images found for {dataset}, stage {stage}: {folder}")
    table = pd.DataFrame(rows)
    if table.empty:
        raise RuntimeError(f"No images found for {dataset}: {root}")
    if table["image_path"].duplicated().any() or table["image_id"].duplicated().any():
        raise ValueError(f"Duplicate image path or image identifier in {dataset}")
    expected = EXPECTED_STAGE_COUNTS.get(dataset)
    if expected is not None:
        observed = table.groupby("stage", observed=True).size().to_dict()
        if observed != expected:
            raise ValueError(
                f"{dataset} stage counts do not match the manuscript cohort: "
                f"expected {expected}, observed {observed}"
            )
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
    seen: set[str] = set()
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"Expected NAME=PATH, received: {spec}")
        name, path = spec.split("=", 1)
        name = name.strip()
        if not name:
            raise ValueError(f"Dataset name is blank: {spec}")
        if not DATASET_NAME_PATTERN.fullmatch(name) or name in {".", ".."}:
            raise ValueError(
                "Dataset names may contain only letters, numbers, dot, underscore, "
                f"and hyphen, and cannot be '.' or '..': {name!r}"
            )
        folded_name = name.casefold()
        if folded_name in seen:
            raise ValueError(f"Duplicate dataset name: {name}")
        root = Path(path).expanduser().resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"Dataset folder not found for {name}: {root}")
        seen.add(folded_name)
        parsed.append((name, root))
    return parsed


def paths_overlap(first: Path, second: Path) -> bool:
    first = first.resolve()
    second = second.resolve()
    return first == second or first in second.parents or second in first.parents


def write_feature_artifacts(
    dataset_dir: Path,
    features: np.ndarray,
    metadata: pd.DataFrame,
    provenance: dict[str, object],
) -> None:
    """Write a self-consistent artifact set, publishing provenance last."""
    dataset_dir.mkdir(parents=True, exist_ok=True)
    temporary_paths: list[Path] = []
    try:
        with tempfile.NamedTemporaryFile(
            dir=dataset_dir, prefix=".features-", suffix=".npy", delete=False
        ) as handle:
            temporary_feature = Path(handle.name)
            np.save(handle, features, allow_pickle=False)
        temporary_paths.append(temporary_feature)
        with tempfile.NamedTemporaryFile(
            dir=dataset_dir,
            prefix=".metadata-",
            suffix=".csv",
            mode="w",
            encoding="utf-8",
            newline="",
            delete=False,
        ) as handle:
            temporary_metadata = Path(handle.name)
            metadata.to_csv(handle, index=False, lineterminator="\n")
        temporary_paths.append(temporary_metadata)

        provenance = {
            **provenance,
            "features_sha256": file_sha256(temporary_feature),
            "metadata_sha256": file_sha256(temporary_metadata),
        }
        with tempfile.NamedTemporaryFile(
            dir=dataset_dir,
            prefix=".extraction-",
            suffix=".json",
            mode="w",
            encoding="utf-8",
            newline="",
            delete=False,
        ) as handle:
            temporary_provenance = Path(handle.name)
            json.dump(provenance, handle, indent=2, ensure_ascii=True)
            handle.write("\n")
        temporary_paths.append(temporary_provenance)

        temporary_feature.replace(dataset_dir / "features.npy")
        temporary_metadata.replace(dataset_dir / "metadata.csv")
        # Downstream readers require this file and verify both preceding
        # hashes, so publishing it last prevents a mixed set from being used.
        temporary_provenance.replace(dataset_dir / "extraction.json")
    finally:
        for temporary in temporary_paths:
            temporary.unlink(missing_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", action="append", required=True, metavar="NAME=PATH")
    parser.add_argument("--output-dir", type=Path, default=TABLE_ROOT / "feature_space" / "features")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0 or args.num_workers < 0:
        raise ValueError("--batch-size must be positive and --num-workers cannot be negative")
    dataset_specs = parse_dataset_specs(args.dataset)
    output_dir = args.output_dir.expanduser().resolve()
    for dataset, source_root in dataset_specs:
        if paths_overlap(source_root, output_dir / dataset):
            raise ValueError(
                f"Feature output for {dataset} must not overlap its source image tree"
            )

    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, checkpoint_sha256 = load_model(device)
    output_dir.mkdir(parents=True, exist_ok=True)

    for dataset, root in dataset_specs:
        metadata = collect(root, dataset)
        features = extract(
            model,
            metadata["image_path"].tolist(),
            device,
            args.batch_size,
            args.num_workers,
        )
        if features.ndim != 2 or features.shape[1] != 512 or not np.isfinite(features).all():
            raise RuntimeError(
                f"Invalid feature array for {dataset}: shape={features.shape}, "
                f"finite={bool(np.isfinite(features).all())}"
            )
        dataset_dir = output_dir / dataset
        write_feature_artifacts(
            dataset_dir,
            features,
            metadata,
            {
                "encoder": "timm.create_model",
                "weights": FEATURE_MODEL_NAME,
                "weights_url": OFFICIAL_WEIGHT_URL,
                "checkpoint_filename": OFFICIAL_WEIGHT_FILENAME,
                "frozen": True,
                "pretraining": "ImageNet-1K (A1 recipe)",
                "checkpoint_sha256": checkpoint_sha256,
                "checkpoint_hash_verified": True,
                "feature_dimension": 512,
                "input_geometry": "curated native square",
                "resize": "direct 224 x 224",
                "normalization": "ImageNet mean/std",
                "l2_normalized": False,
                "random_seed": RANDOM_SEED,
                "device": str(device),
                "torch_version": torch.__version__,
                "torchvision_version": torchvision.__version__,
                "timm_version": timm.__version__,
                "pillow_version": PIL.__version__,
                "numpy_version": np.__version__,
                "n_images": len(metadata),
            },
        )
        print(f"[DONE] {dataset}: {features.shape} -> {dataset_dir}")


if __name__ == "__main__":
    main()
