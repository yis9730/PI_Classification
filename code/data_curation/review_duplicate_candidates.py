"""Generate duplicate-candidate tables and montages from public raw images.

The PIID and Kaggle provider folders are screened before any exclusion or
square curation. Two independent all-pairs searches are performed:

1. L2-normalized ResNet-18 features with cosine similarity >= 0.85.
2. RGB pixels resized directly to 128 x 128 with ``1 - MAE >= 0.85``.

Feature inputs are resized directly to 224 x 224. Complete candidate tables are
sorted from highest similarity down to the requested threshold. To avoid showing
the same highly connected images in hundreds of thousands of redundant pair
panels, the montage queue retains each image's strongest candidate pair and is
also sorted from highest to lowest similarity. Candidate generation never deletes
or rewrites an image. Review decisions remain in ``duplicate_pairs.csv`` and the
two exclusion manifests; this script verifies that those decisions remove exactly
10 PIID and 18 Kaggle images and that every reviewed pair is present in at least
one independently generated candidate table.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import PIL
import timm
import torch
import torchvision
from PIL import Image, ImageDraw, ImageFont, ImageOps
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm


CODE_ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_DIR = CODE_ROOT / "analysis"
if str(ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_DIR))

from extract_resnet18_features import (  # noqa: E402
    FEATURE_MODEL_NAME,
    OFFICIAL_WEIGHT_SHA256,
    load_model,
)


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
FEATURE_SIMILARITY_THRESHOLD = 0.85
PIXEL_SIMILARITY_THRESHOLD = 0.85
FEATURE_INPUT_SIZE = 224
PIXEL_INPUT_SIZE = 128
EXPECTED_EXCLUSIONS = {"PIID": 10, "Kaggle": 18}
EXPECTED_FINAL_COUNTS = {"PIID": 1081, "Kaggle": 141}
EXPECTED_REVIEWED_PAIRS = {"PIID": 8, "Kaggle": 12}
EXPECTED_FINAL_STAGE_COUNTS = {
    "PIID": {1: 229, 2: 311, 3: 273, 4: 268},
    "Kaggle": {1: 27, 2: 46, 3: 41, 4: 27},
}
EXPECTED_RAW_STAGE_COUNTS = {
    "PIID": {1: 230, 2: 313, 3: 275, 4: 273},
    "Kaggle": {1: 28, 2: 53, 3: 46, 4: 32},
}
STAGE_FOLDERS = {
    "PIID": {1: "1", 2: "2", 3: "3", 4: "4"},
    "Kaggle": {1: "Stage_I", 2: "Stage_II", 3: "Stage_III", 4: "Stage_IV"},
}
DATASET_COLORS = {
    "PIID": (198, 55, 55),
    "Kaggle": (38, 105, 190),
}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def paths_overlap(first: Path, second: Path) -> bool:
    first = first.expanduser().resolve()
    second = second.expanduser().resolve()
    return first == second or first in second.parents or second in first.parents


def prepare_output_dir(
    output_dir: Path,
    source_roots: list[Path],
    overwrite: bool,
) -> Path:
    """Prepare one isolated output without permitting source-tree deletion."""
    output_dir = output_dir.expanduser().resolve()
    root = repo_root().resolve()
    allowed_root = (root / "data" / "results").resolve()
    if output_dir == allowed_root or allowed_root not in output_dir.parents:
        raise ValueError(
            "Output directory must be a child of the repository's data/results folder"
        )
    for source in source_roots:
        if paths_overlap(source, output_dir):
            raise ValueError("Output directory must not overlap a source dataset")
    if output_dir.exists():
        if not output_dir.is_dir():
            raise NotADirectoryError(f"Output path is not a directory: {output_dir}")
        if any(output_dir.iterdir()):
            if not overwrite:
                raise FileExistsError(
                    f"Output directory is not empty: {output_dir}. Use --overwrite."
                )
            shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def collect_raw_images(root: Path, dataset: str) -> pd.DataFrame:
    """Collect and validate the provider-stage folders before curation."""
    root = root.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"{dataset} source folder not found: {root}")
    rows: list[dict[str, object]] = []
    observed: dict[int, int] = {}
    for stage, folder_name in STAGE_FOLDERS[dataset].items():
        folder = root / folder_name
        if not folder.is_dir():
            raise FileNotFoundError(f"{dataset} stage folder not found: {folder}")
        images = sorted(
            path
            for path in folder.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        observed[stage] = len(images)
        for path in images:
            try:
                with Image.open(path) as image:
                    image.load()
                    width, height = image.size
                    image_format = image.format
            except (OSError, ValueError):
                raise ValueError(
                    f"{dataset} stage {stage}: unreadable source image found"
                ) from None
            if width <= 0 or height <= 0 or image_format is None:
                raise ValueError(f"{dataset} stage {stage}: invalid source image found")
            rows.append(
                {
                    "dataset": dataset,
                    "stage": stage,
                    "source_folder": folder_name,
                    "filename": path.name,
                    "image_id": path.stem,
                    "image_path": str(path),
                    "width": width,
                    "height": height,
                }
            )
    if observed != EXPECTED_RAW_STAGE_COUNTS[dataset]:
        raise ValueError(
            f"Unexpected raw {dataset} counts: expected "
            f"{EXPECTED_RAW_STAGE_COUNTS[dataset]}, observed {observed}"
        )
    table = pd.DataFrame(rows)
    if table[["dataset", "image_id"]].duplicated().any():
        raise ValueError(f"{dataset} contains duplicate filename stems")
    return table


def source_fingerprint(table: pd.DataFrame) -> str:
    """Hash relative identity and bytes without recording local absolute paths."""
    digest = hashlib.sha256()
    for row in table.sort_values(
        ["dataset", "stage", "source_folder", "filename"], kind="stable"
    ).itertuples(index=False):
        identity = f"{row.dataset}/{row.source_folder}/{row.filename}\0".encode("utf-8")
        digest.update(identity)
        with Path(row.image_path).open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest().upper()


class RawFeatureDataset(Dataset):
    def __init__(self, paths: list[str]):
        self.paths = paths
        self.transform = transforms.Compose(
            [
                transforms.Resize((FEATURE_INPUT_SIZE, FEATURE_INPUT_SIZE)),
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
            return self.transform(image.convert("RGB"))


def extract_normalized_features(
    table: pd.DataFrame,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, str]:
    model, checkpoint_sha256 = load_model(device)
    loader = DataLoader(
        RawFeatureDataset(table["image_path"].tolist()),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    batches: list[np.ndarray] = []
    with torch.inference_mode():
        for images in tqdm(loader, desc="Feature extraction"):
            features = model(images.to(device))
            features = torch.nn.functional.normalize(features, p=2, dim=1)
            batches.append(features.cpu().numpy().astype(np.float32, copy=False))
    matrix = np.concatenate(batches, axis=0)
    if matrix.shape != (len(table), 512) or not np.isfinite(matrix).all():
        raise RuntimeError(f"Invalid feature matrix: {matrix.shape}")
    return matrix, checkpoint_sha256


def feature_candidate_pairs(
    features: np.ndarray,
    threshold: float,
) -> list[tuple[int, int, float]]:
    similarity = features @ features.T
    left, right = np.triu_indices(len(features), k=1)
    scores = similarity[left, right]
    selected = np.flatnonzero(scores >= threshold)
    pairs = [
        (int(left[index]), int(right[index]), float(scores[index]))
        for index in selected
    ]
    pairs.sort(key=lambda item: (-item[2], item[0], item[1]))
    return pairs


def load_pixel_arrays(table: pd.DataFrame, size: int) -> np.ndarray:
    arrays = np.empty((len(table), size, size, 3), dtype=np.uint8)
    for index, path in enumerate(tqdm(table["image_path"], desc="Pixel loading")):
        with Image.open(path) as image:
            resized = image.convert("RGB").resize(
                (size, size), Image.Resampling.LANCZOS
            )
            arrays[index] = np.asarray(resized, dtype=np.uint8)
    return arrays


def pixel_candidate_pairs(
    arrays: np.ndarray,
    threshold: float,
    chunk_size: int,
) -> list[tuple[int, int, float]]:
    """Run an independent all-pairs pixel search with bounded temporary memory."""
    if chunk_size <= 0:
        raise ValueError("--pixel-chunk-size must be positive")
    pairs: list[tuple[int, int, float]] = []
    for left in tqdm(range(len(arrays) - 1), desc="Pixel comparison"):
        reference = arrays[left].astype(np.int16, copy=False)
        for start in range(left + 1, len(arrays), chunk_size):
            stop = min(start + chunk_size, len(arrays))
            batch = arrays[start:stop].astype(np.int16, copy=False)
            mae = np.mean(
                np.abs(batch - reference),
                axis=(1, 2, 3),
                dtype=np.float64,
            ) / 255.0
            scores = 1.0 - mae
            for offset in np.flatnonzero(scores >= threshold):
                pairs.append((left, start + int(offset), float(scores[offset])))
    pairs.sort(key=lambda item: (-item[2], item[0], item[1]))
    return pairs


def pair_key(left: int, right: int) -> tuple[int, int]:
    return (left, right) if left < right else (right, left)


def candidate_table(
    table: pd.DataFrame,
    pairs: list[tuple[int, int, float]],
    method: str,
) -> pd.DataFrame:
    columns = [
        "method",
        "similarity",
        "dataset_1",
        "stage_1",
        "source_folder_1",
        "filename_1",
        "dataset_2",
        "stage_2",
        "source_folder_2",
        "filename_2",
    ]
    if not pairs:
        return pd.DataFrame(columns=columns)
    left = np.fromiter((pair[0] for pair in pairs), dtype=np.int64, count=len(pairs))
    right = np.fromiter((pair[1] for pair in pairs), dtype=np.int64, count=len(pairs))
    scores = np.fromiter((pair[2] for pair in pairs), dtype=np.float64, count=len(pairs))
    values = {
        column: table[column].to_numpy()
        for column in ("dataset", "stage", "source_folder", "filename")
    }
    return pd.DataFrame(
        {
            "method": np.repeat(method, len(pairs)),
            "similarity": scores,
            "dataset_1": values["dataset"][left],
            "stage_1": values["stage"][left],
            "source_folder_1": values["source_folder"][left],
            "filename_1": values["filename"][left],
            "dataset_2": values["dataset"][right],
            "stage_2": values["stage"][right],
            "source_folder_2": values["source_folder"][right],
            "filename_2": values["filename"][right],
        },
        columns=columns,
    )


def connected_components(
    pairs: list[tuple[int, int, float]],
) -> list[dict[str, object]]:
    nodes = sorted({index for left, right, _ in pairs for index in (left, right)})
    parent = {node: node for node in nodes}

    def find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(first: int, second: int) -> None:
        root_first, root_second = find(first), find(second)
        if root_first != root_second:
            parent[root_second] = root_first

    for left, right, _ in pairs:
        union(left, right)
    members: dict[int, set[int]] = defaultdict(set)
    edges: dict[int, list[tuple[int, int, float]]] = defaultdict(list)
    for node in nodes:
        members[find(node)].add(node)
    for left, right, score in pairs:
        edges[find(left)].append((left, right, score))
    components = []
    for root, component_members in members.items():
        component_edges = edges[root]
        components.append(
            {
                "members": sorted(component_members),
                "edges": component_edges,
                "max_similarity": max(score for _, _, score in component_edges),
            }
        )
    components.sort(
        key=lambda item: (
            -float(item["max_similarity"]),
            -len(item["members"]),
            item["members"][0],
        )
    )
    return components


def fitted_image(path: str, width: int, height: int) -> Image.Image:
    with Image.open(path) as image:
        contained = ImageOps.contain(
            image.convert("RGB"), (width, height), Image.Resampling.LANCZOS
        )
    canvas = Image.new("RGB", (width, height), "white")
    x = (width - contained.width) // 2
    y = (height - contained.height) // 2
    canvas.paste(contained, (x, y))
    return canvas


def strongest_neighbor_review_queue(
    pairs: list[tuple[int, int, float]],
) -> tuple[list[tuple[int, int, float]], list[int]]:
    """Select one strongest candidate edge per image for visual review.

    ``pairs`` must already be sorted by decreasing similarity. The union of each
    endpoint's first (therefore strongest) edge removes redundant comparisons
    while retaining a directly inspectable candidate for every image that crossed
    the threshold. Equal-score ties follow the deterministic dataset/stage/file
    ordering used to build the image table.
    """
    best_pair_index: dict[int, int] = {}
    for pair_index, (left, right, _) in enumerate(pairs):
        best_pair_index.setdefault(left, pair_index)
        best_pair_index.setdefault(right, pair_index)
    selected_indices = sorted(set(best_pair_index.values()))
    queue = [pairs[index] for index in selected_indices]
    endpoint_counts = [
        int(best_pair_index.get(left) == pair_index)
        + int(best_pair_index.get(right) == pair_index)
        for pair_index, (left, right, _) in zip(selected_indices, queue)
    ]
    return queue, endpoint_counts


def review_queue_table(
    table: pd.DataFrame,
    pairs: list[tuple[int, int, float]],
    endpoint_counts: list[int],
    method: str,
) -> pd.DataFrame:
    queue = candidate_table(table, pairs, method)
    queue.insert(0, "review_rank", np.arange(1, len(queue) + 1, dtype=np.int64))
    queue.insert(3, "strongest_candidate_for_endpoints", endpoint_counts)
    return queue


def render_pair_review_montages(
    method: str,
    table: pd.DataFrame,
    pairs: list[tuple[int, int, float]],
    output_dir: Path,
    pairs_per_page: int,
) -> None:
    """Render the nonredundant review queue in descending similarity order."""
    if pairs_per_page <= 0:
        raise ValueError("--montage-pairs-per-page must be positive")
    output_dir.mkdir(parents=True, exist_ok=True)
    font = ImageFont.load_default()
    columns = 2
    cell_width, cell_height = 600, 280
    image_width, image_height = 270, 170
    page_header_height = 74
    pages = math.ceil(len(pairs) / pairs_per_page)
    for page_index in range(pages):
        page_pairs = pairs[
            page_index * pairs_per_page : (page_index + 1) * pairs_per_page
        ]
        rows = math.ceil(len(page_pairs) / columns)
        canvas = Image.new(
            "RGB",
            (columns * cell_width, page_header_height + rows * cell_height),
            "white",
        )
        draw = ImageDraw.Draw(canvas)
        draw.rectangle(
            (0, 0, canvas.width, page_header_height), fill=(235, 238, 242)
        )
        first_rank = page_index * pairs_per_page + 1
        last_rank = first_rank + len(page_pairs) - 1
        high_score = page_pairs[0][2]
        low_score = page_pairs[-1][2]
        draw.text(
            (16, 12),
            f"{method.upper()} review queue | page {page_index + 1}/{pages} | "
            f"ranks {first_rank}-{last_rank}",
            fill="black",
            font=font,
        )
        draw.text(
            (16, 34),
            f"similarity {high_score:.6f} down to {low_score:.6f} | "
            "strongest candidate pair per image",
            fill=(55, 55, 55),
            font=font,
        )
        draw.text(
            (16, 54),
            "Original provider images; display scaling preserves the full frame.",
            fill=(80, 80, 80),
            font=font,
        )
        for local_index, (left, right, score) in enumerate(page_pairs):
            column = local_index % columns
            page_row = local_index // columns
            x = column * cell_width
            y = page_header_height + page_row * cell_height
            rank = first_rank + local_index
            draw.rectangle(
                (x + 5, y + 5, x + cell_width - 5, y + cell_height - 5),
                outline=(125, 125, 125),
                width=2,
            )
            draw.text(
                (x + 14, y + 12),
                f"#{rank:05d} | similarity={score:.6f}",
                fill="black",
                font=font,
            )
            for side, image_index in enumerate((left, right)):
                source = table.iloc[image_index]
                image_x = x + 14 + side * 292
                image_y = y + 36
                image = fitted_image(
                    str(source["image_path"]), image_width, image_height
                )
                canvas.paste(image, (image_x, image_y))
                color = DATASET_COLORS[str(source["dataset"])]
                draw.rectangle(
                    (
                        image_x,
                        image_y,
                        image_x + image_width,
                        image_y + image_height,
                    ),
                    outline=color,
                    width=3,
                )
                draw.text(
                    (image_x, y + 214),
                    f"{source['dataset']} | Stage {int(source['stage'])}",
                    fill=color,
                    font=font,
                )
                filename = str(source["filename"])
                if len(filename) > 40:
                    filename = filename[:37] + "..."
                draw.text((image_x, y + 234), filename, fill="black", font=font)
        canvas.save(
            output_dir / f"{method}_review_page_{page_index + 1:04d}.jpg",
            quality=92,
        )


def read_reviewed_decisions(
    root: Path,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    curation_dir = root / "code" / "data_curation"
    pairs = pd.read_csv(curation_dir / "duplicate_pairs.csv", dtype=str)
    required = {
        "dataset",
        "pair_id",
        "image_1",
        "image_2",
        "source_labels_concordant",
        "excluded_image_1",
        "excluded_image_2",
        "curation_outcome",
    }
    if set(pairs.columns) != required:
        raise ValueError("duplicate_pairs.csv columns changed unexpectedly")
    exclusions = {
        "PIID": pd.read_csv(
            curation_dir / "piid_duplicate_exclusions.csv", dtype=str
        ),
        "Kaggle": pd.read_csv(
            curation_dir / "kaggle_duplicate_exclusions.csv", dtype=str
        ),
    }
    expected_columns = {
        "PIID": {"dataset", "stage", "filename", "reason"},
        "Kaggle": {
            "dataset",
            "stage",
            "source_folder",
            "filename",
            "reason",
        },
    }
    for dataset, manifest in exclusions.items():
        if set(manifest.columns) != expected_columns[dataset]:
            raise ValueError(f"{dataset} exclusion manifest columns changed")
    return pairs, exclusions


def validate_reviewed_decisions(
    table: pd.DataFrame,
    feature_pairs: list[tuple[int, int, float]],
    pixel_pairs: list[tuple[int, int, float]],
    feature_montage_pairs: list[tuple[int, int, float]],
    pixel_montage_pairs: list[tuple[int, int, float]],
    features: np.ndarray,
    pixels: np.ndarray,
    root: Path,
) -> tuple[pd.DataFrame, dict[str, object]]:
    decisions, exclusion_manifests = read_reviewed_decisions(root)
    lookup: dict[tuple[str, str], int] = {}
    for index, row in table.iterrows():
        key = (str(row["dataset"]), str(row["image_id"]))
        if key in lookup:
            raise ValueError(f"Ambiguous image identifier: {key}")
        lookup[key] = int(index)
    released_exclusions: dict[str, set[str]] = {}
    for dataset, manifest in exclusion_manifests.items():
        if manifest.empty or manifest.isna().any(axis=None):
            raise ValueError(f"{dataset} exclusion manifest contains blank values")
        if set(manifest["dataset"]) != {dataset}:
            raise ValueError(f"{dataset} exclusion manifest dataset column changed")
        if manifest["filename"].duplicated().any():
            raise ValueError(f"{dataset} exclusion manifest repeats a filename")
        image_ids: set[str] = set()
        for exclusion in manifest.itertuples(index=False):
            filename = str(exclusion.filename)
            if Path(filename).name != filename:
                raise ValueError(f"{dataset} exclusion manifest has an invalid filename")
            image_id = Path(filename).stem
            key = (dataset, image_id)
            if key not in lookup:
                raise ValueError(
                    f"{dataset} exclusion is absent from the raw source: {filename}"
                )
            source = table.iloc[lookup[key]]
            if (
                str(source["filename"]) != filename
                or int(source["stage"]) != int(exclusion.stage)
                or (
                    dataset == "Kaggle"
                    and str(source["source_folder"])
                    != str(exclusion.source_folder)
                )
            ):
                raise ValueError(
                    f"{dataset} exclusion metadata does not match the raw source: "
                    f"{filename}"
                )
            image_ids.add(image_id)
        if len(image_ids) != len(manifest):
            raise ValueError(f"{dataset} exclusion manifest repeats an image identifier")
        released_exclusions[dataset] = image_ids

    if decisions.empty or decisions.isna().any(axis=None):
        raise ValueError("duplicate_pairs.csv contains blank values")
    if decisions["pair_id"].duplicated().any():
        raise ValueError("duplicate_pairs.csv repeats a pair_id")
    feature_lookup = {pair_key(left, right): score for left, right, score in feature_pairs}
    pixel_lookup = {pair_key(left, right): score for left, right, score in pixel_pairs}
    feature_montage_lookup = {
        pair_key(left, right) for left, right, _ in feature_montage_pairs
    }
    pixel_montage_lookup = {
        pair_key(left, right) for left, right, _ in pixel_montage_pairs
    }
    selected_from_pairs: dict[str, set[str]] = {"PIID": set(), "Kaggle": set()}
    seen_pairs: set[tuple[str, str, str]] = set()
    seen_endpoints: set[tuple[str, str]] = set()
    rows = []
    for decision in decisions.itertuples(index=False):
        dataset = str(decision.dataset)
        if dataset not in selected_from_pairs:
            raise ValueError(f"Unexpected dataset in duplicate_pairs.csv: {dataset}")
        key_1 = (dataset, str(decision.image_1))
        key_2 = (dataset, str(decision.image_2))
        if key_1 == key_2:
            raise ValueError(f"Reviewed pair repeats one image: {decision.pair_id}")
        if key_1 not in lookup or key_2 not in lookup:
            raise ValueError(f"Reviewed pair not found in raw source: {decision.pair_id}")
        pair_identity = (dataset, *sorted((key_1[1], key_2[1])))
        if pair_identity in seen_pairs:
            raise ValueError(f"Reviewed pair is repeated: {decision.pair_id}")
        if key_1 in seen_endpoints or key_2 in seen_endpoints:
            raise ValueError(f"Reviewed image occurs in multiple pairs: {decision.pair_id}")
        seen_pairs.add(pair_identity)
        seen_endpoints.update((key_1, key_2))
        left, right = lookup[key_1], lookup[key_2]
        canonical = pair_key(left, right)
        feature_similarity = float(np.dot(features[left], features[right]))
        pixel_similarity = float(
            1.0
            - np.mean(
                np.abs(
                    pixels[left].astype(np.int16) - pixels[right].astype(np.int16)
                ),
                dtype=np.float64,
            )
            / 255.0
        )
        feature_candidate = canonical in feature_lookup
        pixel_candidate = canonical in pixel_lookup
        feature_montage_pair = canonical in feature_montage_lookup
        pixel_montage_pair = canonical in pixel_montage_lookup
        flag_1 = str(decision.excluded_image_1).lower()
        flag_2 = str(decision.excluded_image_2).lower()
        if flag_1 not in {"yes", "no"} or flag_2 not in {"yes", "no"}:
            raise ValueError(f"Invalid exclusion flag: {decision.pair_id}")
        first = table.iloc[left]
        second = table.iloc[right]
        labels_concordant = int(first["stage"]) == int(second["stage"])
        declared_concordance = str(decision.source_labels_concordant).lower()
        if declared_concordance not in {"yes", "no"} or (
            (declared_concordance == "yes") != labels_concordant
        ):
            raise ValueError(f"Label-concordance mismatch: {decision.pair_id}")
        if labels_concordant:
            expected_outcome = {
                ("no", "yes"): "retained_image_1",
                ("yes", "no"): "retained_image_2",
            }.get((flag_1, flag_2))
            if expected_outcome is None or str(decision.curation_outcome) != expected_outcome:
                raise ValueError(
                    f"Concordant-pair decision is inconsistent: {decision.pair_id}"
                )
        elif (
            (flag_1, flag_2) != ("yes", "yes")
            or str(decision.curation_outcome) != "excluded_both_label_conflict"
        ):
            raise ValueError(
                f"Label-conflict decision is inconsistent: {decision.pair_id}"
            )
        if flag_1 == "yes":
            selected_from_pairs[dataset].add(str(decision.image_1))
        if flag_2 == "yes":
            selected_from_pairs[dataset].add(str(decision.image_2))
        rows.append(
            {
                "dataset": dataset,
                "pair_id": decision.pair_id,
                "stage_1": int(first["stage"]),
                "filename_1": first["filename"],
                "stage_2": int(second["stage"]),
                "filename_2": second["filename"],
                "feature_similarity": feature_similarity,
                "feature_candidate": feature_candidate,
                "feature_montage_pair": feature_montage_pair,
                "pixel_similarity": pixel_similarity,
                "pixel_candidate": pixel_candidate,
                "pixel_montage_pair": pixel_montage_pair,
                "candidate_union": feature_candidate or pixel_candidate,
                "montage_queue_union": (
                    feature_montage_pair or pixel_montage_pair
                ),
                "excluded_image_1": decision.excluded_image_1,
                "excluded_image_2": decision.excluded_image_2,
                "curation_outcome": decision.curation_outcome,
            }
        )
    reviewed = pd.DataFrame(rows)
    manifest_match = {
        dataset: selected_from_pairs[dataset] == released_exclusions[dataset]
        for dataset in EXPECTED_EXCLUSIONS
    }
    exclusion_counts = {
        dataset: len(released_exclusions[dataset]) for dataset in EXPECTED_EXCLUSIONS
    }
    final_counts = {
        dataset: int((table["dataset"] == dataset).sum()) - exclusion_counts[dataset]
        for dataset in EXPECTED_EXCLUSIONS
    }
    final_stage_counts: dict[str, dict[int, int]] = {}
    for dataset in EXPECTED_EXCLUSIONS:
        remaining = table[
            (table["dataset"] == dataset)
            & ~table["image_id"].isin(released_exclusions[dataset])
        ]
        final_stage_counts[dataset] = {
            int(stage): int(count)
            for stage, count in remaining.groupby("stage", observed=True).size().items()
        }
    reviewed_pair_counts = {
        str(dataset): int(count)
        for dataset, count in reviewed.groupby("dataset", observed=True).size().items()
    }
    candidate_coverage = int(reviewed["candidate_union"].sum())
    montage_coverage = int(reviewed["montage_queue_union"].sum())
    success = (
        candidate_coverage == len(reviewed)
        and montage_coverage == len(reviewed)
        and all(manifest_match.values())
        and exclusion_counts == EXPECTED_EXCLUSIONS
        and final_counts == EXPECTED_FINAL_COUNTS
        and reviewed_pair_counts == EXPECTED_REVIEWED_PAIRS
        and final_stage_counts == EXPECTED_FINAL_STAGE_COUNTS
    )
    summary: dict[str, object] = {
        "success": success,
        "raw_image_counts": {
            dataset: int((table["dataset"] == dataset).sum())
            for dataset in EXPECTED_EXCLUSIONS
        },
        "reviewed_pair_counts": reviewed_pair_counts,
        "reviewed_pairs_covered_by_candidate_union": candidate_coverage,
        "reviewed_pairs_covered_by_montage_queue_union": montage_coverage,
        "reviewed_pairs_total": len(reviewed),
        "released_exclusion_counts": exclusion_counts,
        "review_flags_match_exclusion_manifests": manifest_match,
        "final_image_counts_after_released_exclusions": final_counts,
        "final_stage_counts_after_released_exclusions": final_stage_counts,
    }
    return reviewed, summary


def render_reviewed_pair_montages(
    reviewed: pd.DataFrame,
    table: pd.DataFrame,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    lookup = {
        (str(row.dataset), str(row.filename)): index
        for index, row in table.iterrows()
    }
    font = ImageFont.load_default()
    for rank, row in enumerate(reviewed.itertuples(index=False), start=1):
        left = table.iloc[lookup[(str(row.dataset), str(row.filename_1))]]
        right = table.iloc[lookup[(str(row.dataset), str(row.filename_2))]]
        canvas = Image.new("RGB", (1040, 560), "white")
        draw = ImageDraw.Draw(canvas)
        draw.rectangle((0, 0, canvas.width, 105), fill=(235, 238, 242))
        draw.text(
            (16, 12),
            f"{row.pair_id} | {row.dataset} | released review decision",
            fill="black",
            font=font,
        )
        draw.text(
            (16, 36),
            f"feature={float(row.feature_similarity):.6f} "
            f"(candidate={bool(row.feature_candidate)}) | "
            f"pixel={float(row.pixel_similarity):.6f} "
            f"(candidate={bool(row.pixel_candidate)})",
            fill=(45, 45, 45),
            font=font,
        )
        draw.text(
            (16, 60),
            f"decision: {row.curation_outcome} | "
            f"exclude left={row.excluded_image_1}, right={row.excluded_image_2}",
            fill=(75, 75, 75),
            font=font,
        )
        draw.text(
            (16, 82),
            "Original provider images; display scaling preserves the full frame.",
            fill=(90, 90, 90),
            font=font,
        )
        for column, source in enumerate((left, right)):
            x = 20 + column * 510
            image = fitted_image(str(source["image_path"]), 490, 370)
            canvas.paste(image, (x, 120))
            color = DATASET_COLORS[str(source["dataset"])]
            draw.rectangle((x, 120, x + 490, 490), outline=color, width=4)
            draw.text(
                (x, 505),
                f"Stage {int(source['stage'])} | {source['filename']}",
                fill=color,
                font=font,
            )
        canvas.save(
            output_dir / f"{rank:02d}_{row.dataset}_{row.pair_id}.jpg",
            quality=94,
        )


def write_json(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_csv(table: pd.DataFrame, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        table.to_csv(temporary, index=False, float_format="%.8f")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Screen raw PIID/Kaggle images independently by feature and pixel "
            "similarity, render review montages, and verify released decisions."
        )
    )
    parser.add_argument("--piid-source", type=Path, required=True)
    parser.add_argument("--kaggle-source", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root() / "data" / "results" / "duplicate_review",
    )
    parser.add_argument(
        "--feature-threshold", type=float, default=FEATURE_SIMILARITY_THRESHOLD
    )
    parser.add_argument(
        "--pixel-threshold", type=float, default=PIXEL_SIMILARITY_THRESHOLD
    )
    parser.add_argument("--feature-batch-size", type=int, default=32)
    parser.add_argument("--pixel-chunk-size", type=int, default=64)
    parser.add_argument("--montage-pairs-per-page", type=int, default=8)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if value == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return torch.device(value)


def main() -> None:
    args = build_parser().parse_args()
    if not 0.0 < args.feature_threshold <= 1.0:
        raise ValueError("--feature-threshold must be in (0, 1]")
    if not 0.0 < args.pixel_threshold <= 1.0:
        raise ValueError("--pixel-threshold must be in (0, 1]")
    if args.feature_batch_size <= 0:
        raise ValueError("--feature-batch-size must be positive")

    piid_source = args.piid_source.expanduser().resolve()
    kaggle_source = args.kaggle_source.expanduser().resolve()
    output_dir = prepare_output_dir(
        args.output_dir, [piid_source, kaggle_source], args.overwrite
    )
    piid = collect_raw_images(piid_source, "PIID")
    kaggle = collect_raw_images(kaggle_source, "Kaggle")
    table = pd.concat([piid, kaggle], ignore_index=True)
    fingerprint_before = source_fingerprint(table)

    device = resolve_device(args.device)
    features, checkpoint_sha256 = extract_normalized_features(
        table, device, args.feature_batch_size
    )
    if checkpoint_sha256 != OFFICIAL_WEIGHT_SHA256:
        raise RuntimeError("Unexpected ResNet-18 checkpoint checksum")
    feature_pairs = feature_candidate_pairs(features, args.feature_threshold)
    pixels = load_pixel_arrays(table, PIXEL_INPUT_SIZE)
    pixel_pairs = pixel_candidate_pairs(
        pixels, args.pixel_threshold, args.pixel_chunk_size
    )

    feature_candidates = candidate_table(table, feature_pairs, "feature")
    pixel_candidates = candidate_table(table, pixel_pairs, "pixel")
    write_csv(feature_candidates, output_dir / "feature_candidate_pairs.csv")
    write_csv(pixel_candidates, output_dir / "pixel_candidate_pairs.csv")

    feature_queue, feature_endpoint_counts = strongest_neighbor_review_queue(
        feature_pairs
    )
    pixel_queue, pixel_endpoint_counts = strongest_neighbor_review_queue(pixel_pairs)
    write_csv(
        review_queue_table(
            table, feature_queue, feature_endpoint_counts, "feature"
        ),
        output_dir / "feature_montage_pairs.csv",
    )
    write_csv(
        review_queue_table(table, pixel_queue, pixel_endpoint_counts, "pixel"),
        output_dir / "pixel_montage_pairs.csv",
    )
    render_pair_review_montages(
        "feature",
        table,
        feature_queue,
        output_dir / "feature_montages",
        args.montage_pairs_per_page,
    )
    render_pair_review_montages(
        "pixel",
        table,
        pixel_queue,
        output_dir / "pixel_montages",
        args.montage_pairs_per_page,
    )

    reviewed, review_summary = validate_reviewed_decisions(
        table,
        feature_pairs,
        pixel_pairs,
        feature_queue,
        pixel_queue,
        features,
        pixels,
        repo_root(),
    )
    write_csv(reviewed, output_dir / "reviewed_pairs_validation.csv")
    render_reviewed_pair_montages(
        reviewed, table, output_dir / "reviewed_pair_montages"
    )

    table_after = pd.concat(
        [
            collect_raw_images(piid_source, "PIID"),
            collect_raw_images(kaggle_source, "Kaggle"),
        ],
        ignore_index=True,
    )
    fingerprint_after = source_fingerprint(table_after)
    if fingerprint_after != fingerprint_before:
        raise RuntimeError("A source dataset changed while duplicate review was running")
    review_summary.update(
        {
            "feature_candidate_pairs": len(feature_pairs),
            "feature_montage_pairs": len(feature_queue),
            "feature_components": len(connected_components(feature_pairs)),
            "pixel_candidate_pairs": len(pixel_pairs),
            "pixel_montage_pairs": len(pixel_queue),
            "pixel_components": len(connected_components(pixel_pairs)),
            "feature_threshold": args.feature_threshold,
            "feature_preprocessing": "direct_resize_224x224_imagenet_normalization_l2",
            "pixel_threshold": args.pixel_threshold,
            "pixel_preprocessing": "direct_resize_128x128_rgb_similarity_1_minus_mae",
            "source_fingerprint_sha256": fingerprint_before,
            "source_unchanged": True,
            "feature_model": FEATURE_MODEL_NAME,
            "feature_checkpoint_sha256": checkpoint_sha256,
            "device": str(device),
            "versions": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "pillow": PIL.__version__,
                "torch": torch.__version__,
                "torchvision": torchvision.__version__,
                "timm": timm.__version__,
            },
        }
    )
    write_json(output_dir / "review_summary.json", review_summary)

    print(f"[OK] Raw PIID images: {len(piid)}")
    print(f"[OK] Raw Kaggle images: {len(kaggle)}")
    print(f"[OK] Feature candidates: {len(feature_pairs)}")
    print(f"[OK] Pixel candidates: {len(pixel_pairs)}")
    print(f"[OK] Feature montage queue: {len(feature_queue)}")
    print(f"[OK] Pixel montage queue: {len(pixel_queue)}")
    print(
        "[OK] Released pairs covered by montage queues: "
        f"{review_summary['reviewed_pairs_covered_by_montage_queue_union']}/"
        f"{review_summary['reviewed_pairs_total']}"
    )
    print(
        "[OK] Released exclusions: PIID "
        f"{review_summary['released_exclusion_counts']['PIID']}, Kaggle "
        f"{review_summary['released_exclusion_counts']['Kaggle']}"
    )
    print(
        "[OK] Final counts: PIID "
        f"{review_summary['final_image_counts_after_released_exclusions']['PIID']}, "
        "Kaggle "
        f"{review_summary['final_image_counts_after_released_exclusions']['Kaggle']}"
    )
    print(f"[DONE] Duplicate review outputs: {output_dir}")
    if not bool(review_summary["success"]):
        raise RuntimeError(
            "Released duplicate-review decisions were not fully reproduced; "
            "inspect reviewed_pairs_validation.csv and review_summary.json"
        )


if __name__ == "__main__":
    main()
