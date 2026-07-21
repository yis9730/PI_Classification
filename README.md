# Pressure Injury Stage Classification Reproduction

This repository is the publication-ready code package for four-class pressure injury staging. It contains the data-curation rules, six-model training pipeline, all 17 augmentation conditions, fold-wise internal and external evaluation, ResNet-18 feature analysis, statistical analysis, and figure-generation code used in the study.

The public PIID and Kaggle images must be downloaded from their original providers. HUMC is a private hospital dataset: no HUMC image, patient identifier, label workbook, or image-level split table is included. Authorized investigators can place HUMC locally and run the same public code without editing source paths.

## Reproducibility scope

| Component | Included | Notes |
|---|---:|---|
| PIID/Kaggle curation and exact exclusion lists | Yes | 20 duplicate pairs and 28 excluded images are documented |
| Center-square crop and 224 x 224 resize | Yes | Applied before training and evaluation |
| PIID split and fold normalization | Yes | Released split tables, seed 40 |
| HUMC patient-level workflow | Code only | Aggregate split metadata and normalization are public; identifiers are private |
| Six architectures x 17 conditions x five folds | Yes | PIID-trained and HUMC-trained entry points |
| Internal and external validation | Yes | PIID, HUMC when locally authorized, and Kaggle |
| No-ensemble fold-wise bootstrap | Yes | Five fold-specific models remain separate |
| Friedman--Iman-Davenport--Nemenyi analysis | Yes | Dataset-specific architecture analysis |
| Critical-difference, Sankey, ROC, confusion, heatmap figures | Yes | Generated from fold-wise prediction files |
| ResNet-18 feature extraction/UMAP | Yes | Checkpoint checksum and transforms documented |
| HUMC clinical data | No | Not shareable |
| Trained weights and generated predictions | No | Recreate locally or distribute separately under an approved data/model agreement |

## Environment

The curated environment uses Python 3.11 and the study versions:

- PyTorch 2.9.0
- TorchVision 0.24.0
- timm 1.0.22
- Albumentations 1.4.4
- random seed/base seed 40

```bash
python -m pip install -r requirements.txt
python code/check_environment.py
```

GPU, CUDA, cuDNN, and deterministic-kernel differences can cause small numerical variation. The model/split configuration is fixed in the released scripts.

Existing local weights can be checked without training or inference:

```bash
python code/check_checkpoint_compatibility.py
```

## Data preparation

Place the downloaded public data at:

```text
data/raw/PIID/original_images/{1,2,3,4}/
data/raw/Kaggle/original_images/{Stage_I,Stage_II,Stage_III,Stage_IV}/
```

Build the analytic folders:

```bash
python code/data_curation/prepare_public_datasets.py --overwrite
```

For every image, EXIF orientation is first corrected. The largest possible square is then cropped from the image center: `side = min(width, height)`, with the crop centered on the midpoint of the long axis. The square is resized to 224 x 224. This is the image representation used by the training and evaluation pipelines.

Expected public analytic counts are 1,081 PIID images and 141 Kaggle images. The complete pair-level decisions are in `code/data_curation/duplicate_pairs.csv`; executable exclusion manifests are in the same directory.

## Splits and normalization

The released PIID split is already in `data/splits/piid_main`. It uses an image-level stratified 15% internal test set and five stratified folds on the remaining 85%.

```bash
python code/experiment/dataset_split_normalization_piid_main.py --use-existing
```

HUMC uses a patient-level 15% held-out test set and five patient-grouped folds. Its public aggregate metadata is in `data/splits/humc_patient_level`; authorized users can regenerate private split tables as follows:

```bash
python code/experiment/dataset_split_normalization_humc_patient_level.py
```

Fold-wise RGB mean and standard deviation are calculated from training images only. See `docs/HUMC_PRIVATE_DATA.md` before using HUMC.

## Training: six architectures and 17 conditions

The architectures are ResNet-50, DenseNet-121, EfficientNetV2-S, ViT-B/16, Swin-T, and ConvNeXt-S. Each is trained over five folds under no augmentation plus 16 augmentation conditions. The exact operator definitions and probabilities are in `docs/AUGMENTATION_CONFIGS.md`.

```bash
# Public PIID development experiment
python code/experiment/train_piid_6models_17augmentations.py

# Private HUMC development experiment, after local data placement
python code/experiment/train_humc_6models_17augmentations.py
```

The default optimizer is AdamW, with weighted cross entropy, batch size 16, learning rate `1e-5`, weight decay `1e-4`, at most 50 epochs, and early-stopping patience 20. The study random seed is fixed at 40. The original fold-specific `DataLoader` shuffle sequence is reproduced with generator seed `40 + fold_id`; this does not change the study-wide seed or the five-fold partition seed.

A quick code-path check is available with:

```bash
python code/experiment/train_piid_6models_17augmentations.py --models resnet50 --augmentations exp00_NoAug --folds 1 --epochs 1
```

## Internal and external evaluation

```bash
# PIID-trained models: PIID internal test, Kaggle, and optional local HUMC
python code/experiment/evaluate_piid_trained_final_results.py

# HUMC-trained models: HUMC internal test, PIID, and Kaggle
python code/experiment/evaluate_humc_trained_final_results.py
```

Each fold model is evaluated separately. Prediction CSVs are written under `data/results/source_archives/inference_results_{piid|humc}` and are the common source for the downstream statistics and figures.

## Statistical analysis and figures

```bash
# Image-level percentile bootstrap of mean fold macro-F1; no prediction ensembling
python code/analysis/bootstrap_macro_f1_foldwise.py --training both --n-bootstrap 1000 --seed 40

# Dataset-specific fold-wise architecture comparison
python code/analysis/friedman_nemenyi_foldwise.py --training both

# Critical-difference diagrams from the preceding tables
python code/visualization/plot_critical_difference.py --training both

# Fold-averaged confusion-flow Sankey diagrams
python code/visualization/plot_sankey_fold_averaged.py --training both

# Confusion matrices, ROC curves, and augmentation heatmaps
python code/visualization/plot_evaluation_results.py --training both

# Direction of adjacent and non-adjacent staging errors
python code/analysis/staging_error_direction.py --training both
```

The bootstrap analysis does not average probabilities or majority-vote predictions across folds. It calculates macro-F1 for each fold-specific model on the same resampled image indices and then averages the five fold scores within each replicate.

## ResNet-18 feature workflow

The duplicate-candidate screen and feature-space analysis use the same ResNet-18 encoder and the same checkpoint. The checkpoint is not tracked because of its size. Place it at:

```text
data/results/checkpoint/feature_extractors/resnet18.pth
```

Then run:

```bash
python code/data_curation/screen_duplicate_candidates.py --dataset PIID=data/processed/analytic_data/PIID --dataset Kaggle=data/processed/analytic_data/Kaggle --output data/results/table/duplicate_screen/public_candidates.csv

python code/analysis/extract_resnet18_features.py --dataset PIID=data/processed/analytic_data/PIID --dataset Kaggle=data/processed/analytic_data/Kaggle --output-dir data/results/source_archives/resnet18_features

python code/analysis/feature_space_analysis.py --feature-root data/results/source_archives/resnet18_features --datasets PIID Kaggle --output-dir data/results/table/feature_space
```

Authorized users can add `HUMC=data/processed/analytic_data/HUMC` to the two extraction commands and add `HUMC` to `--datasets` to recreate the three-dataset study analysis. See `docs/RESNET18_FEATURES.md` for the exact SHA-256 checksum and the intentional preprocessing difference between candidate screening and raw 512-dimensional feature analysis.

## Repository structure

```text
code/
  data_curation/   public dataset preparation and duplicate audit
  development/     shared models and portable path configuration
  experiment/      splits, training, internal/external evaluation
  analysis/        bootstrap, rank tests, feature and error analyses
  visualization/   evaluation, critical-difference, and Sankey figures
data/
  raw/             user-downloaded inputs, excluded from Git
  processed/       generated center-square analytic images, excluded from Git
  private/         optional authorized HUMC inputs, excluded from Git
  splits/          public PIID split and privacy-safe HUMC metadata
  results/         generated checkpoints, predictions, tables, and figures
docs/              method and privacy notes
```

## Public release notes

- All runtime paths are repository-relative; there are no personal workstation or server paths.
- Private data, patient identifiers, images, checkpoints, and generated prediction files are ignored by Git.
- The repository intentionally contains no remote URL, so it can be initialized and uploaded under the official account chosen by the research team.
- An institutional owner should select and approve the software license before or at public release; no license grant is assumed by this package.

Run `python code/validate_release_package.py` and `python code/generate_release_checksums.py --verify` before uploading. The release checklist is in `docs/PUBLIC_RELEASE_CHECKLIST.md`.
