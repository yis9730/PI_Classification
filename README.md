# Pressure Injury Stage Classification Reproduction

This repository is the publication-ready code package for four-class pressure injury staging. It contains the data-curation rules, six-model training pipeline, all 17 augmentation conditions, fold-wise internal and external evaluation, ResNet-18 feature analysis, statistical analysis, and figure-generation code used in the study.

The public PIID and Kaggle images must be downloaded from their original providers. HUMC is a private hospital dataset: no HUMC image, patient identifier, label workbook, or image-level split table is included. Authorized investigators can place HUMC locally and run the same public code without editing source paths.

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

## Data preparation

Download and extract the public datasets from their original pages:

- [PIID original download](https://drive.google.com/drive/u/0/folders/12JouktrzXIo6ywpSe2OYWRYNNIxlEKvK): use the folder containing stage subfolders `1`, `2`, `3`, and `4`.
- [Kaggle Pressure Ulcers Stages](https://www.kaggle.com/datasets/sinemgokoz/pressure-ulcers-stages): use the folder containing `Stage_I`, `Stage_II`, `Stage_III`, and `Stage_IV`.

The downloaded folders may be stored anywhere. Prepare them with:

```bash
python code/data_curation/prepare_public_datasets.py --piid-source /path/to/PIID --kaggle-source /path/to/Kaggle --overwrite
```

Prepared images are saved in `data/piid` and `data/kaggle`. For every image, EXIF orientation is first corrected. The largest possible square is then cropped from the image center: `side = min(width, height)`, with the crop centered on the midpoint of the long axis. The square is resized to 224 x 224. This is the image representation used by the training and evaluation pipelines.

Expected public analytic counts are 1,081 PIID images and 141 Kaggle images. The complete pair-level decisions are in `code/data_curation/duplicate_pairs.csv`; executable exclusion manifests are in the same directory.

## Splits and normalization

The released PIID split is already in `data/splits/piid`. It uses an image-level stratified 15% internal test set and five stratified folds on the remaining 85%.

```bash
python code/experiment/dataset_split_normalization_piid_main.py --use-existing
```

HUMC uses a patient-level 15% held-out test set and five patient-grouped folds. Its public aggregate metadata is in `data/splits/humc`; authorized users can regenerate private split tables as follows:

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

Study-trained classification weights are not distributed in this repository. These commands train new fold-specific weights locally. PIID-trained weights are saved under `data/results/checkpoints/piid_trained`; authorized HUMC training writes to `data/results/checkpoints/humc_trained`.

The default optimizer is AdamW, with weighted cross entropy, batch size 16, learning rate `1e-5`, weight decay `1e-4`, at most 50 epochs, and early-stopping patience 20. The study random seed is fixed at 40. The original fold-specific `DataLoader` shuffle sequence is reproduced with generator seed `40 + fold_id`; this does not change the study-wide seed or the five-fold partition seed.

A quick code-path check is available with:

```bash
python code/experiment/train_piid_6models_17augmentations.py --models resnet50 --augmentations exp00_NoAug --folds 1 --epochs 1
```

## Internal and external evaluation

Run evaluation after the corresponding training step. The scripts automatically load the newly generated local weights from `data/results/checkpoints`.

```bash
# PIID-trained models: PIID internal test, Kaggle, and optional local HUMC
python code/experiment/evaluate_piid_trained_final_results.py

# HUMC-trained models: HUMC internal test, PIID, and Kaggle
python code/experiment/evaluate_humc_trained_final_results.py
```

Each fold model is evaluated separately. Prediction CSVs are written under `data/results/predictions/{piid|humc}` and are the common source for the downstream statistics and figures.

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
data/results/checkpoints/feature_extractors/resnet18.pth
```

Then run:

```bash
python code/data_curation/screen_duplicate_candidates.py --dataset PIID=data/piid --dataset Kaggle=data/kaggle --output data/results/tables/duplicate_screen/public_candidates.csv

python code/analysis/extract_resnet18_features.py --dataset PIID=data/piid --dataset Kaggle=data/kaggle --output-dir data/results/tables/feature_space/features

python code/analysis/feature_space_analysis.py --feature-root data/results/tables/feature_space/features --datasets PIID Kaggle --output-dir data/results/tables/feature_space
```

Authorized users can add `HUMC=data/humc` to the two extraction commands and add `HUMC` to `--datasets` to recreate the three-dataset study analysis. See `docs/RESNET18_FEATURES.md` for the exact SHA-256 checksum and the intentional preprocessing difference between candidate screening and raw 512-dimensional feature analysis.

## Repository structure

```text
code/   analysis and reproduction scripts
data/   prepared images, splits, and generated results
docs/   method and privacy notes
```

## Public release notes

- All runtime paths are repository-relative; there are no personal workstation or server paths.
- Private data, patient identifiers, images, checkpoints, and generated prediction files are ignored by Git.
- An institutional owner should select and approve the software license before or at public release; no license grant is assumed by this package.

Run `python code/validate_release_package.py` before uploading. The release checklist is in `docs/PUBLIC_RELEASE_CHECKLIST.md`.
