# Pressure Injury Stage Classification: reproducible analysis package

This repository is the public, publication-oriented implementation for four-class pressure-injury staging. It is organised around a complete reproduction path: data curation → split generation → training/evaluation → statistical analysis → main figures and tables.

PIID and Kaggle source images remain downloadable from their original providers. HUMC is a controlled hospital dataset and is not distributed here. The code and expected output schemas are public, but HUMC-dependent results require separately authorised local access.

## Start here

1. Read [the repository architecture](docs/REPOSITORY_ARCHITECTURE.md) and [the environment guide](docs/ENVIRONMENTS.md).
2. Create the main environment, then run the public-data curation and PIID training/evaluation workflow.
3. Use the separate `umap` environment only after feature vectors have been exported, to generate Main Figure 3.
4. Before publishing a revision, run `python code/validate_release_package.py`.

The end-to-end command order and expected generated files are in [the reproduction workflow](docs/REPRODUCTION_WORKFLOW.md). The exact script and input contract for each main manuscript result is in [the main-artifact map](docs/MAIN_ARTIFACTS.md).

## Environment

Two environments are deliberate, not optional duplication:

```bash
# Main environment: curation, splitting, training, evaluation, tables, Sankey, Figure 4
python -m venv .venv-main
.\.venv-main\Scripts\Activate.ps1
python -m pip install -r requirements_train_eval.txt
python code/check_environment.py

# Separate: UMAP / Main Figure 3, after feature extraction is complete
python -m venv .venv-umap
.\.venv-umap\Scripts\Activate.ps1
python -m pip install -r requirements_umap_analysis.txt
python code/check_environment.py --environment umap
```

The recorded training/evaluation stack uses NumPy 2.2.6 and OpenCV 4.12. The recorded UMAP stack instead uses NumPy 1.26.4 because its `numba` dependency does not support the newer NumPy stack. Details and scope are documented in [`docs/ENVIRONMENTS.md`](docs/ENVIRONMENTS.md).

## Public data preparation

Download the source datasets from their original providers:

- [PIID](https://drive.google.com/drive/u/0/folders/12JouktrzXIo6ywpSe2OYWRYNNIxlEKvK): stage folders `1`–`4`.
- [Kaggle Pressure Ulcers Stages](https://www.kaggle.com/datasets/sinemgokoz/pressure-ulcers-stages): folders `Stage_I`–`Stage_IV`.

Run the following in the main environment:

```bash
python code/data_curation/prepare_public_datasets.py --piid-source /path/to/PIID --kaggle-source /path/to/Kaggle --overwrite

python code/pipeline/dataset_split_normalization_piid_main.py --use-existing
```

The curation script applies the released duplicate exclusions. Retained PIID
files are copied byte-for-byte. Retained Kaggle images are centre-cropped to
their native short-side square at native resolution. The model pipeline later
maps both datasets directly to `224 x 224` in memory with Albumentations. The
stochastic `A.CenterCrop` belongs only to the centre-zoom training augmentation. The
script validates the target counts (PIID 1,081; Kaggle 141) and rejects
source/output overlap, unexpected raw counts, and unmatched exclusion entries.

## Core study pipeline

All commands below are run in the main environment.

```bash
# PIID development training: six architectures × 17 conditions × five folds
python code/pipeline/train_piid_6models_17augmentations.py

# Publicly accessible evaluation: PIID internal test and Kaggle external validation
python code/pipeline/evaluate_piid_trained_final_results.py

# Optional public two-dataset Sankey code-path check (not Main Figure 2)
python code/visualization/plot_sankey_fold_averaged.py --training piid --datasets PIID_Test Kaggle
```

If HUMC is absent, the evaluator deliberately skips that controlled dataset.
The resulting PIID/Kaggle predictions validate the public path but are not a
complete input set for manuscript Table 2 or Figures 1–2. With authorised HUMC
data available, rerun PIID-trained evaluation, train and evaluate the HUMC
development direction, and then generate the complete manuscript analyses:

```bash
python code/pipeline/evaluate_piid_trained_final_results.py
python code/pipeline/train_humc_6models_17augmentations.py
python code/pipeline/evaluate_humc_trained_final_results.py

python code/analysis/bootstrap_macro_f1_foldwise.py --training piid
python code/analysis/friedman_nemenyi_foldwise.py --training piid
python code/analysis/staging_error_direction.py --training piid
python code/visualization/plot_evaluation_results.py --training piid
python code/visualization/plot_critical_difference.py --training piid
python code/visualization/plot_sankey_fold_averaged.py --main-figure
```

Review [`docs/HUMC_PRIVATE_DATA.md`](docs/HUMC_PRIVATE_DATA.md) before any
controlled-data run. The analysis scripts intentionally require complete
three-dataset inputs when they generate a full manuscript result.

## Feature workflow: Main Figures 3–4 and Table 3

These manuscript analyses resize each prepared analytic image directly to
`224 x 224`.
First export the shared ResNet-18 feature vectors in the main environment:

The script obtains the official timm `resnet18.a1_in1k` weight automatically,
verifies the complete cached-file SHA-256, removes only its classifier tensors,
and exports the same raw pooled 512-D vectors used by the study.

```bash
python code/analysis/extract_resnet18_features.py --dataset PIID=data/piid --dataset Kaggle=data/kaggle --output-dir data/results/tables/feature_space/features

python code/analysis/feature_space_statistics.py --feature-root data/results/tables/feature_space/features --datasets PIID Kaggle --output-dir data/results/tables/feature_space

python code/visualization/plot_centroid_montage.py --representatives data/results/tables/feature_space/centroid_representatives.csv --project-root . --output data/results/figures/public_centroid_representatives.png
```

The command above is the independent numerical rerun. To render the archived
PIID/Kaggle mean-centroid selections used as the public Figure 4 reference, run:

```bash
python code/visualization/plot_centroid_montage.py --representatives data/reference/figure4_public_mean_representatives.csv --project-root . --output data/results/figures/public_centroid_representatives_reference.png
```

The reference manifest contains public filenames only. The corresponding HUMC
selection rows remain controlled and are not committed.

Then activate `umap` and render Main Figure 3 from the exported vectors:

```bash
python code/visualization/plot_umap.py --feature-root data/results/tables/feature_space/features --datasets PIID Kaggle --output-dir data/results/figures/figure_3_umap
```

The public two-dataset commands produce a partial Figure 4 montage. An authorised HUMC holder adds `--dataset HUMC=data/humc` during extraction and includes `HUMC` in the subsequent statistics and UMAP commands to regenerate the full three-dataset Main Figures 3–4.

## Repository layout

```text
code/
  core/            shared paths, model factory, parameter reference
  data_curation/   public data preparation and duplicate decisions
  pipeline/        splits, training, internal/external evaluation
  analysis/        statistics and feature-space numerical analysis
  visualization/   manuscript-facing figures, including UMAP and Sankey
data/              input locations, approved aggregates, public split metadata, generated outputs
docs/              environment, privacy, workflow, and result contracts
```

Generated images, predictions, checkpoints, and private HUMC material are ignored by Git. The released HUMC-derived data are limited to the aggregate counts and percentages already reported in Table 1. The release must not contain controlled images, patient identifiers, or private row-level split/prediction files.
