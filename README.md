# Cross-Dataset Reliability of Deep Learning Models for Pressure Injury Staging: A Multi-Architecture Evaluation and Prediction-Error Analysis

This repository provides the code and supporting materials for the accompanying manuscript. We evaluated six ImageNet-pretrained CNN and Transformer architectures across PIID, an institutional HUMC cohort, and Kaggle to examine whether pressure injury staging performance obtained within one dataset transfers to independently acquired and curated wound images.

The study found substantial cross-dataset performance loss and dataset-dependent staging errors, highlighting the need for independent external validation and more standardised wound-image acquisition. Detailed execution instructions and manuscript-result mappings are linked below.

## Reproducibility scope

- [PIID](https://drive.google.com/drive/u/0/folders/12JouktrzXIo6ywpSe2OYWRYNNIxlEKvK) and [Kaggle Pressure Ulcers Stages](https://www.kaggle.com/datasets/sinemgokoz/pressure-ulcers-stages) remain available from their original providers. Scripts are supplied to prepare and analyse these public datasets.
- HUMC is an institutional dataset and is not distributed. The repository contains no HUMC images, identifiers, or row-level records; HUMC-dependent analyses require institutionally approved local access.
- Only the aggregate HUMC counts and percentages already reported in manuscript Table 1 are included. Generated results and controlled-data outputs remain local and are ignored by Git.

## Quick start

Create the main environment from the repository root:

```powershell
python -m venv .venv-main
.\.venv-main\Scripts\Activate.ps1
python -m pip install -r requirements_train_eval.txt
python code/check_environment.py
```

The main environment covers public-data preparation, training, evaluation, statistical analysis, and all visualisations except UMAP. Main Figure 3 uses the separate UMAP environment described in the [environment guide](docs/ENVIRONMENTS.md).

Follow the [reproduction workflow](docs/REPRODUCTION_WORKFLOW.md) for the complete command sequence. Before publishing a revision, run `python code/validate_release_package.py`.

## Documentation

- [Reproduction workflow](docs/REPRODUCTION_WORKFLOW.md): complete execution order
- [Environment guide](docs/ENVIRONMENTS.md): main and UMAP environments
- [Methods-to-code map](docs/METHODS_IMPLEMENTATION.md): manuscript methods and their implementations
- [Main-artifact map](docs/MAIN_ARTIFACTS.md): code and inputs for each main table and figure
- [HUMC data boundary](docs/HUMC_PRIVATE_DATA.md): controlled-data requirements
- [Validation record](docs/VALIDATION.md): package and representative rerun checks

## Repository layout

```text
code/
  data_curation/   public-dataset curation
  pipeline/        data splitting, training, and evaluation
  analysis/        statistical and feature-space analyses
  visualization/   manuscript figures
  core/            shared model and path utilities
data/              public metadata, local input paths, and generated-output paths
docs/              detailed reproduction documentation
```
