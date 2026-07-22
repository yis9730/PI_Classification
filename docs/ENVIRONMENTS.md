# Two intentionally separate environments

The project uses two pinned environments because the recorded workflows have incompatible NumPy constraints. Do not attempt to install both requirements files into one virtual environment.

| Environment | Requirements file | Purpose | Key compatibility constraint |
|---|---|---|---|
| Main environment | `requirements_train_eval.txt` | curation, split generation, training, evaluation, statistical analysis, Sankey, centroid montage | NumPy 2.2.6 with OpenCV 4.12.0.88 |
| UMAP environment | `requirements_umap_analysis.txt` | Main Figure 3 UMAP from pre-exported features | NumPy 1.26.4 with numba 0.60.0 and UMAP 0.5.6 |

## Create the main environment

```bash
python -m venv .venv-main
.\.venv-main\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements_train_eval.txt
python code/check_environment.py
```

Use this environment for every script except `code/visualization/plot_umap.py`. It is the environment selected by `requirements.txt`.

## Create the UMAP environment

```bash
python -m venv .venv-umap
.\.venv-umap\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements_umap_analysis.txt
python code/check_environment.py --environment umap
```

The UMAP environment consumes `features.npy`, `metadata.csv`, and their
validated `extraction.json` provenance previously written by
`code/analysis/extract_resnet18_features.py`; it does not train a model,
download weights, or alter the numerical feature-space tables.

## Why this separation is required

The original UMAP notebook failed in the training/evaluation environment with the dependency message that numba requires NumPy 2.0 or lower. Pinning UMAP to NumPy 1.26.4 preserves the intended visualisation workflow. Conversely, forcing the training/evaluation stack down to that older NumPy version would no longer represent its recorded OpenCV/NumPy package combination.

For results comparison, record Python, CUDA driver, GPU model, CUDA/cuDNN versions, and package versions. GPU differences can introduce numerical variation, but they should not be used as the sole explanation for a material performance discrepancy without an otherwise matched rerun.
