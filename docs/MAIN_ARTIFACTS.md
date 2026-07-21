# Main manuscript result contract

Every main figure and table needs a runnable code path, declared input files, and an explicit data-access boundary. Public code is necessary for all results; public images are possible only for PIID/Kaggle. HUMC-dependent artefacts require separately authorised local access and are not reproducible from this public package alone.

| Manuscript artefact | Code path | Required inputs | Generated contract | Access boundary |
|---|---|---|---|---|
| Table 1: cohort/split description | `code/data_curation/prepare_public_datasets.py`; `code/pipeline/dataset_split_normalization_*`; `code/analysis/build_cohort_summary_table.py` | prepared public images and PIID metadata; optional local HUMC aggregate summary | `table_1_cohort_summary.csv` | public PIID/Kaggle rows; HUMC only in a controlled local run |
| Table 2: development/internal/external performance | `code/pipeline/evaluate_*_trained_final_results.py`; `code/analysis/bootstrap_macro_f1_foldwise.py` | locally trained fold checkpoints and fold prediction CSVs | per-fold predictions, `bootstrap_summary.csv`, `fold_metrics.csv` | public rerun for PIID/Kaggle; HUMC controlled |
| Table 3: feature-space distances | `code/analysis/extract_resnet18_features.py`; `code/analysis/feature_space_statistics.py` | fixed ResNet-18 feature weight and dataset images | `centroid_distances.csv`, `silhouette_coefficients.csv` | public rerun for PIID/Kaggle; HUMC controlled |
| Figure 1: performance displays | `code/visualization/plot_evaluation_results.py` | fold prediction CSVs | confusion matrices, ROC curves, augmentation heatmaps | same as Table 2 |
| Figure 2: staging-error Sankey | `code/analysis/staging_error_direction.py`; `code/visualization/plot_sankey_fold_averaged.py` | fold prediction CSVs | direction tables and fold-averaged Sankey figures | same as Table 2 |
| Figure 3: UMAP feature embedding | `code/visualization/plot_umap.py` | exported `features.npy` and `metadata.csv` | `umap_coordinates.csv`, PNG, SVG | public rerun for PIID/Kaggle; HUMC controlled |
| Figure 4: centroid-nearest representatives | `code/analysis/feature_space_statistics.py`; `code/visualization/plot_centroid_montage.py` | feature vectors, metadata, locally accessible selected images | `centroid_representatives.csv` and montage PNG | public rerun for PIID/Kaggle; HUMC controlled and not publicly exported |

## Required release behaviour

1. Do not remove UMAP merely to simplify packaging if Main Figure 3, Table 3, Figure 4, or the corresponding manuscript claims remain. Its implementation is isolated in a second environment so it does not block training/evaluation.
2. Do not claim exact numerical identity across hardware unless it has been measured with a matched rerun. Report the environment and compare generated CSVs against the archived study outputs.
3. Do not publish any HUMC-derived file, including images, thumbnails, labels, identifiers, split membership, normalisation statistics, aggregate summaries, feature arrays, checkpoints, or prediction paths. Preserve executable code and expected output schemas instead.

## Minimum pre-submission verification

```bash
python code/check_environment.py
python code/validate_release_package.py
```

For a full analysis run, retain generated CSVs and figures as a non-public archival bundle and record their SHA-256 checksums in the manuscript revision archive. Those runtime outputs remain ignored by Git because they can contain controlled-data-derived information.
