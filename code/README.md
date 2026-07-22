# Code map

## `core`

- `path_config.py`: repository-relative data and output locations.
- `model_pipeline_utils.py`: shared six-backbone model factory and classifier heads.
- `model_params_reference.csv`: parameter-count reference for the six final classification architectures.

## `data_curation`

- `review_duplicate_candidates.py`: independent raw-source feature/pixel
  similarity screens, complete descending candidate CSVs, strongest-pair
  review montages, and verification of the released human decisions.
- `prepare_public_datasets.py`: released exclusions and byte-preserving copies of retained PIID/Kaggle source files.
- `duplicate_pairs.csv` and `*_duplicate_exclusions.csv`: reviewed decisions and executable public exclusion manifests.

## `pipeline`

- `dataset_split_normalization_piid_main.py`: PIID image-level split and per-fold normalisation using Albumentations 224 x 224 resize.
- `dataset_split_normalization_humc_patient_level.py`: controlled HUMC patient-grouped split workflow using the same resize method.
- `train_{piid|humc}_6models_17augmentations.py`: five-fold training entry points for six architectures and 17 augmentation conditions.
- `evaluate_{piid|humc}_trained_final_results.py`: internal and cross-dataset validation from locally generated fold weights.

## `analysis`

- `bootstrap_macro_f1_foldwise.py`: no-ensemble, fold-wise image bootstrap.
- `build_cohort_summary_table.py`: Table 1-ready public/aggregate cohort and split summary.
- `friedman_nemenyi_foldwise.py`: dataset-specific architecture rank analysis.
- `staging_error_direction.py`: adjacent/non-adjacent staging-error summaries.
- `extract_resnet18_features.py`: official timm ResNet-18 A1 raw pooled feature export.
- `feature_space_statistics.py`: silhouette, centroid-distance, and representative-image calculations; it deliberately does not draw UMAP.

## `visualization`

- `plot_evaluation_results.py`: confusion matrices, ROC curves, and augmentation heatmaps.
- `plot_sankey_fold_averaged.py`: fold-averaged confusion-flow Sankey plots.
- `plot_critical_difference.py`: critical-difference diagrams.
- `plot_umap.py`: Main Figure 3 UMAP from already exported features; run in the separate UMAP environment.
- `plot_centroid_montage.py`: Main Figure 4 centroid-nearest image montage.

## Release checks

- `check_environment.py`: checks the main training/evaluation environment.
- `check_checkpoint_compatibility.py`: checks stored checkpoint architecture compatibility before evaluation.
- `validate_release_package.py`: static privacy, layout, and syntax checks before a public commit.

Every executable supports `--help`. Generated files are written under `data/results`, which is excluded from Git.
