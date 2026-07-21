# Code map

## `data_curation`

- `prepare_public_datasets.py`: apply the released exclusions, center-square crop, and resize PIID/Kaggle images.
- `screen_duplicate_candidates.py`: ResNet-18 cosine and pixel-similarity candidate screen for expert review.
- `duplicate_pairs.csv`: all 20 reviewed duplicate pairs and pair-level decisions.
- `*_duplicate_exclusions.csv`: executable exclusion manifests (10 PIID, 18 Kaggle images).

## `development`

- `path_config.py`: repository-relative public/private data and output paths.
- `model_pipeline_utils.py`: shared six-backbone model factory and classifier heads.
- `model_params_reference.csv`: parameter-count reference.

## `experiment`

- `dataset_split_normalization_piid_main.py`: PIID image-level split and fold normalization.
- `dataset_split_normalization_humc_patient_level.py`: private HUMC patient-grouped split workflow.
- `train_{piid|humc}_6models_17augmentations.py`: full five-fold training entry points.
- `evaluate_{piid|humc}_trained_final_results.py`: internal and cross-dataset validation.

## `analysis`

- `bootstrap_macro_f1_foldwise.py`: no-ensemble, fold-wise image bootstrap.
- `friedman_nemenyi_foldwise.py`: dataset-specific rank analysis with Iman-Davenport decision rule.
- `extract_resnet18_features.py` and `feature_space_analysis.py`: shared-encoder feature workflow.
- `staging_error_direction.py`: error-direction summaries.

## `visualization`

- `plot_evaluation_results.py`: confusion matrices, ROC curves, and augmentation heatmaps.
- `plot_critical_difference.py`: critical-difference diagrams.
- `plot_sankey_fold_averaged.py`: fold-averaged confusion-flow diagrams.

Every executable supports `--help`. Generated files are written beneath `data/results`.
