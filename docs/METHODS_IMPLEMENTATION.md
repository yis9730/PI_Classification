# Methods-to-code map

| Study method | Authoritative implementation |
|---|---|
| Authoritative pair decisions and exclusions | `code/data_curation/duplicate_pairs.csv`, exclusion manifests |
| PIID unchanged copy; Kaggle native short-side centre crop | `code/data_curation/prepare_public_datasets.py` |
| Direct 224 x 224 model-input resize | `A.Resize` in `code/pipeline/*.py` |
| Cohort and split summary (Table 1) | `code/analysis/build_cohort_summary_table.py` |
| PIID image-level split | `code/pipeline/dataset_split_normalization_piid_main.py` |
| HUMC patient-level split | `code/pipeline/dataset_split_normalization_humc_patient_level.py` |
| Six-model, 17-condition training | `code/pipeline/train_*_6models_17augmentations.py` |
| Internal/external validation | `code/pipeline/evaluate_*_trained_final_results.py` |
| No-ensemble bootstrap | `code/analysis/bootstrap_macro_f1_foldwise.py` |
| Architecture rank analysis | `code/analysis/friedman_nemenyi_foldwise.py` |
| Critical-difference diagram | `code/visualization/plot_critical_difference.py` |
| Fold-averaged confusion Sankey | `code/visualization/plot_sankey_fold_averaged.py` |
| Frozen official timm ResNet-18 A1 feature extraction | `code/analysis/extract_resnet18_features.py` |
| Raw pooled 512-D feature export | `code/analysis/extract_resnet18_features.py` |
| Silhouette and centroid analysis | `code/analysis/feature_space_statistics.py` |
| UMAP (Main Figure 3) | `code/visualization/plot_umap.py` in the `umap` environment |
| Centroid montage (Main Figure 4) | `code/visualization/plot_centroid_montage.py` |

Evaluation never substitutes a five-fold probability ensemble for a fold-specific model. When a figure pools fold outputs for display, the corresponding code labels that operation explicitly.
