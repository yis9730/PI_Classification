# Reproduction workflow

Run commands from the repository root. Use the main environment unless a step explicitly says `umap`.

## 1. Prepare public datasets

```bash
python code/data_curation/prepare_public_datasets.py --piid-source /path/to/PIID --kaggle-source /path/to/Kaggle --overwrite
```

Expected final counts are PIID 1,081 and Kaggle 141. PIID files are retained
unchanged; Kaggle files are converted to the native-size centre-square
analytic frames used for manuscript external validation. This is not a
`Resize(256) -> CenterCrop(224)` model transform. The script writes curation
manifests under `data/results/manifests/`; those are runtime artefacts and are
not committed.

## 2. Confirm or regenerate splits

```bash
python code/pipeline/dataset_split_normalization_piid_main.py --use-existing
```

The released PIID split uses a stratified image-level 15% test set and five folds in the remaining 85%. HUMC uses patient-disjoint splitting; only an authorised holder can run:

```bash
python code/pipeline/dataset_split_normalization_humc_patient_level.py
```

Rebuild manuscript Table 1 from the released aggregate-only source:

```bash
python code/analysis/build_cohort_summary_table.py
```

The source includes the PIID, HUMC, and Kaggle counts and percentages already
reported in the manuscript. The HUMC row is a published aggregate, not a
patient-level extract; rebuilding the table does not require private HUMC data.

## 3. Train fold-specific models

```bash
# Full public development run
python code/pipeline/train_piid_6models_17augmentations.py

# Short installation/code-path test
python code/pipeline/train_piid_6models_17augmentations.py --models resnet50 --augmentations exp00_NoAug --folds 1 --epochs 1
```

The full run covers six architectures, 17 augmentation conditions, and five folds. It writes fold checkpoints below `data/results/checkpoints/`.

## 4. Evaluate internal and external validation

```bash
python code/pipeline/evaluate_piid_trained_final_results.py
```

This writes one prediction CSV per fold/model/dataset beneath `data/results/predictions/`. These files are the single source of truth for performance tables, bootstrap confidence intervals, rank tests, and error-flow figures; never substitute an implicit five-fold probability ensemble.

## 5. Analyse predictions and generate Figures 1–2

```bash
python code/analysis/bootstrap_macro_f1_foldwise.py --training piid
python code/analysis/friedman_nemenyi_foldwise.py --training piid
python code/analysis/staging_error_direction.py --training piid
python code/visualization/plot_evaluation_results.py --training piid
python code/visualization/plot_sankey_fold_averaged.py --training piid
python code/visualization/plot_critical_difference.py --training piid
```

## 6. Create Table 3 and Figure 4

Place the documented fixed ResNet-18 feature extractor weight at `data/results/checkpoints/feature_extractors/resnet18.pth`, then run:

```bash
python code/analysis/extract_resnet18_features.py --dataset PIID=data/piid --dataset Kaggle=data/kaggle --output-dir data/results/tables/feature_space/features

python code/analysis/feature_space_statistics.py --feature-root data/results/tables/feature_space/features --datasets PIID Kaggle --output-dir data/results/tables/feature_space

python code/visualization/plot_centroid_montage.py --representatives data/results/tables/feature_space/centroid_representatives.csv --project-root . --output data/results/figures/public_centroid_representatives.png
```

## 7. Create Figure 3 in the separate UMAP environment

Activate `umap`, then run:

```bash
python code/visualization/plot_umap.py --feature-root data/results/tables/feature_space/features --datasets PIID Kaggle --output-dir data/results/figures/figure_3_umap
```

The public commands create a two-dataset subset for code-path validation. For the full three-dataset manuscript analysis, include HUMC at feature extraction time and pass `PIID HUMC Kaggle` to both feature-space commands. Then run the centroid montage with the same representatives CSV. See `HUMC_PRIVATE_DATA.md` for the controlled-data boundary.
