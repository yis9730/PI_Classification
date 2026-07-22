# Reproduction workflow

Run commands from the repository root. Use the main environment unless a step explicitly says `umap`.

## 1. Prepare public datasets

```bash
python code/data_curation/prepare_public_datasets.py --piid-source /path/to/PIID --kaggle-source /path/to/Kaggle --overwrite
```

Expected final counts are PIID 1,081 and Kaggle 141. After the released
exclusions, retained PIID files are copied byte-for-byte. Retained Kaggle
images are centre-cropped to their native short-side square at native
resolution. The model pipeline then applies Albumentations `A.Resize(224,
224)`, while the centre-zoom augmentation remains a separately named training
condition. The script writes
curation manifests under `data/results/manifests/`; those are runtime artefacts
and are not committed.

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
python code/pipeline/train_piid_6models_17augmentations.py --models resnet50 --augmentations exp00_NoAug --folds 1 --epochs 1 --run-tag smoke
```

The full run covers six architectures, 17 augmentation conditions, and five folds. It writes fold checkpoints below `data/results/checkpoints/`.

Main Figure 2 also requires the reverse development direction. An authorised
HUMC holder first prepares the private split described above, then runs:

```bash
python code/pipeline/train_humc_6models_17augmentations.py
```

## 4. Evaluate internal and external validation

```bash
python code/pipeline/evaluate_piid_trained_final_results.py
```

Without authorised HUMC data, this public run evaluates the PIID held-out set
and Kaggle and explicitly skips HUMC. That is a valid public code-path check,
but the resulting two-dataset prediction set cannot generate the complete
Table 2 or Figures 1–2. For a full manuscript rerun, place the authorised HUMC
input, rerun the PIID-trained evaluator, and evaluate the HUMC-trained models:

```bash
python code/pipeline/evaluate_humc_trained_final_results.py
```

The evaluators write one prediction CSV per fold/model/dataset beneath
`data/results/predictions/`. These files are the single source of truth for
performance tables, bootstrap confidence intervals, rank tests, and
error-flow figures; never substitute an implicit five-fold probability
ensemble.

## 5. Analyse predictions and generate Figures 1–2

For a public two-dataset visualization check only:

```bash
python code/visualization/plot_sankey_fold_averaged.py --training piid --datasets PIID_Test Kaggle
```

This is not the submitted Main Figure 2. Once all authorised three-dataset
prediction inputs described above exist, run the full manuscript analyses:

```bash
python code/analysis/bootstrap_macro_f1_foldwise.py --training piid
python code/analysis/friedman_nemenyi_foldwise.py --training piid
python code/analysis/staging_error_direction.py --training piid
python code/visualization/plot_evaluation_results.py --training piid
python code/visualization/plot_critical_difference.py --training piid
python code/visualization/plot_sankey_fold_averaged.py --main-figure
```

## 6. Create Table 3 and Figure 4

The extraction script automatically downloads and fully verifies the official
timm `resnet18.a1_in1k` public weight. Run:

```bash
python code/analysis/extract_resnet18_features.py --dataset PIID=data/piid --dataset Kaggle=data/kaggle --output-dir data/results/tables/feature_space/features

python code/analysis/feature_space_statistics.py --feature-root data/results/tables/feature_space/features --datasets PIID Kaggle --output-dir data/results/tables/feature_space

python code/visualization/plot_centroid_montage.py --representatives data/results/tables/feature_space/centroid_representatives.csv --project-root . --output data/results/figures/public_centroid_representatives.png
```

That command independently recalculates the representative identities. To
render the archived public PIID/Kaggle mean-centroid selections with the
manuscript layout, use the released reference manifest:

```bash
python code/visualization/plot_centroid_montage.py --representatives data/reference/figure4_public_mean_representatives.csv --project-root . --output data/results/figures/public_centroid_representatives_reference.png
```

Small floating-point differences between PyTorch/CUDA versions can change a
nearest-image rank when distances are close. Keep the recalculated CSV as the
independent result and compare it with the reference manifest rather than
silently replacing either file.

## 7. Create Figure 3 in the separate UMAP environment

Activate `umap`, then run:

```bash
python code/visualization/plot_umap.py --feature-root data/results/tables/feature_space/features --datasets PIID Kaggle --output-dir data/results/figures/figure_3_umap
```

The public commands create a two-dataset subset for code-path validation. For the full three-dataset manuscript analysis, include HUMC at feature extraction time and pass `PIID HUMC Kaggle` to both feature-space commands. Then run the centroid montage with the same representatives CSV. The archived HUMC representative manifest remains local. See `HUMC_PRIVATE_DATA.md` for the controlled-data boundary.
