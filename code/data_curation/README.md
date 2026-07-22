# Data curation and duplicate screening

`prepare_public_datasets.py` builds the public analytic PIID and Kaggle folders.
It applies the released exclusion manifests, copies retained PIID files
unchanged, and centre-crops only the longer axis of retained Kaggle images to
their native square size. This reproduces the study's `Kaggle_cropped`
external-validation input without a 256-pixel resize. The classification
pipeline subsequently applies the direct 224 x 224 model-input resize in
memory with Albumentations.

The supplementary screening utility uses a frozen ImageNet-pretrained
ResNet-18 encoder (512-D features) with cosine similarity >= 0.85 and reports a
128 x 128 pixel-comparison flag at normalized MAE <= 0.15. Candidate pairs were
reviewed by experts; automatic scores did not determine final exclusion. The
utility directly resizes model inputs to 224 x 224 and does not modify dataset
files. `duplicate_pairs.csv` and the two exclusion manifests are the
authoritative released curation record.

Files:

- `duplicate_pairs.csv`: all 20 public-dataset duplicate pairs and decisions.
- `piid_duplicate_exclusions.csv`: 10 PIID files excluded from the analytic set.
- `kaggle_duplicate_exclusions.csv`: 18 Kaggle files excluded from the analytic set.
- `screen_duplicate_candidates.py`: supplementary candidate-screening utility.
- `prepare_public_datasets.py`: non-destructive public-data preparation.

HUMC identifiers and pair-level records are not distributed because HUMC is a
private institutional dataset. The same screening code can be used locally.
