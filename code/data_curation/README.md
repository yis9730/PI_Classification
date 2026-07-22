# Data curation

`prepare_public_datasets.py` builds the public analytic PIID and Kaggle folders.
It applies the released exclusion manifests, copies retained PIID files
byte-for-byte, and centre-crops retained Kaggle images to a square whose side
equals the image's native shorter side at native resolution.
The classification pipeline subsequently applies the direct 224 x 224
model-input resize in memory with Albumentations. The centre-zoom training
condition additionally uses stochastic `A.CenterCrop` augmentation.

The classification training, validation, and test pipelines separately use
Albumentations `A.Resize(224, 224)`. `duplicate_pairs.csv` and the two exclusion
manifests are the authoritative released curation record.

Files:

- `duplicate_pairs.csv`: all 20 public-dataset duplicate pairs and decisions.
- `piid_duplicate_exclusions.csv`: 10 PIID files excluded from the analytic set.
- `kaggle_duplicate_exclusions.csv`: 18 Kaggle files excluded from the analytic set.
- `prepare_public_datasets.py`: non-destructive public-data preparation.

HUMC identifiers and pair-level records are not distributed because HUMC is a
private institutional dataset. Only the approved aggregate counts and
percentages already reported in Table 1 are public.
