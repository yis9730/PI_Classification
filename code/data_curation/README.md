# Data curation and duplicate screening

`prepare_public_datasets.py` builds the public analytic PIID and Kaggle folders.
It applies the released exclusion manifests, center-crops the long image axis
around its midpoint to obtain the largest square, and resizes the square to
224 x 224 pixels.

The screening procedure used a frozen ImageNet-pretrained ResNet-18 encoder
(512-D features), cosine similarity >= 0.85, 128 x 128 pixel comparison, and
normalized MAE <= 0.15. Candidate pairs were then reviewed by experts. The
automatic scores identified candidates; they did not determine final exclusion.

Files:

- `duplicate_pairs.csv`: all 20 public-dataset duplicate pairs and decisions.
- `piid_duplicate_exclusions.csv`: 10 PIID files excluded from the analytic set.
- `kaggle_duplicate_exclusions.csv`: 18 Kaggle files excluded from the analytic set.
- `screen_duplicate_candidates.py`: reproducible candidate-screening code.
- `prepare_public_datasets.py`: non-destructive public-data preparation.

HUMC identifiers and pair-level records are not distributed because HUMC is a
private institutional dataset. The same screening code can be used locally.
