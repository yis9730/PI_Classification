# Data curation

`review_duplicate_candidates.py` reads the raw PIID and Kaggle provider folders
before exclusion or square curation. It runs two independent all-pairs screens
at similarity `>= 0.85`:

- L2-normalized ResNet-18 features after a direct `224 x 224` resize, compared
  by cosine similarity.
- RGB pixels after a direct `128 x 128` resize, compared as `1 - MAE` on values
  scaled to `[0, 1]`.

Run it from the repository root in the main environment:

```bash
python code/data_curation/review_duplicate_candidates.py --piid-source /path/to/PIID --kaggle-source /path/to/Kaggle --overwrite
```

The complete `feature_candidate_pairs.csv` and `pixel_candidate_pairs.csv`
tables are saved from `1.00` down to `0.85`. For practical visual review, each
method also takes the union of every candidate image's strongest pair, sorts
that queue by decreasing similarity, and writes paginated montages. These
generated files are stored under `data/results/duplicate_review/` and are not
committed.

The threshold creates a review queue rather than an automatic deletion list.
Human inspection of the montages produced the released pair decisions and the
final exclusions: 10 PIID images and 18 Kaggle images. The script checks that
every released pair appears in at least one of the two independently generated
candidate tables, verifies the released counts, and fingerprints the sources
before and after the run. It does not change or delete source images.

`prepare_public_datasets.py` consumes those released exclusion manifests and
builds the public analytic PIID and Kaggle folders. It copies retained PIID
files byte-for-byte and centre-crops retained Kaggle images to a square whose
side equals the image's native shorter side at native resolution. The
classification pipeline subsequently applies the direct 224 x 224 model-input
resize in memory with Albumentations. The centre-zoom training condition
additionally uses stochastic `A.CenterCrop` augmentation.

The classification training, validation, and test pipelines separately use
Albumentations `A.Resize(224, 224)`. `duplicate_pairs.csv` and the two exclusion
manifests are the authoritative released curation record.

Files:

- `review_duplicate_candidates.py`: raw-source feature/pixel candidate tables,
  review montages, and released-decision verification.
- `duplicate_pairs.csv`: all 20 public-dataset duplicate pairs and decisions.
- `piid_duplicate_exclusions.csv`: 10 PIID files excluded from the analytic set.
- `kaggle_duplicate_exclusions.csv`: 18 Kaggle files excluded from the analytic set.
- `prepare_public_datasets.py`: non-destructive public-data preparation.

HUMC identifiers and pair-level records are not distributed because HUMC is a
private institutional dataset. Only the approved aggregate counts and
percentages already reported in Table 1 are public.
