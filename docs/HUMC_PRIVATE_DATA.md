# HUMC private-data boundary

HUMC contains non-public hospital images and patient-linked metadata. This repository releases executable code but must not contain:

- clinical images or thumbnails;
- label workbooks;
- patient or encounter identifiers;
- image-level train/test tables;
- fold membership files that expose image or patient membership;
- prediction files containing private image paths.

Authorized local layout:

```text
data/processed/analytic_data/HUMC/{1,2,3,4}/
data/private/HUMC/labels/humc_labels.xlsx
```

The analytic HUMC images must undergo the same orientation correction, center-square crop around the midpoint of the long axis, and 224 x 224 resize as the public datasets before split generation or model use.

The public aggregate record reports 1,844 images from 500 patients, with 1,556 images/425 patients in train-validation and 288 images/75 patients in the held-out test. The held-out split and five folds are patient-disjoint. Only `split_meta_public.json` and `normalization_stats.csv` are released.

The default patient identifier column is `등록번호`; command-line arguments permit a site-local schema without source edits. All HUMC local files and outputs should be reviewed again before any public commit.
