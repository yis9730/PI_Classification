# HUMC controlled-data boundary

HUMC is not an open dataset. The only HUMC-derived file approved for the
current public release is `data/aggregates/table_1_cohort_counts.csv`. Its HUMC
row contains only the initial, excluded, and final image counts plus the four
stage counts and percentages already reported in manuscript Table 1.

The public aggregate contains no patient-level row, name, medical-record or
encounter number, image identifier, filename or path, date, split membership,
clinical attribute, or prediction. Known historical HUMC data blobs reviewed
for this release contain de-identified total/train/test patient and image
counts, stage counts, fold-level RGB normalisation summaries, and a synthetic
template row. No identified or row-level HUMC data blob was found in that
review. These older aggregates remain reachable unless Git history is
rewritten, although they are intentionally absent from the current release
tree.

The paper reports the HUMC cohort and its results. Reproducing the HUMC-dependent portions of the study requires a separate institutional authorisation process and a local controlled-data workspace. This repository is not a data-access mechanism.

An authorised project member keeps all controlled inputs and outputs outside Git, including local files under these ignored locations:

```text
data/humc/
data/splits/humc/
data/results/
```

## Authorised local input contract

The split script expects the final analytic HUMC cohort in this exact local
layout by default:

```text
data/humc/
  labels.xlsx
  1/                    # final Stage 1 images
  2/                    # final Stage 2 images
  3/                    # final Stage 3 images
  4/                    # final Stage 4 images
```

After authorised duplicate removal, each rectangular image in the final
1,844-image HUMC cohort is deterministically centre-cropped to a square whose
side equals its native shorter side; already-square images remain square. This
curation operation retains the native short-side resolution. The later
classification pipeline applies Albumentations `A.Resize(224, 224)` to every
model input, while stochastic `A.CenterCrop` appears only in the separately
named centre-zoom augmentation. The HUMC curation outputs remain controlled
and are not distributed.

Images must be direct children of their stage folder. Supported extensions are
`.jpg`, `.jpeg`, `.png`, `.bmp`, `.tif`, `.tiff`, and `.webp`, matched without
case sensitivity. The four folders must all exist and each must contain at
least one supported image. The validated cohort must contain exactly 1,844
images with the published Stage 1-4 counts `233/709/575/327`, matching the
aggregate values already reported in Table 1.

The default workbook columns are:

| Purpose | Default column | Requirement |
|---|---|---|
| Image key | `number` | Non-null, non-blank, and unique |
| Patient grouping key | `등록번호` | Non-null and non-blank |
| Declared stage | `stage` | Integer 1, 2, 3, or 4 |

Each image filename without its extension must equal exactly one value in the
`number` column. Numeric Excel keys such as `123.0` are normalised to `123`;
text keys are stripped of surrounding whitespace. Every eligible image must
have one workbook row, every workbook row must have one image, no filename stem
may occur more than once across the four folders, and the workbook stage must
equal the containing folder number.

Run the defaults from the repository root:

```bash
python code/pipeline/dataset_split_normalization_humc_patient_level.py
```

The corresponding explicit arguments are:

```bash
python code/pipeline/dataset_split_normalization_humc_patient_level.py \
  --labels data/humc/labels.xlsx \
  --image-root data/humc \
  --number-col number \
  --patient-id-col 등록번호 \
  --stage-col stage
```

Use the column-name options only when an authorised local workbook uses
different headers. Validation errors report counts and workbook row numbers
where practical; they do not print patient identifiers.

The HUMC pipeline is retained so an authorised holder can execute the same
patient-grouped split, training, and evaluation logic locally. Before any
future public release, confirm that the approved Table 1 aggregate is the only
HUMC-derived data file staged. Normalisation values and every row-level or
image-level output remain controlled. The label workbook, identifiers, images,
split tables, normalisation statistics, checkpoints, predictions, and all
derived feature or figure files must remain uncommitted even after a successful
local run.
