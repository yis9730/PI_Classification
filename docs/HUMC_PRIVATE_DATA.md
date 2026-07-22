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

The HUMC pipeline is retained so an authorised holder can execute the same
patient-grouped split, training, and evaluation logic locally. Before any
future public release, confirm that the approved Table 1 aggregate is the only
HUMC-derived data file staged. Normalisation values and every row-level or
image-level output remain controlled.
