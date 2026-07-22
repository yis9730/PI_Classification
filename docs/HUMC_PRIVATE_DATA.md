# HUMC controlled-data boundary

HUMC is not an open dataset. The current release tree intentionally contains
no HUMC image, thumbnail, label file, patient or encounter identifier, split
membership, normalisation statistic, aggregate cohort metadata, feature
vector, checkpoint, or prediction file.

This statement applies to the current release tree, not automatically to every
older Git revision. Before publication, the repository owner must separately
review reachable history and decide whether any previously committed,
de-identified HUMC aggregate summaries are institutionally approved for public
retention. Removing a file in a later commit does not remove it from Git
history.

The paper reports the HUMC cohort and its results. Reproducing the HUMC-dependent portions of the study requires a separate institutional authorisation process and a local controlled-data workspace. This repository is not a data-access mechanism.

An authorised project member keeps all controlled inputs and outputs outside Git, including local files under these ignored locations:

```text
data/humc/
data/splits/humc/
data/results/
```

The HUMC pipeline is retained so an authorised holder can execute the same patient-grouped split, training, and evaluation logic locally. Before any future public release, confirm that no HUMC-derived file is staged, including aggregate JSON/CSV summaries and normalisation values.
