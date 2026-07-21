# HUMC split metadata

Only privacy-safe aggregate metadata and fold-wise RGB normalization values are public here. Image-level tables, patient identifiers, and fold membership are generated locally and ignored by Git.

- Base random seed: 40
- Split unit: patient
- Held-out test ratio: 15%
- Cross-validation: five patient-grouped folds
- Normalization: training images only, separately for each fold

See `docs/HUMC_PRIVATE_DATA.md`.
