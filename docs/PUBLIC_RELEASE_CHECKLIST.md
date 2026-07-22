# Public release checklist

- [ ] Run `python code/validate_release_package.py` and resolve every failure.
- [ ] Confirm the remote belongs to the official institutional/project account.
- [ ] Confirm the approved Table 1 aggregate is the only HUMC-derived data file staged.
- [ ] Confirm no HUMC patient rows, identifiers, filenames/paths, split membership, normalisation values, checkpoints, or predictions are staged.
- [ ] Confirm whether non-identifying historical HUMC aggregates may remain reachable; rewrite history only if the Table 1-only boundary must apply retroactively.
- [ ] Confirm commit author/committer email addresses are suitable for public release.
- [ ] Confirm `requirements_train_eval.txt` and `requirements_umap_analysis.txt` remain separate and locked.
- [ ] Confirm each main table and figure has a current row in `docs/MAIN_ARTIFACTS.md`.
- [ ] Confirm the manuscript names, dataset labels, split rules, and code release URL agree.
- [ ] Select an institutionally approved software license.
- [ ] Add citation metadata after the manuscript DOI and final author list are available.
- [ ] Tag the exact manuscript release, for example `v1.0.0`.
- [ ] Archive the tagged release in a DOI-bearing repository if required by the journal.
