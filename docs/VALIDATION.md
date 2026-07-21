# Package validation record

The release package was checked without retraining models.

- Static release validation passed: Python syntax, required files, 17 conditions per trainer, dependency locks, duplicate counts, private-path scan, and private HUMC split scan.
- The public PIID/Kaggle curation workflow was executed from the downloaded source folders. After the 28 released exclusions, all 1,222 retained outputs were SHA-256-identical to their source files; the expected PIID (1,081) and Kaggle (141) stage counts passed.
- The 20 released reviewed pairs were re-scored with direct 224 x 224 ResNet-18 inputs: all 20 met the cosine candidate threshold and 19 met the corroborating pixel threshold. This confirms why expert decisions and the released manifests, not an automatic intersection of scores, define the final exclusions.
- The local checkpoint archive contained 102 PIID-trained and 102 HUMC-trained run folders. Each contained exactly five fold weights (510 weights per training source).
- Fold 1 of the no-augmentation condition for all six architectures was strictly loaded for both PIID-trained and HUMC-trained archives: 12 representative checkpoints with no missing or unexpected state-dictionary keys.
- The ResNet-18 feature weight matched the documented SHA-256 value.
- Using existing fold prediction files, reduced bootstrap smoke testing and full code-path tests for Friedman--Nemenyi, critical-difference, Sankey, staging-direction, confusion matrix, ROC, and augmentation-heatmap generation completed successfully.

These checks establish packaging and code-path compatibility. They are not a new training run or an independent numerical reproduction of the manuscript results. The original classification-weight and prediction archives are not distributed. Public visitors download PIID and Kaggle, train new PIID weights locally, and then run evaluation. Recreating the HUMC portions additionally requires authorized HUMC access.

Seed audit: the study seed and split seed are 40. As in the original training code, each fold's `DataLoader` shuffle generator uses `40 + fold_id`; the full model/NumPy/Python seed is not reset to 41-45 for each fold.
