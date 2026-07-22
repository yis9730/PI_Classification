# Package validation record

The structural checks below were supplemented by one complete representative
five-fold retraining run.

- Static release validation passed: Python syntax, required files, 17 conditions per trainer, dependency locks, duplicate counts, private-path scan, and private HUMC split scan.
- The public PIID/Kaggle curation workflow was executed from the downloaded source folders. After the 28 released exclusions, all 1,081 retained PIID files were SHA-256-identical to source. All 141 generated Kaggle native-square images matched the study analytic set in dimensions and decoded RGB pixels. The expected stage counts passed.
- The 20 released reviewed pairs were re-scored with direct 224 x 224 ResNet-18 inputs: all 20 met the cosine candidate threshold and 19 met the corroborating pixel threshold. This confirms why expert decisions and the released manifests, not an automatic intersection of scores, define the final exclusions.
- The local checkpoint archive contained 102 PIID-trained and 102 HUMC-trained run folders. Each contained exactly five fold weights (510 weights per training source).
- Fold 1 of the no-augmentation condition for all six architectures was strictly loaded for both PIID-trained and HUMC-trained archives: 12 representative checkpoints with no missing or unexpected state-dictionary keys.
- The ResNet-18 feature weight matched the documented SHA-256 value.
- Using existing fold prediction files, reduced bootstrap smoke testing and full code-path tests for Friedman--Nemenyi, critical-difference, Sankey, staging-direction, confusion matrix, ROC, and augmentation-heatmap generation completed successfully.

These checks establish packaging and code-path compatibility. The original
classification-weight and prediction archives are not distributed. Public
visitors download PIID and Kaggle, train new PIID weights locally, and then run
evaluation. Recreating the HUMC portions additionally requires authorized HUMC
access.

## Representative post-correction retraining

On 2026-07-22, DenseNet-121 with `exp00_NoAug` was trained for all five PIID
folds with the released split, direct Albumentations `A.Resize(224, 224)`,
batch size 16, AdamW at 1e-5, weight decay 1e-4, `drop_last=True`, at most 50
epochs, and patience 20. The run used Python 3.11.6, PyTorch 2.9.0+cu128, and
an NVIDIA GeForce RTX 2080.

- Recomputed fold normalisation values agreed with the released six-decimal
  CSV to a maximum absolute difference below `5.0e-7`.
- Best validation macro-F1 across folds was 0.7992, 0.7803, 0.7479, 0.7777,
  and 0.7862 (mean 0.7783). The archived study mean was approximately 0.7768.
- All five generated checkpoints loaded with strict state-dictionary checks.
- Mean fold-specific PIID test macro-F1 was 0.6599 versus archived 0.6891
  (difference -0.0292).
- Mean fold-specific Kaggle macro-F1 was 0.5060 versus archived 0.5195
  (difference -0.0136).
- An empty private HUMC placeholder was correctly skipped; PIID (163 images)
  and Kaggle (141 images) were evaluated.

This is a representative numerical reproduction of one of the 102 PIID
model/augmentation conditions, not a rerun of all study conditions. The
historical trainer declared a seed helper but did not call it, so the archived
classifier initialisation and dropout RNG state cannot be reconstructed from
the preserved executable alone. The public trainer stabilises future runs by
setting the full RNG state once to 40 and uses `40 + fold_id` for each fold's
DataLoader shuffle generator. Consequently, the remaining numerical gap must
not be attributed to GPU model alone, and bitwise identity with the archived
weights is not claimed.
