# Package validation record

The structural checks below were supplemented by one complete representative
five-fold retraining run.

- Static release validation passed: Python syntax, required files, 17 conditions per trainer, dependency locks, duplicate counts, private-path scan, and private HUMC split scan.
- The released Table 1 source contains only the three datasets' manuscript counts and stage percentages; its HUMC row has no patient-level or image-level field.
- The raw-source duplicate-review entry point validates 1,091 PIID and 159
  Kaggle provider images, runs independent feature and pixel all-pairs searches
  at `0.85`, and writes complete descending candidate CSVs plus strongest-pair
  montage queues. The full run produced 6,473 feature candidates and 299,913
  pixel candidates; the nonredundant montage queues contained 694 and 1,151
  pairs, respectively. All 20 released human-reviewed pairs occurred in both
  the complete-candidate union and the montage-queue union. The released
  decisions excluded 10 PIID and 18 Kaggle images, and the source fingerprint
  was unchanged.
- The public PIID/Kaggle curation workflow was executed from the downloaded source folders. After the 28 released exclusions, all 1,081 retained PIID files were SHA-256-identical to source. All 141 generated Kaggle images had dimensions and decoded-pixel SHA-256 values identical to the archived analytic dataset; 135 rectangular retained images were native-square cropped and six were already square. The expected stage counts passed.
- The local checkpoint archive contained 102 PIID-trained and 102 HUMC-trained run folders. Each contained exactly five fold weights (510 weights per training source).
- Fold 1 of the no-augmentation condition for all six architectures was strictly loaded for both PIID-trained and HUMC-trained archives: 12 representative checkpoints with no missing or unexpected state-dictionary keys.
- The official timm `resnet18.a1_in1k` cache file matched the documented full
  SHA-256
  `D63EAFA07A6E32A39D328E364F8C9F89D671444ECC7F02AA0F7EB8882AF3DD29`.
- Removing only the classifier tensors from that public checkpoint reproduced
  all 120 tensors of the preserved headless study encoder bit-for-bit. A full
  re-extraction of all 3,066 PIID/HUMC/Kaggle images preserved image identity
  and order; mean absolute feature drift by dataset was below `0.00042` and
  mean cosine agreement exceeded `0.999997`, consistent with version/device
  numerical drift under the same model contract.
- All nine inter-cluster centroid distances and all seven intra-cluster
  pairwise-mean distances reproduced the manuscript's two-decimal Table 3
  values. The independent mean-centroid representative selection agreed with
  the archived CSV for all 12 PIID and all 12 HUMC positions and seven of 12
  Kaggle positions; the released public reference manifest retains the
  archived PIID/Kaggle selection instead of silently replacing it.
- The complete three-dataset UMAP workflow generated the submitted six-panel
  layout. Repeating it in the locked UMAP environment produced bit-identical
  coordinates; similarity-Procrustes comparison with the preserved embedding
  gave `R² = 0.99396`, reflecting small feature/runtime drift while preserving
  the reported structure.
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
