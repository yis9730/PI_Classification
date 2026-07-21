# ResNet-18 workflows

The manuscript feature-space analysis and the separate duplicate-candidate
screen used the same frozen ResNet-18 encoder and study checkpoint, but they did
not use the same image preprocessing or feature postprocessing.

The expected checkpoint path is:

`data/results/checkpoints/feature_extractors/resnet18.pth`

SHA-256:

`69E2B9D2711F7CFB70B67091D16027EFAA781BCE78E1084A780AC1D1839B82F9`

The 44.8 MB checkpoint is not committed. A user must obtain the authorised
study checkpoint separately and verify its hash.

## Manuscript feature-space workflow

Main Figure 3 (UMAP), Main Table 3 (feature-space distances), Main Figure 4
(centroid-nearest representatives), and the silhouette analysis all use the
vectors exported by `code/analysis/extract_resnet18_features.py`:

1. directly resize each RGB image to `224 x 224` with
   `transforms.Resize((224, 224))`;
2. apply ImageNet mean/std normalisation;
3. retain the raw pooled 512-dimensional vector without L2 normalisation.

There is **no intermediate resize to 256 and no centre crop** in this
manuscript feature-space workflow.

## Separate duplicate-candidate screen

`code/data_curation/screen_duplicate_candidates.py` reproduces a distinct data
curation step:

1. `transforms.Resize(256)` resizes the shorter image side to 256 while
   preserving aspect ratio; it does not forcibly reshape every image to
   `256 x 256`;
2. `transforms.CenterCrop(224)` extracts the central `224 x 224` region;
3. ImageNet mean/std normalisation is applied;
4. the pooled 512-dimensional vector is L2-normalised for cosine screening.

This 256/centre-crop route belongs only to duplicate-candidate screening. It is
not used to generate UMAP, silhouette, centroid-distance, or representative-
image results.
