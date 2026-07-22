# Image resizing and ResNet-18 workflows

PIID curation preserves retained files unchanged. Kaggle curation first
creates the study's native-size centre-square analytic frame. Model-input
resizing is then performed in memory by the relevant analysis or
classification transform.

The expected checkpoint path is:

`data/results/checkpoints/feature_extractors/resnet18.pth`

SHA-256:

`69E2B9D2711F7CFB70B67091D16027EFAA781BCE78E1084A780AC1D1839B82F9`

The 44.8 MB checkpoint is not committed. A user must obtain the authorised
study checkpoint separately and verify its hash.

## Classification training and evaluation

The PIID and HUMC classification pipelines load each prepared analytic image
as RGB and apply Albumentations `A.Resize(224, 224)`. The same direct resize is
used when calculating fold-specific normalisation statistics and during
validation/test inference. No `Resize(256) -> CenterCrop(224)` preprocessing
sequence is used. Kaggle's separate native-size square curation happens before
this shared model-input transform.

`A.CenterCrop` appears only in the explicitly named centre zoom-in training
augmentation. When that condition is selected, a stochastic 158 x 158 centre
crop is resized back to 224 x 224. It is not a dataset-curation step and is not
applied to every image.

## Manuscript feature-space workflow

Main Figure 3 (UMAP), Main Table 3 (feature-space distances), Main Figure 4
(centroid-nearest representatives), and the silhouette analysis all use the
vectors exported by `code/analysis/extract_resnet18_features.py`:

1. directly resize each RGB image to `224 x 224` with
   `transforms.Resize((224, 224))`;
2. apply ImageNet mean/std normalisation;
3. retain the raw pooled 512-dimensional vector without L2 normalisation.

There is no intermediate 256-pixel resize and no model-input centre crop in
this workflow.

## Supplementary duplicate-candidate screening utility

`code/data_curation/screen_duplicate_candidates.py` generates candidates for
expert review without modifying dataset files:

1. directly resize each RGB model input to `224 x 224`;
2. apply ImageNet mean/std normalisation;
3. L2-normalise the pooled 512-dimensional vector and retain pairs meeting the
   cosine threshold;
4. compare those candidates again at 128 x 128 and report whether each passes
   the corroborating pixel-MAE threshold.

The 128 x 128 operation is an in-memory comparison used only to score a
candidate pair. It does not replace or rewrite either source image. The
released pair-decision table and exclusion manifests, rather than either
automatic score, define the final analytic datasets.
