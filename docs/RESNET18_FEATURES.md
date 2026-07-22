# Image resizing and ResNet-18 workflows

PIID curation preserves retained files byte-for-byte. Retained rectangular
Kaggle images, and authorised HUMC images under the same private contract, are
centre-cropped to their native short-side square at native resolution. Model-input
resizing is performed later in memory by the relevant analysis or
classification transform.

The feature extractor fails closed on unreadable or non-square inputs before
loading the encoder. This prevents accidentally pointing the manuscript
workflow at raw rectangular HUMC/Kaggle folders; controlled filenames and paths
are not included in these validation errors.

## Classification training and evaluation

The PIID and HUMC classification pipelines load each prepared analytic image
as RGB and apply Albumentations `A.Resize(224, 224)`. The same direct resize is
used when calculating fold-specific normalisation statistics and during
validation/test inference.

`A.CenterCrop` appears only in the explicitly named centre zoom-in training
augmentation. When that condition is selected, a stochastic 158 x 158 centre
crop is resized back to 224 x 224.

## Manuscript feature-space workflow

Main Figure 3 (UMAP), Main Table 3 (feature-space distances), Main Figure 4
(centroid-nearest representatives), and the silhouette analysis all use the
vectors exported by `code/analysis/extract_resnet18_features.py`:

1. download and fully verify the official timm `resnet18.a1_in1k` checkpoint,
   remove only `fc.weight` and `fc.bias`, and strictly load the remaining 120
   tensors into a headless timm ResNet-18;
2. directly resize each RGB image to `224 x 224` with
   `transforms.Resize((224, 224))`;
3. apply ImageNet mean/std normalisation;
4. retain the raw pooled 512-dimensional vector without L2 normalisation.

The extraction script downloads the public weight automatically from the
official pytorch-image-models GitHub release and verifies the complete cached-
file SHA-256 before use:

`https://github.com/huggingface/pytorch-image-models/releases/download/v0.1-rsb-weights/resnet18_a1_0-d63eafa0.pth`

Complete-file SHA-256:

`D63EAFA07A6E32A39D328E364F8C9F89D671444ECC7F02AA0F7EB8882AF3DD29`

Users do not need to obtain or place a separate study checkpoint.

The preserved study extraction metadata records timm 1.0.22 and PyTorch 2.5.0
with CUDA 12.1. The public main environment uses timm 1.0.22 with the currently
validated PyTorch stack. Both use the same public A1 tensors and mathematical
preprocessing, but small floating-point differences can change a nearest-image
rank. The generated
`centroid_representatives.csv` is therefore retained as the independent rerun;
`data/reference/figure4_public_mean_representatives.csv` records the archived
public PIID/Kaggle selection for direct Figure 4 comparison.
