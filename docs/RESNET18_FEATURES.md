# ResNet-18 feature extraction

All feature-based analyses used a frozen ImageNet-pretrained ResNet-18 encoder
with a 512-dimensional pooled output. The exact study checkpoint is expected at:

`data/results/checkpoint/feature_extractors/resnet18.pth`

SHA-256:

`69E2B9D2711F7CFB70B67091D16027EFAA781BCE78E1084A780AC1D1839B82F9`

The 44.8 MB checkpoint is not committed to avoid duplicating pretrained model
weights. A user must obtain the authorized study checkpoint and verify the hash.

Two intentional processing modes were used:

1. Duplicate candidate screening: resize the shorter side to 256, center-crop
   224 x 224, apply ImageNet normalization, and L2-normalize the 512-D vector.
2. Exploratory feature-space analysis: directly resize to 224 x 224, apply
   ImageNet normalization, and retain the raw pooled 512-D vector. The same raw
   vectors feed UMAP, silhouette, centroid-distance, and representative-image
   selection.

The duplicate screen and the exploratory feature analysis therefore use the
same frozen encoder and weights but different, documented preprocessing and
postprocessing appropriate to their respective analyses.
