# Augmentation configurations

All conditions begin with a 224 x 224 resize and end with fold-specific normalization plus tensor conversion. Each enabled stochastic operation has probability 0.5 and is applied in the listed code order.

| ID | Operations |
|---|---|
| `exp00_NoAug` | none |
| `exp01_Flip` | random horizontal, vertical, or combined flip |
| `exp02_Rotate90` | random 0/90/180/270-degree rotation |
| `exp03a_RandomZoomIn` | random resized crop, scale 0.50-0.99, square aspect |
| `exp03b_CenterZoomIn` | centered 158 x 158 crop, then resize to 224 |
| `exp04_ZoomOut` | affine scale 0.70-0.99, reflected border |
| `exp05_Brightness` | brightness limit ±0.20 |
| `exp06_Contrast` | contrast limit ±0.20 |
| `exp07_F_R` | flip + rotation |
| `exp08_F_R_ZI` | flip + rotation + random zoom-in |
| `exp09_F_R_ZI_ZO` | preceding + zoom-out |
| `exp10_F_R_ZI_ZO_B` | preceding + brightness |
| `exp11_F_R_ZI_ZO_B_C` | preceding + contrast |
| `exp12_F_R_CZI` | flip + rotation + center zoom-in |
| `exp13_F_R_CZI_ZO` | preceding + zoom-out |
| `exp14_F_R_CZI_ZO_B` | preceding + brightness |
| `exp15_F_R_CZI_ZO_B_C` | preceding + contrast |

The authoritative executable definitions are `AUGMENTATION_CONFIGS` and `build_train_transform` in both training scripts.
