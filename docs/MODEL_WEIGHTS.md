# Locally generated model weights

Study-trained classification weights are not distributed in this repository. Visitors recreate them by downloading the permitted data and running the public training scripts. A complete local recreation produces:

| Training source | Run folders | Fold weights | Expected structure |
|---|---:|---:|---|
| PIID | 102 | 510 | 6 architectures x 17 conditions x 5 folds |
| HUMC | 102 | 510 | 6 architectures x 17 conditions x 5 folds |

Training and evaluation use this local layout:

```text
data/results/checkpoints/
  piid_trained/{backbone}_Baseline_{augmentation}_bs16_lr1e-05_wd1e-04/
    best_models_weights/best_model_fold_{1..5}.pth
  humc_trained/{backbone}_Baseline_{augmentation}_bs16_lr1e-05_wd1e-04/
    best_models_weights/best_model_fold_{1..5}.pth
```

The classifier contract is:

```text
timm backbone (num_classes=0, global_pool=''), except the study's
TorchVision EfficientNetV2-S implementation
-> architecture-appropriate pooling
-> Linear(C, C/2) -> ReLU -> Dropout(0.5) -> Linear(C/2, 4)
```

All newly generated classification weights remain local and are excluded by
`.gitignore`. The evaluation scripts load them directly from this structure.
Feature-space analysis uses the official timm `resnet18.a1_in1k` public weight,
downloaded automatically and verified against the full checksum recorded in
`RESNET18_FEATURES.md`.
