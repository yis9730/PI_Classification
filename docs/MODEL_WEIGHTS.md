# Model weights

The study workstation archive was checked against the public loading convention without copying weights into this repository.

| Training source | Run folders | Fold weights | Expected structure |
|---|---:|---:|---|
| PIID | 102 | 510 | 6 architectures x 17 conditions x 5 folds |
| HUMC | 102 | 510 | 6 architectures x 17 conditions x 5 folds |

Expected local placement:

```text
data/results/checkpoint/
  piid_trained/{backbone}_Baseline_{augmentation}_bs16_lr1e-05_wd1e-04/
    best_models_weights/best_model_fold_{1..5}.pth
  humc_trained/{backbone}_Baseline_{augmentation}_bs16_lr1e-05_wd1e-04/
    best_models_weights/best_model_fold_{1..5}.pth
  feature_extractors/resnet18.pth
```

The classifier contract is:

```text
timm backbone (num_classes=0, global_pool=''), except the study's
TorchVision EfficientNetV2-S implementation
-> architecture-appropriate pooling
-> Linear(C, C/2) -> ReLU -> Dropout(0.5) -> Linear(C/2, 4)
```

The large weights are excluded by `.gitignore`. If they are distributed separately, provide a versioned model archive and its checksums under an approved license/data agreement. The ResNet-18 feature checkpoint checksum is recorded in `RESNET18_FEATURES.md`.
