# Analysis source data

The analysis scripts consume fold-specific prediction CSVs produced by the evaluation entry points. These generated files are not committed because PIID/HUMC/Kaggle runs can contain local paths and HUMC image-level information.

Expected pattern:

```text
data/results/source_archives/inference_results_{piid|humc}/
  {model}_exp00_NoAug/predictions/{dataset}_fold{1..5}_predictions.csv
```

Each prediction file must contain `image_path`, `true_label`, `predicted_label`, and four class-probability columns. The bootstrap, rank tests, Sankey, ROC, confusion, and error-direction scripts validate or consume this common format.
