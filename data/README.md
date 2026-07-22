# Data

```text
data/
  piid/       prepared PIID images
  kaggle/     prepared Kaggle images
  humc/       private HUMC images and labels
  splits/     released PIID split information
  results/    generated outputs
  templates/  example input files
```

PIID and Kaggle images are prepared with `code/data_curation/prepare_public_datasets.py`. Source downloads may be stored anywhere. After applying the released exclusions, retained PIID files are copied unchanged to `data/piid`; retained Kaggle files are centre-cropped on their longer axis to the native square analytic frame used by the study and written to `data/kaggle`. The model pipelines then resize both prepared datasets to 224 x 224 in memory.

HUMC data and every HUMC-derived split/normalisation file remain local and are ignored by Git. The public package contains neither HUMC aggregate metadata nor an input template; investigators with authorised access create these local inputs according to their institutional data-governance process.

Sources:

- [PIID](https://drive.google.com/drive/u/0/folders/12JouktrzXIo6ywpSe2OYWRYNNIxlEKvK): stage folders `1` through `4`.
- [Kaggle](https://www.kaggle.com/datasets/sinemgokoz/pressure-ulcers-stages): folders `Stage_I` through `Stage_IV`.
