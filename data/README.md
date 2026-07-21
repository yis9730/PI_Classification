# Data

```text
data/
  piid/       prepared PIID images
  kaggle/     prepared Kaggle images
  humc/       private HUMC images and labels
  splits/     released split information
  results/    generated outputs
  templates/  example input files
```

PIID and Kaggle images are prepared with `code/data_curation/prepare_public_datasets.py`. Source downloads may be stored anywhere; the script writes center-cropped 224 x 224 images to `data/piid` and `data/kaggle`.

HUMC data remain local and are ignored by Git. Only privacy-safe aggregate split metadata and fold-wise normalization values are public.

Sources:

- [PIID](https://drive.google.com/drive/u/0/folders/12JouktrzXIo6ywpSe2OYWRYNNIxlEKvK): stage folders `1` through `4`.
- [Kaggle](https://www.kaggle.com/datasets/sinemgokoz/pressure-ulcers-stages): folders `Stage_I` through `Stage_IV`.
