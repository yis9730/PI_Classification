# Public repository architecture

## Design rule

Each executable has one responsibility and one place in the workflow. The public repository excludes notebook checkpoints, server-specific copies, intermediate experiments, and ambiguous duplicate scripts. The curated release replaces the former broad `development`/`experiment` grouping with `core` and `pipeline`, making the execution order evident from the directory tree.

```text
.
├── requirements.txt                    # default environment pointer
├── requirements_train_eval.txt         # main environment
├── requirements_umap_analysis.txt      # UMAP-only environment
├── code/
│   ├── core/                            # paths, model factory, model metadata
│   ├── data_curation/                   # dataset assembly and exclusion decisions
│   ├── pipeline/                        # splits → training → evaluation
│   ├── analysis/                        # numerical summaries and inferential tests
│   ├── visualization/                   # manuscript-facing plots
│   └── *check*.py                       # environment/release/checkpoint checks
├── data/
│   ├── {piid,kaggle,humc}/              # local images; ignored by Git
│   ├── aggregates/                      # approved manuscript-level counts only
│   ├── splits/                          # public PIID metadata; local HUMC outputs ignored
│   ├── results/                         # generated, ignored by Git
│   └── templates/                       # controlled-data input templates
└── docs/                                # execution, privacy, and result contracts
```

## Workflow ownership

| Step | Owner directory | Main input | Main generated output |
|---|---|---|---|
| Public image curation | `code/data_curation` | original PIID/Kaggle downloads | prepared images and curation manifests |
| PIID/HUMC split creation | `code/pipeline` | prepared images; controlled HUMC labels | split tables and normalisation statistics |
| Model development | `code/pipeline` + `code/core` | splits and images | fold-specific checkpoints and training records |
| Internal/external validation | `code/pipeline` | checkpoints and test/external images | fold-specific prediction CSVs |
| Statistical tests | `code/analysis` | prediction CSVs | table-ready summary CSVs |
| Feature-space computation | `code/analysis` | images and official timm ResNet-18 A1 public weight | feature arrays, Table 3 values, representatives |
| Main figures | `code/visualization` | analysis outputs | figures 1–4 and supporting figures |

## Naming rules

- Use a verb-led file name for a runnable action: `train_`, `evaluate_`, `extract_`, `plot_`, `build_`, or `check_`.
- Keep reusable imports only in `core`; workflow scripts must not contain personal paths or host-specific configuration.
- Save numerical tables in `analysis` before plotting them in `visualization`. UMAP is a plot, so it belongs in `visualization`; silhouette and centroid distances are statistics, so they belong in `analysis`.
- A new main-manuscript output requires a command, inputs, output path, and public/controlled-data status in `MAIN_ARTIFACTS.md` before release.
