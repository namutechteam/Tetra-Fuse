# Prediction-Level Late Fusion Preserves Task-Specific Strengths across Heterogeneous Molecular Encoders for ADMET Prediction — Code Release

Reproduction package for the paper *"Prediction-Level Late Fusion Preserves Task-Specific
Strengths across Heterogeneous Molecular Encoders for ADMET Prediction."*

The paper studies representation- and prediction-level fusion across five molecular foundation
models on 21 TDC ADMET tasks, and validates the resulting uniform-mean late-fusion ensemble
(**LateFusion-k5**) on the independent Polaris ADME-Fang benchmark (6 regression endpoints,
officially submitted to the Polaris Hub).

## Two levels of reproducibility

**1. Figures and tables (no GPU, no network) — start here.**
Every figure and table in the paper is a function of the derived summary files in `data/`.
Two self-contained notebooks regenerate them:

```bash
pip install -r requirements.txt          # numpy, pandas, matplotlib, openpyxl, jupyter
jupyter notebook notebooks/reproduce_figures.ipynb   # -> figures/figure1..6.png
jupyter notebook notebooks/reproduce_tables.ipynb    # -> tables/*.csv + paper_tables.xlsx
```

Or run headless:

```bash
jupyter nbconvert --to notebook --execute notebooks/reproduce_figures.ipynb
jupyter nbconvert --to notebook --execute notebooks/reproduce_tables.ipynb
```

Both notebooks read only from `data/` and write into `figures/` and `tables/`. The tables
notebook contains `assert` checks that verify the numbers match the manuscript
(e.g. the ensemble wins 3 of 6 Polaris endpoints; k=5 composition).

**2. Model training (GPU) — see `code/TRAINING.md`.**
The full `qadmet` training/evaluation engine (`code/qadmet/`), all recipe YAMLs, the pinned
MoLFormer revision, and the raw Polaris data (`code/data_polaris_adme_fang/`) are included.
Reproducing the trained checkpoints requires a GPU and the `admet-mtl` conda environment;
training is driven by `python -m qadmet.cli.main train --recipe <recipe.yaml>`.

## Package layout

```
release/
├── README.md                     ← this file
├── requirements.txt              ← minimal deps for the notebooks
├── environment.yml               ← conda env for the notebooks
├── data/                         ← all ground-truth summary files (see DATA_MANIFEST.md)
│   ├── DATA_MANIFEST.md
│   ├── ground_truth.json, pertask.json, tables_data.json, stat_tests.json
│   ├── all1023.csv, uniform_val_test_unified.csv, solo10_within_cross.json, ...
│   └── polaris_*.csv / polaris_leaderboard_full.json   (external benchmark)
├── notebooks/
│   ├── reproduce_figures.ipynb   ← Figure 1–6
│   └── reproduce_tables.ipynb    ← Table 1–5, S1–S3, S6–S7
├── figures/                      ← generated PNGs (figure1..6.png)
├── tables/                       ← generated CSVs + paper_tables.xlsx
└── code/
    ├── TRAINING.md               ← model-training reproduction guide
    ├── qadmet/                   ← the full qadmet training/evaluation engine
    │   ├── src/qadmet/           ← 113 modules: core, datasets, features, modules/backbones,
    │   │                            ensembles, evaluation, precompute, runners, tasks, training, cli
    │   ├── recipes/              ← 490 recipe YAMLs (train/ has TDC-22 + Polaris ADME-Fang;
    │   │                            _base/, ensemble/, evaluate/, precompute/, groupings/, ablations/)
    │   ├── pyproject.toml        ← qadmet package metadata (CLI entry point qadmet.cli.main)
    │   └── README.md
    ├── polaris_fang_run_all.sh   ← end-to-end training run order
    └── data_polaris_adme_fang/   ← raw Polaris ADME-Fang CSVs (train/val + test) + summary
```

The five backbones behind the ensemble, their exact recipe paths inside `qadmet/recipes/`,
and the pinned MoLFormer revision are documented in `code/TRAINING.md`.

## Figures and tables produced

| Output | What it shows |
|---|---|
| Figure 1 | No single molecular encoder dominates (within-seed solo test norm_score, 10 models) |
| Figure 2 | MAP joint fusion raises the floor but rarely tops the best single backbone |
| Figure 3 | Ensemble-size sweep; diminishing marginal gain, elbow at k=5 |
| Figure 4 | Validation→test reliability of subset selection (1,023 subsets, post-hoc) |
| Figure 5 | Per-member Leave-One-Out and Shapley contributions in the final k=5 ensemble |
| Figure 6 | LateFusion-k5 vs all public submissions on the 6 Polaris ADME-Fang leaderboards |
| Table 1–5 | Endpoint groups; headline result; k=5 members; per-task ensemble; Polaris ranks |
| Table S1–S3, S6–S7 | Per-backbone detail; representation aggregation; k-sweep; Polaris datasets & per-model |

## Data sources

- **TDC** — Therapeutics Data Commons (https://tdcommons.ai), ADMET benchmark group.
- **Polaris ADME-Fang** — Biogen ADME-Fang benchmark on the Polaris Hub
  (https://polarishub.io), benchmarks `polaris/adme-fang-{hclint,rclint,perm,solu,hppb,rppb}-1`.
  Our ensemble was officially submitted as `LateFusion-k5`.

## License

Released under the MIT License (see `LICENSE`). The `qadmet` engine under `code/qadmet/`
is also MIT-licensed.

## Notes

- `data/` contains **derived summary statistics** (per-model/per-task norm_scores, correlations,
  Shapley/LOO values, official Polaris predictions). It does not contain trained model weights;
  those are reproducible via `code/TRAINING.md`.
- The leaderboard scores in `polaris_leaderboard_full.json` reflect publicly listed submissions
  parsed from the benchmark pages at the time of analysis and may change as new submissions arrive.
