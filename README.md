# Prediction-Level Late Fusion Preserves Task-Specific Strengths across Heterogeneous Molecular Encoders for ADMET Prediction — Code Release

Reproduction package for the paper *"Prediction-Level Late Fusion Preserves Task-Specific
Strengths across Heterogeneous Molecular Encoders for ADMET Prediction."*

The paper studies representation- and prediction-level fusion across five molecular foundation
models on 21 TDC ADMET tasks, and validates the resulting uniform-mean late-fusion ensemble
(**LateFusion-k5**) on the independent Polaris ADME-Fang benchmark (6 regression endpoints,
officially submitted to the Polaris Hub).

## Two levels of reproducibility

**1. Figures and tables (no GPU, no network) — start here.**
Every figure and table in the paper is recomputed from the raw per-seed model predictions
and leaderboard entries in `data/recompute_data/` (no hard-coded summary numbers). A single
self-contained notebook regenerates all figures and tables:

```bash
pip install -r requirements.txt          # numpy, pandas, scipy, scikit-learn, statsmodels, matplotlib, openpyxl, jupyter
jupyter notebook notebooks/regenerate_paper_materials.ipynb   # -> figures/figure1..6.png + figureS1..S7.png + tables/*.csv + *.xlsx
```

Or run headless:

```bash
jupyter nbconvert --to notebook --execute notebooks/regenerate_paper_materials.ipynb
```

The notebook reads raw predictions from `data/recompute_data/` and writes into `figures/`
and `tables/`. The setup cell recomputes every value — solo/ensemble
norm_scores, the uniform-mean late-fusion ensemble, LOO/Shapley, correlations, the k-sweep,
and the Wilcoxon statistics — from those raw files, and `assert` blocks verify each result
against the manuscript numbers before drawing (e.g. the k=5 composition; the ensemble wins
3 of 6 Polaris endpoints). The derived summary files (`ground_truth.json`, `pertask.json`,
`stat_tests.json`) are retained as a verification reference, not as the notebook input.

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
├── data/                         ← reproduction inputs (see DATA_MANIFEST.md)
│   ├── DATA_MANIFEST.md
│   ├── recompute_data/           ← RAW source the notebooks recompute from:
│   │   ├── predictions/{model}.parquet   (10 models; seed,task,split,smiles,y_true,y_pred)
│   │   ├── leaderboards/{model}.csv       (10 models; dataset,metric,ours,ours_std,rank,n_total,Top,Top_model)
│   │   ├── clean_stage1_current.csv, all1023.csv   (1,023-subset sweeps; validation-only vs val+test — complementary, not duplicates)
│   │   └── tdc_leaderboard_snapshot.json, task_registry.json
│   ├── ground_truth.json, pertask.json, tables_data.json, stat_tests.json   (verification reference)
│   ├── uniform_val_test_ALL1023.csv   (per-subset val/test norm, all 1,023 uniform subsets)
│   │   (per-model/per-task solo norm — formerly data/solo_pertask_within_cross.csv /
│   │    data/fig1_within_pertask.csv — is now recomputed in-notebook from predictions+leaderboards)
│   └── polaris_*.csv / polaris_leaderboard_full.json   (external benchmark)
├── notebooks/
│   ├── regenerate_paper_materials.ipynb   ← ALL figures + tables (Figure 1–6, Figure S1–S7;
│   │                                          Table 1–5, Table S1–S8) recomputed from data/recompute_data/
│   └── ensemble_subset_files_guide.md ← what the two 1,023-subset files (clean_stage1_current.csv, all1023.csv) are + how they are used
├── figures/                      ← generated PNGs (figure1..6.png, figureS1..S7.png)
├── tables/                       ← generated CSVs + supplementary_tables.xlsx
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
| Figure 4 | Validation→test reliability (1,023 subsets, post-hoc) + per-member Leave-One-Out, Shapley, and winner distribution |
| Figure 5 | Statistical validation: paired Wilcoxon of the ensemble gain vs each deployable single model and the per-task oracle |
| Figure 6 | LateFusion-k5 vs all public submissions on the 6 Polaris ADME-Fang leaderboards |
| Figure S1–S7 | Supplementary figures: per-backbone/aggregation heatmaps (S1–S3), inter-model correlation (S4), member-role heatmap (S5), raw-score vs TDC-best box plots (S6), leaderboard coverage vs mean rank (S7) |
| Table 1–5 | Positioning vs prior fusion work; mechanism grouping; ensemble summary; per-task ensemble; k=5 member roles |
| Table S1–S8 | Per-backbone detail; representation aggregation; k-sweep; inter-model correlation; reproducibility; Polaris datasets; dataset provenance; ensemble-vs-baseline significance |

## Data sources

- **TDC** — Therapeutics Data Commons (https://tdcommons.ai), ADMET benchmark group.
- **Polaris ADME-Fang** — Biogen ADME-Fang benchmark on the Polaris Hub
  (https://polarishub.io), benchmarks `polaris/adme-fang-{hclint,rclint,perm,solu,hppb,rppb}-1`.
  Our ensemble was officially submitted as `LateFusion-k5`.

## License

Released under the MIT License (see `LICENSE`). The `qadmet` engine under `code/qadmet/`
is also MIT-licensed.

## Notes

- `data/recompute_data/` contains the **raw per-seed model predictions and leaderboard entries**
  that the notebooks recompute every figure/table value from; the top-level `data/*.json`
  summary files are kept only as a verification reference. `data/` does not contain trained
  model weights; those (and the raw predictions/leaderboards themselves) are reproducible via
  `code/TRAINING.md` — see "How the notebook inputs are produced" there.
- The leaderboard scores in `polaris_leaderboard_full.json` reflect publicly listed submissions
  parsed from the benchmark pages at the time of analysis and may change as new submissions arrive.
