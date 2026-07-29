# Tetra-Fuse: Late Fusion Preserves Task-Specific Strengths of Heterogeneous Molecular Encoders for ADMET Prediction — Code Release

Reproduction package for the paper *"Tetra-Fuse: Late Fusion Preserves Task-Specific
Strengths of Heterogeneous Molecular Encoders for ADMET Prediction."*

The paper studies representation- and prediction-level fusion across five molecular foundation
models on 21 TDC ADMET tasks, and validates the resulting uniform-mean late-fusion ensemble
(**Tetra-Fuse**) on the independent Polaris ADME-Fang benchmark (6 regression endpoints;
submitted to the Polaris Hub under the label `LateFusion-k5`).

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
├── CITATION.cff                  ← citation metadata
├── requirements.txt              ← minimal deps for the notebooks
├── environment.yml               ← conda env for the notebooks
├── LICENSE                       ← MIT
├── data/                         ← reproduction inputs (see DATA_MANIFEST.md)
│   ├── DATA_MANIFEST.md
│   ├── recompute_data/           ← RAW source the notebooks recompute from:
│   │   ├── predictions/{model}.parquet   (10 models; seed,task,split,smiles,y_true,y_pred)
│   │   ├── leaderboards/{model}.csv       (10 models; dataset,metric,ours,ours_std,rank,n_total,Top,Top_model)
│   │   ├── clean_stage1_current.csv, all1023.csv   (1,023-subset sweeps; validation-only vs val+test — complementary, not duplicates)
│   │   ├── tdc_leaderboard_snapshot.json, task_registry.json, task_metadata.json
│   │   └── polaris_leaderboard_full.json   (Polaris public submissions, notebook input for Figure 6)
│   └── polaris_external/          ← Polaris audit/reference tables (not notebook inputs; see its README.md)
├── notebooks/
│   ├── regenerate_paper_materials.ipynb   ← ALL figures + tables (Figure 1–6, Figure S1–S7;
│   │                                          Table 1–5, Table S1–S8) recomputed from data/recompute_data/
│   └── ensemble_subset_files_guide.md ← what the two 1,023-subset files (clean_stage1_current.csv, all1023.csv) are + how they are used
├── figures/                      ← generated PNGs (figure1..6.png, figureS1..S7.png)
├── tables/                       ← generated CSVs + supplementary_tables.xlsx
└── code/
    ├── TRAINING.md               ← model-training reproduction guide (weights, splits, pipeline)
    ├── qadmet/                   ← the full qadmet training/evaluation engine
    │   ├── src/qadmet/           ← core, datasets, features, modules/backbones,
    │   │                            ensembles, evaluation, precompute, runners, tasks, training, cli
    │   ├── recipes/              ← 38 recipe YAMLs (the paper-reproduction set):
    │   │                            train/ 15 (TDC-22: 5 MTL + 5 STL; Polaris: 5 members),
    │   │                            evaluate/ 10, _base/ 9, groupings/ 2, precompute/ 2
    │   ├── pyproject.toml        ← qadmet package metadata (CLI entry point qadmet.cli.main)
    │   └── README.md
    ├── polaris_fang_run_all.sh   ← end-to-end training run order
    └── data_polaris_adme_fang/   ← raw Polaris ADME-Fang CSVs (train/val + test) + summary
```

The five backbones behind the ensemble, their exact recipe paths inside `qadmet/recipes/`,
the pretrained-weight download instructions (CheMeleon, GraphMVP), the dataset-split provenance,
and the pinned MoLFormer revision are all documented in `code/TRAINING.md`.

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
  that the notebooks recompute every figure/table value from. `data/` does not contain trained
  model weights or the pretrained backbone weights; the trained checkpoints (and the raw
  predictions/leaderboards themselves) are reproducible via `code/TRAINING.md`. `rdkit2d`, `xtb`,
  and `unimol2` descriptor/embedding files are **not** required — none of the 10 paper models use
  them (the two joint-fusion MAP models use only the CheMeleon graph, MoLFormer tokens, and a
  GraphMVP encoder built on-the-fly from SMILES).
- Dataset splits reproduce deterministically per seed: TDC-22 is downloaded at runtime by the
  `tdc` package (fixed official test split; seeded train/val), and the Polaris ADME-Fang
  train/val CSVs ship in `code/data_polaris_adme_fang/` (test labels are withheld by the Polaris
  Hub). Hosting the raw splits on Zenodo is therefore unnecessary — see `code/TRAINING.md`.
- The leaderboard scores in `polaris_leaderboard_full.json` reflect publicly listed submissions
  parsed from the benchmark pages at the time of analysis and may change as new submissions arrive.
