# Data Manifest — Reproduction Bundle

All figures and tables in the paper are reproduced by the six section notebooks in
`../notebooks/` (`00_pre_analysis.ipynb` + `31`–`35_*.ipynb`, one per Results subsection).
Each notebook **recomputes every quantity from raw source predictions** — no hard-coded summary
numbers. GPU compute and network are NOT required.

## Source data for full recomputation — `recompute_data/`

The notebooks' shared setup cell derives all figure/table values from these raw source files:

| File | Content |
|---|---|
| `recompute_data/predictions/{model}.parquet` | 10 base models. Raw per-seed, per-molecule predictions. Cols: `task,seed,split,smiles,y_true,y_pred` (seeds 0-4, 22 tasks, val+test). These are the primary source: all solo/ensemble norm_scores, LOO/Shapley, correlations, winner distributions, and per-task tables are computed from these by uniform-mean fusion and TDC-leaderboard rank normalization. |
| `recompute_data/leaderboards/{model}.csv` | 10 base models. Each model's own TDC leaderboard entry (direct training-pipeline output). Cols: `dataset,metric,ours,ours_std,rank,n_total,Top,Top_model`. Source for the per-backbone norm_score (Figure 1 / Table S1) and the MAP-vs-mean/max aggregation (Figure 2 / Table S2), via norm = (n_total - rank + 1)/n_total. |
| `recompute_data/all1023.csv` | All 1,023 uniform-mean subsets of the 10 base models, scored on validation and held-out test in both tracks. Cols: `combo_key,k,val_within,val_cross,test_within,test_cross,val_comb,test_comb`. Source for the ensemble-size sweep (Figure 3 / Table S3, validation top-20 mean per k) and the validation→test reliability check (Figure 5). Fully regenerable from `predictions/` via `code/make_all1023.py`; see `../notebooks/ensemble_subset_files_guide.md`. |
| `recompute_data/tdc_leaderboard_snapshot.json` | TDC ADMET public leaderboard snapshot (23 tasks, captured 2026-04-29) — the rank-normalization ruler for all TDC norm_scores. |
| `recompute_data/task_registry.json` | Per-task metric (mae/auroc/auprc/spearman) and type. |
| `recompute_data/task_metadata.json` | Per-task display names and grouping used by the figures/tables. |
| `recompute_data/polaris_leaderboard_full.json` | Full public Polaris leaderboard scores parsed per benchmark (`hclint,rclint,perm,solu,hppb,rppb`), each a list of `[model_name, pearson_r]`. Notebook input for Figure 6 (all-submission distribution). |

The notebook setup cell recomputes `_GT`/`_PT`/`_ST`/`_FIG1` from the above and asserts every
value against the manuscript numbers before drawing — so nothing in the figures/tables is
hard-coded.

## External-benchmark (Polaris ADME-Fang) reference — `polaris_external/`

Per-molecule predictions, per-endpoint scores and official leaderboard ranks behind the
external validation (Figure 6, Table 5). **Not read by the notebook** — retained for
transparency and to make the Polaris submission independently auditable. See
`polaris_external/README.md` for the per-file breakdown.

## Provenance

- TDC benchmark: Therapeutics Data Commons (tdcommons.ai), ADMET group, scaffold split, 5 seeds.
- Polaris ADME-Fang: Biogen ADME-Fang benchmark on the Polaris Hub (polarishub.io),
  official `polaris/adme-fang-{hclint,rclint,perm,solu,hppb,rppb}-1` benchmarks;
  random split, Pearson r metric; ensemble officially submitted as `LateFusion-k5`.
- Leaderboard scores in `polaris_leaderboard_full.json` were parsed from the public
  benchmark pages and reflect publicly listed submissions at parse time.
