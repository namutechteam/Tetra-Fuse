# Data Manifest — Reproduction Bundle

All figures and tables in the paper are reproduced by the seven notebooks in
`../notebooks/` (`00_pre_analysis.ipynb`, `31`–`35_*.ipynb` and `36_controlled_comparisons.ipynb`).
Each notebook **recomputes every quantity from raw source predictions** — no hard-coded summary
numbers. No GPU is required. The only external input is the TDC ADMET training/validation
SMILES used by two descriptive analyses of notebook 36 (similarity strata, Note S3.4, and the
uncertainty analysis, Note S3.6); see *External input* below.

## Source data for full recomputation — `recompute_data/`

The notebooks' shared setup cell derives all figure/table values from these raw source files:

| File | Content |
|---|---|
| `recompute_data/predictions/{model}.parquet` | 10 base models. Raw per-seed, per-molecule predictions. Cols: `task,seed,split,smiles,y_true,y_pred` (seeds 0-4, 22 tasks, val+test). These are the primary source: all solo/ensemble norm_scores, LOO/Shapley, correlations, winner distributions, and per-task tables are computed from these by uniform-mean fusion and TDC-leaderboard rank normalization. |
| `recompute_data/leaderboards/{model}.csv` | 10 base models. Each model's own TDC leaderboard entry (direct training-pipeline output). Cols: `dataset,metric,ours,ours_std,rank,n_total,Top,Top_model`. Source for the per-backbone norm_score (Figure 1 / Table S1) and the MAP-vs-mean/max aggregation (Figure S12 / Table S2), via norm = (n_total - rank + 1)/n_total. |
| `recompute_data/all1023.csv` | All 1,023 uniform-mean subsets of the 10 base models, scored on validation and held-out test in both tracks. Cols: `combo_key,k,val_within,val_cross,test_within,test_cross,val_comb,test_comb`. Source for the ensemble-size sweep (Figure 4 / Table S3, validation top-20 mean per k), the validation→test reliability check (Figure 6A) and the seed-averaging effect (Table 2). Fully regenerable from `predictions/` via `code/make_all1023.py`; see `../notebooks/ensemble_subset_files_guide.md`. |
| `recompute_data/tdc_leaderboard_snapshot.json` | TDC ADMET public leaderboard snapshot (23 tasks, captured 2026-04-29) — the rank-normalization ruler for all TDC norm_scores. |
| `recompute_data/task_registry.json` | Per-task metric (mae/auroc/auprc/spearman) and type. |
| `recompute_data/task_metadata.json` | Per-task display names and grouping used by the figures/tables. |
| `recompute_data/polaris_leaderboard_full.json` | Full public Polaris leaderboard scores parsed per benchmark (`hclint,rclint,perm,solu,hppb,rppb`), each a list of `[model_name, pearson_r]`. Notebook input for Figure 7 (all-submission distribution). |

The notebook setup cell recomputes `_GT`/`_PT`/`_ST`/`_FIG1` from the above and asserts every
value against the manuscript numbers before drawing — so nothing in the figures/tables is
hard-coded.

## Controlled comparisons (Section 2.7) — no extra data

Figure 2, Figure 3, Table 2, Table S10a–c and Supplementary Note S3 are recomputed by
`../code/followup/` from the same `predictions/` parquets, `all1023.csv` (validation-only
choice of the controls; seed-averaging effect) and the leaderboard snapshot.

## External input — TDC ADMET training/validation SMILES

Analysis C (maximum Tanimoto similarity of each test molecule to the training data) and the
uncertainty analysis stratify test molecules by similarity, so they need each task's
`train_val.csv` from the TDC ADMET benchmark group. Notebook 36 looks for them in
`$TDC_ADMET_GROUP`, then `data/tdc_admet_group/`, then `data/admet_group/`, and otherwise
downloads them once with PyTDC (`tdc.benchmark_group.admet_group`). They are not
redistributed here (gitignored).

## External-benchmark (Polaris ADME-Fang) reference — `polaris_external/`

Per-molecule predictions, per-endpoint scores and official leaderboard ranks behind the
external validation (Figure 6, Table S6). **Not read by the notebook** — retained for
transparency and to make the Polaris submission independently auditable. See
`polaris_external/README.md` for the per-file breakdown.

## Provenance

- TDC benchmark: Therapeutics Data Commons (tdcommons.ai), ADMET group, scaffold split, 5 seeds.
- Polaris ADME-Fang: Biogen ADME-Fang benchmark on the Polaris Hub (polarishub.io),
  official `polaris/adme-fang-{hclint,rclint,perm,solu,hppb,rppb}-1` benchmarks;
  random split, Pearson r metric; ensemble officially submitted as `LateFusion-k5`.
- Leaderboard scores in `polaris_leaderboard_full.json` were parsed from the public
  benchmark pages and reflect publicly listed submissions at parse time.
