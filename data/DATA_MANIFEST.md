# Data Manifest — Reproduction Bundle

All figures and tables in the paper are reproduced from the derived summary files in
this directory. No raw model checkpoints or GPU compute are required to regenerate the
figures/tables; the two notebooks in `../notebooks/` read only these files.

The values here are the ground-truth numbers used in the manuscript, computed on
Node02 from the trained models' held-out-test and nested-OOF-validation predictions
(21 valid TDC ADMET tasks + the Polaris ADME-Fang external benchmark).

## Primary-benchmark (TDC 21-task) summary files

| File | Content |
|---|---|
| `ground_truth.json` | Master results dict. Keys: `per_model_within_cross_solo` (10 models x [within,cross] solo test norm_score — Figure 1 basis), `ksweep` (10 rows, top-20-mean norm by ensemble size k — Figure 3), `final_ensemble`/`final_composition` (the k=5 members: 2 MTL + 3 STL), `headline` (21-task ensemble norm & mean rank, within/cross), `sota_tasks`, `loo_shapley` (per-member Leave-One-Out and Shapley), `corr` (prediction/residual off-diagonal correlation), `best_single`, `val_test_corr` (Figure 4a reliability rho), `k4_variant`, `k5_selection`. |
| `pertask.json` | Per-task rows for tables. `t1`=21 (per-task ensemble test), `t2`=6, `t3`=21 (Table 3 members), `t4`=22 (all endpoints incl. Bioavailability_Ma auxiliary). |
| `tables_data.json` | Pre-formatted table rows: `T1`-`T4` (main tables), `S1`-`S3` (supplementary), with `cols_s1`/`cols_s2` headers. |
| `stat_tests.json` | Wilcoxon signed-rank results: `deployable_single` (ensemble vs 3 strongest singles), `per_task_oracle`, `n_tasks`, `ensemble_mean`. One-sided, Holm-corrected. |
| `all1023.csv` | All 1,023 non-empty subsets of the 10 base models (uniform-mean). Cols: `combo_key,k,val_within,val_cross,test_within,test_cross,val_comb,test_comb`. Basis for Figure 4a and the k-sweep. |
| `tdc_leaderboard_snapshot.json` | TDC ADMET Benchmark Group public leaderboard snapshot (23 tasks, captured 2026-04-29), used as the rank-normalization ruler for all TDC norm_scores. |
| `uniform_val_test_unified.csv` | 133-subset validation/test table (subset used in an earlier draft; retained for provenance). Same columns as all1023. |
| `solo10_within_cross.json` | 10-model solo within/cross norm_score (redundant with ground_truth for cross-check). |
| `reconciliation_flags.json` | 10 draft-vs-ground-truth discrepancies that were fixed during revision (provenance/audit trail). |
| `additions.json` | Manuscript text blocks (title, contributions, statistical paragraph, Box 1) in EN/KO. |

## External-benchmark (Polaris ADME-Fang, 6 endpoints) files

| File | Content |
|---|---|
| `polaris_all_model_test_preds.csv` | Seed-averaged per-molecule test predictions, 2076 rows. Cols: `task,smiles,mtl_dmpnn,mtl_molformer,stl_chemeleon,stl_graphmvp,stl_map,ensemble`. The `ensemble` column is the uniform mean of the five models (LateFusion-k5). This is the file submitted to the Polaris Hub. |
| `polaris_official_pertask_pearson.csv` | Per-model & ensemble Pearson r on each official Polaris test set (6 endpoints x 6 columns). Basis for Table S7 and the main-text 3/6 win claim. |
| `polaris_official_overall.csv` | 6-task mean Pearson r per model. |
| `polaris_official_submitted_rank.csv` | LateFusion-k5 official leaderboard rank per endpoint. Cols: `benchmark,slug,our_pearson,rank,n_entries,top_model,top_pearson,gap_to_top`. Basis for Table 5. |
| `polaris_ensemble_pertask_val_pearson.csv` | Per-model & ensemble Pearson on the Polaris validation split (within-seed). |
| `polaris_leaderboard_full.json` | Full public leaderboard scores parsed per benchmark (`hclint,rclint,perm,solu,hppb,rppb`), each a list of `[model_name, pearson_r]`. Basis for Figure 6 (all-submission distribution). |

## Provenance

- TDC benchmark: Therapeutics Data Commons (tdcommons.ai), ADMET group, scaffold split, 5 seeds.
- Polaris ADME-Fang: Biogen ADME-Fang benchmark on the Polaris Hub (polarishub.io),
  official `polaris/adme-fang-{hclint,rclint,perm,solu,hppb,rppb}-1` benchmarks;
  random split, Pearson r metric; ensemble officially submitted as `LateFusion-k5`.
- Leaderboard scores in `polaris_leaderboard_full.json` were parsed from the public
  benchmark pages and reflect publicly listed submissions at parse time.
