# Polaris ADME-Fang — external-validation reference files

These files document the paper's external validation on the **Polaris ADME-Fang**
benchmark (Figure 6, Table 5). They are **not read by the reproduction notebook** —
the notebook regenerates Figure 6 from `../polaris_leaderboard_full.json` (kept at the
top level of `data/` because it is a notebook input). The files here are the underlying
per-molecule predictions, per-endpoint scores, and official leaderboard ranks, retained
for transparency and to make the Polaris submission independently auditable.

| File | Content |
|---|---|
| `polaris_all_model_test_preds.csv` | Seed-averaged per-molecule test predictions (2,076 rows). Cols: `task,smiles,mtl_dmpnn,mtl_molformer,stl_chemeleon,stl_graphmvp,stl_map,ensemble`. The `ensemble` column is the uniform mean of the five models (LateFusion-k5). **This is the exact file submitted to the Polaris Hub.** |
| `polaris_official_pertask_pearson.csv` | Per-model & ensemble Pearson r on each official Polaris test set (6 endpoints). Basis for the main-text 3/6 win claim and Table S-level per-model detail. |
| `polaris_official_overall.csv` | 6-endpoint mean Pearson r per model. |
| `polaris_official_submitted_rank.csv` | LateFusion-k5 official leaderboard rank per endpoint. Cols: `benchmark,slug,our_pearson,rank,n_entries,top_model,top_pearson,gap_to_top`. Basis for Table 5. |
| `polaris_ensemble_pertask_val_pearson.csv` | Per-model & ensemble Pearson on the Polaris validation split (within-seed). |

## Provenance
- Benchmark: Biogen ADME-Fang on the Polaris Hub (polarishub.io), official
  `polaris/adme-fang-{hclint,rclint,perm,solu,hppb,rppb}-1` benchmarks; random split,
  Pearson r metric.
- Our ensemble was officially submitted as **LateFusion-k5** (uniform mean of the five
  final models).
- Endpoints: hclint (human liver-microsomal clearance), rclint (rat), perm (MDR1-MDCK
  permeability), solu (solubility), hppb (human plasma-protein binding), rppb (rat).
