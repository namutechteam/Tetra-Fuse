# Ensemble-subset scoring files (`clean_stage1_current.csv` & `all1023.csv`) — what they are, how the notebook uses them, how they were made

Two 1,023-row files under `recompute_data/`, one row per non-empty uniform-mean
subset of the 10 backbones. They are **complementary, not duplicates**:

| file | scored on | key columns | consumed by |
|---|---|---|---|
| `clean_stage1_current.csv` | **validation only** | `k`, `bases`, `simple_{within,cross}_{norm,rank,ntask}` (+ legacy `weighted_*`/`ridge_*`, unused) | Figure 3 + Table S3 (k-sweep elbow) |
| `all1023.csv` | **validation AND held-out test** | `combo_key`, `k`, `val_{within,cross}`, `test_{within,cross}`, `val_comb`, `test_comb` | Figure 4a (val→test rank reliability) |

The validation columns differ numerically between the two files (max abs diff
~0.038) because they come from different scoring passes; only `all1023.csv`
carries the `test_*` columns Figure 4a needs.

---

## A. Analysis / use commands (reproduction package — deterministic, no GPU)

Both files are read inside the shared setup cell that every section notebook
carries, and turned into the published numbers. The analyses that actually
consume them are the ensemble-size sweep (§3.2, `32_fusion_comparison.ipynb` —
Figure 3 / Table S3) and the validation→test reliability check (§3.4,
`34_gain_analysis.ipynb` — Figure 5). To reproduce those:

```bash
cd paper_release/notebooks
# each notebook's setup cell reads clean_stage1_current.csv + all1023.csv
# from ../data/recompute_data; run the two that use them (or any section notebook):
jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=1800 32_fusion_comparison.ipynb
jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=1800 34_gain_analysis.ipynb
```

The two analysis functions that consume the files (in the setup cell) are:

```python
# clean_stage1_current.csv  ->  ensemble-size sweep (Figure 3 / Table S3)
#   top-20 mean validation norm_score per k, and the marginal gain per +1 member
ksweep()          # returns [[k, within_top20mean, within_marg, cross_top20mean, cross_marg], ...]

# all1023.csv  ->  validation->test rank reliability (Figure 4a)
#   Spearman rho / Kendall tau of validation vs held-out-test norm_score, per track
val_test_corr()   # returns {"within":{rho,tau}, "cross":{...}, "combined":{...}, "n_configs":1023}
```

Stand-alone (only these two files, no figure rendering):

```bash
cd paper_release
python - <<'PY'
import pandas as pd
from scipy.stats import spearmanr, kendalltau
D = "data/recompute_data"

# --- clean_stage1_current.csv : ensemble-size sweep (top-20 mean per k) ---
s1 = pd.read_csv(f"{D}/clean_stage1_current.csv")
s1["k"] = s1.bases.apply(lambda b: len(b.split(",")))
def top20(col): return {k: round(s1[s1.k==k].nlargest(min(20,(s1.k==k).sum()), col)[col].mean(),4)
                        for k in range(1,11)}
tw, tc = top20("simple_within_norm"), top20("simple_cross_norm")
print("k-sweep (top-20 mean validation norm_score):")
for k in range(1,11):
    # combined marginal = within marginal + cross marginal (matches Figure 3 / GT["ksweep"])
    dm = round((tw[k]-tw[k-1]) + (tc[k]-tc[k-1]),4) if k>1 else None
    print(f"  k={k:2d}  within={tw[k]:.4f}  cross={tc[k]:.4f}  comb_marg={dm}")
# elbow at k=5: k4->5 = +0.0104 (> epsilon 0.01), k5->6 = +0.0029 (< epsilon)

# --- all1023.csv : validation -> test rank reliability ---
a = pd.read_csv(f"{D}/all1023.csv")
for name, vc, tc_ in [("within","val_within","test_within"),
                      ("cross","val_cross","test_cross"),
                      ("combined","val_comb","test_comb")]:
    rho = spearmanr(a[vc], a[tc_])[0]; tau = kendalltau(a[vc], a[tc_])[0]
    print(f"{name:9s} rho={rho:.4f} tau={tau:.4f}  (n={len(a)})")
PY
```

---

## B. Source-generation commands (development session — NOT in the release package)

The two files were produced in the ablation development session
`qadmet/docs/sessions/2026-06-01_stack_ablation/` (conda env `admet-mtl`),
not by the public reproduction package. They are provided pre-computed under
`recompute_data/` because regeneration is heavy (per-subset nested-OOF /
200-trial scoring across 1,023 subsets) and depends on session-local runner
scripts.

```bash
cd /nas/tk_ahn/ADMET-MTL/qadmet/docs/sessions/2026-06-01_stack_ablation
PY=/nas/tk_ahn/miniconda3/envs/admet-mtl/bin/python

# clean_stage1_current.csv  <-  Stage-1 of the clean selection pipeline
#   (all base combos x {simple, weighted, ridge}, Optuna n1=80, nested-OOF on validation;
#    writes results/clean_stage1_<tag>.csv)
$PY clean_pipeline.py --json artifacts_full10.json --tag current --n1 80 --n2 200 --workers 24

# all1023.csv  <-  uniform-mean sweep over all 1,023 subsets (val + test)
#   assembled from the batch uniform runners; the per-subset worker is:
#   batch_uniform60.py  (reliability._pertask -> u_within/u_cross norm+rank per subset).
#   The published all1023.csv is the merged uniform val+test table over all 1,023 subsets
#   (columns renamed to val_/test_{within,cross} + val_/test_comb); it is a curated
#   aggregate of these runner outputs, so there is no single one-line regeneration command.
```

> Note: `runner_eoe.py` only **reads** `clean_stage1_current.csv`; the producer is
> `clean_pipeline.py`. `all1023.csv` is an assembled table (uniform val+test over all
> subsets), so treat `recompute_data/all1023.csv` as the authoritative released source
> rather than re-deriving it.
