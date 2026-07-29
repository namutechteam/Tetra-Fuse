# Ensemble-subset sweep file (`all1023.csv`) — what it is, how the notebooks use it, how to regenerate it

`recompute_data/all1023.csv` has one row per non-empty uniform-mean subset of the
10 base models (2¹⁰−1 = 1,023 rows). Each subset is scored on both validation and
the held-out TDC test split, in the matched-seed (**within**) and seed-averaged
(**cross**) tracks:

| column | meaning |
|---|---|
| `combo_key` | comma-joined member list (the subset) |
| `k` | subset size (number of members) |
| `val_within`, `val_cross` | mean norm_score on **validation**, within / cross track |
| `test_within`, `test_cross` | mean norm_score on **held-out test**, within / cross track |
| `val_comb`, `test_comb` | `within + cross` on validation / test |

It is the single source for both ensemble-size analyses: the **validation** columns
drive the k-sweep elbow (Figure 3 / Table S3), and the **validation→test** columns
drive the reliability scatter (Figure 5).

---

## A. Regenerate `all1023.csv` from the released predictions

`all1023.csv` is shipped as a result file, but is fully closed over the 10
`predictions/*.parquet` — regenerate it deterministically with:

```bash
cd paper_release
python code/make_all1023.py        # -> data/recompute_data/all1023.csv (1023 rows, ~40 min single-core)
```

`make_all1023.py` reuses the notebooks' exact scoring functions (`metric` /
`norm_pct` / `merge_uniform`), so it reproduces the shipped `all1023.csv`
byte-for-byte — the `predictions -> script -> csv` closure, no GPU/network, no
randomness (running it twice gives identical output).

---

## B. How the notebooks use it (deterministic, no GPU)

`all1023.csv` is read inside the shared setup cell that every section notebook
carries. The two analysis functions that consume it are the ensemble-size sweep
(§3.2, `32_fusion_comparison.ipynb` — Figure 3 / Table S3) and the validation→test
reliability check (§3.4, `34_gain_analysis.ipynb` — Figure 5):

```python
# all1023.csv (validation columns)  ->  ensemble-size sweep (Figure 3 / Table S3)
#   top-20 mean validation norm_score per k, and the marginal gain per +1 member
ksweep()          # returns [[k, within_top20mean, within_marg, cross_top20mean, cross_marg], ...]

# all1023.csv (val -> test)  ->  validation->test rank reliability (Figure 5)
#   Spearman rho / Kendall tau of validation vs held-out-test norm_score, per track
val_test_corr()   # returns {"within":{rho,tau}, "cross":{...}, "combined":{...}, "n_configs":1023}
```

Stand-alone (only `all1023.csv`, no figure rendering):

```bash
cd paper_release
python - <<'PY'
import pandas as pd
from scipy.stats import spearmanr, kendalltau
D = "data/recompute_data"
a = pd.read_csv(f"{D}/all1023.csv")

# --- ensemble-size sweep: top-20 mean validation norm_score per k ---
def top20(col): return {k: round(a[a.k==k].nlargest(min(20,(a.k==k).sum()), col)[col].mean(),4)
                        for k in range(1,11)}
tw, tc = top20("val_within"), top20("val_cross")
print("k-sweep (top-20 mean validation norm_score):")
for k in range(1,11):
    # combined marginal = within marginal + cross marginal (matches Figure 3 / GT["ksweep"])
    dm = round((tw[k]-tw[k-1]) + (tc[k]-tc[k-1]),4) if k>1 else None
    print(f"  k={k:2d}  within={tw[k]:.4f}  cross={tc[k]:.4f}  comb_marg={dm}")
# elbow at k=5: k4->5 > epsilon (0.01), k5->6 < epsilon

# --- validation -> test rank reliability ---
for name, vc, tc_ in [("within","val_within","test_within"),
                      ("cross","val_cross","test_cross"),
                      ("combined","val_comb","test_comb")]:
    rho = spearmanr(a[vc], a[tc_])[0]; tau = kendalltau(a[vc], a[tc_])[0]
    print(f"{name:9s} rho={rho:.4f} tau={tau:.4f}  (n={len(a)})")
PY
```
