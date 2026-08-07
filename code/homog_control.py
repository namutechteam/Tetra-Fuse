#!/usr/bin/env python
"""Homogeneous-ensemble control for the heterogeneity claim.

Both arms average exactly FIVE prediction vectors and are scored once, so the
number of model instances (5) and the scoring procedure are identical; the only
difference is the SOURCE of diversity:

  heterogeneous : 5 different encoders at one shared seed (Tetra-Fuse). Scored on
                  the paper's matched-seed basis: the five per-seed RAW metric
                  values are averaged and normalized once (score -> mean -> norm),
                  which is what pertask_norm(..., within=True) computes.
  homogeneous   : one encoder, its five seeds averaged      (== that model's
                  seed-averaged prediction, i.e. a 5-member same-architecture
                  ensemble)

Note the asymmetry is conservative: the homogeneous arm gets the variance
reduction of seed averaging, while each heterogeneous per-seed score does not.

Paired one-sided Wilcoxon over the 21 evaluation tasks, Holm-corrected within
this new family (family B) so the eight prespecified comparisons of Table S8
(family A) keep their published p-values.
"""
import json, os, re, sys
import numpy as np, pandas as pd
from scipy.stats import wilcoxon

NB = "/nas/tk_ahn/ADMET-MTL/paper_release/notebooks/33_heldout_test.ipynb"

# ---- reuse the notebook's own setup/helpers verbatim (no reimplementation) ----
cells = [c for c in json.load(open(NB))["cells"] if c["cell_type"] == "code"]
srcs = ["".join(c["source"]) for c in cells]
# The whole setup (TASKS, REG, SNAP, PRED, metric, norm_pct, merge_uniform, ENSEMBLE)
# lives in the first cell, which also writes a figure at the end. Run cells up to and
# including the one defining merge_uniform, with a non-interactive backend so any
# plotting is harmless, and keep figures out of the way.
import matplotlib
matplotlib.use("Agg")
stop = next(i for i, s in enumerate(srcs) if "def merge_uniform" in s)
ns = {}
for s in srcs[: stop + 1]:
    exec(compile(s, "<nb>", "exec"), ns)

TASKS, ENSEMBLE = ns["TASKS"], ns["ENSEMBLE"]
REG, metric, norm_pct = ns["REG"], ns["metric"], ns["norm_pct"]
merge_uniform, PRED = ns["merge_uniform"], ns["PRED"]
print(f"loaded: {len(TASKS)} tasks | ensemble = {ENSEMBLE}", flush=True)

# the four pure backbones exactly as they enter the ensemble (MAP is excluded from
# the test: it is itself a 3-encoder fusion, so it is not a homogeneous control)
BACKBONES = [("CheMeleon·STL", "stl_chemeleon"), ("D-MPNN·MTL", "mtl_dmpnn"),
             ("MoLFormer·MTL", "mtl_molformer"), ("GraphMVP·STL", "stl_graphmvp")]
MEMO = [("MAP·STL (memo)", "stl_map")]

pertask_norm = ns["pertask_norm"]

def het_raw_per_seed(task_list):
    """Per-seed RAW metric values of the heterogeneous ensemble (for reporting seed
    spread only). The matched-seed norm_score itself averages these and normalizes
    once — see het_norms()."""
    base = merge_uniform(list(ENSEMBLE), within=True)
    out = {}
    for task in task_list:
        met = REG[task]["metric"]
        t = base[(base.task == task) & (base.split == "test")].dropna(subset=["y_true"])
        out[task] = [metric(met, g.y_true, g.yhat) for _, g in t.groupby("seed")]
    return out

# Both arms go through the notebook's own scorer, so the basis is identical to the
# paper by construction: within=True is the matched-seed basis of the heterogeneous
# ensemble; within=False on a single model averages that model's five seed
# predictions and scores them once = a 5-member homogeneous ensemble.
het_d = pertask_norm(list(ENSEMBLE), True)
het_mean = np.array([het_d[t] for t in TASKS])
raw = het_raw_per_seed(TASKS)
het_raw_sd = np.array([np.std(raw[t], ddof=0) for t in TASKS])

# Seed-level normalized scores, for DISPLAY of seed variability only. The published
# matched-seed value is norm_pct(mean(raw)) (a single number per task); these are
# norm_pct applied to each seed separately. Because norm_pct is monotone in the
# direction-aligned score, norm_pct(mean) always lies within [min, max] of these,
# so the interval brackets the plotted point (asserted below).
het_seed_norms = {t: [norm_pct(v, t) for v in raw[t]] for t in TASKS}
het_lo = np.array([min(het_seed_norms[t]) for t in TASKS])
het_hi = np.array([max(het_seed_norms[t]) for t in TASKS])
bad = [t for i, t in enumerate(TASKS) if not (het_lo[i] - 1e-12 <= het_mean[i] <= het_hi[i] + 1e-12)]
assert not bad, f"published matched-seed value outside per-seed norm range: {bad}"
print(f"per-seed norm interval brackets the published value on all {len(TASKS)} tasks; "
      f"mean width = {(het_hi - het_lo).mean():.4f}", flush=True)
print(f"heterogeneous (matched-seed, paper basis) mean norm = {het_mean.mean():.4f}",
      flush=True)

rows, pvals = [], []
for label, model in BACKBONES + MEMO:
    hom = pertask_norm([model], False)
    h = np.array([hom[t] for t in TASKS])
    diff = het_mean - h
    wins = int((diff > 1e-9).sum()); ties = int((abs(diff) <= 1e-9).sum())
    loss = int((diff < -1e-9).sum())
    memo = label.endswith("(memo)")
    if not memo:
        W, p = wilcoxon(het_mean, h, alternative="greater")
        pvals.append(p)
    else:
        W, p = np.nan, np.nan
    rows.append(dict(control=label, model=model, homog_norm=h.mean(),
                     het_norm=het_mean.mean(), mean_delta=diff.mean(),
                     median_delta=float(np.median(diff)), wtl=f"{wins}/{ties}/{loss}",
                     W=W, p_raw=p))
    print(f"  {label:16} homog={h.mean():.4f}  Δ={diff.mean():+.4f}  "
          f"W/T/L={wins}/{ties}/{loss}  p={p if memo else f'{p:.3e}'}", flush=True)

# Holm within family B (the four homogeneous controls only)
order = np.argsort(pvals); m = len(pvals); adj = [None]*m; run = 0.0
for k, idx in enumerate(order):
    v = max(run, min(1.0, (m - k) * pvals[idx])); adj[idx] = v; run = v
j = 0
for r in rows:
    if not r["control"].endswith("(memo)"):
        r["p_holm_familyB"] = adj[j]; j += 1
    else:
        r["p_holm_familyB"] = np.nan

df = pd.DataFrame(rows)
out = "/nas/tk_ahn/ADMET-MTL/paper_release/verification/homogeneous_control.csv"
os.makedirs(os.path.dirname(out), exist_ok=True)
df.to_csv(out, index=False)
print("\n=== Table S9 (family B) ===")
print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

pt = pd.DataFrame({"task": TASKS, "het_matched_norm": het_mean,
                   "het_seed_norm_lo": het_lo, "het_seed_norm_hi": het_hi,
                   "het_raw_seed_sd": het_raw_sd,
                   **{lbl: [pertask_norm([mo], False)[t] for t in TASKS]
                      for lbl, mo in BACKBONES}})
pt.to_csv("/nas/tk_ahn/ADMET-MTL/paper_release/verification/homogeneous_control_pertask.csv",
          index=False)
print(f"\nwrote {out} and homogeneous_control_pertask.csv")
