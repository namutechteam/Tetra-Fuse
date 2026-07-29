#!/usr/bin/env python
"""Regenerate all1023.csv — the 1,023-subset uniform-mean fusion sweep.

Reproduces data/recompute_data/all1023.csv from the 10 base-model prediction
files alone (data/recompute_data/predictions/{model}.parquet) plus the TDC
leaderboard snapshot used as the norm_score ruler. For every one of the 1,023
non-empty subsets of the 10 models it computes the uniform-mean fusion's mean
norm_score on BOTH the validation and the held-out TDC test split, under the
matched-seed (within) and seed-averaged (cross) tracks.

The scoring functions here are byte-for-byte the ones the section notebooks'
setup cell uses (metric / norm_pct / merge_uniform / pertask_norm), so the
output matches the shipped all1023.csv. This is the "predictions -> script ->
csv" closure: nothing here is hard-coded, and no GPU/network is needed.

Output columns: combo_key,k,val_within,val_cross,test_within,test_cross,
val_comb,test_comb  (val_comb = val_within+val_cross; test_comb likewise).

Usage (from paper_release/):
    python code/make_all1023.py
    # writes data/recompute_data/all1023.csv (1023 rows, ~40 min single-core)
Optional: pass an output path as argv[1].
"""
import os, sys, json, time, itertools
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, mean_absolute_error
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data", "recompute_data")     # release data dir
OUT  = sys.argv[1] if len(sys.argv) > 1 else os.path.join(DATA, "all1023.csv")

# The 10 base models (order fixes combo_key member spelling only; sweep is order-free)
MODELS = ["mtl_dmpnn", "mtl_chemeleon", "mtl_graphmvp", "mtl_molformer", "mtl_map",
          "stl_dmpnn", "stl_chemeleon", "stl_graphmvp", "stl_molformer", "stl_map"]

# --- load raw inputs (predictions + leaderboard ruler + per-task metric) ---
SNAP = json.load(open(os.path.join(DATA, "tdc_leaderboard_snapshot.json")))
REG  = json.load(open(os.path.join(DATA, "task_registry.json")))
# 21 evaluation tasks: registered tasks present in the snapshot, minus the excluded auxiliary
EXCLUDE = {"Bioavailability_Ma"}
TASKS = [t for t in REG if isinstance(SNAP.get(t), dict) and SNAP[t].get("models") and t not in EXCLUDE]

PRED = {}
for m in MODELS:
    d = pd.read_parquet(os.path.join(DATA, "predictions", f"{m}.parquet"))
    # synthesize a within-group replicate index so duplicate SMILES merge 1:1 (matches notebook)
    PRED[m] = d.assign(rep=d.groupby(["seed", "task", "split", "smiles"]).cumcount())


def metric(name, y, p):
    """One TDC metric (mae/auroc/auprc/spearman) after dropping NaN labels."""
    y = np.asarray(y, float); p = np.asarray(p, float); m = ~np.isnan(y); y, p = y[m], p[m]
    if name == "mae":      return float(mean_absolute_error(y, p))
    if name == "auroc":    return float(roc_auc_score(y, p))
    if name == "auprc":    return float(average_precision_score(y, p))
    if name == "spearman": return float(spearmanr(y, p)[0])
    raise KeyError(name)


def norm_pct(score, task):
    """norm_score = fraction of TDC leaderboard models we beat (metric-direction aware)."""
    e = SNAP.get(task)
    if e is None or np.isnan(score): return np.nan
    s = [mm["score"] for mm in e["models"].values()]; met = REG[task]["metric"]
    worse = (sum(1 for x in s if x > score) if met == "mae" else sum(1 for x in s if x < score))
    return worse / len(s)


def merge_uniform(models, within):
    """Inner-join the models' per-molecule predictions; `yhat` = their uniform mean.
    within=True keeps the seed axis (matched-seed); within=False averages across seeds first."""
    keys = ["seed", "task", "split", "smiles", "rep"] if within else ["task", "split", "smiles", "rep"]
    if within:
        base = PRED[models[0]][keys + ["y_true", "y_pred"]].rename(columns={"y_pred": f"p_{models[0]}"})
        for m in models[1:]:
            base = base.merge(PRED[m][keys + ["y_pred"]].rename(columns={"y_pred": f"p_{m}"}), on=keys, how="inner")
    else:
        sm = {m: PRED[m].groupby(["task", "split", "smiles", "rep"])
                        .agg(y_true=("y_true", "first"), y_pred=("y_pred", "mean")).reset_index() for m in models}
        base = sm[models[0]][keys + ["y_true", "y_pred"]].rename(columns={"y_pred": f"p_{models[0]}"})
        for m in models[1:]:
            base = base.merge(sm[m][keys + ["y_pred"]].rename(columns={"y_pred": f"p_{m}"}), on=keys, how="inner")
    base["yhat"] = base[[f"p_{m}" for m in models]].mean(axis=1)
    return base


def mean_norm(models, within, split):
    """Mean norm_score over the 21 tasks for the uniform-mean fusion, on `split`.
    within=True: score each seed then average the per-seed norm (matched-seed track);
    within=False: score once on the seed-averaged predictions (seed-averaged track)."""
    base = merge_uniform(models, within); vals = []
    for task in TASKS:
        met = REG[task]["metric"]
        t = base[(base.task == task) & (base.split == split)].dropna(subset=["y_true"])
        if len(t) == 0: continue
        if within:
            seed_sc = [metric(met, g.y_true, g.yhat) for _, g in t.groupby("seed")]
            vals.append(norm_pct(float(np.mean(seed_sc)), task))
        else:
            vals.append(norm_pct(metric(met, t.y_true, t.yhat), task))
    vals = [v for v in vals if not np.isnan(v)]
    return float(np.mean(vals)) if vals else np.nan


def main():
    t0 = time.time(); rows = []
    subsets = [c for k in range(1, len(MODELS) + 1) for c in itertools.combinations(MODELS, k)]
    print(f"tasks={len(TASKS)} models={len(MODELS)} subsets={len(subsets)}", flush=True)
    for j, combo in enumerate(subsets):
        combo = list(combo)
        vw = mean_norm(combo, True,  "val");  vc = mean_norm(combo, False, "val")
        tw = mean_norm(combo, True,  "test"); tc = mean_norm(combo, False, "test")
        rows.append({"combo_key": ",".join(sorted(combo)), "k": len(combo),
                     "val_within": vw, "val_cross": vc, "test_within": tw, "test_cross": tc,
                     "val_comb": vw + vc, "test_comb": tw + tc})
        if (j + 1) % 100 == 0:
            print(f"  {j+1}/{len(subsets)}  {time.time()-t0:.0f}s", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)
    print(f"DONE {len(df)} rows -> {OUT}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
