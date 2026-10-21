"""Shared loaders/scorers for Tetra-Fuse follow-up analyses.
Scoring functions copied verbatim in logic from paper_release/code/make_all1023.py."""
import os, json, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, mean_absolute_error
from scipy.stats import spearmanr

def load(root):
    D = os.path.join(root, "data", "recompute_data")
    SNAP = json.load(open(os.path.join(D, "tdc_leaderboard_snapshot.json")))
    REG = json.load(open(os.path.join(D, "task_registry.json")))
    TASKS = [t for t in REG if isinstance(SNAP.get(t), dict) and SNAP[t].get("models") and t != "Bioavailability_Ma"]
    MODELS = ["mtl_dmpnn","mtl_chemeleon","mtl_graphmvp","mtl_molformer","mtl_map",
              "stl_dmpnn","stl_chemeleon","stl_graphmvp","stl_molformer","stl_map"]
    PRED = {}
    for m in MODELS:
        d = pd.read_parquet(os.path.join(D, "predictions", f"{m}.parquet"))
        PRED[m] = d.assign(rep=d.groupby(["seed","task","split","smiles"]).cumcount())
    A1023 = pd.read_csv(os.path.join(D, "all1023.csv"))
    return dict(SNAP=SNAP, REG=REG, TASKS=TASKS, MODELS=MODELS, PRED=PRED, A1023=A1023)

def metric(name, y, p):
    y = np.asarray(y, float); p = np.asarray(p, float); m = ~np.isnan(y); y, p = y[m], p[m]
    if name == "mae": return float(mean_absolute_error(y, p))
    if name == "auroc": return float(roc_auc_score(y, p))
    if name == "auprc": return float(average_precision_score(y, p))
    if name == "spearman": return float(spearmanr(y, p)[0])
    raise KeyError(name)

def norm_pct(score, task, SNAP, REG):
    e = SNAP.get(task)
    if e is None or np.isnan(score): return np.nan
    s = [mm["score"] for mm in e["models"].values()]; met = REG[task]["metric"]
    worse = (sum(1 for x in s if x > score) if met == "mae" else sum(1 for x in s if x < score))
    return worse / len(s)

def wide(PRED, models, split="test"):
    """Per-molecule wide table: index (task, smiles, rep); columns (model, seed) -> y_pred; plus y_true."""
    frames = []
    for m in models:
        d = PRED[m][PRED[m].split == split]
        w = d.pivot_table(index=["task","smiles","rep"], columns="seed", values="y_pred", aggfunc="first")
        w.columns = pd.MultiIndex.from_product([[m], w.columns])
        frames.append(w)
    W = pd.concat(frames, axis=1, join="inner")
    y = PRED[models[0]][PRED[models[0]].split == split].groupby(["task","smiles","rep"])["y_true"].first()
    return W, y.reindex(W.index)
