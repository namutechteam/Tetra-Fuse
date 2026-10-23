"""Tetra-Fuse controlled comparisons (manuscript Section 2.7, Fig. 2, Fig. 3C, Table 3, Table S10).

Everything here is recomputed from the shipped per-molecule predictions
(``data/recompute_data/predictions/*.parquet``) and the archived leaderboard snapshot.

Scoring convention (the manuscript's; identical to ``pertask_norm(..., within=True)`` in the
notebooks and to ``code/homog_control.py``): a configuration with several realizations
(splits, or assignments of runs to members) is scored by computing the official RAW metric
for every realization, averaging the raw values, and rank-normalizing ONCE against the
leaderboard ("score -> mean -> norm").

Runs: run i (i = 0..4) uses TDC split seed i and training seed i; the five runs are paired,
not crossed, and share the same held-out test set, so their predictions can be averaged per
molecule.
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from tf_common import load as _load, metric as _metric, norm_pct as _norm_pct, wide as _wide

SEED = 20261002
B = 2000
SEEDS = [0, 1, 2, 3, 4]
POOL = ["stl_chemeleon", "mtl_dmpnn", "mtl_molformer", "stl_graphmvp"]  # Tetra-Fuse's four encoder types
TF = ["mtl_dmpnn", "mtl_molformer", "stl_chemeleon", "stl_graphmvp", "stl_map"]
NAMES = {"stl_chemeleon": "CheMeleon·STL", "mtl_chemeleon": "CheMeleon·MTL", "stl_map": "MAP·STL", "mtl_map": "MAP·MTL",
         "stl_graphmvp": "GraphMVP·STL", "mtl_graphmvp": "GraphMVP·MTL", "stl_molformer": "MoLFormer·STL",
         "mtl_molformer": "MoLFormer·MTL", "stl_dmpnn": "D-MPNN·STL", "mtl_dmpnn": "D-MPNN·MTL"}


class Ctx:
    """Per-task test arrays and scoring helpers, built once from the release data."""

    def __init__(self, root: str):
        L = _load(root)
        self.root = root
        self.TASKS, self.SNAP, self.REG = L["TASKS"], L["SNAP"], L["REG"]
        self.PRED, self.MODELS, self.A1023 = L["PRED"], L["MODELS"], L["A1023"]
        assert len(self.TASKS) == 21
        W, y = _wide(self.PRED, self.MODELS, "test")
        self.ARR = {}
        for t in self.TASKS:
            yt = y.loc[t].to_numpy(float)
            keep = ~np.isnan(yt)
            self.ARR[t] = (yt[keep], {(m, s): W.loc[t][(m, s)].to_numpy(float)[keep]
                                      for m in self.MODELS for s in SEEDS})
        self.BI = np.random.default_rng(SEED).integers(0, len(self.TASKS), size=(B, len(self.TASKS)))

    # ---- scoring ---------------------------------------------------------------------
    def raw(self, t, members):
        yt, X = self.ARR[t]
        return _metric(self.REG[t]["metric"], yt, np.mean([X[i] for i in members], axis=0))

    def norm(self, t, r):
        return _norm_pct(r, t, self.SNAP, self.REG)

    def ms(self, t, realisations):
        """score -> mean -> norm over the realizations of one configuration."""
        return self.norm(t, float(np.mean([self.raw(t, R) for R in realisations])))

    # ---- named configurations ----------------------------------------------------------
    def multi_run(self, t, model, k):
        """Same-model ensemble of k runs: all C(5, k) run subsets."""
        return self.ms(t, [[(model, s) for s in c] for c in itertools.combinations(SEEDS, k)])

    def matched_seed(self, t, models):
        """Uniform mean of one member per model type, all from the same run (5 realizations)."""
        return self.ms(t, [[(m, s) for m in models] for s in SEEDS])

    # ---- statistics --------------------------------------------------------------------
    def contrast(self, a, b, label=""):
        a, b = np.asarray(a, float), np.asarray(b, float)
        d = a - b
        bs = d[self.BI].mean(1)
        return dict(contrast=label, mean_a=a.mean(), mean_b=b.mean(), delta=d.mean(),
                    ci_low=np.percentile(bs, 2.5), ci_high=np.percentile(bs, 97.5),
                    wins=int((d > 1e-12).sum()), ties=int((np.abs(d) <= 1e-12).sum()),
                    losses=int((d < -1e-12).sum()))


# =====================================================================================
# (a)/(b) member count x encoder heterogeneity, with same-encoder controls (Fig. 2, Table S10)
# =====================================================================================
def _assignments(combo, m, rng, cap=2000, nsamp=400):
    d = len(combo)
    q, r = divmod(m, d)
    pats = sorted(set(itertools.permutations([q + 1] * r + [q] * (d - r))))
    sp = []
    for pat in pats:
        for ch in itertools.product(*[list(itertools.combinations(SEEDS, c)) for c in pat]):
            sp.append([(bb, s) for bb, ss in zip(combo, ch) for s in ss])
    if len(sp) > cap:
        sp = [sp[i] for i in sorted(rng.choice(len(sp), nsamp, replace=False))]
    return sp


def validation_selection(C: Ctx, pool=None):
    """Validation-only choice of same-encoder controls (no test labels)."""
    rows = []
    for mm in C.MODELS:
        d = C.PRED[mm][C.PRED[mm].split == "val"].dropna(subset=["y_true"])
        for (t, s), g in d.groupby(["task", "seed"]):
            if t in C.TASKS:
                rows.append(dict(model=mm, task=t, seed=s, raw=_metric(C.REG[t]["metric"], g.y_true, g.y_pred)))
    V = pd.DataFrame(rows).groupby(["model", "task"]).raw.mean().reset_index()
    V["vnorm"] = [C.norm(t, r) for t, r in zip(V.task, V.raw)]
    V["sgn_raw"] = [(-r if C.REG[t]["metric"] == "mae" else r) for t, r in zip(V.task, V.raw)]

    def per_task_best(cands):
        sub = V[V.model.isin(cands)].sort_values(["task", "vnorm", "sgn_raw"], ascending=[True, False, False])
        return sub.groupby("task").model.first().to_dict()

    k1 = C.A1023[C.A1023.k == 1].set_index("combo_key").val_within
    pool = list(pool or POOL)
    return dict(global_pool=k1.loc[pool].idxmax(), global_all=k1.idxmax(),
                task_pool=per_task_best(pool), task_all=per_task_best(C.MODELS))


def heterogeneity_grid(C: Ctx, pool=None):
    """Per-task table of every arm used in Fig. 2 and Table S10a (manuscript convention)."""
    pool = list(pool or POOL)
    sel = validation_selection(C, pool)
    rng = np.random.default_rng(SEED)
    grid = [(m, d) for m in range(1, 6) for d in range(1, min(m, 4) + 1)]
    draws = {(m, d): [_assignments(c, m, rng) for c in itertools.combinations(pool, d)] for m, d in grid}
    rows = []
    for t in C.TASKS:
        rec = dict(task=t, metric=C.REG[t]["metric"])
        for m, d in grid:
            rec[f"Het_{m}_{d}"] = np.mean([C.ms(t, sp) for sp in draws[(m, d)]])
        for d in range(1, 5):
            rec[f"HetMS_{d}"] = np.mean([C.matched_seed(t, c) for c in itertools.combinations(pool, d)])
        for m in range(1, 6):
            hb = {b: C.multi_run(t, b, m) for b in C.MODELS}
            rec[f"HomMean_{m}"] = np.mean([hb[b] for b in pool])
            rec[f"HomGlobalPool_{m}"] = hb[sel["global_pool"]]
            rec[f"HomGlobalAll_{m}"] = hb[sel["global_all"]]
            rec[f"HomTaskValPool_{m}"] = hb[sel["task_pool"][t]]
            rec[f"HomTaskValAll_{m}"] = hb[sel["task_all"][t]]
            rec[f"HomTaskTestPool_{m}"] = max(hb[b] for b in pool)
            rec[f"HomTaskTestAll_{m}"] = max(hb.values())
            for b in C.MODELS:
                rec[f"Hom[{b}]_{m}"] = hb[b]
        rec["TF_ms"] = C.matched_seed(t, TF)
        rec["tvb_pool"], rec["tvb_all"] = sel["task_pool"][t], sel["task_all"][t]
        rows.append(rec)
    P = pd.DataFrame(rows).set_index("task")
    return P, sel


def heterogeneity_contrasts(C: Ctx, P: pd.DataFrame):
    c = lambda a, b, lab: C.contrast(P[a], P[b], lab)
    out = []
    for m in (4, 5):
        het = f"Het_{m}_4"
        tag = "m=4" if m == 4 else "m=5 (four-encoder pool, S(5,4))"
        out += [c(het, f"HomMean_{m}", f"{tag}: heterogeneous vs same-encoder, mean of 4 encoders"),
                c(het, f"HomGlobalPool_{m}", f"{tag}: heterogeneous vs same-encoder, validation-best encoder (CheMeleon·STL)"),
                c(het, f"HomTaskValPool_{m}", f"{tag}: heterogeneous vs same-encoder, per-task validation-best encoder"),
                c(het, f"HomTaskTestPool_{m}", f"{tag}: heterogeneous vs same-encoder, per-task test-best encoder (oracle)"),
                c(het, f"HomGlobalAll_{m}", f"{tag}: heterogeneous vs same-model, validation-best of all 10 (MAP·STL)"),
                c(het, f"HomTaskValAll_{m}", f"{tag}: heterogeneous vs same-model, per-task validation-best of all 10")]
    out += [c("HetMS_4", "HomMean_4", "matched-seed heterogeneous (4) vs same-encoder, mean of 4 encoders"),
            c("HetMS_4", "HomGlobalPool_4", "matched-seed heterogeneous (4) vs same-encoder, validation-best encoder"),
            c("HetMS_4", "HomTaskValPool_4", "matched-seed heterogeneous (4) vs same-encoder, per-task validation-best"),
            c("HetMS_4", "HomTaskTestPool_4", "matched-seed heterogeneous (4) vs same-encoder, per-task test-best (oracle)"),
            c("TF_ms", "HomMean_5", "Tetra-Fuse vs same-encoder 5 runs, mean of 4 encoders"),
            c("TF_ms", "HomGlobalPool_5", "Tetra-Fuse vs same-encoder 5 runs, validation-best encoder (CheMeleon·STL)"),
            c("TF_ms", "HomTaskValPool_5", "Tetra-Fuse vs same-encoder 5 runs, per-task validation-best encoder"),
            c("TF_ms", "HomGlobalAll_5", "Tetra-Fuse vs same-model 5 runs, validation-best of all 10 (MAP·STL)"),
            c("TF_ms", "HomTaskValAll_5", "Tetra-Fuse vs same-model 5 runs, per-task validation-best of all 10"),
            c("TF_ms", "HomTaskTestAll_5", "Tetra-Fuse vs same-model 5 runs, per-task test-best of all 10 (oracle)")]
    for m in range(2, 6):
        out.append(c(f"Het_{m}_{min(m, 4)}", f"Het_{m}_1", f"E_div({m})"))
    out.append(c("Het_4_4", "Het_1_1", "N = S(4,4) - S(1,1)"))
    out.append(c("Het_4_1", "Het_1_1", "E_cnt = S(4,1) - S(1,1)"))
    CT = pd.DataFrame(out)
    num = (P["Het_4_1"] - P["Het_1_1"]).to_numpy()
    den = (P["Het_4_4"] - P["Het_1_1"]).to_numpy()
    sh = num[C.BI].mean(1) / den[C.BI].mean(1)
    CT = pd.concat([CT, pd.DataFrame([dict(contrast="count share E_cnt/N (descriptive)", delta=num.mean() / den.mean(),
                                           ci_low=np.percentile(sh, 2.5), ci_high=np.percentile(sh, 97.5))])],
                   ignore_index=True)
    G = pd.DataFrame([dict(m=m, d=d, S=P[f"Het_{m}_{d}"].mean()) for m in range(1, 6) for d in range(1, min(m, 4) + 1)])
    return CT, G


# =====================================================================================
# (c) fusion level at matched member count (Fig. 3C)
# =====================================================================================
def fusion_level(C: Ctx):
    """Per-task scores of MAP (1 model), Late-3, MAP x3, MAP x5 per paradigm and Tetra-Fuse."""
    rows = []
    for t in C.TASKS:
        rec = dict(task=t)
        for fam in ("stl", "mtl"):
            three = [f"{fam}_chemeleon", f"{fam}_graphmvp", f"{fam}_molformer"]
            rec[f"{fam}_MAP1"] = C.multi_run(t, f"{fam}_map", 1)
            rec[f"{fam}_Late3"] = C.matched_seed(t, three)
            rec[f"{fam}_MAP3"] = C.multi_run(t, f"{fam}_map", 3)
            rec[f"{fam}_MAP5"] = C.multi_run(t, f"{fam}_map", 5)
        rec["TF"] = C.matched_seed(t, TF)
        rows.append(rec)
    F = pd.DataFrame(rows).set_index("task")
    out = []
    for fam in ("stl", "mtl"):
        u = fam.upper()
        out += [C.contrast(F[f"{fam}_Late3"], F[f"{fam}_MAP1"], f"{u}: Late-3 vs one MAP model (unmatched)"),
                C.contrast(F[f"{fam}_MAP3"], F[f"{fam}_MAP1"], f"{u}: MAP x3 vs one MAP model (ensembling)"),
                C.contrast(F[f"{fam}_Late3"], F[f"{fam}_MAP3"], f"{u}: Late-3 vs MAP x3 (matched)")]
    out.append(C.contrast(F["TF"], F["stl_MAP5"], "Tetra-Fuse vs MAP·STL x5"))
    cost = pd.DataFrame([("MAP (1 model)", 1, 3), ("Late-3", 3, 3), ("MAP x3", 3, 9), ("MAP x5", 5, 15),
                         ("Tetra-Fuse", 5, 7)], columns=["arm", "members", "encoder_forward_passes"])
    return F, pd.DataFrame(out), cost


# =====================================================================================
# (d) averaging over runs
# =====================================================================================
def seed_averaging(C: Ctx):
    A = C.A1023.copy()
    A["members"] = A.combo_key.str.split(",").apply(frozenset)
    k1 = A[A.k == 1][["combo_key", "test_within", "test_cross"]].assign(gain=lambda d: d.test_cross - d.test_within)
    tf = A[A.members == frozenset(TF)].iloc[0]
    return k1, dict(tf_matched=float(tf.test_within), tf_seed_avg=float(tf.test_cross),
                    tf_gain=float(tf.test_cross - tf.test_within),
                    single_mean_gain=float(k1.gain.mean()), single_min=float(k1.gain.min()),
                    single_max=float(k1.gain.max()))


# =====================================================================================
# Table 3 (effects) and Table S10 (per-task / contrasts / grid)
# =====================================================================================
def table3(CT, FC, sa):
    g = lambda lab: CT[CT.contrast == lab].iloc[0]
    f = lambda lab: FC[FC.contrast == lab].iloc[0]
    u = lambda x: x.replace("-", "\u2212")
    fmt = lambda r: u(f"{r.delta:+.3f} ({r.ci_low:.3f} to {r.ci_high:.3f})")
    wtl = lambda r: f"{int(r.wins)}/{int(r.ties)}/{int(r.losses)}"
    rows = [
        ("Adding runs of one encoder", "1 \u2192 4 runs, mean of the four pool encoders", fmt(g("E_cnt = S(4,1) - S(1,1)")), wtl(g("E_cnt = S(4,1) - S(1,1)"))),
        ("Encoder heterogeneity, average control", "4 members; 4 encoders vs 4 runs of one encoder (mean of four)",
         fmt(g("m=4: heterogeneous vs same-encoder, mean of 4 encoders")), wtl(g("m=4: heterogeneous vs same-encoder, mean of 4 encoders"))),
        ("Encoder heterogeneity, strongest validation-selected control", "4 members; vs 4 runs of the validation-best encoder (CheMeleon·STL)",
         fmt(g("m=4: heterogeneous vs same-encoder, validation-best encoder (CheMeleon·STL)")),
         wtl(g("m=4: heterogeneous vs same-encoder, validation-best encoder (CheMeleon·STL)"))),
        ("", "4 members; vs 4 runs of the per-task validation-best encoder",
         fmt(g("m=4: heterogeneous vs same-encoder, per-task validation-best encoder")),
         wtl(g("m=4: heterogeneous vs same-encoder, per-task validation-best encoder"))),
        ("Encoder heterogeneity, test-label oracle (descriptive)", "4 members; vs 4 runs of the per-task test-best encoder",
         fmt(g("m=4: heterogeneous vs same-encoder, per-task test-best encoder (oracle)")),
         wtl(g("m=4: heterogeneous vs same-encoder, per-task test-best encoder (oracle)"))),
        ("Fusion level, unmatched", "Late-3 (3 members) vs one MAP model, STL", fmt(f("STL: Late-3 vs one MAP model (unmatched)")), "–"),
        ("Ensembling the fusion model", "MAP\u00d73 vs one MAP model, STL", fmt(f("STL: MAP x3 vs one MAP model (ensembling)")), "–"),
        ("Fusion level, matched", "Late-3 vs MAP\u00d73, STL / MTL",
         fmt(f("STL: Late-3 vs MAP x3 (matched)")) + " / " + fmt(f("MTL: Late-3 vs MAP x3 (matched)")), "–"),
        ("Averaging over runs", "Seed-averaged minus matched-seed, mean of 10 base models / Tetra-Fuse",
         f"+{sa['single_mean_gain']:.3f} (range {sa['single_min']:.3f} to {sa['single_max']:.3f}) / +{sa['tf_gain']:.3f}", "–"),
    ]
    return pd.DataFrame(rows, columns=["Effect", "Comparison (what is held fixed)", "Delta norm_score (95% CI)", "W/T/L"])


def table_s10(P, CT, G):
    cols = [("Het_4_4", "Het. 4"), ("HetMS_4", "Het. 4, matched-seed"), ("HomMean_4", "Same-enc. 4, mean"),
            ("HomGlobalPool_4", "CheMeleon·STL \u00d74"), ("HomTaskValPool_4", "Per-task val-best \u00d74"),
            ("HomTaskTestPool_4", "Per-task test-best \u00d74 (oracle)"), ("TF_ms", "Tetra-Fuse"),
            ("HomGlobalPool_5", "CheMeleon·STL \u00d75"), ("HomTaskValPool_5", "Per-task val-best \u00d75"),
            ("HomGlobalAll_5", "MAP·STL \u00d75"), ("HomTaskValAll_5", "Per-task val-best (all 10) \u00d75"),
            ("HomTaskTestAll_5", "Per-task test-best (all 10) \u00d75 (oracle)")]
    A = P[[c for c, _ in cols]].rename(columns=dict(cols)).round(3)
    A.insert(0, "Metric", P.metric.str.upper().replace({"SPEARMAN": "Spearman"}))
    A["Val-best encoder (pool)"] = P.tvb_pool.map(NAMES)
    A["Val-best model (all 10)"] = P.tvb_all.map(NAMES)
    num = P[[c for c, _ in cols]].rename(columns=dict(cols)).mean().round(3)  # means of unrounded values
    A.loc["Mean"] = pd.Series({**num.to_dict(), "Metric": "", "Val-best encoder (pool)": "", "Val-best model (all 10)": ""})
    A.index.name = "Task"
    Bt = CT[["contrast", "delta", "ci_low", "ci_high", "wins", "ties", "losses"]].round(6)
    Ct = G.pivot(index="m", columns="d", values="S").round(3)
    Ct.columns = [f"d = {d}" for d in Ct.columns]
    return A.reset_index(), Bt, Ct.reset_index()
