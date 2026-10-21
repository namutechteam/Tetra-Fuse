"""Pre-registered ANALYSIS C (protocol v1 + addendum v1.1):
late-fusion gain vs. test-molecule similarity to the training data.

Usage:  python analysis_C_distance.py <bundle_root> [out_dir]   (bundle_root contains data/recompute_data, admet_group/)
Requires tf_common.py on sys.path, RDKit, pandas, pyarrow, matplotlib.
Outputs (out_dir, default cwd): C_per_task.csv, C_per_task_bin.csv, C_summary.csv, C_tanimoto_distribution.csv,
               C_molecule_level.parquet, fig_C_distance_gain.png/.pdf
"""
import os, sys, numpy as np, pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tf_common as tc

RDLogger.DisableLog("rdApp.*")
ROOT = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else "bundle")
ADMET_DIR = os.environ.get("TDC_ADMET_GROUP", os.path.join(ROOT, "data", "tdc_admet_group"))  # TDC admet_group cache (train_val.csv per task)
OUT = os.path.abspath(sys.argv[2] if len(sys.argv) > 2 else ".")
os.makedirs(OUT, exist_ok=True)
B, RNG_SEED = 2000, 20261002
MEMBERS = ["mtl_dmpnn", "mtl_molformer", "stl_chemeleon", "stl_graphmvp", "stl_map"]
BACKBONES = ["mtl_dmpnn", "mtl_molformer", "stl_chemeleon", "stl_graphmvp"]
G, G2 = "stl_map", "stl_chemeleon"
SEEDS = [0, 1, 2, 3, 4]
CONTRASTS = {"primary_G_stl_map": "loss_G", "secondary_stl_chemeleon": "loss_chemeleon",
             "secondary_backbone_mean": "loss_bbmean"}

L = tc.load(ROOT); TASKS, REG, PRED = L["TASKS"], L["REG"], L["PRED"]
assert len(TASKS) == 21

# ---------------- Step 1: max Tanimoto to train_val ----------------
gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
def fp(smi):
    m = Chem.MolFromSmiles(smi) if isinstance(smi, str) else None
    return None if m is None else gen.GetFingerprint(m)

W, Y = tc.wide(PRED, MEMBERS, "test")
keep = W.index.get_level_values("task").isin(TASKS)   # drop Bioavailability_Ma (not an evaluation task)
W, Y = W[keep], Y[keep]
assert W.shape[1] == len(MEMBERS) * len(SEEDS) and not W.isna().any().any(), "missing (model,seed) preds"
assert set(W.index.get_level_values("task")) == set(TASKS)

sim_rows, dist_rows = [], []
for t in TASKS:
    d = os.path.join(ADMET_DIR, t.lower())
    assert os.path.isfile(os.path.join(d, "train_val.csv")), f"no admet_group dir for {t}"
    tv = pd.read_csv(os.path.join(d, "train_val.csv"))
    tv_fps = [fp(s) for s in tv["Drug"]]
    n_tv_bad = sum(f is None for f in tv_fps); tv_fps = [f for f in tv_fps if f is not None]
    test_smis = pd.Index(W.loc[t].index.get_level_values("smiles")).unique()
    n_test_csv = len(pd.read_csv(os.path.join(d, "test.csv")))
    n_bad = 0
    for s in test_smis:
        f = fp(s)
        if f is None:
            n_bad += 1; sim_rows.append((t, s, np.nan)); continue
        sim_rows.append((t, s, float(max(DataStructs.BulkTanimotoSimilarity(f, tv_fps)))))
    dist_rows.append(dict(task=t, admet_dir=t.lower(), n_train_val=len(tv), n_train_val_unparsable=n_tv_bad,
                          n_test_rows_pred=int(len(W.loc[t])), n_test_csv=n_test_csv,
                          n_test_unique_smiles=len(test_smis), n_test_unparsable=n_bad))
SIM = pd.DataFrame(sim_rows, columns=["task", "smiles", "max_tanimoto"])

# ---------------- Step 2: per-molecule matched-seed losses ----------------
Wv = W.copy(); idx = Wv.index
yt = Y.values.astype(float)
task_of = idx.get_level_values("task")
is_cls = np.array([REG[t]["type"] == "classification" for t in task_of])
sd = pd.Series(yt, index=task_of).groupby(level=0).std(ddof=1)  # SD(y_test) per task
sd_vec = sd.reindex(task_of).values

def loss(p):  # p: array aligned with idx
    return np.where(is_cls, (p - yt) ** 2, np.abs(p - yt) / sd_vec)

P = {(m, s): Wv[(m, s)].values.astype(float) for m in MEMBERS for s in SEEDS}
for m in MEMBERS:
    if any(REG[t]["type"] == "classification" for t in TASKS):
        pc = np.concatenate([P[(m, s)][is_cls] for s in SEEDS]); assert pc.min() >= 0 and pc.max() <= 1
loss_m = {m: np.mean([loss(P[(m, s)]) for s in SEEDS], axis=0) for m in MEMBERS}
loss_tf = np.mean([loss(np.mean([P[(m, s)] for m in MEMBERS], axis=0)) for s in SEEDS], axis=0)

M = pd.DataFrame({"task": task_of, "smiles": idx.get_level_values("smiles"), "rep": idx.get_level_values("rep"),
                  "y_true": yt, "loss_TF": loss_tf, "loss_G": loss_m[G], "loss_chemeleon": loss_m[G2],
                  "loss_bbmean": np.mean([loss_m[m] for m in BACKBONES], axis=0)})
for m in MEMBERS: M[f"loss_{m}"] = loss_m[m]
M = M.merge(SIM, on=["task", "smiles"], how="left")
n_excl = int(M.max_tanimoto.isna().sum()); M = M[M.max_tanimoto.notna()].copy()
for c, col in CONTRASTS.items(): M[f"delta_{c}"] = M[col] - M["loss_TF"]

# quintile bins per task: Q1 = most similar ... Q5 = least similar
def qbin(s):   # primary: value-quantile edges (pd.qcut on max-Tanimoto; ties stay together)
    q = pd.qcut(s, 5, labels=False, duplicates="raise")
    return 5 - q  # highest-similarity quintile -> 1
def qbin_rank(s):  # sensitivity: equal-count bins, ties broken by order of appearance
    return 5 - pd.qcut(s.rank(method="first"), 5, labels=False)
M["bin"] = M.groupby("task")["max_tanimoto"].transform(qbin).astype(int)
M["bin_rank"] = M.groupby("task")["max_tanimoto"].transform(qbin_rank).astype(int)

# distribution of max-Tanimoto per task
for r in dist_rows:
    s = M.loc[M.task == r["task"], "max_tanimoto"]
    r.update(n_analysed=len(s), sim_mean=s.mean(), sim_min=s.min(), sim_q20=s.quantile(.2), sim_q40=s.quantile(.4),
             sim_median=s.median(), sim_q60=s.quantile(.6), sim_q80=s.quantile(.8), sim_max=s.max(),
             frac_sim_eq_1=(s >= 0.999999).mean(),
             **{f"n_bin{b}": int((M.loc[M.task == r["task"], "bin"] == b).sum()) for b in range(1, 6)})
DIST = pd.DataFrame(dist_rows)

# ---------------- per-task bin means & OLS slopes ----------------
X = np.arange(1, 6, dtype=float)
def slope(v): return float(np.polyfit(X, v, 1)[0])

bin_rows, task_rows = [], []
for t in TASKS:
    Mt = M[M.task == t]
    row = dict(task=t, type=REG[t]["type"], metric=REG[t]["metric"])
    for bcol in ["bin", "bin_rank"]:
        g = Mt.groupby(bcol)
        assert sorted(g.groups) == [1, 2, 3, 4, 5]
        for c in CONTRASTS:
            v = g[f"delta_{c}"].mean().reindex(range(1, 6)).values
            sfx = "" if bcol == "bin" else "_rankbins"
            row[f"slope_{c}{sfx}"] = slope(v)
            if bcol == "bin":
                for b in range(1, 6): row[f"{c}_Q{b}"] = v[b - 1]
                row[f"{c}_overall"] = Mt[f"delta_{c}"].mean()
        if bcol == "bin":
            for b, gb in g:
                bin_rows.append(dict(task=t, bin=b, n=len(gb), sim_min=gb.max_tanimoto.min(), sim_max=gb.max_tanimoto.max(),
                                     loss_TF=gb.loss_TF.mean(), loss_G=gb.loss_G.mean(), loss_chemeleon=gb.loss_chemeleon.mean(),
                                     loss_bbmean=gb.loss_bbmean.mean(),
                                     **{f"delta_{c}": gb[f"delta_{c}"].mean() for c in CONTRASTS}))
    task_rows.append(row)
PT = pd.DataFrame(task_rows); PTB = pd.DataFrame(bin_rows)

# ---------------- task bootstrap ----------------
rng = np.random.default_rng(RNG_SEED)
BI = rng.integers(0, len(TASKS), size=(B, len(TASKS)))  # shared draws for all estimands (paired)
def boot(v):
    v = np.asarray(v, float); bs = v[BI].mean(axis=1)
    return float(v.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))

summ = []
def add(name, contrast, v, **kw):
    e, lo, hi = boot(v); summ.append(dict(estimand=name, contrast=contrast, estimate=e, ci_low=lo, ci_high=hi,
                                          n_tasks=len(v), **kw))
for c in CONTRASTS:
    add("mean_slope", c, PT[f"slope_{c}"].values, n_pos=int((PT[f"slope_{c}"] > 0).sum()))
    add("mean_slope_rankbins(sens)", c, PT[f"slope_{c}_rankbins"].values, n_pos=int((PT[f"slope_{c}_rankbins"] > 0).sum()))
    for typ in ["classification", "regression"]:
        sub = PT[PT.type == typ]
        e = sub[f"slope_{c}"].mean()
        summ.append(dict(estimand=f"mean_slope_{typ}(descr)", contrast=c, estimate=e, ci_low=np.nan, ci_high=np.nan,
                         n_tasks=len(sub), n_pos=int((sub[f"slope_{c}"] > 0).sum())))
    add("mean_delta_overall", c, PT[f"{c}_overall"].values)
    for b in range(1, 6):
        pooled_mol = M.loc[M.bin == b, f"delta_{c}"].mean()
        add(f"mean_delta_Q{b}", c, PT[f"{c}_Q{b}"].values, molecule_pooled=pooled_mol,
            n_molecules=int((M.bin == b).sum()))
    add("Q5_minus_Q1", c, (PT[f"{c}_Q5"] - PT[f"{c}_Q1"]).values)
    v = PT[f"slope_{c}"].values  # descriptive leave-one-task-out range of the mean slope
    jk = np.array([np.delete(v, i).mean() for i in range(len(v))])
    summ.append(dict(estimand="mean_slope_leave1out_min/max(descr)", contrast=c, estimate=v.mean(), ci_low=jk.min(),
                     ci_high=jk.max(), n_tasks=20, note=f"min when dropping {TASKS[int(jk.argmin())]}; max when dropping {TASKS[int(jk.argmax())]}"))
SUM = pd.DataFrame(summ)

# ---------------- verdict (primary only) ----------------
p = SUM[(SUM.estimand == "mean_slope") & (SUM.contrast == "primary_G_stl_map")].iloc[0]
if p.ci_low > 0 and p.n_pos >= 14: verdict = "C+"
elif p.ci_high < 0: verdict = "C-"
else: verdict = "inconclusive"
VERDICT = dict(verdict=verdict, mean_slope=p.estimate, ci_low=p.ci_low, ci_high=p.ci_high, n_pos=int(p.n_pos),
               n_excluded_unparsable=n_excl)
SUM = pd.concat([SUM, pd.DataFrame([dict(estimand="VERDICT_" + verdict, contrast="primary_G_stl_map",
                                         estimate=p.estimate, ci_low=p.ci_low, ci_high=p.ci_high, n_tasks=21,
                                         n_pos=int(p.n_pos))])], ignore_index=True)

PT = PT.merge(DIST, on="task")
os.chdir(OUT)
PT.to_csv("C_per_task.csv", index=False); PTB.to_csv("C_per_task_bin.csv", index=False)
SUM.to_csv("C_summary.csv", index=False); DIST.to_csv("C_tanimoto_distribution.csv", index=False)
M.to_parquet("C_molecule_level.parquet", index=False)
print(VERDICT)

# ---------------- figure ----------------
import matplotlib as mpl; mpl.use("Agg"); import matplotlib.pyplot as plt
try:
    apply_figure_style(sizes=(8, 7, 6))  # provided by the figure-style kernel plugin when available
except NameError:
    mpl.rcParams.update({"font.size": 7, "axes.titlesize": 7, "axes.labelsize": 7, "xtick.labelsize": 6, "ytick.labelsize": 6, "legend.fontsize": 6, "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "svg.fonttype": "none"})
LAB = {"primary_G_stl_map": "vs MAP·STL (G, primary)", "secondary_stl_chemeleon": "vs CheMeleon·STL",
       "secondary_backbone_mean": "vs mean of 4 single backbones"}
COL = {"primary_G_stl_map": "#C0392B", "secondary_stl_chemeleon": "#2E86C1", "secondary_backbone_mean": "#7F8C8D"}
fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.0, 2.9), gridspec_kw=dict(width_ratios=[1.6, 1.0], wspace=0.42))
offs = dict(zip(CONTRASTS, [-0.12, 0.0, 0.12]))
for c in CONTRASTS:
    s = SUM[(SUM.contrast == c) & SUM.estimand.str.match(r"mean_delta_Q\d$")]
    x = X + offs[c]; e = s.estimate.values
    a1.errorbar(x, e, yerr=[e - s.ci_low.values, s.ci_high.values - e], color=COL[c], marker="o", ms=4,
                lw=1.6 if c.startswith("primary") else 1.0, capsize=2, label=LAB[c],
                zorder=3 if c.startswith("primary") else 2)
a1.axhline(0, color="0.6", lw=0.6, zorder=0)
a1.set_xticks(X); a1.set_xticklabels(["Q1\nmost similar", "Q2", "Q3", "Q4", "Q5\nleast similar"])
a1.set_xlabel("Quintile of max Tanimoto to train_val (per task)")
a1.set_ylabel("Mean Δ loss (reference − Tetra-Fuse)")
a1.set_title("Gain over MAP·STL is flat across similarity quintiles", loc="left")
a1.legend(frameon=False, fontsize=7, loc="upper left")
a1.text(0.99, 0.98, "Δ > 0: Tetra-Fuse better", transform=a1.transAxes, ha="right", va="top", fontsize=6, color="0.35")
a1.margins(x=0.04)
jit = np.random.default_rng(0)
for k, c in enumerate(CONTRASTS):
    v = PT[f"slope_{c}"].values; e = SUM[(SUM.contrast == c) & (SUM.estimand == "mean_slope")].iloc[0]
    a2.scatter(np.full(len(v), k) + jit.uniform(-0.15, 0.15, len(v)), v, s=9, color=COL[c], alpha=0.6, lw=0)
    a2.errorbar(k + 0.3, e.estimate, yerr=[[e.estimate - e.ci_low], [e.ci_high - e.estimate]], fmt="D", ms=4,
                color="k", capsize=2)
a2.axhline(0, color="0.6", lw=0.6, zorder=0)
a2.set_xticks(range(3)); a2.set_xticklabels(["MAP·STL\n(G)", "CheMeleon\n·STL", "backbone\nmean"])
a2.set_ylabel("Per-task slope of Δ on quintile")
npos = [int(SUM[(SUM.contrast == c) & (SUM.estimand == "mean_slope")].n_pos.iloc[0]) for c in CONTRASTS]
ymax = PT[[f"slope_{c}" for c in CONTRASTS]].values.max()
for k, n in enumerate(npos): a2.text(k, ymax * 1.12 if ymax > 0 else 0.01, f"{n}/21 > 0", ha="center", fontsize=6)
a2.set_title("Per-task slopes centre near zero", loc="left")
a2.text(0.98, 0.02, "◆ mean, 95% CI", transform=a2.transAxes, ha="right", va="bottom", fontsize=6)
a2.margins(y=0.12); a2.set_xlim(-0.5, 2.6)
for ax, l in [(a1, "a"), (a2, "b")]:
    ax.text(-0.14 if l == "a" else -0.28, 1.06, l, transform=ax.transAxes, fontweight="bold", fontsize=10, va="bottom")
fig.savefig("fig_C_distance_gain.png", dpi=300, bbox_inches="tight"); fig.savefig("fig_C_distance_gain.pdf", bbox_inches="tight")
