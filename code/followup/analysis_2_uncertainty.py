"""Pre-registered ANALYSIS 3 (protocol v2, 2026-10-03):
uncertainty quality of heterogeneous (HET) vs homogeneous (HOM) 5-member ensembles.

Usage:  python analysis3_uncertainty.py <bundle_root> <out_dir>
  bundle_root contains data/recompute_data/ and admet_group/; tf_common.py must sit next to this script.
Outputs (out_dir): A3_per_task.csv, A3_summary.csv, A3_similarity_quintile.csv, A3_molecule_level.parquet,
                   fig_A3_uncertainty.png/.pdf

Definitions (protocol v2, Analysis 3):
  HET_s = {(m, s) : m in Tetra-Fuse 5 types}, s = 0..4          (5 ensembles)
  HOM_m = {(m, s) : s = 0..4}, m in Tetra-Fuse 5 types           (5 ensembles)
  u_i   = SD of the 5 member predictions (regression raw scale; classification probabilities), ddof=1
  e_i   = |y - yhat_ens| / SD(y_test) (regression; SD ddof=1 over the task's test rows),  (y - p_ens)^2 (classification)
  rho   = per-task Spearman(u, e) (primary);  AURC relative reduction vs random ordering (secondary)
  Delta_rho_t = mean_s rho_HET_s,t - mean_m rho_HOM_m,t
  Verdict: U+ if CI_low(task-mean Delta_rho) > 0 and #(Delta_rho_t > 0) >= 14/21; U- if CI_high < 0; else inconclusive.
Bootstrap: tasks resampled, B = 2000, rng seed 20261002, 95 % percentile CI (shared draws for all 21-task estimands).
"""
import os, sys, numpy as np, pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tf_common as tc

ROOT = sys.argv[1] if len(sys.argv) > 1 else "bundle"
ADMET_DIR = os.environ.get("TDC_ADMET_GROUP", os.path.join(ROOT, "data", "tdc_admet_group"))  # TDC admet_group cache (train_val.csv per task)
OUT = sys.argv[2] if len(sys.argv) > 2 else "."
os.makedirs(OUT, exist_ok=True)
B, RNG_SEED = 2000, 20261002
TYPES = ["mtl_dmpnn", "mtl_molformer", "stl_chemeleon", "stl_graphmvp", "stl_map"]
SEEDS = [0, 1, 2, 3, 4]

L = tc.load(ROOT); TASKS, REG, PRED = L["TASKS"], L["REG"], L["PRED"]
assert len(TASKS) == 21 and "Bioavailability_Ma" not in TASKS

W, Y = tc.wide(PRED, TYPES, "test")
keep = W.index.get_level_values("task").isin(TASKS)
W, Y = W[keep], Y[keep]
assert W.shape[1] == 25 and not W.isna().any().any(), "missing (type, seed) predictions"
assert set(W.index.get_level_values("task")) == set(TASKS)
assert not Y.isna().any()

# ---------------- ensembles ----------------
ARMS = {f"HET_s{s}": [(m, s) for m in TYPES] for s in SEEDS}
ARMS.update({f"HOM_{m}": [(m, s) for s in SEEDS] for m in TYPES})
HET = [f"HET_s{s}" for s in SEEDS]; HOM = [f"HOM_{m}" for m in TYPES]


def rc_rel_reduction(u, e):
    """AURC of the risk-coverage curve (sort by u ascending, risk = mean e of retained set, coverage k/n,
    k = 1..n), expressed as relative reduction vs random ordering (expected AURC = mean e).
    Ties in u are resolved by their expectation under random tie-breaking (e replaced by tie-group mean)."""
    u = np.asarray(u, float); e = np.asarray(e, float); n = len(e)
    o = np.argsort(u, kind="mergesort"); us, es = u[o], e[o]
    _, inv = np.unique(us, return_inverse=True)
    es = (np.bincount(inv, es) / np.bincount(inv))[inv]
    aurc = np.mean(np.cumsum(es) / np.arange(1, n + 1))
    rnd = e.mean()
    return float(aurc), float(rnd), float((rnd - aurc) / rnd)


task_of = W.index.get_level_values("task")
rows, mol = [], {}
for t in TASKS:
    Wt = W.loc[t]; y = Y.loc[t].values.astype(float)
    is_cls = REG[t]["type"] == "classification"
    sd_y = float(np.std(y, ddof=1))
    row = dict(task=t, type=REG[t]["type"], metric=REG[t]["metric"], n_test=len(y), sd_y_test=sd_y)
    for a, mem in ARMS.items():
        P = np.column_stack([Wt[c].values.astype(float) for c in mem])
        if is_cls: assert P.min() >= 0 and P.max() <= 1, (t, a)
        p_ens = P.mean(axis=1)
        u = P.std(axis=1, ddof=1)
        e = (y - p_ens) ** 2 if is_cls else np.abs(y - p_ens) / sd_y
        rho = spearmanr(u, e)[0]
        aurc, aurc_rnd, rel = rc_rel_reduction(u, e)
        row[f"rho_{a}"] = float(rho); row[f"aurcrel_{a}"] = rel
        row[f"aurc_{a}"] = aurc; row[f"aurcrandom_{a}"] = aurc_rnd
        row[f"meanu_{a}"] = float(u.mean()); row[f"meane_{a}"] = float(e.mean())
        row[f"n_u_ties_{a}"] = int(len(u) - len(np.unique(u)))
        mol[(t, a)] = (u, e)
    for k in ["rho", "aurcrel", "meanu", "meane"]:
        row[f"{k}_HET_mean"] = np.mean([row[f"{k}_{a}"] for a in HET])
        row[f"{k}_HOM_mean"] = np.mean([row[f"{k}_{a}"] for a in HOM])
    row["rho_HOM_best"] = max(row[f"rho_{a}"] for a in HOM)
    row["rho_HOM_best_type"] = max(HOM, key=lambda a: row[f"rho_{a}"])[4:]
    row["aurcrel_HOM_best"] = max(row[f"aurcrel_{a}"] for a in HOM)
    row["delta_rho"] = row["rho_HET_mean"] - row["rho_HOM_mean"]                          # PRIMARY
    row["delta_rho_vs_chemeleon"] = row["rho_HET_mean"] - row["rho_HOM_stl_chemeleon"]
    row["delta_rho_vs_bestHOM"] = row["rho_HET_mean"] - row["rho_HOM_best"]
    row["delta_aurcrel"] = row["aurcrel_HET_mean"] - row["aurcrel_HOM_mean"]
    row["delta_aurcrel_vs_chemeleon"] = row["aurcrel_HET_mean"] - row["aurcrel_HOM_stl_chemeleon"]
    row["delta_aurcrel_vs_bestHOM"] = row["aurcrel_HET_mean"] - row["aurcrel_HOM_best"]
    row["u_ratio_HET_over_HOM"] = row["meanu_HET_mean"] / row["meanu_HOM_mean"]
    rows.append(row)
PT = pd.DataFrame(rows)

# ---------------- bootstrap ----------------
rng = np.random.default_rng(RNG_SEED)
BI = rng.integers(0, len(TASKS), size=(B, len(TASKS)))   # shared (paired) draws for all 21-task estimands


def boot(v, idx=None):
    v = np.asarray(v, float)
    if idx is None:
        bi = BI
    else:  # subset (classification / regression): resample tasks within the subset
        bi = np.random.default_rng(RNG_SEED).integers(0, len(v), size=(B, len(v)))
    bs = v[bi].mean(axis=1)
    return float(v.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


summ = []
def add(estimand, role, v, subset="all21", **kw):
    v = np.asarray(v, float)
    e, lo, hi = boot(v, None if subset == "all21" else "sub")
    summ.append(dict(estimand=estimand, role=role, subset=subset, estimate=e, ci_low=lo, ci_high=hi,
                     n_tasks=len(v), n_pos=int((v > 0).sum()), **kw))

add("delta_rho (HET mean_s - HOM mean_m)", "PRIMARY", PT.delta_rho)
add("rho_HET_mean", "descriptive", PT.rho_HET_mean)
add("rho_HOM_mean", "descriptive", PT.rho_HOM_mean)
for a in HET + HOM: add(f"rho_{a}", "descriptive", PT[f"rho_{a}"])
add("delta_rho_vs_HOM_stl_chemeleon", "secondary", PT.delta_rho_vs_chemeleon)
add("delta_rho_vs_taskwise_best_HOM (optimistic)", "secondary", PT.delta_rho_vs_bestHOM)
add("delta_aurcrel (HET - HOM mean)", "secondary", PT.delta_aurcrel)
add("aurcrel_HET_mean", "secondary", PT.aurcrel_HET_mean)
add("aurcrel_HOM_mean", "secondary", PT.aurcrel_HOM_mean)
add("delta_aurcrel_vs_HOM_stl_chemeleon", "secondary", PT.delta_aurcrel_vs_chemeleon)
add("delta_aurcrel_vs_taskwise_best_HOM", "secondary", PT.delta_aurcrel_vs_bestHOM)
for typ in ["classification", "regression"]:
    sub = PT[PT.type == typ]
    for col, name in [("delta_rho", "delta_rho"), ("rho_HET_mean", "rho_HET_mean"), ("rho_HOM_mean", "rho_HOM_mean"),
                      ("delta_rho_vs_chemeleon", "delta_rho_vs_HOM_stl_chemeleon"),
                      ("delta_rho_vs_bestHOM", "delta_rho_vs_taskwise_best_HOM"), ("delta_aurcrel", "delta_aurcrel")]:
        add(name, "secondary (split; CI resamples tasks within subset)", sub[col], subset=typ)
# leave-one-task-out range of primary (descriptive)
v = PT.delta_rho.values; jk = np.array([np.delete(v, i).mean() for i in range(len(v))])
summ.append(dict(estimand="delta_rho leave-one-task-out min/max (descr)", role="descriptive", subset="all21",
                 estimate=v.mean(), ci_low=jk.min(), ci_high=jk.max(), n_tasks=20, n_pos=np.nan,
                 note=f"min when dropping {TASKS[int(jk.argmin())]}; max when dropping {TASKS[int(jk.argmax())]}"))

# ---------------- verdict ----------------
p = [r for r in summ if r["role"] == "PRIMARY"][0]
if p["ci_low"] > 0 and p["n_pos"] >= 14: verdict = "U+"
elif p["ci_high"] < 0: verdict = "U-"
else: verdict = "inconclusive"
summ.append(dict(estimand=f"VERDICT_{verdict}", role="PRIMARY", subset="all21", estimate=p["estimate"],
                 ci_low=p["ci_low"], ci_high=p["ci_high"], n_tasks=21, n_pos=p["n_pos"],
                 note="U+: CI_low>0 and n_pos>=14; U-: CI_high<0; else inconclusive"))

# ---------------- similarity quintiles (v1 analysis C definition) ----------------
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator
RDLogger.DisableLog("rdApp.*")
gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
def fp(smi):
    m = Chem.MolFromSmiles(smi) if isinstance(smi, str) else None
    return None if m is None else gen.GetFingerprint(m)

ML = []
for t in TASKS:
    d = os.path.join(ADMET_DIR, t.lower())
    tv = pd.read_csv(os.path.join(d, "train_val.csv"))
    tv_fps = [f for f in (fp(s) for s in tv["Drug"]) if f is not None]
    smis = W.loc[t].index.get_level_values("smiles")
    cache = {}
    for s in pd.unique(smis):
        f = fp(s)
        cache[s] = np.nan if f is None else float(max(DataStructs.BulkTanimotoSimilarity(f, tv_fps)))
    df = pd.DataFrame({"task": t, "smiles": smis, "rep": W.loc[t].index.get_level_values("rep"),
                       "y_true": Y.loc[t].values, "max_tanimoto": [cache[s] for s in smis]})
    for a in ARMS: df[f"u_{a}"], df[f"e_{a}"] = mol[(t, a)]
    df["u_HET_mean"] = df[[f"u_{a}" for a in HET]].mean(axis=1)
    df["u_HOM_mean"] = df[[f"u_{a}" for a in HOM]].mean(axis=1)
    ML.append(df)
ML = pd.concat(ML, ignore_index=True)
n_unparsable = int(ML.max_tanimoto.isna().sum())
MS = ML[ML.max_tanimoto.notna()].copy()
MS["bin"] = MS.groupby("task")["max_tanimoto"].transform(
    lambda s: 5 - pd.qcut(s, 5, labels=False, duplicates="raise")).astype(int)   # Q1 = most similar

UCOLS = ["u_HET_mean", "u_HOM_mean", "u_HOM_stl_chemeleon"]
qrows = []
X = np.arange(1, 6, dtype=float)
for t in TASKS:
    Mt = MS[MS.task == t]
    g = Mt.groupby("bin")
    assert sorted(g.groups) == [1, 2, 3, 4, 5]
    r = dict(task=t, type=REG[t]["type"], n_analysed=len(Mt))
    for c in UCOLS:
        raw = g[c].mean().reindex(range(1, 6)).values
        rel = raw / Mt[c].mean()     # relative to the task's overall mean u (scale-free across tasks)
        for b in range(1, 6):
            r[f"{c}_Q{b}_raw"] = raw[b - 1]; r[f"{c}_Q{b}_rel"] = rel[b - 1]
        r[f"{c}_slope_rel"] = float(np.polyfit(X, rel, 1)[0])
        r[f"{c}_Q5overQ1"] = rel[4] / rel[0]
    for b in range(1, 6):
        gb = Mt[Mt.bin == b]
        r[f"n_Q{b}"] = len(gb); r[f"sim_min_Q{b}"] = gb.max_tanimoto.min(); r[f"sim_max_Q{b}"] = gb.max_tanimoto.max()
    qrows.append(r)
QT = pd.DataFrame(qrows)
for c in UCOLS:
    for b in range(1, 6):
        add(f"rel_mean_u_Q{b} [{c}]", "secondary (similarity quintile)", QT[f"{c}_Q{b}_rel"])
    add(f"slope_rel_u_on_quintile [{c}]", "secondary (similarity quintile)", QT[f"{c}_slope_rel"])
    add(f"Q5_minus_Q1_rel_u [{c}]", "secondary (similarity quintile)", QT[f"{c}_Q5_rel"] - QT[f"{c}_Q1_rel"])
add("HET/HOM mean-u ratio minus 1", "descriptive", PT.u_ratio_HET_over_HOM - 1)

SUM = pd.DataFrame(summ)
PT = PT.merge(QT[["task", "n_analysed"] + [c for c in QT.columns if c.endswith(("slope_rel", "Q5overQ1"))]],
              on="task")
PT.to_csv(os.path.join(OUT, "A3_per_task.csv"), index=False)
SUM.to_csv(os.path.join(OUT, "A3_summary.csv"), index=False)
QT.to_csv(os.path.join(OUT, "A3_similarity_quintile.csv"), index=False)
ML.to_parquet(os.path.join(OUT, "A3_molecule_level.parquet"), index=False)
print(dict(verdict=verdict, delta_rho=p["estimate"], ci=(p["ci_low"], p["ci_high"]), n_pos=p["n_pos"],
           n_unparsable_excluded_from_quintiles=n_unparsable, n_rows=len(ML)))

# ---------------- figure ----------------
import matplotlib as mpl; mpl.use("Agg"); import matplotlib.pyplot as plt
try:
    apply_figure_style(sizes=(8, 7, 6))   # figure-style plugin, when run inside a session kernel
except NameError:                          # standalone: the same rcParams
    mpl.rcParams.update({"font.family": "sans-serif", "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
                         "legend.fontsize": 7, "xtick.labelsize": 6, "ytick.labelsize": 6, "axes.linewidth": 0.6,
                         "xtick.major.size": 3, "ytick.major.size": 3, "xtick.major.width": 0.6,
                         "ytick.major.width": 0.6, "axes.spines.top": False, "axes.spines.right": False,
                         "legend.frameon": False, "savefig.dpi": 300, "savefig.bbox": "tight",
                         "axes.titlelocation": "left", "lines.linewidth": 1.2, "pdf.fonttype": 42, "ps.fonttype": 42})
C_HET, C_HOM, GREY = "#C0392B", "#5D6D7E", "#888888"
SHORT = {"HIA_Hou": "HIA", "Pgp_Broccatelli": "Pgp", "BBB_Martins": "BBB", "CYP2C9_Veith": "CYP2C9 inh.",
         "CYP2D6_Veith": "CYP2D6 inh.", "CYP3A4_Veith": "CYP3A4 inh.", "CYP2C9_Substrate_CarbonMangels": "CYP2C9 sub.",
         "CYP2D6_Substrate_CarbonMangels": "CYP2D6 sub.", "CYP3A4_Substrate_CarbonMangels": "CYP3A4 sub.",
         "hERG": "hERG", "AMES": "AMES", "DILI": "DILI", "Caco2_Wang": "Caco-2", "Lipophilicity_AstraZeneca": "Lipophilicity",
         "Solubility_AqSolDB": "Solubility", "PPBR_AZ": "PPBR", "VDss_Lombardo": "VDss", "Half_Life_Obach": "Half-life",
         "Clearance_Hepatocyte_AZ": "CL hepatocyte", "Clearance_Microsome_AZ": "CL microsome", "LD50_Zhu": "LD50"}

fig = plt.figure(figsize=(7.2, 5.0))
gs = fig.add_gridspec(2, 2, width_ratios=[1.05, 1.0], height_ratios=[1.0, 1.0], wspace=0.62, hspace=0.66)
a1 = fig.add_subplot(gs[:, 0]); a2 = fig.add_subplot(gs[0, 1]); a3 = fig.add_subplot(gs[1, 1])

# (a) per-task paired dots, grouped by task type, sorted by delta within type
PTs = PT.copy(); PTs["grp"] = (PTs.type == "regression").astype(int)
PTs = PTs.sort_values(["grp", "delta_rho"], ascending=[False, True]).reset_index(drop=True)
ypos = np.arange(len(PTs), dtype=float); ypos[PTs.grp.values == 0] += 1.0   # gap between regression and classification
for yv, (_, r) in zip(ypos, PTs.iterrows()):
    a1.plot([r.rho_HOM_mean, r.rho_HET_mean], [yv, yv], color="0.75", lw=0.9, zorder=1)
a1.scatter(PTs.rho_HOM_mean, ypos, s=16, color=C_HOM, facecolor="white", lw=0.9, zorder=2,
           label="Same-encoder (5 runs of one type; mean of 5 types)")
a1.scatter(PTs.rho_HET_mean, ypos, s=16, color=C_HET, zorder=3, label="Heterogeneous (5 Tetra-Fuse types; mean of 5 seeds)")
a1.set_yticks(ypos); a1.set_yticklabels([SHORT[t] for t in PTs.task])
a1.axvline(0, color="0.6", lw=0.6, zorder=0)
a1.set_xlabel("Spearman ρ(uncertainty u, error e)")
nreg = int((PTs.grp == 1).sum())
a1.text(1.0, ypos[nreg - 1] + 0.55, "regression", transform=a1.get_yaxis_transform(), ha="right", va="center",
        fontsize=6, color="0.35", style="italic")
a1.text(1.0, ypos[-1] + 0.9, "classification", transform=a1.get_yaxis_transform(), ha="right", va="bottom",
        fontsize=6, color="0.35", style="italic")
a1.set_ylim(-0.8, ypos[-1] + 1.6)
a1.legend(loc="upper center", bbox_to_anchor=(0.42, -0.085), fontsize=6, handletextpad=0.3, borderaxespad=0)
npos = int((PT.delta_rho > 0).sum())
a1.set_title(f"Heterogeneous > same-encoder on {npos}/21 tasks")
a1.margins(x=0.04)

# (b) task-mean contrasts with 95 % CI
def srow(est, subset="all21"):
    return SUM[(SUM.estimand == est) & (SUM.subset == subset)].iloc[0]
items = [("All 21 tasks\n(primary)", srow("delta_rho (HET mean_s - HOM mean_m)"), True),
         ("Classification (12)", srow("delta_rho", "classification"), False),
         ("Regression (9)", srow("delta_rho", "regression"), False),
         ("vs CheMeleon·STL only", srow("delta_rho_vs_HOM_stl_chemeleon"), False),
         ("vs task-wise best type", srow("delta_rho_vs_taskwise_best_HOM (optimistic)"), False)]
yy = np.arange(len(items))[::-1].astype(float); yy[1:] -= 0.35
for (lab, r, prim), yv in zip(items, yy):
    a2.errorbar(r.estimate, yv, xerr=[[r.estimate - r.ci_low], [r.ci_high - r.estimate]], fmt="o" if prim else "o",
                ms=5 if prim else 3.5, color=C_HET if prim else "0.25", mfc=C_HET if prim else "white",
                capsize=2, lw=1.4 if prim else 0.9, zorder=3)
    a2.text(r.ci_high + 0.012, yv, f"{int(r.n_pos)}/{int(r.n_tasks)}", va="center", fontsize=6, color="0.35")
a2.axvline(0, color="0.6", lw=0.6, zorder=0)
a2.set_yticks(yy); a2.set_yticklabels([i[0] for i in items])
a2.set_xlabel("Δρ, heterogeneous − same-encoder (task mean, 95% CI)")
pr = items[0][1]
a2.set_title(f"Primary Δρ = {pr.estimate:.3f} [{pr.ci_low:.3f}, {pr.ci_high:.3f}]")
a2.margins(x=0.08, y=0.12)
a2.text(0.99, 0.02, "heterogeneous better →", transform=a2.transAxes, ha="right", va="bottom", fontsize=6, color="0.35")

# (c) relative mean u by similarity quintile
for c, col, lab, off in [("u_HET_mean", C_HET, "Heterogeneous", -0.06), ("u_HOM_mean", C_HOM, "Same-encoder", 0.06)]:
    rr = [srow(f"rel_mean_u_Q{b} [{c}]") for b in range(1, 6)]
    e = np.array([r.estimate for r in rr]); lo = np.array([r.ci_low for r in rr]); hi = np.array([r.ci_high for r in rr])
    a3.errorbar(X + off, e, yerr=[e - lo, hi - e], color=col, marker="o", ms=3.5, capsize=2, lw=1.1,
                mfc=col if c == "u_HET_mean" else "white")
    a3.text(5.25, e[-1], lab, color=col, fontsize=6, va="center")
a3.axhline(1, color="0.6", lw=0.6, zorder=0)
a3.set_xticks(X); a3.set_xticklabels(["Q1\nmost\nsimilar", "Q2", "Q3", "Q4", "Q5\nleast\nsimilar"])
a3.set_xlim(0.7, 6.2)
a3.set_xlabel("Quintile of max Tanimoto to train_val (per task)", labelpad=2)
a3.set_ylabel("Mean u / task mean u")
a3.set_title("Disagreement rises in the least-similar quintile")
fig.canvas.draw(); _r = fig.canvas.get_renderer()
for ax, l in [(a1, "a"), (a2, "b"), (a3, "c")]:
    tb = ax.get_tightbbox(_r).transformed(fig.transFigure.inverted())
    fig.text(tb.x0 - 0.005, tb.y1 + 0.005, l, fontweight="bold", fontsize=10, ha="left", va="bottom")
fig.savefig(os.path.join(OUT, "fig_A3_uncertainty.png"), dpi=300, bbox_inches="tight")
fig.savefig(os.path.join(OUT, "fig_A3_uncertainty.pdf"), bbox_inches="tight")
