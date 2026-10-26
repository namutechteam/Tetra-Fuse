"""Pre-registered ANALYSIS B (protocol v1 + addendum v1.1): exact ensemble-error decomposition.

T = L_G - L_E = S + O,  S = L_G - L_b(t),  O = L_b(t) - L_E = Ambiguity - (Lbar - L_b(t)),
Krogh-Vedelsby: L_E = Lbar - Ambiguity (uniform mean, squared loss).
Loss: regression = MSE / var(y_test) (population variance, ddof=0); classification = Brier on y_pred.
Primary: Tetra-Fuse, matched-seed (per-seed decomposition, then seed mean per task), G = stl_map,
b(t) = member with lowest test loss on task t (chosen per seed in the matched-seed track).
Verdict B+: task-mean (S - O) 95% bootstrap CI lower bound > 0 (21-task bootstrap, B=2000, seed 20261002).

Usage: python analysis_B_decomposition.py <bundle_root> [outdir]
"""
import sys, os, itertools, json
import numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tf_common as tc

ROOT = sys.argv[1] if len(sys.argv) > 1 else "."
OUT = sys.argv[2] if len(sys.argv) > 2 else "."
os.makedirs(OUT, exist_ok=True)
TOL = 1e-9
B_BOOT, RNG_SEED = 2000, 20261002
TF = ["mtl_dmpnn", "mtl_molformer", "stl_chemeleon", "stl_graphmvp", "stl_map"]
G_PRIMARY, G_SECONDARY = "stl_map", "stl_chemeleon"
SEEDS = [0, 1, 2, 3, 4]

BUN = tc.load(ROOT)
TASKS, REG, MODELS = BUN["TASKS"], BUN["REG"], BUN["MODELS"]
assert len(TASKS) == 21
W, Y = tc.wide(BUN["PRED"], MODELS, "test")

# ---- per-task tensors P[model, seed, n], y[n], normaliser v ----
DATA = {}
for t in TASKS:
    Wt = W.loc[t]; y = Y.loc[t].to_numpy(float)
    assert not np.isnan(y).any() and not Wt.isna().any().any()
    P = np.stack([np.stack([Wt[(m, s)].to_numpy(float) for s in SEEDS]) for m in MODELS])  # (10,5,n)
    cls = REG[t]["type"] == "classification"
    if cls:
        assert P.min() >= 0 and P.max() <= 1, t  # y_pred already probabilities
        assert set(np.unique(y)) <= {0.0, 1.0}
        v = 1.0
    else:
        v = float(np.var(y))
    DATA[t] = dict(P=P, y=y, v=v, cls=cls, n=len(y))
MIDX = {m: i for i, m in enumerate(MODELS)}


def loss(f, y, v):
    return float(np.mean((f - y) ** 2) / v)


def decompose(F, fG, y, v, names):
    """F: (k, n) member predictions; fG: reference predictions. Returns dict of terms."""
    fE = F.mean(0)
    Li = np.array([loss(f, y, v) for f in F])
    LE, LG, Lbar = loss(fE, y, v), loss(fG, y, v), Li.mean()
    A = float(np.mean((F - fE) ** 2) / v)           # mean_i mean_n (f_i - fE)^2 / v
    ib = int(np.argmin(Li)); Lb = Li[ib]
    T, S, O = LG - LE, LG - Lb, Lb - LE
    assert abs(LE - (Lbar - A)) < TOL, ("ambiguity identity", LE, Lbar - A)
    assert abs(O - (A - (Lbar - Lb))) < TOL, ("O identity", O)
    assert abs(T - (S + O)) < TOL, ("T=S+O", T, S + O)
    R = F - y
    C = np.corrcoef(R)
    rho = float(C[np.triu_indices(len(F), 1)].mean())
    return dict(L_G=LG, L_E=LE, Lbar=Lbar, L_b=Lb, b=names[ib], A=A, D=Lbar - Lb, T=T, S=S, O=O, rho=rho, Li=Li)


NUM = ["L_G", "L_E", "Lbar", "L_b", "A", "D", "T", "S", "O", "rho"]


def task_terms(t, members, G, track):
    d = DATA[t]; P, y, v = d["P"], d["y"], d["v"]
    mi = [MIDX[m] for m in members]; gi = MIDX[G]
    if track == "matched":
        rows = [decompose(P[mi, s], P[gi, s], y, v, members) for s in range(len(SEEDS))]
        out = {k: float(np.mean([r[k] for r in rows])) for k in NUM}
        bs = [r["b"] for r in rows]
        out["b"] = ";".join(bs)
        out["b_mode"] = max(set(bs), key=bs.count)
        # sensitivity: b fixed across seeds = member with lowest seed-mean test loss
        Lmean = np.mean([r["Li"] for r in rows], axis=0); jb = int(np.argmin(Lmean))
        out["b_fixed"] = members[jb]
        out["S_fixb"] = out["L_G"] - Lmean[jb]; out["O_fixb"] = Lmean[jb] - out["L_E"]
        assert abs(out["T"] - (out["S_fixb"] + out["O_fixb"])) < TOL
        # identities also hold exactly after seed averaging (linear)
        assert abs(out["T"] - (out["S"] + out["O"])) < TOL
        assert abs(out["L_E"] - (out["Lbar"] - out["A"])) < TOL
        assert abs(out["O"] - (out["A"] - out["D"])) < TOL
    else:  # seed-averaged track: each member = mean over its 5 seed instances
        r = decompose(P[mi].mean(1), P[gi].mean(0), y, v, members)
        out = {k: r[k] for k in NUM}; out["b"] = r["b"]; out["b_mode"] = r["b"]
        out["b_fixed"] = r["b"]; out["S_fixb"] = r["S"]; out["O_fixb"] = r["O"]
    out["S_minus_O"] = out["S"] - out["O"]
    out["S_minus_O_fixb"] = out["S_fixb"] - out["O_fixb"]
    return out


def per_task_table(members, G, track):
    rows = []
    for t in TASKS:
        r = task_terms(t, members, G, track)
        r.update(task=t, type=REG[t]["type"], metric=REG[t]["metric"], n_test=DATA[t]["n"],
                 reference=G, track=track)
        rows.append(r)
    return pd.DataFrame(rows)


RNG = np.random.default_rng(RNG_SEED)
BOOT_IDX = RNG.integers(0, 21, size=(B_BOOT, 21))   # same resample indices for every estimand


def boot(x):
    x = np.asarray(x, float); assert len(x) == 21
    bm = x[BOOT_IDX].mean(1)
    return float(x.mean()), float(np.percentile(bm, 2.5)), float(np.percentile(bm, 97.5))


# ---- 1. per-task tables: primary + secondaries ----
configs = [("primary", G_PRIMARY, "matched"), ("sec_ref_chemeleon", G_SECONDARY, "matched"),
           ("sec_seedavg", G_PRIMARY, "cross"), ("sec_seedavg_ref_chemeleon", G_SECONDARY, "cross")]
PT = {}
for name, G, tr in configs:
    PT[name] = per_task_table(TF, G, tr).assign(analysis=name)
pertask = pd.concat(PT.values(), ignore_index=True)
cols = ["analysis", "track", "reference", "task", "type", "metric", "n_test", "L_G", "L_E", "Lbar", "L_b",
        "b", "b_mode", "A", "D", "T", "S", "O", "S_minus_O", "rho",
        "b_fixed", "S_fixb", "O_fixb", "S_minus_O_fixb"]
pertask = pertask[cols]
pertask.to_csv(os.path.join(OUT, "B_pertask_decomposition.csv"), index=False)

# ---- 2. summary with bootstrap CIs ----
summ = []
for name, df in PT.items():
    for est in ["S_minus_O", "S", "O", "T", "A", "D", "Lbar", "L_E", "L_G", "rho",
                "S_minus_O_fixb", "S_fixb", "O_fixb"]:
        e, lo, hi = boot(df[est])
        summ.append(dict(analysis=name, subset="Tetra-Fuse", reference=df.reference.iloc[0], track=df.track.iloc[0],
                         estimand=f"task_mean_{est}", estimate=e, ci_low=lo, ci_high=hi, n_tasks=21))
    for est in ["S_minus_O"]:
        x = df[est].to_numpy()
        summ.append(dict(analysis=name, subset="Tetra-Fuse", reference=df.reference.iloc[0], track=df.track.iloc[0],
                         estimand="n_tasks_S_gt_O", estimate=float((x > 0).sum()), ci_low=np.nan, ci_high=np.nan, n_tasks=21))
    # descriptive strata means (no CI: strata bootstraps are not pre-registered)
    for typ in ["regression", "classification"]:
        sub = df[df.type == typ]
        for est in ["S", "O", "S_minus_O", "A", "D", "rho"]:
            summ.append(dict(analysis=name, subset="Tetra-Fuse", reference=df.reference.iloc[0], track=df.track.iloc[0],
                             estimand=f"{typ}_mean_{est}", estimate=float(sub[est].mean()), ci_low=np.nan,
                             ci_high=np.nan, n_tasks=len(sub)))

# ---- 3. all 252 k=5 subsets of the 10 models (G = stl_map fixed), both tracks ----
A1023 = BUN["A1023"].set_index("combo_key")
G_norm = {"matched": A1023.loc[G_PRIMARY, "test_within"], "cross": A1023.loc[G_PRIMARY, "test_cross"]}
sub_rows, pooled = [], []
for combo in itertools.combinations(sorted(MODELS), 5):
    key = ",".join(combo)
    for tr in ["matched", "cross"]:
        df = per_task_table(list(combo), G_PRIMARY, tr)
        nrm = A1023.loc[key, "test_within" if tr == "matched" else "test_cross"]
        r = dict(combo_key=key, track=tr, contains_G=G_PRIMARY in combo, is_tetrafuse=sorted(combo) == sorted(TF),
                 dnorm=float(nrm - G_norm[tr]))
        for est in ["T", "S", "O", "A", "D", "Lbar", "S_minus_O", "rho"]:
            r[est] = float(df[est].mean())
        r["n_tasks_S_gt_O"] = int((df.S_minus_O > 0).sum())
        sub_rows.append(r)
        pooled.append(df[["task", "T", "S", "A", "Lbar"]].assign(combo_key=key, track=tr))
SUB = pd.DataFrame(sub_rows)
assert SUB.groupby("track").size().eq(252).all()
SUB.to_csv(os.path.join(OUT, "B_k5_subsets.csv"), index=False)
POOL = pd.concat(pooled, ignore_index=True)


def zs(x):
    x = np.asarray(x, float); return (x - x.mean()) / x.std()


def std_ols(yv, Xs):
    X = np.column_stack([zs(x) for x in Xs]); yz = zs(yv)
    X1 = np.column_stack([np.ones(len(yz)), X])
    beta, *_ = np.linalg.lstsq(X1, yz, rcond=None)
    r2 = 1 - np.sum((yz - X1 @ beta) ** 2) / np.sum(yz ** 2)
    return beta[1:], float(r2)


reg_rows = []
for tr in ["matched", "cross"]:
    S5 = SUB[SUB.track == tr]
    for q in ["S", "O", "S_minus_O", "A", "T"]:
        x = S5[q].to_numpy()
        reg_rows.append(dict(track=tr, kind="distribution", quantity=f"task_mean_{q}", min=x.min(),
                             q25=np.percentile(x, 25), median=np.median(x), q75=np.percentile(x, 75), max=x.max(),
                             frac_gt0=float((x > 0).mean()), n_subsets=len(x)))
    # pre-registered: standardized regression T ~ S + Ambiguity at fixed k=5 (subset level, task means)
    specs = [("T ~ S + A (subset-level task means) [pre-registered]", S5["T"], [S5["S"], S5["A"]], ["S", "A"]),
             ("dnorm ~ S + A (subset-level; official-metric link, exploratory)", S5["dnorm"], [S5["S"], S5["A"]], ["S", "A"]),
             ("dnorm ~ Lbar + A (member-mean loss controlled; exploratory)", S5["dnorm"], [S5["Lbar"], S5["A"]], ["Lbar", "A"])]
    PL = POOL[POOL.track == tr].copy()
    for c in ["T", "S", "A"]:
        PL[c + "_z"] = PL.groupby("task")[c].transform(lambda s: (s - s.mean()) / s.std(ddof=0))
    specs.append(("T ~ S + A (subset x task, within-task z; sensitivity)", PL["T_z"], [PL["S_z"], PL["A_z"]], ["S", "A"]))
    for lab, yv, Xs, nm in specs:
        beta, r2 = std_ols(yv, Xs)
        reg_rows.append(dict(track=tr, kind="std_regression", quantity=lab, **{f"beta_{n}": b for n, b in zip(nm, beta)},
                             R2=r2, n_obs=len(yv)))
REGT = pd.DataFrame(reg_rows)
REGT.to_csv(os.path.join(OUT, "B_k5_regression_and_distribution.csv"), index=False)

# Tetra-Fuse position within the 252 (rank of S_minus_O)
for tr in ["matched", "cross"]:
    S5 = SUB[SUB.track == tr]
    tfrow = S5[S5.is_tetrafuse].iloc[0]
    summ.append(dict(analysis=f"k5_{tr}", subset="Tetra-Fuse vs 252", reference=G_PRIMARY, track=tr,
                     estimand="TF_percentile_of_task_mean_S_minus_O", estimate=float((S5.S_minus_O < tfrow.S_minus_O).mean()),
                     ci_low=np.nan, ci_high=np.nan, n_tasks=21))
    for q in ["S", "O", "A"]:
        x = S5[q].to_numpy()
        summ.append(dict(analysis=f"k5_{tr}", subset="252 k=5 subsets", reference=G_PRIMARY, track=tr,
                         estimand=f"median_task_mean_{q}", estimate=float(np.median(x)), ci_low=float(x.min()),
                         ci_high=float(x.max()), n_tasks=21))
SUMM = pd.DataFrame(summ)
SUMM.to_csv(os.path.join(OUT, "B_summary_estimates.csv"), index=False)

prim = SUMM[(SUMM.analysis == "primary") & (SUMM.estimand == "task_mean_S_minus_O")].iloc[0]
verdict = "B+" if prim.ci_low > 0 else "not B+"
json.dump(dict(verdict=verdict, estimate=prim.estimate, ci_low=prim.ci_low, ci_high=prim.ci_high),
          open(os.path.join(OUT, "B_verdict.json"), "w"), indent=1)
print(verdict, prim.estimate, prim.ci_low, prim.ci_high)

# ---- 4. figure: per-task S and O (T marked), Tetra-Fuse primary ----
import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl, matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
# figure-style mechanics (role-mapped size ladder 8/7/6, open frame, 300 dpi, Type-42 fonts)
mpl.rcParams.update({"font.family": "sans-serif", "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
                     "legend.fontsize": 7, "xtick.labelsize": 6, "ytick.labelsize": 6, "axes.linewidth": 0.6,
                     "xtick.direction": "out", "ytick.direction": "out", "xtick.major.size": 3, "ytick.major.size": 3,
                     "xtick.major.width": 0.6, "ytick.major.width": 0.6, "axes.spines.top": False,
                     "axes.spines.right": False, "legend.frameon": False, "savefig.dpi": 300, "savefig.bbox": "tight",
                     "axes.titlelocation": "left", "patch.linewidth": 0.6, "pdf.fonttype": 42, "ps.fonttype": 42})
C_S, C_O, C_T = "#3B6FB6", "#E08A2E", "#1A1A1A"
SHORT = {"CYP2C9_Substrate_CarbonMangels": "CYP2C9 substrate", "CYP2D6_Substrate_CarbonMangels": "CYP2D6 substrate",
         "CYP3A4_Substrate_CarbonMangels": "CYP3A4 substrate", "Lipophilicity_AstraZeneca": "Lipophilicity",
         "Solubility_AqSolDB": "Solubility", "Clearance_Hepatocyte_AZ": "Clearance (hepatocyte)",
         "Clearance_Microsome_AZ": "Clearance (microsome)", "Pgp_Broccatelli": "Pgp", "BBB_Martins": "BBB",
         "HIA_Hou": "HIA", "CYP2C9_Veith": "CYP2C9 inhibition", "CYP2D6_Veith": "CYP2D6 inhibition",
         "CYP3A4_Veith": "CYP3A4 inhibition", "Caco2_Wang": "Caco-2", "PPBR_AZ": "PPBR", "VDss_Lombardo": "VDss",
         "Half_Life_Obach": "Half-life", "LD50_Zhu": "LD50", "AMES": "AMES", "DILI": "DILI", "hERG": "hERG"}
pt = PT["primary"]
fig = plt.figure(figsize=(7.2, 3.2))
gs = fig.add_gridspec(1, 3, width_ratios=[12, 9, 4.2], wspace=0.42)
for j, (typ, ylab) in enumerate([("classification", "Brier-score difference"),
                                 ("regression", "MSE / var(y) difference")]):
    ax = fig.add_subplot(gs[0, j])
    d = pt[pt.type == typ].sort_values("S_minus_O", ascending=False).reset_index(drop=True)
    x = np.arange(len(d))
    # diverging stack: positive parts stacked above 0, negative parts below 0; T = net (diamond)
    Sp, Sn = d.S.clip(lower=0), d.S.clip(upper=0)
    ax.bar(x, d.S, width=0.7, color=C_S, zorder=2)
    ax.bar(x, d.O, bottom=np.where(d.O >= 0, Sp, Sn), width=0.7, color=C_O, zorder=2)
    ax.scatter(x, d["T"], marker="D", s=12, color=C_T, zorder=3, linewidths=0)
    ax.axhline(0, color="0.4", lw=0.6, zorder=1)
    ax.set_xticks(x); ax.set_xticklabels([SHORT[t] for t in d.task], rotation=60, ha="right", rotation_mode="anchor")
    ax.set_ylabel(ylab); ax.margins(x=0.03, y=0.06)
    ax.set_title(f"{typ.capitalize()} tasks (n = {len(d)})")
    ax.text(-0.15 if j == 0 else -0.2, 1.06, "ab"[j], transform=ax.transAxes, fontweight="bold", fontsize=10, va="bottom")
    if j == 0:
        fig.legend(handles=[Patch(color=C_S, label="S: specialization (G − best member)"),
                           Patch(color=C_O, label="O: beyond best member (best − ensemble)"),
                           Line2D([], [], marker="D", ls="", color=C_T, ms=3.5, label="T = S + O: total gain over G")],
                   loc="lower center", bbox_to_anchor=(0.5, 0.98), ncol=3, handlelength=1.2, columnspacing=1.5)
axc = fig.add_subplot(gs[0, 2])
SP = SUMM[(SUMM.analysis == "primary")].set_index("estimand")
items = [("task_mean_S", "S", C_S), ("task_mean_O", "O", C_O), ("task_mean_S_minus_O", "S − O", "0.25")]
for i, (k, lab, c) in enumerate(items):
    r = SP.loc[k]
    axc.errorbar(i, r.estimate, yerr=[[r.estimate - r.ci_low], [r.ci_high - r.estimate]], fmt="o", color=c,
                 ms=4, capsize=2, lw=1)
axc.axhline(0, color="0.4", lw=0.6)
axc.set_xticks(range(3)); axc.set_xticklabels([i[1] for i in items]); axc.set_xlim(-0.6, 2.6)
axc.set_ylabel("Task-mean loss difference")
axc.set_title("Mean over 21 tasks")
r = SP.loc["task_mean_S_minus_O"]
axc.annotate(f"{r.estimate:.3f}\n[{r.ci_low:.3f}, {r.ci_high:.3f}]", (2, r.ci_high), xytext=(0, 4),
             textcoords="offset points", ha="center", va="bottom", fontsize=6)
axc.margins(y=0.15)
axc.text(-0.55, 1.06, "c", transform=axc.transAxes, fontweight="bold", fontsize=10, va="bottom")
fig.savefig(os.path.join(OUT, "B_decomposition_figure.png"))
fig.savefig(os.path.join(OUT, "B_decomposition_figure.pdf"))

# ---- 5. results markdown (numbers pulled from the CSV tables above) ----
def E(an, est, nd=4):
    r = SUMM[(SUMM.analysis == an) & (SUMM.estimand == est)].iloc[0]
    if np.isnan(r.ci_low):
        return f"{r.estimate:.{nd}f}"
    return f"{r.estimate:.{nd}f} [{r.ci_low:.{nd}f}, {r.ci_high:.{nd}f}]"


def NT(an, col="S_minus_O"):
    return int((PT[an][col] > 0).sum())


def RG(tr, lab_start):
    r = REGT[(REGT.track == tr) & (REGT.kind == "std_regression") & REGT.quantity.str.startswith(lab_start)].iloc[0]
    return r


def DS(tr, q):
    return REGT[(REGT.track == tr) & (REGT.kind == "distribution") & (REGT.quantity == f"task_mean_{q}")].iloc[0]


pm = PT["primary"]
best_counts = pm.b.str.split(";").explode().value_counts().to_dict()
lines = []
L = lines.append
L("# Analysis B — Ensemble Error Decomposition Results (Pre-registered Protocol v1 + Appendix v1.1)\n")
L("## Pre-registered Verdict\n")
L(f"- **Verdict: {verdict}** (specialization dominant).")
L(f"- Basis: In Tetra-Fuse, matched-seed, G = stl_map(MAP·STL), task-average (S − O) = {E('primary','task_mean_S_minus_O')} "
  "(21-task bootstrap, B = 2,000, seed 20261002, 95% percentile CI). Pre-registration criterion is CI lower bound > 0.")
L("- Per protocol, B is for explanatory purposes and is not part of the journal verdict rule.")
L("- **Note: G = stl_map is itself a Tetra-Fuse member.** Therefore, for every task L_b(t) ≤ L_G, and S ≥ 0 is guaranteed by construction. "
  "S measures 'the difference when a member other than G performs better on that task.'\n")
L("## Definitions and Numerical Verification\n")
L("- Loss: regression = MSE / var(y_test) (population variance, ddof = 0), classification = Brier score on y_pred (already [0,1] probability; range assert passed).")
L("- T = L_G − L_E, S = L_G − L_b(t), O = L_b(t) − L_E = A − (L̄ − L_b(t)), A = mean of per-member (f_i − f_E)² (Krogh–Vedelsby: L_E = L̄ − A).")
L("- matched-seed: decompose using the 5 members (same seed) per seed, then take the seed average per task. b(t) is the member with minimum test loss per seed (descriptive oracle).")
L("- For all tasks, seeds, and subsets (Tetra-Fuse's 4 configurations + k=5's 252 × 2 tracks), L_E = L̄ − A, O = A − (L̄ − L_b), T = S + O were asserted with tolerance 1e-9, and all passed.\n")
L("## Primary Results (Tetra-Fuse, matched-seed, G = stl_map)\n")
L("| Estimator (task average) | Estimate [95% CI] |\n|---|---|")
for est, lab in [("S_minus_O", "S − O (verdict target)"), ("S", "S (specialization)"), ("O", "O (best member excess)"),
                 ("T", "T (total gain relative to G)"), ("A", "Ambiguity A"), ("D", "L̄ − L_b (member average − best member)"),
                 ("rho", "mean pairwise residual correlation among members (for context)")]:
    L(f"| {lab} | {E('primary', 'task_mean_' + est)} |")
L("")
L(f"- Tasks where S > O: {NT('primary')}/21. Regression (9 tasks) average S = {E('primary','regression_mean_S')}, O = {E('primary','regression_mean_O')}; "
  f"classification (12 tasks) average S = {E('primary','classification_mean_S')}, O = {E('primary','classification_mean_O')}.")
L(f"- Ambiguity (task average {E('primary','task_mean_A')}) is large but is nearly offset by the gap between the member average and the best member, L̄ − L_b({E('primary','task_mean_D')}), "
  "so O's CI includes 0. That is, the extent to which Tetra-Fuse beats G mostly comes from 'a different member being best on each task,' and "
  "the pure diversity gain beyond the per-task best member is, on average, indistinguishable from 0.")
L(f"- This is consistent with the high residual correlation (task average {E('primary','task_mean_rho')}), which makes Ambiguity small relative to member loss.")
L(f"- b(t) frequency per seed (21 tasks × 5 seeds = 105): " + ", ".join(f"{k} {v}" for k, v in sorted(best_counts.items(), key=lambda kv: -kv[1])) + ".")
L("- Per-task values: Half_Life_Obach (S − O = %.4f), Caco2_Wang (%.4f), VDss_Lombardo (%.4f) are the largest. Since regression tasks use raw-scale MSE divided by variance, heavy-tail tasks (Half-life, VDss, Clearance) have a large influence."
  % tuple(pm.set_index("task").loc[t, "S_minus_O"] for t in ["Half_Life_Obach", "Caco2_Wang", "VDss_Lombardo"]))
L("")
L("## Secondary / Sensitivity Results\n")
L("| Configuration | S − O | S | O | T | Number of tasks with S > O |\n|---|---|---|---|---|---|")
for an, lab in [("primary", "Primary: matched-seed, G = stl_map"), ("sec_ref_chemeleon", "matched-seed, G = stl_chemeleon (Appendix v1.1)"),
                ("sec_seedavg", "seed-averaged, G = stl_map"), ("sec_seedavg_ref_chemeleon", "seed-averaged, G = stl_chemeleon")]:
    L(f"| {lab} | {E(an,'task_mean_S_minus_O')} | {E(an,'task_mean_S')} | {E(an,'task_mean_O')} | {E(an,'task_mean_T')} | {NT(an)}/21 |")
L("")
L(f"- **Fixed-b(t) sensitivity (outside pre-registration, for interpretive aid)**: In matched-seed, choosing b(t) per seed means the test-oracle selection occurs 5 times, biasing S upward. "
  f"If b(t) is fixed to the single member with the minimum seed-averaged loss, S − O = {E('primary','task_mean_S_minus_O_fixb')}, S = {E('primary','task_mean_S_fixb')}, "
  f"O = {E('primary','task_mean_O_fixb')}, tasks with S > O: {NT('primary','S_minus_O_fixb')}/21. The direction is the same but the magnitude shrinks and the CI lower bound approaches 0. "
  f"For G = stl_chemeleon, the same fixed-b value is {E('sec_ref_chemeleon','task_mean_S_minus_O_fixb')}, whose CI includes 0.")
L(f"- In the seed-averaged track, since G itself is a 5-seed average, T shrinks to {E('sec_seedavg','task_mean_T')}, and O becomes negative at {E('sec_seedavg','task_mean_O')} (the best seed-averaged member outperforms the ensemble).")
L("")
L("## k = 5 Subsets, 252 (10 models, G = stl_map fixed)\n")
L("| Track | Task-average S median [min, max] | Task-average O median [min, max] | Fraction O > 0 | Fraction S − O > 0 | Ambiguity median [min, max] | Tetra-Fuse's S − O percentile |\n|---|---|---|---|---|---|---|")
for tr, an in [("matched", "k5_matched"), ("cross", "k5_cross")]:
    L(f"| {tr} | {E(an,'median_task_mean_S')} | {E(an,'median_task_mean_O')} | {DS(tr,'O').frac_gt0:.3f} | {DS(tr,'S_minus_O').frac_gt0:.3f} | "
      f"{E(an,'median_task_mean_A')} | {E(an,'TF_percentile_of_task_mean_S_minus_O',3)} |")
L("")
L("(In subsets that do not include G, S can be negative. The brackets in the table above are not CIs but the min–max across the 252 subsets.)\n")
L("k = 5 fixed standardized regression (coefficients only, no p-values; k is controlled by design):\n")
L("| Track | Model | β_S | β_A | β_L̄ | R² | n |\n|---|---|---|---|---|---|---|")
for tr in ["matched", "cross"]:
    for lab in ["T ~ S + A (subset-level", "T ~ S + A (subset x task", "dnorm ~ S + A", "dnorm ~ Lbar + A"]:
        r = RG(tr, lab)
        f = lambda v: "—" if pd.isna(v) else f"{v:.3f}"
        L(f"| {tr} | {r.quantity} | {f(r.get('beta_S'))} | {f(r.get('beta_A'))} | {f(r.get('beta_Lbar'))} | {r.R2:.3f} | {int(r.n_obs)} |")
L("")
L("- In the pre-registered regression (T ~ S + A, at the subset level), the standardized coefficient of S is larger than that of Ambiguity (both tracks).")
L("- dnorm = the subset's test norm_score (test_within/test_cross in all1023.csv) − stl_map's norm_score. The link to the official metric space is exploratory and not a pre-registered estimator. "
  "When included alongside S, the Ambiguity coefficient is negative (subsets with high diversity tend to have weaker member averages), and once member-average loss L̄ is controlled for, Ambiguity's additional explanatory power is small.")
L("")
L("## Limitations\n")
L("- The decomposition space (MSE/var(y), Brier) differs from the official metrics (MAE, AUROC, AUPRC, Spearman). Brier/MSE gains do not correspond 1:1 to rank-based metric gains.")
L("- Since b(t) is an oracle chosen by test loss, S leans toward an optimistic upper bound and O toward a pessimistic lower bound (more so when chosen per seed in matched-seed; see the fixed-b sensitivity above).")
L("- Since G = stl_map is a Tetra-Fuse member, S ≥ 0 is guaranteed by construction in the primary analysis.")
L("- Since regression loss is on the raw scale, heavy-tail tasks (Half-life, VDss, Clearance) carry a large weight in the task average.")
L("- Panel c and the task average are values obtained by averaging classification (Brier) and regression (MSE/var y) units together.")
open(os.path.join(OUT, "B_results.md"), "w").write("\n".join(lines) + "\n")
