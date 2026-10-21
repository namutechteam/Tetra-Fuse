"""Plotting helpers for the controlled-comparison figures (Fig. 2 and Fig. 3C of the manuscript)."""
from __future__ import annotations

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

FOC = "#C2410C"      # heterogeneous ensembles
TFC = "#7C2D12"      # Tetra-Fuse
GREY = "#6B7280"
RC = {"font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "xtick.labelsize": 6, "ytick.labelsize": 6,
      "legend.fontsize": 6, "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 300,
      "savefig.bbox": "tight", "axes.titlelocation": "left", "legend.frameon": False}


def _letter(ax, s):
    ax.text(-0.12, 1.06, s, transform=ax.transAxes, fontsize=10, fontweight="bold", va="bottom", ha="right")


def figure2(C, P, G):
    """Fig. 2: different encoders vs repeated runs of one encoder at equal member count."""
    mpl.rcParams.update(RC)
    ms = range(1, 6)
    series = [("HomMean", "Same encoder: mean of 4", "#9CA3AF", "-"),
              ("HomGlobalPool", "Same encoder: CheMeleon·STL", "#2563EB", "-"),
              ("HomTaskValPool", "Same encoder: per-task val-best", "#0F766E", "-"),
              ("HomTaskTestPool", "Same encoder: per-task test-best (oracle)", "#9CA3AF", ":")]
    het = [P[f"Het_{m}_{min(m, 4)}"].mean() for m in ms]
    fig = plt.figure(figsize=(7.2, 5.6))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.25, 1], height_ratios=[1, 0.9], wspace=0.35, hspace=0.55)

    ax = fig.add_subplot(gs[0, 0])
    for key, lab, col, ls in series:
        ax.plot(ms, [P[f"{key}_{m}"].mean() for m in ms], ls=ls, color=col, marker="o", ms=3, lw=1.2)
    ax.plot(ms, het, color=FOC, marker="o", ms=4, lw=2)
    ax.set_xticks(list(ms)); ax.set_xlabel("Ensemble members (m)")
    ax.set_ylabel("TDC norm_score\n(mean of 21 tasks)"); ax.set_xlim(0.7, 5.3); ax.set_ylim(0.5, 0.92)
    ax.set_title("Different encoders beat validation-selected runs")
    h = [Line2D([], [], color=FOC, lw=2, marker="o", ms=3, label="Different encoders")] + \
        [Line2D([], [], color=c, ls=s, lw=1.2, marker="o", ms=3, label=l) for _, l, c, s in series]
    ax.legend(handles=h, loc="lower right", handlelength=1.6, borderaxespad=0.2)
    _letter(ax, "A")

    ax2 = fig.add_subplot(gs[0, 1])
    M = G.pivot(index="m", columns="d", values="S").reindex(index=[5, 4, 3, 2, 1])
    ax2.imshow(M.values, cmap="viridis", vmin=0.55, vmax=0.85, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M.values[i, j]
            ax2.text(j, i, "—" if np.isnan(v) else f"{v:.2f}", ha="center", va="center", fontsize=6,
                     color=GREY if np.isnan(v) else ("white" if v < 0.75 else "black"))
    ax2.set_xticks(range(4)); ax2.set_xticklabels([1, 2, 3, 4]); ax2.set_yticks(range(5)); ax2.set_yticklabels([5, 4, 3, 2, 1])
    ax2.set_xlabel("Distinct encoders (d)"); ax2.set_ylabel("Members (m)"); ax2.set_title("Score rises mainly along d")
    for s in ax2.spines.values():
        s.set_visible(False)
    _letter(ax2, "B")

    ax3 = fig.add_subplot(gs[1, :])
    rows = [("Same encoder: mean of 4 encoders' runs", "HomMean_4", "HomMean_5"),
            ("Same encoder: val-best (CheMeleon·STL)", "HomGlobalPool_4", "HomGlobalPool_5"),
            ("Same encoder: per-task val-best", "HomTaskValPool_4", "HomTaskValPool_5"),
            ("Same model, all 10: val-best (MAP·STL, a fusion model)", "HomGlobalAll_4", "HomGlobalAll_5"),
            ("Same model, all 10: per-task val-best", "HomTaskValAll_4", "HomTaskValAll_5"),
            ("Same encoder: per-task test-best (oracle)", "HomTaskTestPool_4", "HomTaskTestAll_5")]
    for i, (lab, b4, b5) in enumerate(rows):
        y0 = len(rows) - 1 - i
        for a, b, dy, col, mk in [("Het_4_4", b4, 0.17, FOC, "o"), ("TF_ms", b5, -0.17, TFC, "D")]:
            c = C.contrast(P[a], P[b])
            ax3.plot([c["ci_low"], c["ci_high"]], [y0 + dy] * 2, color=col, lw=1)
            ax3.plot(c["delta"], y0 + dy, mk, color=col, ms=3.5, mfc=("white" if i == 5 else col))
            ax3.text(0.29, y0 + dy, f"{c['delta']:+.3f}  ({c['wins']}/{c['ties']}/{c['losses']})", fontsize=6, va="center", color=col)
    ax3.axvline(0, color=GREY, lw=0.6)
    ax3.set_yticks(range(len(rows))); ax3.set_yticklabels([r[0] for r in rows][::-1])
    ax3.set_xlabel("Δ norm_score vs multi-run ensemble of equal size (95% CI; win/tie/loss over 21 tasks)")
    ax3.set_title("Margin holds against every validation-selected control, not against a test-selected oracle")
    ax3.legend(handles=[Line2D([], [], marker="o", color=FOC, ls="", ms=3.5, label="4 different encoders vs 4 runs"),
                        Line2D([], [], marker="D", color=TFC, ls="", ms=3.5, label="Tetra-Fuse (5 members) vs 5 runs")],
               loc="upper left", handlelength=1)
    ax3.set_xlim(-0.13, 0.40); ax3.set_ylim(-0.6, 5.6)
    ax3.text(0.26, -0.55, "higher = different encoders better →", ha="right", fontsize=6, color=GREY)
    _letter(ax3, "C")
    return fig


def panel_fusion_matched(ax, F, FC):
    """Fig. 3C: MAP (1 model), Late-3, MAP x3, MAP x5 per paradigm, and Tetra-Fuse.
    The third tick-label line gives encoder forward passes per prediction (Section 3.3, Note S3.5)."""
    cols = {"MAP1": "#CBD5E1", "Late3": "#2563EB", "MAP3": "#94A3B8", "MAP5": "#64748B"}
    labs = {"MAP1": "MAP\n(1 model)\n3 passes", "Late3": "Late-3\n(3 encoders)\n3 passes",
            "MAP3": "MAP×3\n(3 runs)\n9 passes", "MAP5": "MAP×5\n(5 runs)\n15 passes"}
    x = 0; xt, xl, pos = [], [], {}
    for fam in ("stl", "mtl"):
        for arm in ("MAP1", "Late3", "MAP3", "MAP5"):
            v = F[f"{fam}_{arm}"].mean()
            ax.bar(x, v, 0.8, color=cols[arm], edgecolor="#1F2937", lw=0.6, hatch="" if arm == "Late3" else "///")
            ax.text(x, v + 0.012, f"{v:.3f}", ha="center", fontsize=6)
            pos[(fam, arm)] = x; xt.append(x); xl.append(labs[arm]); x += 1
        ax.text(x - 2.5, 1.04, fam.upper(), ha="center", fontsize=8, fontweight="bold")
        x += 0.6
    v = F["TF"].mean()
    ax.bar(x, v, 0.8, color=FOC, edgecolor="#1F2937", lw=0.6)
    ax.text(x, v + 0.012, f"{v:.3f}", ha="center", fontsize=6, fontweight="bold")
    pos["TF"] = x; xt.append(x); xl.append("Tetra-Fuse\n(5 members)\n7 passes")
    ax.set_xticks(xt); ax.set_xticklabels(xl, fontsize=6)
    ax.set_ylim(0, 1.09); ax.set_ylabel("Matched-seed norm_score\n(mean of 21 tasks)")

    def br(x1, x2, y, r):
        ax.plot([x1, x1, x2, x2], [y - 0.012, y, y, y - 0.012], color="#374151", lw=0.7)
        ax.text((x1 + x2) / 2, y + 0.008, f"Δ {r.delta:+.3f} ({r.ci_low:.3f}, {r.ci_high:.3f})", ha="center", fontsize=6)
    g = lambda lab: FC[FC.contrast == lab].iloc[0]
    br(pos[("stl", "Late3")], pos[("stl", "MAP3")], 0.84, g("STL: Late-3 vs MAP x3 (matched)"))
    br(pos[("mtl", "Late3")], pos[("mtl", "MAP3")], 0.84, g("MTL: Late-3 vs MAP x3 (matched)"))
    br(pos[("stl", "MAP5")], pos["TF"], 0.955, g("Tetra-Fuse vs MAP·STL x5"))
    ax.set_title("At matched member count, Late-3 and MAP×3 do not differ detectably")
