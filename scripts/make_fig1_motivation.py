#!/usr/bin/env python3
"""Figure 1: three-panel motivation figure, precisely drawn with real numbers.

(a) the hope: per-query adaptive controller routing to a cheap embedding judge
    or an expensive LLM judge.
(b) the reality: on the MetaQA mixed stream (7B, real measured F1), a tuned
    global budget is the non-dominated frontier; every adaptive/fixed policy
    falls below it.
(c) the ceiling: oracle upside is large but realizable routing collapses to
    always-LLM, because I(sigma;Y*) ~ 0 -- the deciding information appears
    only after the expensive step.

All numbers are the measured results used in the paper. Okabe-Ito palette.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch

plt.rcParams.update({
    "font.size": 9, "figure.dpi": 200, "savefig.bbox": "tight",
    "pdf.fonttype": 42, "font.family": "sans-serif",
})
C = {"emb": "#009E73", "adaptive": "#E69F00", "tuned": "#0072B2",
     "oracle": "#D55E00", "grey": "#7F7F7F", "red": "#C0392B"}
RED = C["red"]

fig = plt.figure(figsize=(13.2, 4.3))
gs = fig.add_gridspec(1, 3, width_ratios=[1.02, 1.05, 1.12], wspace=0.22)

# =============================================================== Panel (a)
axa = fig.add_subplot(gs[0, 0]); axa.set_xlim(0, 10); axa.set_ylim(0, 10)
axa.axis("off")
axa.set_title("(a)  Per-query adaptation, the hope", fontsize=11, fontweight="bold", loc="left")
# KG motif: seed + 4 nodes
seed = (1.7, 5.2)
nodes = {"A": (0.7, 7.2), "B": (3.1, 6.3), "C": (0.7, 3.2), "E": (2.6, 3.1)}
for _, p in nodes.items():
    axa.plot([seed[0], p[0]], [seed[1], p[1]], color="#555", lw=1.0, zorder=1)
for lab, p in nodes.items():
    axa.add_patch(Circle(p, 0.42, fc="white", ec="#555", lw=1.1, zorder=2))
    axa.text(p[0], p[1], lab, ha="center", va="center", fontsize=8)
axa.add_patch(Circle(seed, 0.5, fc=C["emb"], ec="#333", lw=1.1, zorder=3))
axa.text(seed[0], seed[1]-1.15, "query\n(seed)", ha="center", va="top",
         fontsize=8, style="italic", color=C["emb"])
# controller box
cb = FancyBboxPatch((4.1, 4.5), 2.1, 1.35, boxstyle="round,pad=0.08,rounding_size=0.12",
                    fc="white", ec="#555", lw=1.2)
axa.add_patch(cb); axa.text(5.15, 5.17, "adaptive\ncontroller", ha="center", va="center", fontsize=8.5)
axa.add_patch(FancyArrowPatch((2.35, 5.1), (4.05, 5.15), arrowstyle="->", mutation_scale=13, color="#333", lw=1.3))
# branches
axa.add_patch(FancyArrowPatch((6.25, 5.55), (7.9, 7.0), arrowstyle="->", mutation_scale=12, color=C["emb"], lw=1.4))
axa.add_patch(FancyArrowPatch((6.25, 4.8), (7.9, 3.2), arrowstyle="->", mutation_scale=12, color=C["adaptive"], lw=1.4))
# judge doc icons
for (x, y), col in [((8.5, 7.0), C["emb"]), ((8.5, 3.2), C["adaptive"])]:
    axa.add_patch(FancyBboxPatch((x-0.55, y-0.6), 1.0, 1.2, boxstyle="round,pad=0.02,rounding_size=0.05",
                                 fc="white", ec=col, lw=1.3))
    for i in range(4):
        axa.plot([x-0.35, x+0.15], [y+0.35-i*0.22]*2, color=col, lw=1.3)
    axa.add_patch(Circle((x+0.42, y-0.5), 0.16, fc="none", ec="#333", lw=1.0))
    axa.plot([x+0.53, x+0.68], [y-0.61, y-0.76], color="#333", lw=1.2)
axa.text(7.5, 8.15, "cheap:\nembedding judge", ha="center", va="center", fontsize=7.8, color=C["emb"], style="italic")
axa.text(7.5, 1.95, "expensive:\nLLM judge", ha="center", va="center", fontsize=7.8, color=C["adaptive"], style="italic")
axa.text(5.0, 0.7, "\u201cspend compute only on hard queries?\u201d", ha="center", va="center",
         fontsize=9.5, style="italic", color=RED)

# =============================================================== Panel (b)
axb = fig.add_subplot(gs[0, 1])
axb.set_title("(b)  Reality: tuned global dominates", fontsize=11, fontweight="bold", loc="left")
# real measured MetaQA mixed 7B numbers (tokens, F1)
fixed = [(159, 0.397), (398, 0.254), (1005, 0.396), (1610, 0.406)]
adaptive = [(886, 0.394), (1402, 0.431)]
emb = (159, 0.397)  # embedding-only is the cheapest fixed point (green)
tuned = (1101, 0.478)
gx = [f[0] for f in fixed[1:]]; gy = [f[1] for f in fixed[1:]]
axb.scatter(gx, gy, s=70, c=C["grey"], marker="o", edgecolors="black", lw=0.5, zorder=3, label="fixed budgets")
axb.scatter([a[0] for a in adaptive], [a[1] for a in adaptive], s=95, c=C["adaptive"],
            marker="^", edgecolors="black", lw=0.5, zorder=3, label="adaptive")
axb.scatter([emb[0]], [emb[1]], s=90, c=C["emb"], marker="s", edgecolors="black", lw=0.5, zorder=3, label="embedding-only")
axb.scatter([tuned[0]], [tuned[1]], s=260, c=C["tuned"], marker="*", edgecolors="black", lw=0.6, zorder=4,
            label="Fixed $K{=}2,B{=}8$ (best)")
axb.axhline(0.478, ls="--", c=C["tuned"], lw=1.3, zorder=1)
axb.text(180, 0.487, "tuned global frontier", color=C["tuned"], fontsize=8.5)
axb.annotate("", xy=(900, 0.470), xytext=(760, 0.415),
             arrowprops=dict(arrowstyle="->", color=RED, lw=1.4,
                             connectionstyle="arc3,rad=-0.3"))
axb.text(430, 0.35, "all adaptive policies\nfall below", color=RED, fontsize=9, style="italic")
axb.set_xscale("log"); axb.set_xlim(90, 2200); axb.set_ylim(0.20, 0.52)
axb.set_xlabel("cost (tokens / query)"); axb.set_ylabel("F1")
axb.grid(alpha=0.25, lw=0.6)
axb.spines[["top", "right"]].set_visible(False)
axb.legend(fontsize=7.2, loc="lower right", framealpha=0.9, ncol=1)

# =============================================================== Panel (c)
axc = fig.add_subplot(gs[0, 2]); axc.set_xlim(0, 10); axc.set_ylim(0, 10); axc.axis("off")
axc.set_title("(c)  Why: the predictability ceiling", fontsize=11, fontweight="bold", loc="left")
# small inset bar chart (oracle vs realizable headroom over always-LLM baseline 0.411)
axins = axc.inset_axes([0.02, 0.34, 0.32, 0.55])
# headroom over always-LLM (0.411): oracle 0.474 -> +0.063 ; best realizable 0.370 -> below (~0)
axins.bar([0], [0.474-0.411], width=0.55, color=C["oracle"], edgecolor="black", lw=0.5)
axins.bar([1], [0.001], width=0.55, color=C["grey"], edgecolor="black", lw=0.5)
axins.set_xticks([0, 1]); axins.set_xticklabels(["oracle", "realiz."], fontsize=7.5)
axins.set_ylim(0, 0.075); axins.set_ylabel("F1 headroom", fontsize=8)
axins.tick_params(labelsize=7); axins.spines[["top", "right"]].set_visible(False)
# info-flow diagram: sigma -> X -> Y*
def box(x, y, w, h, text, ec="#555"):
    axc.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05,rounding_size=0.1",
                                 fc="white", ec=ec, lw=1.2))
    axc.text(x+w/2, y+h/2, text, ha="center", va="center", fontsize=8)
box(4.0, 6.2, 2.1, 1.5, "pre-\ncomputation\nsignal  $\\sigma$")
box(7.9, 6.2, 2.0, 1.5, "optimal\ndecision\n$Y^\\star$")
axc.add_patch(FancyArrowPatch((6.15, 6.95), (7.85, 6.95), arrowstyle="->", mutation_scale=12, color="#333", lw=1.3))
axc.add_patch(Circle((7.0, 6.95), 0.42, fc="white", ec=RED, lw=2.0, zorder=5))
axc.plot([6.75, 7.25], [6.7, 7.2], color=RED, lw=2.2, zorder=6)
axc.plot([6.75, 7.25], [7.2, 6.7], color=RED, lw=2.2, zorder=6)
axc.text(7.0, 5.55, "$\\approx 0$ bits", ha="center", fontsize=8.5, color=RED, fontweight="bold")
axc.text(7.0, 4.75, "$I(\\sigma;\\, Y^\\star) \\approx 0$", ha="center", fontsize=11)
# before/after timeline
axc.add_patch(FancyArrowPatch((0.6, 3.0), (9.6, 3.0), arrowstyle="->", mutation_scale=14, color="#333", lw=1.5))
axc.plot([5.1, 5.1], [2.75, 3.25], color="#333", lw=1.2)
# small drawn clock marker at the before/after boundary
axc.add_patch(Circle((5.1, 3.75), 0.34, fc="white", ec="#333", lw=1.3, zorder=6))
axc.plot([5.1, 5.1], [3.75, 3.98], color="#333", lw=1.3, zorder=7)   # minute hand
axc.plot([5.1, 5.26], [3.75, 3.68], color="#333", lw=1.3, zorder=7)  # hour hand
axc.text(2.4, 2.4, "cheap signals\n(before)", ha="center", va="top", fontsize=8)
axc.text(7.9, 2.4, "true difficulty revealed\n(after expensive step)", ha="center", va="top", fontsize=8)
axc.text(5.0, 0.75, "the deciding information appears\nonly after the cost is paid",
         ha="center", va="center", fontsize=9.5, style="italic", color=RED)

OUT = "/tmp/work/prj_01KXAZFT6R21GYEJMTNF94NW86/kdd2027_paper/figures/fig1_motivation.pdf"
fig.savefig(OUT); print("wrote", OUT)
fig.savefig(OUT.replace(".pdf", ".png"), dpi=200); print("wrote png")
