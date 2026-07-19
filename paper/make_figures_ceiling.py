#!/usr/bin/env python3
"""Figures for the predictability-ceiling analysis paper."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

plt.rcParams.update({
    "font.family": "serif", "font.size": 11, "axes.labelsize": 11,
    "axes.titlesize": 11, "legend.fontsize": 8.5, "xtick.labelsize": 10,
    "ytick.labelsize": 10, "figure.dpi": 200, "savefig.dpi": 300,
    "savefig.bbox": "tight", "axes.spines.top": False, "axes.spines.right": False,
})
TEAL = "#0d7377"; AMBER = "#e8a317"; GRAY = "#888888"; RED = "#c0392b"; BLUE = "#2c6fbb"
import os; OUT = "figures"; os.makedirs(OUT, exist_ok=True)

# ===== Figure 1: predictability ceiling (oracle vs realizable) =====
def fig_ceiling():
    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    # (label, F1, color, marker)
    pts = [
        ("Always-emb",            0.329, GRAY,  "o"),
        ("Trained router\n(signals)", 0.329, GRAY, "v"),
        ("Gap-routing",           0.354, AMBER, "s"),
        ("Trained router\n(question emb.)", 0.370, AMBER, "D"),
        ("Always-LLM",            0.411, BLUE,  "o"),
        ("Oracle (pick best/query)", 0.474, RED, "*"),
    ]
    ys = list(range(len(pts)))
    for y, (lab, f1, c, m) in zip(ys, pts):
        ax.scatter(f1, y, s=(230 if m=="*" else 90), color=c, ec="k", lw=.5, marker=m, zorder=3)
        ax.annotate(lab, (f1, y), textcoords="offset points", xytext=(10, 0),
                    fontsize=8, va="center")
    # always-LLM reference line
    ax.axvline(0.411, color=BLUE, ls="--", lw=1, alpha=.5)
    # shade the "unreachable" gap between always-LLM and oracle
    ax.axvspan(0.411, 0.474, color=RED, alpha=.08)
    ax.annotate("unrealized\nupside", (0.44, 2.5), fontsize=8, color=RED, ha="center")
    ax.set_yticks([]); ax.set_xlabel("Answer F1 (MetaQA mixed, 7B)")
    ax.set_xlim(0.30, 0.52)
    ax.set_title("The predictability ceiling: no realizable policy\nreaches the oracle; all fall below Always-LLM")
    ax.grid(True, axis="x", alpha=.25, linestyle="--")
    fig.savefig(f"{OUT}/fig_ceiling.pdf"); fig.savefig(f"{OUT}/fig_ceiling.png")
    plt.close(fig); print("saved fig_ceiling")

# ===== Figure 2: embedding vs LLM judge by depth =====
def fig_judge():
    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    settings = ["MetaQA\n1-hop", "MetaQA\n2-hop", "MetaQA\n3-hop", "CWQ"]
    emb = [0.793, 0.209, 0.056, 0.062]  # strong engine
    llm = [0.685, 0.248, 0.083, 0.085]  # strong engine
    x = np.arange(len(settings)); w = 0.36
    b1 = ax.bar(x - w/2, emb, w, label="Embedding judge (free)", color=TEAL, ec="k", lw=.5)
    b2 = ax.bar(x + w/2, llm, w, label="LLM judge (expensive)", color=AMBER, ec="k", lw=.5)
    # mark winner
    for i in range(len(settings)):
        win = max(emb[i], llm[i])
        ax.annotate("emb wins" if emb[i] > llm[i] else "LLM wins",
                    (x[i], win + 0.02), fontsize=7, ha="center",
                    color=TEAL if emb[i] > llm[i] else AMBER)
    ax.set_xticks(x); ax.set_xticklabels(settings)
    ax.set_ylabel("Answer F1"); ax.set_ylim(0, 0.88)
    ax.legend(loc="upper right", framealpha=.9)
    ax.set_title("Under a strong (ToG-2.0-style) engine, the LLM judge\nwins in 3 of 4 regimes; embedding wins only at 1-hop")
    ax.grid(True, axis="y", alpha=.25, linestyle="--")
    fig.savefig(f"{OUT}/fig_judge.pdf"); fig.savefig(f"{OUT}/fig_judge.png")
    plt.close(fig); print("saved fig_judge")

# ===== Figure 3: mixed-stream Pareto (adaptive dominated by tuned global) =====
def fig_pareto():
    fig, ax = plt.subplots(figsize=(5.4, 3.7))
    # (label, toks, F1, kind) kind: global / adaptive / oracle
    data = [
        ("Fixed-emb K=3", 159, 0.397, "global"),
        ("Fixed K=1", 398, 0.254, "global"),
        ("AdaptiveJudge .55", 886, 0.394, "adaptive"),
        ("Fixed K=2 B=4", 1005, 0.396, "global"),
        ("Oracle-depth", 999, 0.444, "oracle"),
        ("Fixed K=2 B=8", 1101, 0.478, "global"),
        ("AdaptiveJudge .75", 1402, 0.431, "adaptive"),
        ("Fixed K=3 B=4", 1610, 0.406, "global"),
    ]
    col = {"global": BLUE, "adaptive": AMBER, "oracle": RED}
    mk = {"global": "o", "adaptive": "^", "oracle": "*"}
    for lab, t, f, k in data:
        ax.scatter(t, f, s=(220 if k=="oracle" else 90), color=col[k], ec="k", lw=.5,
                   marker=mk[k], zorder=3)
    # global Pareto frontier (non-dominated global points): emb, K2B8
    front = sorted([(159,0.397),(1101,0.478)])
    ax.plot([p[0] for p in front],[p[1] for p in front], "--", color=BLUE, lw=1.3, alpha=.6, zorder=2)
    ax.annotate("Fixed K=2,B=8\n(non-dominated)", (1101,0.478), textcoords="offset points",
                xytext=(-6,8), fontsize=7.5, color=BLUE, ha="right")
    ax.annotate("Oracle-depth\n(dominated!)", (999,0.444), textcoords="offset points",
                xytext=(8,-14), fontsize=7.5, color=RED)
    ax.annotate("AdaptiveJudge\n(dominated)", (886,0.394), textcoords="offset points",
                xytext=(0,-22), fontsize=7.5, color=AMBER, ha="center")
    handles = [Line2D([0],[0],marker="o",color="w",mfc=BLUE,ms=8,label="Global (tuned)"),
               Line2D([0],[0],marker="^",color="w",mfc=AMBER,ms=9,label="Adaptive (ours)"),
               Line2D([0],[0],marker="*",color="w",mfc=RED,ms=12,label="Oracle-depth")]
    ax.legend(handles=handles, loc="lower right", framealpha=.9)
    ax.set_xlabel("Retrieval cost (input tokens / query)")
    ax.set_ylabel("Answer F1")
    ax.set_title("Tuned global budget dominates every adaptive policy\n(MetaQA mixed, 7B)")
    ax.grid(True, alpha=.25, linestyle="--")
    fig.savefig(f"{OUT}/fig_pareto_mixed.pdf"); fig.savefig(f"{OUT}/fig_pareto_mixed.png")
    plt.close(fig); print("saved fig_pareto_mixed")

if __name__ == "__main__":
    fig_ceiling(); fig_judge(); fig_pareto()
    print("ALL FIGURES DONE")
