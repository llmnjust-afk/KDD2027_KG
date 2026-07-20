#!/usr/bin/env python3
"""Generate a rich set of figures for the predictability-ceiling paper.

All numbers are the actual measured results (copied from results_*/*.json).
Outputs colorblind-safe (Okabe-Ito) vector PDFs into figures/.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

plt.rcParams.update({
    "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.3, "grid.linewidth": 0.6,
    "figure.dpi": 150, "savefig.bbox": "tight", "pdf.fonttype": 42,
})
# Okabe-Ito colorblind-safe palette
C = {"emb": "#009E73", "adaptive": "#E69F00", "tuned": "#0072B2",
     "oracle": "#D55E00", "llm": "#CC79A7", "grey": "#999999", "k3": "#56B4E9"}
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "kdd2027_paper", "figures")
os.makedirs(OUT, exist_ok=True)
def save(fig, name):
    p = os.path.join(OUT, name); fig.savefig(p); plt.close(fig); print("wrote", p)


# ---------------------------------------------------------------- Fig 1: Pareto
def fig_pareto():
    # MetaQA mixed 7B (full_pareto_metaqa_mixed.json)
    pts = {
        "Fixed-emb K=3":      (159.2, 0.397, C["emb"], "s"),
        "Fixed-LLM K=1 B=4":  (398.2, 0.254, C["grey"], "o"),
        "Fixed-LLM K=2 B=4":  (1005.3, 0.396, C["grey"], "o"),
        "Fixed-LLM K=3 B=4":  (1610.4, 0.406, C["grey"], "o"),
        "AdaptiveJudge T=.55":(886.0, 0.394, C["adaptive"], "^"),
        "AdaptiveJudge T=.75":(1402.4, 0.431, C["adaptive"], "^"),
        "Fixed-LLM K=2 B=8":  (1101.1, 0.478, C["tuned"], "*"),
    }
    fig, ax = plt.subplots(figsize=(5.0, 3.6))
    for name, (x, y, c, m) in pts.items():
        ax.scatter(x, y, s=190 if m == "*" else 90, c=c, marker=m,
                   edgecolors="black", linewidths=0.6, zorder=3,
                   label=name)
    # frontier line through the tuned global point (horizontal ceiling)
    ax.axhline(0.478, ls="--", c=C["tuned"], lw=1.2, alpha=0.8, zorder=1)
    ax.text(1500, 0.482, "tuned-global frontier", color=C["tuned"], fontsize=8)
    ax.annotate("adaptive & all fixed\nlie below", xy=(1101, 0.478),
                xytext=(430, 0.44), fontsize=8,
                arrowprops=dict(arrowstyle="->", color="black", lw=0.8))
    ax.set_xlabel("Cost (input tokens / query)"); ax.set_ylabel("F1")
    ax.set_title("MetaQA mixed stream (7B): accuracy--cost")
    ax.legend(fontsize=6.7, loc="lower right", ncol=1, framealpha=0.9)
    save(fig, "fig_pareto_rich.pdf")


# ------------------------------------------------ Fig 2: cross-family grouped bar
def fig_family():
    fams = ["Qwen2.5-7B", "Llama-3.1-8B", "Mistral-7B"]
    emb   = [0.397, 0.223, 0.442]
    adap  = [0.394, 0.239, 0.453]   # best AdaptiveJudge
    tuned = [0.478, 0.248, 0.569]   # Fixed-K2B8
    x = np.arange(len(fams)); w = 0.26
    fig, ax = plt.subplots(figsize=(5.2, 3.5))
    ax.bar(x - w, emb, w, label="Fixed-emb $K{=}3$", color=C["emb"], edgecolor="black", lw=0.5)
    ax.bar(x,     adap, w, label="best AdaptiveJudge", color=C["adaptive"], edgecolor="black", lw=0.5)
    ax.bar(x + w, tuned, w, label="tuned Fixed-$K{=}2,B{=}8$", color=C["tuned"], edgecolor="black", lw=0.5)
    for i in range(len(fams)):
        ax.text(x[i]+w, tuned[i]+0.008, f"{tuned[i]:.3f}", ha="center", fontsize=7.5, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(fams); ax.set_ylabel("F1")
    ax.set_title("Cross-family: tuned global dominates on all three")
    ax.legend(fontsize=8, loc="upper center", ncol=1)
    ax.set_ylim(0, 0.66)
    save(fig, "fig_family.pdf")


# --------------------------------------------- Fig 3: multi-seed with bootstrap CI
def fig_multiseed():
    seeds = {
        "Fixed-emb":   [0.3755, 0.3458, 0.3496],
        "AdaptiveJudge":[0.3750, 0.3594, 0.3473],
        "Fixed-K3":    [0.3770, 0.3821, 0.3552],
        "Fixed-K2B8":  [0.4283, 0.4123, 0.4047],
    }
    order = ["Fixed-emb", "AdaptiveJudge", "Fixed-K3", "Fixed-K2B8"]
    cols = [C["emb"], C["adaptive"], C["k3"], C["tuned"]]
    fig, ax = plt.subplots(figsize=(5.2, 3.5))
    for i, k in enumerate(order):
        vals = seeds[k]; m = np.mean(vals); sd = np.std(vals)
        ax.errorbar(i, m, yerr=sd, fmt="o", color=cols[i], ms=9, capsize=5,
                    elinewidth=1.5, mec="black", mew=0.6, zorder=3)
        ax.scatter([i]*3, vals, c=cols[i], s=28, alpha=0.55, zorder=2)
        ax.text(i+0.12, m, f"{m:.3f}", fontsize=8, va="center")
    ax.set_xticks(range(4)); ax.set_xticklabels(order, rotation=12)
    ax.set_ylabel("F1 (mean $\\pm$ std over 3 subsamples)")
    ax.set_title("Multi-seed ($n{=}500\\times3$): tuned global significantly best\n"
                 "paired bootstrap $\\Delta{=}{+}0.055$, 95% CI $[.039,.070]$, $p{<}10^{-4}$",
                 fontsize=9)
    save(fig, "fig_multiseed.pdf")


# ------------------------------------------ Fig 4: strong-engine per-hop grouped
def fig_engine():
    regimes = ["MetaQA\n1-hop", "MetaQA\n2-hop", "MetaQA\n3-hop", "CWQ"]
    emb_w = [0.787, 0.250, 0.095, 0.046]
    emb_s = [0.793, 0.209, 0.056, 0.062]
    llm_w = [0.672, 0.285, 0.090, 0.024]
    llm_s = [0.685, 0.248, 0.083, 0.085]
    x = np.arange(len(regimes)); w = 0.2
    fig, ax = plt.subplots(figsize=(6.6, 3.5))
    ax.bar(x-1.5*w, emb_w, w, label="emb / naive", color=C["emb"], alpha=0.55, edgecolor="black", lw=0.4)
    ax.bar(x-0.5*w, emb_s, w, label="emb / strong", color=C["emb"], edgecolor="black", lw=0.4)
    ax.bar(x+0.5*w, llm_w, w, label="LLM / naive", color=C["llm"], alpha=0.55, edgecolor="black", lw=0.4)
    ax.bar(x+1.5*w, llm_s, w, label="LLM / strong", color=C["llm"], edgecolor="black", lw=0.4)
    ax.set_xticks(x); ax.set_xticklabels(regimes); ax.set_ylabel("F1")
    ax.set_title("Strong-engine ablation: LLM judge wins 3/4 regimes under fair candidate selection")
    ax.legend(fontsize=8, ncol=2)
    save(fig, "fig_engine.pdf")


# ------------------------- Fig 5: richer-router AUROC + mutual information (2 panel)
def fig_ceiling_info():
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.8, 3.3))
    # panel 1: AUROC of routers (all near 0.5)
    routers = ["gap-\nrouting", "LogReg", "MLP", "RandForest"]
    auroc = [0.512, 0.559, 0.559, 0.533]
    a1.bar(routers, auroc, color=C["adaptive"], edgecolor="black", lw=0.5)
    a1.axhline(0.5, ls="--", c="black", lw=1.0); a1.text(2.4, 0.505, "chance", fontsize=8)
    a1.set_ylim(0.45, 0.75); a1.set_ylabel("CV-AUROC (predict better judge)")
    a1.set_title("(a) Every router is near-random", fontsize=9)
    for i, v in enumerate(auroc):
        a1.text(i, v+0.006, f"{v:.3f}", ha="center", fontsize=7.5)
    # panel 2: mutual information vs permutation null
    labels = ["observed\n$\\hat I(\\sigma;Y^\\star)$", "permutation\nnull mean", "$H(Y^\\star)$"]
    vals = [0.017, 0.076, 0.754]
    cols = [C["oracle"], C["grey"], C["tuned"]]
    a2.bar(labels, vals, color=cols, edgecolor="black", lw=0.5)
    for i, v in enumerate(vals):
        a2.text(i, v+0.015, f"{v:.3f}", ha="center", fontsize=8)
    a2.set_ylabel("bits"); a2.set_title("(b) Signal carries ~0 information\n$p{=}0.92$, bias-corrected $I{=}0$", fontsize=9)
    save(fig, "fig_ceiling_info.pdf")


# ----------------------------------------------- Fig 6: cost-latency bubble
def fig_cost():
    data = {
        "Fixed-emb K=3":     (0.89, 0.323, 0.0),
        "AdaptiveJudge T=.55":(1.30, 0.371, 0.31),
        "Fixed-LLM K=2 B=8": (3.16, 0.380, 2.0),
    }
    cols = [C["emb"], C["adaptive"], C["tuned"]]
    fig, ax = plt.subplots(figsize=(5.2, 3.5))
    for (name, (lat, f1, calls)), c in zip(data.items(), cols):
        ax.scatter(lat, f1, s=120 + calls*260, c=c, edgecolors="black",
                   linewidths=0.7, alpha=0.85, zorder=3, label=name)
        ax.text(lat, f1+0.006, f"{calls:.2f} judge/q", ha="center", fontsize=7.5)
    ax.set_xlabel("Latency (s / query)"); ax.set_ylabel("F1")
    ax.set_title("Cost--latency: adaptation is cheaper but still\nless accurate than the tuned global point", fontsize=9)
    ax.legend(fontsize=8, loc="lower right"); ax.set_xlim(0.4, 3.6); ax.set_ylim(0.30, 0.40)
    save(fig, "fig_cost.pdf")


# ----------------------------------- Fig 7: three-benchmark summary (tuned/adaptive/oracle)
def fig_benchmarks():
    bench = ["MetaQA\nmixed", "CWQ", "WebQSP"]
    emb    = [0.397, 0.061, 0.251]
    adap   = [0.394, 0.052, 0.293]
    oracle = [0.474, 0.072, 0.324]
    tuned  = [0.478, 0.072, 0.338]
    x = np.arange(len(bench)); w = 0.2
    fig, ax = plt.subplots(figsize=(6.4, 3.5))
    ax.bar(x-1.5*w, emb, w, label="Fixed-emb", color=C["emb"], edgecolor="black", lw=0.4)
    ax.bar(x-0.5*w, adap, w, label="best Adaptive", color=C["adaptive"], edgecolor="black", lw=0.4)
    ax.bar(x+0.5*w, oracle, w, label="oracle (unreachable)", color=C["oracle"], alpha=0.6, edgecolor="black", lw=0.4, hatch="//")
    ax.bar(x+1.5*w, tuned, w, label="tuned global", color=C["tuned"], edgecolor="black", lw=0.4)
    ax.set_xticks(x); ax.set_xticklabels(bench); ax.set_ylabel("F1")
    ax.set_title("Across three benchmarks: tuned global $\\geq$ adaptive; oracle unreached")
    ax.legend(fontsize=8, ncol=2)
    save(fig, "fig_benchmarks.pdf")


if __name__ == "__main__":
    fig_pareto(); fig_family(); fig_multiseed(); fig_engine()
    fig_ceiling_info(); fig_cost(); fig_benchmarks()
    print("all figures written to", OUT)
