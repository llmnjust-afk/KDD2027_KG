#!/usr/bin/env python3
"""Generate data figures locally using known experiment values."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

plt.rcParams.update({
    "font.family": "serif", "font.size": 11, "axes.labelsize": 11,
    "axes.titlesize": 12, "legend.fontsize": 9, "xtick.labelsize": 10,
    "ytick.labelsize": 10, "figure.dpi": 200, "savefig.dpi": 300,
    "savefig.bbox": "tight", "axes.spines.top": False, "axes.spines.right": False,
})
TEAL = "#0d7377"; AMBER = "#e8a317"; GRAY = "#888888"; RED = "#c0392b"; BLUE = "#2c6fbb"
OUT = "figures"
import os; os.makedirs(OUT, exist_ok=True)

# Known experiment values (from results)
# (F1, toks) per setting
DATA = {
    "7B 1-hop":  {"fixed": (0.732, 980),  "adaptive": (0.790, 729)},
    "7B 2-hop":  {"fixed": (0.343, 1111), "adaptive": (0.378, 853)},
    "7B 3-hop":  {"fixed": (0.075, 1354), "adaptive": (0.080, 1028)},
    "1.5B 2-hop":{"fixed": (0.183, 1095), "adaptive": (0.226, 888)},
}
COLORS = {"7B 1-hop": BLUE, "7B 2-hop": TEAL, "7B 3-hop": AMBER, "1.5B 2-hop": RED}

# ===== Figure 2: Pareto frontier =====
def fig_pareto():
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    for label in ["1.5B 2-hop", "7B 3-hop", "7B 2-hop", "7B 1-hop"]:
        d = DATA[label]
        fx, ff = d["fixed"]; axx, af = d["adaptive"]
        ax.scatter(fx, ff, marker="s", s=70, color=COLORS[label], ec="k", lw=.5, zorder=3)
        ax.scatter(axx, af, marker="^", s=95, color=COLORS[label], ec="k", lw=.5, zorder=4)
        ax.annotate("", xy=(axx, af), xytext=(fx, ff),
                    arrowprops=dict(arrowstyle="->", color=COLORS[label], lw=1.3, alpha=.6))
        ax.annotate(label, (axx, af), textcoords="offset points", xytext=(7, 3), fontsize=7.5, color=COLORS[label])
    handles = [Line2D([0],[0], marker="s", color="w", mfc="gray", ms=8, label="Fixed (ToG)"),
               Line2D([0],[0], marker="^", color="w", mfc="gray", ms=9, label="AquaRAG (ours)")]
    ax.legend(handles=handles, loc="upper left", framealpha=.9)
    ax.set_xlabel("Retrieval cost (input tokens / query)  [log scale]")
    ax.set_ylabel("F1")
    ax.set_title("Pareto frontier: accuracy vs. cost")
    ax.set_xscale("log")
    ax.grid(True, alpha=.25, linestyle="--")
    fig.savefig(f"{OUT}/fig_pareto.pdf"); fig.savefig(f"{OUT}/fig_pareto.png")
    plt.close(fig); print("saved fig_pareto")

# ===== Figure 3: Ablation bar chart (7B 2-hop) =====
def fig_ablation():
    # 7B 2-hop ablation: system -> (F1, toks)
    ABL = {"Fixed": (0.343, 1111), "w/o (D3)": (0.361, 1118),
           "w/o (D2)": (0.358, 854), "AquaRAG": (0.378, 853)}
    labels = list(ABL.keys())
    f1s = [ABL[l][0] for l in labels]; toks = [ABL[l][1] for l in labels]
    colors = [GRAY, "#d4a373", "#a98467", TEAL]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 3.2))
    x = np.arange(len(labels))
    bars1 = ax1.bar(x, f1s, color=colors, ec="k", lw=.5)
    ax1.set_xticks(x); ax1.set_xticklabels(labels, rotation=20, ha="right")
    ax1.set_ylabel("F1"); ax1.set_title("(a) Accuracy (7B, 2-hop)")
    ax1.set_ylim(0, max(f1s)*1.18)
    for b, v in zip(bars1, f1s): ax1.text(b.get_x()+b.get_width()/2, v+0.005, f"{v:.3f}", ha="center", fontsize=8)
    bars2 = ax2.bar(x, toks, color=colors, ec="k", lw=.5)
    ax2.set_xticks(x); ax2.set_xticklabels(labels, rotation=20, ha="right")
    ax2.set_ylabel("Input tokens / query"); ax2.set_title("(b) Cost (7B, 2-hop)")
    ax2.set_ylim(0, max(toks)*1.12)
    for b, v in zip(bars2, toks): ax2.text(b.get_x()+b.get_width()/2, v+8, f"{v:.0f}", ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig_ablation.pdf"); fig.savefig(f"{OUT}/fig_ablation.png")
    plt.close(fig); print("saved fig_ablation")

# ===== Figure 4: Sensitivity sweep =====
def fig_sensitivity():
    DELTA = {"0.01": (0.214, 737), "0.03": (0.214, 737), "0.05": (0.214, 737),
             "0.1": (0.204, 720), "0.2": (0.175, 561)}
    THETA = {"0.1": (0.214, 737), "0.2": (0.214, 737), "0.3": (0.214, 737),
             "0.4": (0.214, 737), "0.5": (0.214, 737)}
    BEAM = {"2": (0.144, 642), "4": (0.214, 737), "6": (0.249, 745), "8": (0.271, 754)}
    panels = [(DELTA, r"Early-stop margin $\delta$"),
              (THETA, r"Use-graph threshold $\theta_{\rm low}$"),
              (BEAM, r"Base beam $B$")]
    fig, axes = plt.subplots(1, 3, figsize=(8.5, 2.9))
    for ax, (data, label) in zip(axes, panels):
        xv = [float(k) for k in data.keys()]
        f1 = [v[0] for v in data.values()]; toks = [v[1] for v in data.values()]
        ax.plot(xv, f1, "o-", color=TEAL, lw=1.8, ms=6, label="F1")
        ax.set_xlabel(label); ax.set_ylabel("F1", color=TEAL)
        ax.tick_params(axis="y", labelcolor=TEAL)
        ax2 = ax.twinx(); ax2.spines["top"].set_visible(False)
        ax2.plot(xv, toks, "s--", color=AMBER, lw=1.5, ms=5)
        ax2.set_ylabel("tokens/q", color=AMBER); ax2.tick_params(axis="y", labelcolor=AMBER)
        ax.grid(True, alpha=.2, linestyle="--")
    fig.suptitle("Hyperparameter sensitivity (1.5B, 2-hop, n=100)", fontsize=11, y=1.03)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig_sensitivity.pdf"); fig.savefig(f"{OUT}/fig_sensitivity.png")
    plt.close(fig); print("saved fig_sensitivity")

if __name__ == "__main__":
    fig_pareto(); fig_ablation(); fig_sensitivity()
    print("ALL FIGURES DONE")
