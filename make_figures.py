#!/usr/bin/env python3
"""Generate all data figures for the KDD paper from experiment results."""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

plt.rcParams.update({
    "font.family": "serif", "font.size": 11, "axes.labelsize": 11,
    "axes.titlesize": 12, "legend.fontsize": 9, "xtick.labelsize": 10,
    "ytick.labelsize": 10, "figure.dpi": 200, "savefig.dpi": 300,
    "savefig.bbox": "tight", "axes.spines.top": False, "axes.spines.right": False,
})
TEAL = "#0d7377"; AMBER = "#e8a317"; GRAY = "#888888"; RED = "#c0392b"; BLUE = "#2c6fbb"
LAB = "/data/lab/adaptive_graphrag"
OUT = "/data/lab/adaptive_graphrag/figures"
os.makedirs(OUT, exist_ok=True)


def load(split, system, root=LAB):
    p = os.path.join(root, f"metaqa_{split}_{system}", "reports.jsonl")
    if not os.path.exists(p):
        return []
    return [json.loads(l) for l in open(p) if l.strip()]


def agg(reps, key):
    return float(np.mean([r[key] for r in reps])) if reps else 0.0


# ============ Figure 2: Pareto frontier (F1 vs cost) ============
def fig_pareto():
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    # 7B points
    pts = {
        "7B 1-hop": ("1-hop", "results_7b"),
        "7B 2-hop": ("2-hop", "results_7b"),
        "7B 3-hop": ("3-hop", "results_7b"),
        "1.5B 2-hop": ("2-hop", "results"),
    }
    markers = {"Fixed": "s", "Adaptive": "^"}
    colors = {"7B 1-hop": BLUE, "7B 2-hop": TEAL, "7B 3-hop": AMBER, "1.5B 2-hop": RED}
    for label, (split, root) in pts.items():
        root_full = os.path.join(LAB, root)
        f = load(split, "fixed", root_full)
        a = load(split, "adaptive", root_full)
        if not f or not a:
            continue
        ax.scatter(agg(f, "n_input_tokens"), agg(f, "f1"), marker="s", s=70,
                   color=colors[label], edgecolor="k", linewidth=.5, zorder=3)
        ax.scatter(agg(a, "n_input_tokens"), agg(a, "f1"), marker="^", s=90,
                   color=colors[label], edgecolor="k", linewidth=.5, zorder=4)
        # arrow from fixed to adaptive
        ax.annotate("", xy=(agg(a, "n_input_tokens"), agg(a, "f1")),
                    xytext=(agg(f, "n_input_tokens"), agg(f, "f1")),
                    arrowprops=dict(arrowstyle="->", color=colors[label], lw=1.2, alpha=.6))
        ax.annotate(label, (agg(a, "n_input_tokens"), agg(a, "f1")),
                    textcoords="offset points", xytext=(6, 4), fontsize=7.5, color=colors[label])
    # legend
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], marker="s", color="w", markerfacecolor="gray", markersize=8, label="Fixed (ToG)"),
               Line2D([0], [0], marker="^", color="w", markerfacecolor="gray", markersize=9, label="AquaRAG (ours)")]
    ax.legend(handles=handles, loc="upper left", framealpha=.9)
    ax.set_xlabel("Retrieval cost (input tokens / query)")
    ax.set_ylabel("F1")
    ax.set_title("Pareto frontier: accuracy vs. cost")
    ax.set_xscale("log")
    ax.grid(True, alpha=.25, linestyle="--")
    fig.savefig(os.path.join(OUT, "fig_pareto.pdf"))
    fig.savefig(os.path.join(OUT, "fig_pareto.png"))
    plt.close(fig)
    print("saved fig_pareto")


# ============ Figure 3: Ablation bar chart (7B 2-hop) ============
def fig_ablation():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 3.2))
    systems = ["fixed", "abl-nostop", "abl-fixbeam", "adaptive"]
    labels = ["Fixed", "w/o (D3)", "w/o (D2)", "AquaRAG"]
    colors = [GRAY, "#d4a373", "#a98467", TEAL]
    f1s, toks = [], []
    for s in systems:
        reps = load("2-hop", s, os.path.join(LAB, "results_7b"))
        f1s.append(agg(reps, "f1"))
        toks.append(agg(reps, "n_input_tokens"))
    x = np.arange(len(labels))
    bars1 = ax1.bar(x, f1s, color=colors, edgecolor="k", linewidth=.5)
    ax1.set_xticks(x); ax1.set_xticklabels(labels, rotation=20, ha="right")
    ax1.set_ylabel("F1"); ax1.set_title("(a) Accuracy (7B, 2-hop)")
    ax1.set_ylim(0, max(f1s)*1.18)
    for b, v in zip(bars1, f1s):
        ax1.text(b.get_x()+b.get_width()/2, v+0.005, f"{v:.3f}", ha="center", fontsize=8)
    bars2 = ax2.bar(x, toks, color=colors, edgecolor="k", linewidth=.5)
    ax2.set_xticks(x); ax2.set_xticklabels(labels, rotation=20, ha="right")
    ax2.set_ylabel("Input tokens / query"); ax2.set_title("(b) Cost (7B, 2-hop)")
    ax2.set_ylim(0, max(toks)*1.12)
    for b, v in zip(bars2, toks):
        ax2.text(b.get_x()+b.get_width()/2, v+8, f"{v:.0f}", ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_ablation.pdf"))
    fig.savefig(os.path.join(OUT, "fig_ablation.png"))
    plt.close(fig)
    print("saved fig_ablation")


# ============ Figure 4: Sensitivity sweep ============
def fig_sensitivity():
    path = os.path.join(LAB, "results", "sweep", "sweep_2-hop.json")
    if not os.path.exists(path):
        print("no sweep data; skipping"); return
    data = json.load(open(path))
    fig, axes = plt.subplots(1, 3, figsize=(8.5, 2.8))
    params = [("delta", r"Early-stop margin $\delta$"),
              ("theta_low", r"Use-graph threshold $\theta_{\rm low}$"),
              ("base_beam", r"Base beam $B$")]
    for ax, (p, label) in zip(axes, params):
        if p not in data:
            ax.set_visible(False); continue
        vals = sorted(data[p].keys(), key=lambda x: float(x))
        f1 = [data[p][v]["f1"] for v in vals]
        toks = [data[p][v]["toks"] for v in vals]
        xv = [float(v) for v in vals]
        ax.plot(xv, f1, "o-", color=TEAL, lw=1.8, ms=6, label="F1")
        ax.set_xlabel(label); ax.set_ylabel("F1", color=TEAL)
        ax.tick_params(axis="y", labelcolor=TEAL)
        ax2 = ax.twinx(); ax2.spines["top"].set_visible(False)
        ax2.plot(xv, toks, "s--", color=AMBER, lw=1.5, ms=5, label="tokens")
        ax2.set_ylabel("tokens/q", color=AMBER)
        ax2.tick_params(axis="y", labelcolor=AMBER)
        ax.grid(True, alpha=.2, linestyle="--")
    fig.suptitle("Hyperparameter sensitivity (1.5B, 2-hop)", fontsize=11, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_sensitivity.pdf"))
    fig.savefig(os.path.join(OUT, "fig_sensitivity.png"))
    plt.close(fig)
    print("saved fig_sensitivity")


if __name__ == "__main__":
    fig_pareto()
    fig_ablation()
    fig_sensitivity()
    print("ALL FIGURES DONE")
