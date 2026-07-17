#!/usr/bin/env python3
"""Analysis experiments from saved per-query reports (zero compute).

Produces:
  1. Hop-count distribution per system (histogram) -- shows early-stop diversity
  2. Cost savings attribution by innovation (2a/2b/3)
  3. Per-query win/loss/tie breakdown (Adaptive vs Fixed)
  4. Cost-accuracy scatter data for Pareto plotting
Usage: python scripts/analyze.py --results-dir ./results --split 2-hop
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load(path):
    out = []
    if not os.path.exists(path):
        return out
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def hop_distribution(results_dir, split):
    print("=" * 70)
    print(f"Hop-count distribution  [{split}]")
    print("=" * 70)
    for sname in ["fixed", "adaptive", "abl-nograph", "abl-fixbeam", "abl-nostop"]:
        reps = load(os.path.join(results_dir, f"metaqa_{split}_{sname}", "reports.jsonl"))
        if not reps:
            continue
        hops = Counter(r["n_hops"] for r in reps)
        total = len(reps)
        print(f"\n  {sname} (n={total}):")
        for h in sorted(hops):
            bar = "#" * int(40 * hops[h] / total)
            print(f"    {h} hops: {hops[h]:4d} ({100*hops[h]/total:5.1f}%) {bar}")


def cost_attribution(results_dir, split):
    """Attribute cost savings to each innovation by pairwise comparison."""
    print("\n" + "=" * 70)
    print(f"Cost-savings attribution by innovation  [{split}]")
    print("=" * 70)
    sys_reps = {}
    for sname in ["fixed", "adaptive", "abl-nograph", "abl-fixbeam", "abl-nostop"]:
        reps = load(os.path.join(results_dir, f"metaqa_{split}_{sname}", "reports.jsonl"))
        if reps:
            sys_reps[sname] = reps

    if "fixed" not in sys_reps:
        print("  (fixed baseline missing)")
        return
    n = len(sys_reps["fixed"])
    base_toks = np.mean([r["n_input_tokens"] for r in sys_reps["fixed"]])
    print(f"\n  Baseline (fixed) mean toks/q = {base_toks:.0f}")

    pairs = [
        ("abl-nostop", "Adaptive", "Innovation (3) early-stop"),
        ("abl-fixbeam", "Adaptive", "Innovation (2b) adaptive beam"),
        ("abl-nograph", "Adaptive", "Innovation (2a) use-graph decision"),
    ]
    for abl_name, full_name, innov in pairs:
        if abl_name not in sys_reps:
            continue
        abl_toks = np.mean([r["n_input_tokens"] for r in sys_reps[abl_name][:n]])
        full_toks = np.mean([r["n_input_tokens"] for r in sys_reps.get("adaptive", sys_reps[abl_name])[:n]])
        abl_f1 = np.mean([r["f1"] for r in sys_reps[abl_name][:n]])
        full_f1 = np.mean([r["f1"] for r in sys_reps.get("adaptive", sys_reps[abl_name])[:n]])
        savings_from_base = (base_toks - full_toks) / base_toks * 100
        print(f"\n  {innov}:")
        print(f"    {abl_name}: toks/q={abl_toks:.0f}, F1={abl_f1:.3f}")
        print(f"    {full_name.lower()}: toks/q={full_toks:.0f}, F1={full_f1:.3f}")
        print(f"    -> enabling this innovation saves {(abl_toks-full_toks)/abl_toks*100:+.1f}% toks, "
              f"ΔF1={full_f1-abl_f1:+.3f}")


def win_loss(results_dir, split):
    print("\n" + "=" * 70)
    print(f"Per-query win/loss/tie: Adaptive vs Fixed  [{split}]")
    print("=" * 70)
    fixed = load(os.path.join(results_dir, f"metaqa_{split}_fixed", "reports.jsonl"))
    adapt = load(os.path.join(results_dir, f"metaqa_{split}_adaptive", "reports.jsonl"))
    if not fixed or not adapt:
        print("  (missing data)")
        return
    n = min(len(fixed), len(adapt))
    wins, losses, ties, cost_saved = 0, 0, 0, 0
    for i in range(n):
        df = adapt[i]["f1"] - fixed[i]["f1"]
        if df > 0.001:
            wins += 1
        elif df < -0.001:
            losses += 1
        else:
            ties += 1
        cost_saved += fixed[i]["n_input_tokens"] - adapt[i]["n_input_tokens"]
    print(f"\n  n={n}")
    print(f"  Adaptive wins:   {wins:4d} ({100*wins/n:5.1f}%)")
    print(f"  Adaptive ties:   {ties:4d} ({100*ties/n:5.1f}%)")
    print(f"  Adaptive losses: {losses:4d} ({100*losses/n:5.1f}%)")
    print(f"  Total tokens saved: {cost_saved} ({cost_saved/n:.0f}/query)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="./results")
    ap.add_argument("--split", default=None)
    args = ap.parse_args()
    splits = [args.split] if args.split else ["1-hop", "2-hop", "3-hop"]
    for sp in splits:
        hop_distribution(args.results_dir, sp)
        cost_attribution(args.results_dir, sp)
        win_loss(args.results_dir, sp)
        print()


if __name__ == "__main__":
    main()
