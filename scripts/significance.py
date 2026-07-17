#!/usr/bin/env python3
"""Statistical significance: bootstrap CIs + paired tests on existing reports.

Reads per-query reports.jsonl for each system, computes:
  - bootstrap 95% CI for mean F1, Hit@1, toks/q, calls/q
  - paired bootstrap test: Adaptive vs Fixed (and vs each ablation)
  - per-query cost reduction distribution

Zero extra compute -- operates entirely on saved reports.
Usage: python scripts/significance.py --results-dir ./results --split 1-hop
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import numpy as np
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_reports(path):
    reports = []
    if not os.path.exists(path):
        return reports
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                reports.append(json.loads(line))
    return reports


def bootstrap_ci(values, n_boot=10000, ci=0.95, seed=42):
    """Bootstrap confidence interval for the mean."""
    values = np.asarray(values, dtype=float)
    n = len(values)
    if n == 0:
        return (0.0, 0.0, 0.0)
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        means[i] = values[idx].mean()
    alpha = (1 - ci) / 2
    lo = np.percentile(means, 100 * alpha)
    hi = np.percentile(means, 100 * (1 - alpha))
    return float(values.mean()), float(lo), float(hi)


def paired_bootstrap_test(a, b, n_boot=10000, seed=42):
    """Paired bootstrap test: P(mean(a-b) <= 0) under resampling.
    Returns (observed_diff, p_value_one_sided).
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    n = min(len(a), len(b))
    if n == 0:
        return 0.0, 1.0
    a, b = a[:n], b[:n]
    diff = a - b
    observed = diff.mean()
    rng = np.random.default_rng(seed)
    count = 0
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        if diff[idx].mean() <= 0:
            count += 1
    p = count / n_boot
    return float(observed), float(p)


def analyze_split(results_dir, split, n_boot=10000):
    systems = {}
    for d in sorted(os.listdir(results_dir)):
        if not d.startswith(f"metaqa_{split}_"):
            continue
        sname = d.replace(f"metaqa_{split}_", "")
        reports = load_reports(os.path.join(results_dir, d, "reports.jsonl"))
        if reports:
            systems[sname] = reports

    if not systems:
        print(f"No reports found for {split}")
        return

    metrics = ["f1", "hit1", "n_input_tokens", "n_llm_calls", "n_hops"]
    metric_labels = {"f1": "F1", "hit1": "Hit@1", "n_input_tokens": "toks/q",
                     "n_llm_calls": "calls/q", "n_hops": "hops/q"}

    print("=" * 80)
    print(f"Bootstrap 95% CIs  [{split}]  (n_boot={n_boot})")
    print("=" * 80)
    for sname in sorted(systems):
        reps = systems[sname]
        print(f"\n  {sname} (n={len(reps)}):")
        for m in metrics:
            vals = [r[m] for r in reps]
            mean, lo, hi = bootstrap_ci(vals, n_boot=n_boot)
            print(f"    {metric_labels[m]:<10} = {mean:.4f}  [{lo:.4f}, {hi:.4f}]")

    # paired tests: adaptive vs everything
    if "adaptive" in systems and "fixed" in systems:
        n = min(len(systems["adaptive"]), len(systems["fixed"]))
        print("\n" + "=" * 80)
        print(f"Paired bootstrap tests: Adaptive vs Baselines  [{split}]  (n={n})")
        print("=" * 80)
        ref = systems["adaptive"][:n]
        for other in ["fixed", "abl-nograph", "abl-fixbeam", "abl-nostop"]:
            if other not in systems:
                continue
            oth = systems[other][:n]
            print(f"\n  Adaptive vs {other}:")
            for m in metrics:
                a_vals = [r[m] for r in ref]
                b_vals = [r[m] for r in oth]
                diff, p = paired_bootstrap_test(a_vals, b_vals, n_boot=n_boot)
                sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
                # for cost metrics (tokens, calls, hops), negative diff = improvement
                cost_m = m in ("n_input_tokens", "n_llm_calls", "n_hops")
                better = "lower" if cost_m else "higher"
                print(f"    {metric_labels[m]:<10}: Δ={diff:+.4f}  p={p:.4f}  {sig}  "
                      f"(adaptive {'lower' if cost_m and diff<0 else 'higher' if not cost_m and diff>0 else ''})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="./results")
    ap.add_argument("--split", default=None, help="e.g. 1-hop; if None, do all")
    ap.add_argument("--n-boot", type=int, default=10000)
    args = ap.parse_args()

    splits = [args.split] if args.split else ["1-hop", "2-hop", "3-hop"]
    for sp in splits:
        analyze_split(args.results_dir, sp, args.n_boot)


if __name__ == "__main__":
    main()
