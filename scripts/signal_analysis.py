#!/usr/bin/env python3
"""Signal quality analysis (Point 4): AUROC of D3's stopping signal.

For each query where we know the ground-truth answer, we compute whether
stopping at hop k (vs continuing) would have produced a correct answer.
The stopping signal (top-score gain t_k - t_{k-1}) should discriminate
"safe to stop" from "must continue". We report AUROC.

Also: entropy H_k vs whether the optimal beam width (found by oracle)
correlates with our adaptive beam allocation.
"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load(path):
    if not os.path.exists(path): return []
    return [json.loads(l) for l in open(path) if l.strip()]


def auroc(scores, labels):
    """AUROC: can `scores` discriminate positive from negative labels?"""
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    if len(set(labels)) < 2:
        return float('nan'), "degenerate (one class)"
    # rank-based AUROC
    order = np.argsort(-scores)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(scores)+1)
    # handle ties via average rank
    unique_scores = np.unique(scores)
    for s in unique_scores:
        mask = scores == s
        ranks[mask] = np.mean(np.where(mask)[0] + 1)
    n_pos = labels.sum()
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float('nan'), "degenerate"
    sum_ranks_pos = ranks[labels == 1].sum()
    auroc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return float(auroc), f"n_pos={n_pos} n_neg={n_neg}"


def analyze_stop_signal(results_dir, split):
    """Analyze D3 stopping signal quality.

    For the Adaptive system, we look at n_hops (how many hops were executed).
    Queries that stopped early (n_hops < 3) and still got the right answer
    are "safe to stop" (positive). Queries that ran all 3 hops and got the
    right answer are "needed continuation" (negative).

    The signal: we use hop_signals from the retriever's diagnostics, but
    since reports.jsonl doesn't store per-hop scores, we use a proxy:
    queries where Adaptive stopped early AND got F1>0 are "correctly stopped"
    (positive class), and queries where Adaptive ran all 3 hops AND got F1=0
    are "should have stopped earlier" (negative class).

    We compute AUROC of "did Adaptive stop early?" (binary) as a discriminator
    of "was the answer correct?" (binary).
    """
    fixed = load(os.path.join(results_dir, f"metaqa_{split}_fixed", "reports.jsonl"))
    ada = load(os.path.join(results_dir, f"metaqa_{split}_adaptive", "reports.jsonl"))
    if not fixed or not ada:
        print(f"  no data for {split}")
        return

    n = min(len(fixed), len(ada))
    # For each query: did Adaptive stop early? (n_hops < 3) and was F1 > 0?
    stopped_early = [1 if ada[i]["n_hops"] < 3 else 0 for i in range(n)]
    correct_ada = [1 if ada[i]["f1"] > 0 else 0 for i in range(n)]
    correct_fix = [1 if fixed[i]["f1"] > 0 else 0 for i in range(n)]

    # Q1: Does stopping early correlate with correctness?
    # If stopping is safe, queries where we stopped should still be correct.
    stopped_correct = sum(1 for i in range(n) if stopped_early[i] and correct_ada[i])
    stopped_total = sum(stopped_early)
    continued_correct = sum(1 for i in range(n) if not stopped_early[i] and correct_ada[i])
    continued_total = n - stopped_total

    print(f"\n  [{split}] n={n}")
    print(f"    Stopped early: {stopped_total} queries, {stopped_correct} correct ({100*stopped_correct/max(stopped_total,1):.1f}%)")
    print(f"    Ran all 3 hops: {continued_total} queries, {continued_correct} correct ({100*continued_correct/max(continued_total,1):.1f}%)")

    # Q2: AUROC of "stop signal" (n_hops < 3) as discriminator of "answer correct under Fixed"
    # If Fixed also gets these right, stopping is safe
    safe_to_stop = [1 if (ada[i]["n_hops"] < 3 and fixed[i]["f1"] > 0) else 0 for i in range(n)]
    # signal = whether Adaptive stopped (1=stopped, 0=continued)
    # We want: did stopping predict that Fixed also got it right?
    # Actually the better question: for queries where Adaptive stopped early,
    # was the 3rd hop unnecessary (Fixed also correct without it)?
    # But Fixed always runs 3 hops. So we check: Adaptive F1 == Fixed F1 for stopped queries
    same_answer = [1 if ada[i]["f1"] == fixed[i]["f1"] else 0 for i in range(n)]

    # AUROC: can "Adaptive stopped early" discriminate "same answer as Fixed"?
    auc, info = auroc(stopped_early, same_answer)
    print(f"    AUROC(stop_signal → same_answer_as_Fixed): {auc:.3f} ({info})")

    # Q3: Cost saving analysis per query
    cost_saved = [fixed[i]["n_input_tokens"] - ada[i]["n_input_tokens"] for i in range(n)]
    print(f"    Mean tokens saved/query: {np.mean(cost_saved):.0f}")
    print(f"    Tokens saved on stopped queries: {np.mean([cost_saved[i] for i in range(n) if stopped_early[i]]):.0f}")
    print(f"    Tokens saved on continued queries: {np.mean([cost_saved[i] for i in range(n) if not stopped_early[i]]):.0f}")

    # Q4: Win/loss/tie
    wins = sum(1 for i in range(n) if ada[i]["f1"] > fixed[i]["f1"] + 0.001)
    losses = sum(1 for i in range(n) if ada[i]["f1"] < fixed[i]["f1"] - 0.001)
    ties = n - wins - losses
    print(f"    Win/Loss/Tie: {wins}/{losses}/{ties} ({100*wins/max(n,1):.1f}%/{100*losses/max(n,1):.1f}%/{100*ties/max(n,1):.1f}%)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="./results")
    ap.add_argument("--splits", nargs="+", default=["1-hop", "2-hop", "3-hop"])
    args = ap.parse_args()

    print("="*70)
    print("Signal Quality Analysis (D3 stopping signal AUROC)")
    print("="*70)
    for split in args.splits:
        analyze_stop_signal(args.results_dir, split)

    # Also analyze 7B results
    print("\n" + "="*70)
    print("7B Results")
    print("="*70)
    for split in ["1-hop", "2-hop", "3-hop"]:
        analyze_stop_signal("./results_7b", split)

    # Seeds
    print("\n" + "="*70)
    print("Seed consistency (7B/2-hop)")
    print("="*70)
    for s in [1, 2, 3]:
        analyze_stop_signal("./results_seeds", f"2-hop_s{s}")


if __name__ == "__main__":
    main()
