#!/usr/bin/env python3
"""Judge reliability measurement (P0.3 — directly measure judge quality).

The plan asks: instead of using model size as a proxy for "judge quality",
directly measure how reliable each judge's relevance scores are, and
correlate this with AquaRAG's benefit over Fixed.

For each judge model, we measure:
  1. Gold triple ranking AUC: can the judge's scores rank gold-answer
     triples above non-gold triples?  (MetaQA has gold answers → we can
     identify which triples lead to the answer.)
  2. Score calibration: are high scores (≥80) actually high-precision?
  3. Cross-hop score comparability: is the top score trajectory t_k
     monotone when the true evidence is accumulating?
  4. Marginal gain ↔ true F1 gain correlation: does g_k = t_k - t_{k-1}
     correlate with ΔF1_{k→k+1}?  (This is the signal D3 relies on.)
  5. "Continue helps" AUROC: can g_k predict whether continuing improves F1?

Then we correlate judge reliability metrics with:
  ΔF1 = Adaptive_F1 - Fixed_F1
  Δcost = (Fixed_toks - Adaptive_toks) / Fixed_toks

Usage:
  python scripts/judge_reliability.py --gen-models Qwen/Qwen2.5-1.5B-Instruct Qwen/Qwen2.5-7B-Instruct Qwen/Qwen2.5-14B-Instruct --split 2-hop --limit 150
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agr import (load_metaqa, build_backend, build_controller,
                 GraphRAGRetriever, generate_answer, extract_answer_entities,
                 score_query)


def auroc(scores, labels):
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    if len(set(labels)) < 2:
        return float('nan')
    order = np.argsort(-scores)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    for s in np.unique(scores):
        mask = scores == s
        ranks[mask] = np.mean(np.where(mask)[0] + 1)
    n_pos = labels.sum(); n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float('nan')
    return float((ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def measure_judge(kg, backend, examples, max_hops=3):
    """Run force-continue retrieval + per-hop F1; measure judge reliability."""
    controller = build_controller({"name": "adaptive", "base_beam": 4,
                                   "max_hops": max_hops, "theta_low": 0.30,
                                   "delta": 999.0, "ablate_early_stop": True})
    retriever = GraphRAGRetriever(kg, backend, controller, link_topk=5,
                                  judge_with_llm=True)

    all_gains = []        # marginal gains g_k
    all_f1_deltas = []    # ΔF1_{k→k+1}
    all_labels = []       # 1 if continuing helped (ΔF1 > 0)
    hop_top_scores = []   # top scores per hop (for cross-hop comparability)
    gold_scores = []      # scores of gold-bearing triples
    nongold_scores = []   # scores of non-gold triples
    high_score_precision = []  # precision of score ≥ 80

    for i, ex in enumerate(examples):
        ret = retriever.retrieve(ex.question)
        hs = ret.hop_signals
        # per-hop F1
        hop_f1 = []
        for kh in range(1, max_hops + 1):
            cfg_kh = {"name": "adaptive", "base_beam": 4, "max_hops": kh,
                      "theta_low": 0.30, "delta": 999.0, "ablate_early_stop": True}
            ctrl_kh = build_controller(cfg_kh)
            ret_kh = GraphRAGRetriever(kg, backend, ctrl_kh, link_topk=5,
                                       judge_with_llm=True)
            r = ret_kh.retrieve(ex.question)
            ans = generate_answer(backend, ex.question, r)
            pred = extract_answer_entities(ans.text)
            sc = score_query(pred, ex.answers)
            hop_f1.append(sc["f1"])

        hop_scores = [h["top_score"] for h in hs]
        hop_top_scores.append(hop_scores)
        for k in range(1, len(hop_scores)):
            g = hop_scores[k] - hop_scores[k - 1]
            all_gains.append(g)
            if k < len(hop_f1):
                d = hop_f1[k] - hop_f1[k - 1]
                all_f1_deltas.append(d)
                all_labels.append(1 if d > 0 else 0)

        # gold vs nongold triple scores (approximation: triples whose tail
        # entity is in gold answers are "gold-bearing")
        gold_set = set(ex.answers)
        for h in hs:
            # we don't have per-triple scores in hop_signals, only top_score
            # approximate: top-scoring triple is "gold-bearing" if F1 improved
            pass  # skip gold ranking for now — requires per-triple score logging

        if (i + 1) % 25 == 0:
            print(f"  [{i+1}/{len(examples)}] processing...", flush=True)

    # compute metrics
    metrics = {}
    # 1. "Continue helps" AUROC: can g_k predict ΔF1 > 0?
    if all_gains and all_labels:
        metrics["continue_helps_auroc"] = auroc(all_gains, all_labels)
        metrics["n_pairs"] = len(all_gains)
        metrics["n_helped"] = int(sum(all_labels))
    # 2. gain-f1_delta correlation
    if all_gains and all_f1_deltas:
        g = np.array(all_gains); d = np.array(all_f1_deltas)
        metrics["gain_f1_corr"] = float(np.corrcoef(g, d)[0, 1]) if len(g) > 2 else float('nan')
    # 3. cross-hop score comparability: fraction of queries where t_k is
    #    monotone non-decreasing (evidence accumulating → scores should rise)
    monotone_count = 0
    for ts in hop_top_scores:
        if all(ts[k] <= ts[k + 1] for k in range(len(ts) - 1)):
            monotone_count += 1
    metrics["score_monotonicity"] = monotone_count / max(len(hop_top_scores), 1)
    # 4. mean top score at each hop
    max_len = max(len(ts) for ts in hop_top_scores) if hop_top_scores else 0
    metrics["mean_top_scores_by_hop"] = [
        float(np.mean([ts[k] for ts in hop_top_scores if k < len(ts)]))
        for k in range(max_len)
    ]
    return metrics


def run_fixed_adaptive(kg, backend, examples, max_hops=3):
    """Run Fixed and Adaptive, return (F1, toks) for each."""
    results = {}
    for name, cfg in [("fixed", {"name": "fixed", "beam": 4, "max_hops": max_hops}),
                       ("adaptive", {"name": "adaptive", "base_beam": 4,
                                     "max_hops": max_hops, "theta_low": 0.30, "delta": 0.05})]:
        controller = build_controller(cfg)
        retriever = GraphRAGRetriever(kg, backend, controller, link_topk=5,
                                      judge_with_llm=True)
        f1s, toks = [], []
        for ex in examples:
            ret = retriever.retrieve(ex.question)
            ans = generate_answer(backend, ex.question, ret)
            pred = extract_answer_entities(ans.text)
            sc = score_query(pred, ex.answers)
            f1s.append(sc["f1"]); toks.append(ret.n_input_tokens)
        results[name] = {"f1": float(np.mean(f1s)), "toks": float(np.mean(toks))}
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="./data/MetaQA")
    ap.add_argument("--split", default="2-hop")
    ap.add_argument("--gen-models", nargs="+", required=True,
                    help="judge models to evaluate")
    ap.add_argument("--limit", type=int, default=150)
    ap.add_argument("--max-hops", type=int, default=3)
    ap.add_argument("--out-dir", default="./results_judge_reliability")
    args = ap.parse_args()

    kg, examples = load_metaqa(args.data_dir, args.split)
    import random as _r
    _r.Random(0).shuffle(examples)
    examples = examples[:args.limit]
    print(f"KG: {len(kg)} triples | {len(examples)} questions", flush=True)
    os.makedirs(args.out_dir, exist_ok=True)

    all_results = {}
    for model in args.gen_models:
        print(f"\n{'='*72}", flush=True)
        print(f"=== Judge: {model} ===", flush=True)
        print(f"{'='*72}", flush=True)
        backend = build_backend({"kind": "hf", "gen_model": model,
                                 "embed_model": "sentence-transformers/all-MiniLM-L6-v2",
                                 "device": "cuda", "dtype": "bfloat16"})
        print("  Measuring judge reliability...", flush=True)
        metrics = measure_judge(kg, backend, examples, args.max_hops)
        print("  Running Fixed vs Adaptive...", flush=True)
        perf = run_fixed_adaptive(kg, backend, examples, args.max_hops)
        delta_f1 = perf["adaptive"]["f1"] - perf["fixed"]["f1"]
        delta_cost = (perf["fixed"]["toks"] - perf["adaptive"]["toks"]) / perf["fixed"]["toks"]
        print(f"  Fixed F1={perf['fixed']['f1']:.3f} toks={perf['fixed']['toks']:.0f}", flush=True)
        print(f"  Adapt F1={perf['adaptive']['f1']:.3f} toks={perf['adaptive']['toks']:.0f}", flush=True)
        print(f"  ΔF1={delta_f1:+.3f}  Δcost={delta_cost:+.1%}", flush=True)
        print(f"  continue_helps_auroc={metrics.get('continue_helps_auroc','?')}", flush=True)
        print(f"  gain_f1_corr={metrics.get('gain_f1_corr','?')}", flush=True)
        print(f"  score_monotonicity={metrics.get('score_monotonicity','?')}", flush=True)

        all_results[model] = {
            "judge_metrics": metrics,
            "fixed": perf["fixed"], "adaptive": perf["adaptive"],
            "delta_f1": delta_f1, "delta_cost": delta_cost,
        }

    # correlation: judge reliability vs benefit
    print(f"\n{'='*72}", flush=True)
    print("JUDGE RELIABILITY vs ADAPTIVE BENEFIT", flush=True)
    print(f"{'='*72}", flush=True)
    aurocs = [r["judge_metrics"].get("continue_helps_auroc", float('nan')) for r in all_results.values()]
    delta_f1s = [r["delta_f1"] for r in all_results.values()]
    valid = [(a, d) for a, d in zip(aurocs, delta_f1s) if not np.isnan(a)]
    if len(valid) >= 2:
        a_arr, d_arr = zip(*valid)
        corr = np.corrcoef(a_arr, d_arr)[0, 1]
        print(f"  Corr(continue_helps_auroc, ΔF1) = {corr:.3f}", flush=True)
        print(f"  Judges: {list(all_results.keys())}", flush=True)
        print(f"  AUROCs: {[f'{a:.3f}' for a in aurocs]}", flush=True)
        print(f"  ΔF1s:   {[f'{d:+.3f}' for d in delta_f1s]}", flush=True)

    out = os.path.join(args.out_dir, "judge_reliability.json")
    with open(out, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to {out}", flush=True)


if __name__ == "__main__":
    main()
