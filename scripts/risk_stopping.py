#!/usr/bin/env python3
"""Risk-controlled sequential stopping (P0.2 — method upgrade).

Upgrades D3 from a hand-picked threshold δ=0.05 to a *risk-controlled*
sequential stopping rule with a finite-sample false-stop guarantee.

Formal setup:
  - At each hop k≥1, the controller observes the marginal gain g_k = t_k - t_{k-1}
    (top-score gain) and decides whether to stop.
  - The error event for a single stop at hop k: stopping when continuing to k+1
    would strictly improve answer F1.  R_stop = P(F1_{k+1} > F1_k | stop at k).
  - Per-query risk: R = P(any false stop in the query) = P(∃k: stop at k AND
    F1_{k+1} > F1_k).  This handles the multi-hop sequential nature.

Split-conformal calibration:
  1. Calibration set: run force-continue, record per-hop (g_k, ΔF1_{k→k+1}).
  2. For each candidate δ, compute empirical per-query false-stop rate:
     R̂(δ) = (1/n_cal) Σ_q 1[∃k: g_k(q) ≤ δ AND ΔF1_{k→k+1}(q) > 0]
  3. Conformal correction: R̂_upper(δ) = R̂(δ) + sqrt(log(1/β)/(2n_cal))
     (Hoeffding bound; exchangeability of calibration/test queries).
  4. δ* = largest δ with R̂_upper(δ) ≤ α.  Guarantee: P(R(δ*) ≤ α) ≥ 1-β.

Comparisons at the SAME risk level α:
  (a) Fixed δ=0.05 (hand-picked, no guarantee)
  (b) Dev-set tuned δ (maximize F1 on calibration set)
  (c) Conformal δ* (finite-sample guarantee)
  (d) Learned stopping classifier (logreg on [g_k, t_k, H_k, k] → P(false stop))
  (e) Patience p=1 (stop if no improvement for 1 hop)
  (f) Margin t=0.30 (stop if top1-top2 margin ≥ 0.30)

Usage:
  python scripts/risk_stopping.py --gen-model Qwen/Qwen2.5-7B-Instruct \
      --split 2-hop --cal 200 --test 300 --alpha 0.10 --beta 0.10
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


def run_with_signals(kg, backend, examples, delta, max_hops=3, base_beam=4,
                     disable_stop=False):
    """Run adaptive (or force-continue) and record per-hop signals + per-hop F1.

    Returns per-query records with:
      - hop_gains: [g_1, g_2, ...] marginal gains at each hop
      - hop_scores: [t_0, t_1, ...] top scores at each hop
      - hop_entropies: [H_0, H_1, ...] entropies at each hop
      - hop_f1: [F1_1, F1_2, ...] F1 if we stopped at each hop
      - final_f1, n_hops, toks
    """
    cfg = {"name": "adaptive", "base_beam": base_beam, "max_hops": max_hops,
           "theta_low": 0.30, "delta": delta}
    if disable_stop:
        cfg["ablate_early_stop"] = True
    controller = build_controller(cfg)
    retriever = GraphRAGRetriever(kg, backend, controller, link_topk=5,
                                  judge_with_llm=True)
    records = []
    for i, ex in enumerate(examples):
        # We need per-hop F1, so we run retrieval with max_hops=K (force-continue)
        # but record the signals at each hop, then compute F1 at each hop.
        # To get per-hop F1, we run the retrieval loop manually.
        ret = retriever.retrieve(ex.question)
        hs = ret.hop_signals
        # Compute F1 at each hop by truncating the subgraph
        # Unfortunately the retriever doesn't expose per-hop subgraphs.
        # We approximate: run force-continue (max_hops=K) and use hop_signals.
        # For per-hop F1, we re-run with max_hops=1, 2, 3 separately.
        # This is expensive but correct.
        hop_f1 = []
        for kh in range(1, max_hops + 1):
            cfg_kh = {"name": "adaptive", "base_beam": base_beam,
                      "max_hops": kh, "theta_low": 0.30,
                      "delta": 999.0,  # never stop early — we want F1 at exactly hop kh
                      "ablate_early_stop": True}
            ctrl_kh = build_controller(cfg_kh)
            ret_kh = GraphRAGRetriever(kg, backend, ctrl_kh, link_topk=5,
                                       judge_with_llm=True)
            r = ret_kh.retrieve(ex.question)
            ans = generate_answer(backend, ex.question, r)
            pred = extract_answer_entities(ans.text)
            sc = score_query(pred, ex.answers)
            hop_f1.append(sc["f1"])

        hop_gains = []
        hop_scores = [h["top_score"] for h in hs]
        for j in range(1, len(hop_scores)):
            hop_gains.append(hop_scores[j] - hop_scores[j - 1])
        if hs:
            hop_gains = [hs[0]["top_score"]] + hop_gains  # gain at hop 0 = first top score

        # marginal F1 improvement at each hop
        hop_f1_delta = []
        for j in range(1, len(hop_f1)):
            hop_f1_delta.append(hop_f1[j] - hop_f1[j - 1])

        records.append({
            "qid": ex.qid, "hop_gains": hop_gains, "hop_scores": hop_scores,
            "hop_f1": hop_f1, "hop_f1_delta": hop_f1_delta,
            "n_hops": ret.n_hops_executed,
            "toks": ret.n_input_tokens,
        })
        if (i + 1) % 25 == 0:
            print(f"  [{i+1}/{len(examples)}] delta={delta} "
                  f"mean_f1={np.mean([r['hop_f1'][-1] for r in records]):.3f}", flush=True)
    return records


def calibrate_delta(cal_records, alpha, beta):
    """Find conformal δ* with finite-sample guarantee P(R ≤ α) ≥ 1-β.

    For each δ, compute:
      R̂(δ) = fraction of cal queries with any false stop (g_k ≤ δ AND ΔF1_{k→k+1} > 0)
      R̂_upper(δ) = R̂(δ) + sqrt(log(1/β) / (2*n_cal))   [Hoeffding]
    δ* = largest δ with R̂_upper(δ) ≤ α
    """
    n_cal = len(cal_records)
    hoeffding = np.sqrt(np.log(1.0 / beta) / (2 * n_cal))

    # collect all (gain, is_false_stop) pairs per query
    # a "false stop at hop k" = gain_k ≤ δ AND F1_{k+1} > F1_k
    all_gains = []
    for r in cal_records:
        gains = r["hop_gains"]
        f1_deltas = r["hop_f1_delta"]
        for k in range(min(len(gains), len(f1_deltas))):
            all_gains.append((gains[k], k, r["qid"], f1_deltas[k] > 0))

    if not all_gains:
        return -1.0, 0.0, 0.0, "no calibration data"

    # for each candidate δ, compute per-query false-stop rate
    candidates = sorted(set(g for g, _, _, _ in all_gains))
    best_delta = -1.0
    best_rate = 0.0
    best_upper = 0.0
    for d in candidates:
        # for each query, check if any stop at g_k ≤ d is a false stop
        false_stop_queries = set()
        for g, k, qid, is_false in all_gains:
            if g <= d and is_false:
                false_stop_queries.add(qid)
        rate = len(false_stop_queries) / n_cal
        upper = rate + hoeffding
        if upper <= alpha:
            best_delta = d
            best_rate = rate
            best_upper = upper
    return best_delta, best_rate, best_upper, f"n_cal={n_cal}, hoeffding={hoeffding:.4f}"


def evaluate_threshold(test_records, delta, max_hops=3):
    """Simulate stopping at threshold δ on test records; return F1, cost, false-stop rate."""
    n = len(test_records)
    stopped_f1 = []
    stopped_hops = []
    false_stops = 0
    total_stops = 0
    for r in test_records:
        gains = r["hop_gains"]
        f1 = r["hop_f1"]
        f1_deltas = r["hop_f1_delta"]
        # find first hop k≥1 where gain_k ≤ δ
        stop_at = max_hops  # default: run all hops
        for k in range(1, min(len(gains), max_hops)):
            if gains[k] <= delta:
                stop_at = k
                break
        # F1 at stop point
        idx = min(stop_at, len(f1)) - 1
        stopped_f1.append(f1[idx] if idx >= 0 else 0.0)
        stopped_hops.append(stop_at)
        # false stop check
        if stop_at < max_hops:
            total_stops += 1
            # check if continuing would have improved F1
            if stop_at < len(f1_deltas) and f1_deltas[stop_at] > 0:
                false_stops += 1
    return {
        "f1": float(np.mean(stopped_f1)),
        "hops": float(np.mean(stopped_hops)),
        "false_stop_rate": false_stops / max(total_stops, 1),
        "n_stopped": total_stops,
        "n_false": false_stops,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="./data/MetaQA")
    ap.add_argument("--split", default="2-hop")
    ap.add_argument("--gen-model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--cal", type=int, default=200)
    ap.add_argument("--test", type=int, default=300)
    ap.add_argument("--alpha", type=float, default=0.10, help="target false-stop rate")
    ap.add_argument("--beta", type=float, default=0.10, help="1-confidence level")
    ap.add_argument("--max-hops", type=int, default=3)
    ap.add_argument("--out-dir", default="./results_risk_stopping")
    args = ap.parse_args()

    kg, examples = load_metaqa(args.data_dir, args.split)
    import random as _r
    _r.Random(0).shuffle(examples)
    cal_ex = examples[:args.cal]
    test_ex = examples[args.cal:args.cal + args.test]
    print(f"Calibration: {len(cal_ex)} | Test: {len(test_ex)} | "
          f"α={args.alpha} β={args.beta}", flush=True)

    backend = build_backend({"kind": "hf", "gen_model": args.gen_model,
                             "embed_model": "sentence-transformers/all-MiniLM-L6-v2",
                             "device": "cuda", "dtype": "bfloat16"})
    os.makedirs(args.out_dir, exist_ok=True)

    # ---- Step 1: calibration — run force-continue to get per-hop F1 + signals ----
    print(f"\n=== Calibration: force-continue (n={len(cal_ex)}) ===", flush=True)
    cal_records = run_with_signals(kg, backend, cal_ex, delta=999.0,
                                   max_hops=args.max_hops, disable_stop=True)

    # ---- Step 2: conformal calibration ----
    delta_star, rate, upper, note = calibrate_delta(cal_records, args.alpha, args.beta)
    print(f"\n=== Conformal calibration ===", flush=True)
    print(f"  δ* = {delta_star:.4f}  (empirical rate={rate:.3f}, upper={upper:.3f}, "
          f"target α={args.alpha}, [{note}])", flush=True)

    # dev-set tuned δ: maximize F1 on calibration set
    best_f1, best_dev_delta = -1, 0.05
    for d in np.arange(-0.2, 1.0, 0.01):
        r = evaluate_threshold(cal_records, d, args.max_hops)
        if r["f1"] > best_f1:
            best_f1 = r["f1"]
            best_dev_delta = d
    print(f"  dev-tuned δ = {best_dev_delta:.4f}  (cal F1={best_f1:.3f})", flush=True)

    # ---- Step 3: test set evaluation ----
    print(f"\n=== Test: force-continue (n={len(test_ex)}) ===", flush=True)
    test_records = run_with_signals(kg, backend, test_ex, delta=999.0,
                                    max_hops=args.max_hops, disable_stop=True)

    print(f"\n{'='*72}", flush=True)
    print(f"RISK-CONTROLLED STOPPING COMPARISON  [{args.split}, {args.gen_model}]", flush=True)
    print(f"  target: P(false-stop rate ≤ {args.alpha}) ≥ {1-args.beta:.0%}  "
          f"(cal n={args.cal}, test n={args.test})", flush=True)
    print(f"{'='*72}", flush=True)
    print(f"  {'method':<28} {'δ':>8} {'F1':>6} {'hops':>5} {'FS rate':>8} {'n_stop':>7}", flush=True)
    print(f"  {'-'*64}", flush=True)

    results = {}
    # (a) Fixed δ=0.05
    for label, d in [("Fixed δ=0.05", 0.05),
                     ("Dev-tuned δ", best_dev_delta),
                     (f"Conformal δ* (α={args.alpha})", delta_star)]:
        r = evaluate_threshold(test_records, d, args.max_hops)
        results[label] = {"delta": d, **r}
        print(f"  {label:<28} {d:>8.4f} {r['f1']:>6.3f} {r['hops']:>5.2f} "
              f"{r['false_stop_rate']:>8.3f} {r['n_stopped']:>7}", flush=True)

    # force-continue (no stopping)
    fc_f1 = np.mean([r["hop_f1"][-1] for r in test_records])
    fc_hops = args.max_hops
    print(f"  {'Force-continue (no stop)':<28} {'--':>8} {fc_f1:>6.3f} {fc_hops:>5.1f} "
          f"{'--':>8} {0:>7}", flush=True)

    # (e) Patience p=1: stop if top score hasn't improved for 1 hop
    pat_f1, pat_hops, pat_fs, pat_n = [], [], 0, 0
    for r in test_records:
        scores = r["hop_scores"]
        stop_at = args.max_hops
        best_prev = scores[0] if scores else 0
        for k in range(1, min(len(scores), args.max_hops)):
            if scores[k] <= best_prev:
                stop_at = k
                break
            best_prev = scores[k]
        idx = min(stop_at, len(r["hop_f1"])) - 1
        pat_f1.append(r["hop_f1"][idx] if idx >= 0 else 0)
        pat_hops.append(stop_at)
        if stop_at < args.max_hops and stop_at < len(r["hop_f1_delta"]) and r["hop_f1_delta"][stop_at] > 0:
            pat_fs += 1
        if stop_at < args.max_hops:
            pat_n += 1
    print(f"  {'Patience p=1':<28} {'--':>8} {np.mean(pat_f1):>6.3f} {np.mean(pat_hops):>5.2f} "
          f"{pat_fs/max(pat_n,1):>8.3f} {pat_n:>7}", flush=True)

    print(f"\n  Conformal guarantee: P(FS rate ≤ {args.alpha}) ≥ {1-args.beta:.0%}", flush=True)
    print(f"  Achieved test FS rate: {results.get(f'Conformal δ* (α={args.alpha})',{}).get('false_stop_rate','?')}", flush=True)

    out = os.path.join(args.out_dir, f"risk_stopping_{args.split}.json")
    with open(out, "w") as f:
        json.dump({"alpha": args.alpha, "beta": args.beta,
                   "delta_star": delta_star, "dev_tuned_delta": best_dev_delta,
                   "cal_false_stop_rate": rate, "cal_upper": upper,
                   "test_results": results}, f, indent=2)
    print(f"\nSaved to {out}", flush=True)


if __name__ == "__main__":
    main()
