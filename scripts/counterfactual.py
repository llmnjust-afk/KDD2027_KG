#!/usr/bin/env python3
"""Counterfactual stopping analysis (reviewer-requested, addresses review #5).

The reviewer correctly notes that comparing P(correct|stopped) vs
P(correct|continued) suffers selection bias (different difficulty queries).
The correct test is a *counterfactual*: take the queries D3 decided to stop,
FORCE them to continue to max_hops, and measure whether the answer improves.

We do this by running, on the SAME queries:
  - Adaptive       : our controller (stops early on some queries)
  - Force-continue : identical controller with early-stop DISABLED (abl-nostop)
                     -> same beam logic (D2), same use-graph (D1), only D3 off

On the stopped subset (adaptive n_hops < force n_hops) we report:
  - false-stop rate : fraction where force-continue STRICTLY improves F1
  - answer gain     : mean(F1_force - F1_adaptive) on stopped queries
  - answer change   : fraction where the generated answer changed at all
  - AUROC           : can the marginal-gain stop signal discriminate
                      "force-continue helps" from "force-continue doesn't"?

Usage:
  python scripts/counterfactual.py --gen-model Qwen/Qwen2.5-7B-Instruct \
      --split 2-hop --limit 200
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
        return float('nan'), "one-class"
    order = np.argsort(-scores)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    for s in np.unique(scores):
        mask = scores == s
        ranks[mask] = np.mean(np.where(mask)[0] + 1)
    n_pos = labels.sum(); n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float('nan'), "degenerate"
    auroc = (ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return float(auroc), f"n_pos={int(n_pos)} n_neg={int(n_neg)}"


def run_mode(kg, backend, examples, cfg, limit, max_hops=3):
    """Run one controller mode; return per-query records with hop_signals."""
    controller = build_controller(cfg)
    retriever = GraphRAGRetriever(kg, backend, controller, link_topk=5,
                                  judge_with_llm=True)
    records = []
    n = min(limit, len(examples))
    for i, ex in enumerate(examples[:n]):
        ret = retriever.retrieve(ex.question)
        ans = generate_answer(backend, ex.question, ret)
        pred = extract_answer_entities(ans.text)
        sc = score_query(pred, ex.answers)
        # marginal gain at the stop point: last hop top_score minus previous
        hs = ret.hop_signals
        stop_gain = None
        if len(hs) >= 2:
            stop_gain = hs[-1]["top_score"] - hs[-2]["top_score"]
        elif len(hs) == 1:
            stop_gain = hs[-1]["top_score"]
        records.append({
            "qid": ex.qid, "n_hops": ret.n_hops_executed,
            "f1": sc["f1"], "hit1": sc["hit1"], "exact": sc["exact"],
            "pred": pred, "answer": ans.text,
            "stop_gain": stop_gain,
            "top_scores": [h["top_score"] for h in hs],
        })
        if (i + 1) % 50 == 0 or i == n - 1:
            f1 = np.mean([r["f1"] for r in records])
            nh = np.mean([r["n_hops"] for r in records])
            print(f"  [{i+1}/{n}] F1={f1:.3f} hops={nh:.2f} ({time.time():.0f})", flush=True)
    return records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="./data/MetaQA")
    ap.add_argument("--split", default="2-hop")
    ap.add_argument("--gen-model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--max-hops", type=int, default=3)
    ap.add_argument("--out-dir", default="./results_counterfactual")
    args = ap.parse_args()

    kg, examples = load_metaqa(args.data_dir, args.split)
    print(f"KG: {len(kg)} triples | {len(examples)} questions (using {args.limit})", flush=True)
    backend = build_backend({"kind": "hf", "gen_model": args.gen_model,
                             "embed_model": "sentence-transformers/all-MiniLM-L6-v2",
                             "device": "cuda", "dtype": "bfloat16"})
    os.makedirs(args.out_dir, exist_ok=True)

    # ---- Adaptive (with D3 early stopping) ----
    print("\n=== Adaptive (D3 active) ===", flush=True)
    adapt = run_mode(kg, backend, examples,
                     {"name": "adaptive", "base_beam": 4, "max_hops": args.max_hops,
                      "theta_low": 0.30, "delta": 0.05}, args.limit, args.max_hops)

    # ---- Force-continue (D3 disabled, same D1/D2) ----
    print("\n=== Force-continue (D3 disabled) ===", flush=True)
    force = run_mode(kg, backend, examples,
                     {"name": "adaptive", "base_beam": 4, "max_hops": args.max_hops,
                      "theta_low": 0.30, "delta": 0.05,
                      "ablate_early_stop": True}, args.limit, args.max_hops)

    # ---- align by qid ----
    fa = {r["qid"]: r for r in adapt}
    ff = {r["qid"]: r for r in force}
    qids = [r["qid"] for r in adapt if r["qid"] in ff]

    stopped = [q for q in qids if fa[q]["n_hops"] < ff[q]["n_hops"]]
    fullrun = [q for q in qids if fa[q]["n_hops"] == ff[q]["n_hops"]]

    print("\n" + "=" * 72, flush=True)
    print(f"COUNTERFACTUAL STOPPING ANALYSIS  [{args.split}, n={len(qids)}, 7B]", flush=True)
    print("=" * 72, flush=True)
    print(f"  Adaptive stopped early on : {len(stopped)}/{len(qids)} queries "
          f"({100*len(stopped)/max(len(qids),1):.1f}%)", flush=True)
    print(f"  Adaptive ran to max on    : {len(fullrun)}/{len(qids)} queries", flush=True)

    if stopped:
        # false-stop: force-continue STRICTLY improves F1
        false_stops = [q for q in stopped if ff[q]["f1"] > fa[q]["f1"]]
        helped = [q for q in stopped if ff[q]["f1"] > fa[q]["f1"]]
        hurt = [q for q in stopped if ff[q]["f1"] < fa[q]["f1"]]
        unchanged = [q for q in stopped if ff[q]["f1"] == fa[q]["f1"]]
        gains = [ff[q]["f1"] - fa[q]["f1"] for q in stopped]

        print(f"\n  On the {len(stopped)} stopped queries:", flush=True)
        print(f"    force-continue improved F1 : {len(helped)} ({100*len(helped)/len(stopped):.1f}%)  <- false stops", flush=True)
        print(f"    force-continue hurt F1     : {len(hurt)} ({100*len(hurt)/len(stopped):.1f}%)", flush=True)
        print(f"    force-continue no change   : {len(unchanged)} ({100*len(unchanged)/len(stopped):.1f}%)", flush=True)
        print(f"    false-stop rate            : {len(false_stops)/len(stopped):.3f}", flush=True)
        print(f"    mean answer gain (force-adapt): {np.mean(gains):+.4f}", flush=True)
        print(f"    median answer gain         : {np.median(gains):+.4f}", flush=True)

        # answer text change
        changed = [q for q in stopped if ff[q]["answer"].strip() != fa[q]["answer"].strip()]
        print(f"    answer text changed        : {len(changed)}/{len(stopped)} "
              f"({100*len(changed)/len(stopped):.1f}%)", flush=True)

        # AUROC: stop signal (marginal gain) predicts "force-continue helps"
        # label = 1 if force improved F1 (false stop), 0 if not
        # score = marginal gain at stop point (LOW gain -> should stop -> predict no help)
        # so we expect NEGATIVE correlation: low gain -> safe to stop
        labels = [1 if ff[q]["f1"] > fa[q]["f1"] else 0 for q in stopped]
        scores = [fa[q]["stop_gain"] if fa[q]["stop_gain"] is not None else 0.0
                  for q in stopped]
        a, note = auroc(scores, labels)
        # AUROC of "gain predicts help": if low gain => safe to stop => NOT help,
        # then AUROC(gain, help) should be < 0.5. Report 1-AUROC as "safe-to-stop AUROC".
        print(f"\n  AUROC (gain predicts 'continue helps'): {a:.3f}  [{note}]", flush=True)
        print(f"  AUROC (gain predicts 'safe to stop' = 1-AUROC): {1-a:.3f}", flush=True)

        # also AUROC on ALL queries (not just stopped) using stop decision
        all_labels = [1 if ff[q]["f1"] > fa[q]["f1"] else 0 for q in qids]
        all_scores = [-(fa[q]["stop_gain"] or 0.0) for q in qids]  # negative: low gain -> stop
        a2, note2 = auroc(all_scores, all_labels)
        print(f"  AUROC (all queries, safe-to-stop signal): {a2:.3f}  [{note2}]", flush=True)

    # overall F1 comparison
    f1a = np.mean([fa[q]["f1"] for q in qids])
    f1f = np.mean([ff[q]["f1"] for q in qids])
    ta = np.mean([fa[q]["n_hops"] for q in qids])
    print(f"\n  Overall: Adaptive F1={f1a:.3f} (hops={ta:.2f})  "
          f"Force F1={f1f:.3f} (hops={args.max_hops:.1f})", flush=True)
    print(f"  => stopping saves {args.max_hops - ta:.2f} hops/query at "
          f"{(f1a-f1f)*100:+.1f} F1 delta", flush=True)

    # save raw
    out = os.path.join(args.out_dir, f"counterfactual_{args.split}.json")
    with open(out, "w") as f:
        json.dump({"adaptive": adapt, "force": force,
                   "n_stopped": len(stopped), "n_total": len(qids)}, f, indent=2)
    print(f"\nSaved to {out}", flush=True)


if __name__ == "__main__":
    main()
