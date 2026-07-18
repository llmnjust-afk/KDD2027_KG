#!/usr/bin/env python3
"""Adaptive B=8 sweep on the mixed stream (the tie-breaker experiment).

B-tier showed Fixed K=2 B=8 (0.474/1103) dominates Adaptive B=4 (0.395/1084).
Hypothesis: Adaptive's base_beam=4 is too narrow — at high-entropy hops it
needs a wider beam to compete. This runs Adaptive with B in {4,6,8,10} on the
SAME mixed stream to test whether a wider base_beam lets Adaptive reach or
beat the K=2 B=8 frontier.

Usage: python scripts/adaptive_beam_sweep.py --gen-model Qwen/Qwen2.5-7B-Instruct --per 80
"""
from __future__ import annotations
import argparse, json, os, sys, time, random
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agr import (load_metaqa, build_backend, build_controller, GraphRAGRetriever,
                 generate_answer, extract_answer_entities, score_query, aggregate)


def build_mixed(data_dir, per, seed=0):
    _, ex1 = load_metaqa(data_dir, "1-hop")
    _, ex2 = load_metaqa(data_dir, "2-hop")
    _, ex3 = load_metaqa(data_dir, "3-hop")
    rng = random.Random(seed)
    rng.shuffle(ex1); rng.shuffle(ex2); rng.shuffle(ex3)
    examples = [(ex, 1) for ex in ex1[:per]] + [(ex, 2) for ex in ex2[:per]] + [(ex, 3) for ex in ex3[:per]]
    rng.shuffle(examples)
    return examples


def run_adaptive(kg, backend, examples, base_beam, max_hops=3):
    cfg = {"name": "adaptive", "base_beam": base_beam, "max_hops": max_hops,
           "theta_low": 0.30, "delta": 0.05, "min_beam": 1, "max_beam": max(8, base_beam * 2)}
    controller = build_controller(cfg)
    retriever = GraphRAGRetriever(kg, backend, controller, link_topk=5, judge_with_llm=True)
    reports = []
    for i, (ex, depth) in enumerate(examples):
        c0 = backend.usage.n_calls; tk0 = backend.usage.n_input_tokens
        ret = retriever.retrieve(ex.question)
        ans = generate_answer(backend, ex.question, ret)
        pred = extract_answer_entities(ans.text)
        sc = score_query(pred, ex.answers)
        reports.append({"qid": ex.qid, "f1": sc["f1"], "hit1": sc["hit1"],
                        "n_hops": ret.n_hops_executed,
                        "toks": backend.usage.n_input_tokens - tk0})
        if (i + 1) % 50 == 0 or i == len(examples) - 1:
            f1 = np.mean([r["f1"] for r in reports]); t = np.mean([r["toks"] for r in reports])
            nh = np.mean([r["n_hops"] for r in reports])
            print(f"  [{i+1}/{len(examples)}] B={base_beam} F1={f1:.3f} toks={t:.0f} hops={nh:.2f} ({time.time():.0f})", flush=True)
    return reports


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="./data/MetaQA")
    ap.add_argument("--gen-model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--per", type=int, default=80)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--beams", nargs="+", type=int, default=[4, 6, 8, 10])
    ap.add_argument("--out-dir", default="./results_adaptive_beam")
    args = ap.parse_args()

    examples = build_mixed(args.data_dir, args.per, seed=args.seed)
    kg, _ = load_metaqa(args.data_dir, "3-hop")
    print(f"Mixed: {len(examples)} queries, beams={args.beams}", flush=True)
    backend = build_backend({"kind": "hf", "gen_model": args.gen_model,
                             "embed_model": "sentence-transformers/all-MiniLM-L6-v2",
                             "device": "cuda", "dtype": "bfloat16"})
    os.makedirs(args.out_dir, exist_ok=True)
    results = {}
    for B in args.beams:
        print(f"\n=== Adaptive base_beam={B} ===", flush=True)
        reps = run_adaptive(kg, backend, examples, B)
        agg_f1 = np.mean([r["f1"] for r in reps]); agg_t = np.mean([r["toks"] for r in reps])
        agg_h = np.mean([r["n_hops"] for r in reps])
        results[f"Adaptive B={B}"] = {"f1": float(agg_f1), "toks": float(agg_t), "hops": float(agg_h)}
        print(f"  -> F1={agg_f1:.3f} toks={agg_t:.0f} hops={agg_h:.2f}", flush=True)

    # compare to K=2 B=8 (known from B-tier)
    print("\n=== COMPARISON vs Fixed K=2 B=8 (B-tier: F1=0.474, toks=1103) ===", flush=True)
    for lab, r in sorted(results.items(), key=lambda kv: kv[1]["toks"]):
        dom_by_k2b8 = (0.474 >= r["f1"] and 1103 <= r["toks"])
        mark = "DOMINATED by K=2 B=8" if dom_by_k2b8 else "NOT dominated by K=2 B=8"
        print(f"  {lab}: F1={r['f1']:.3f} toks={r['toks']:.0f}  [{mark}]", flush=True)

    out = os.path.join(args.out_dir, "adaptive_beam_sweep.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out}", flush=True)


if __name__ == "__main__":
    main()
