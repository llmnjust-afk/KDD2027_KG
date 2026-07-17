#!/usr/bin/env python3
"""Hyperparameter sensitivity sweep for the AdaptiveController.

Sweeps delta (early-stop threshold), theta_low (use-graph threshold), and
base_beam, running a short MetaQA subset for each. Demonstrates robustness --
the method should be stable across a wide range of hyperparameters.

Usage: python scripts/sweep.py --split 2-hop --limit 100
"""
from __future__ import annotations

import argparse
import os
import sys
import json
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agr import (load_metaqa, load_toy, build_backend, build_controller,
                 GraphRAGRetriever, generate_answer, extract_answer_entities,
                 score_query, aggregate)


def quick_eval(kg, examples, backend, cfg, limit, split):
    controller = build_controller(cfg)
    retriever = GraphRAGRetriever(kg, backend, controller, link_topk=5,
                                  judge_with_llm=True)
    from agr import QueryReport
    reports = []
    for ex in examples[:limit]:
        ret = retriever.retrieve(ex.question)
        ans = generate_answer(backend, ex.question, ret)
        pred = extract_answer_entities(ans.text)
        sc = score_query(pred, ex.answers)
        reports.append(QueryReport(qid="", question="", gold=[], pred=[],
                       hit1=sc["hit1"], f1=sc["f1"], exact=sc["exact"],
                       used_graph=True, n_hops=ret.n_hops_executed,
                       n_llm_calls=0, n_input_tokens=ret.n_input_tokens,
                       n_output_tokens=0))
    agg = aggregate(reports) if reports else None
    return {"f1": agg.mean_f1, "hit1": agg.mean_hit1,
            "toks": agg.mean_n_input_tokens, "hops": agg.mean_n_hops}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="2-hop")
    ap.add_argument("--data-dir", default="./data/MetaQA")
    ap.add_argument("--gen-model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--out-dir", default="./results/sweep")
    args = ap.parse_args()

    kg, examples = load_metaqa(args.data_dir, args.split)
    print(f"KG: {len(kg)} triples | {len(examples)} questions (using {args.limit})", flush=True)

    backend = build_backend({"kind": "hf", "gen_model": args.gen_model,
                             "embed_model": "sentence-transformers/all-MiniLM-L6-v2",
                             "device": "cuda", "dtype": "bfloat16"})

    sweeps = {
        "delta": [0.01, 0.03, 0.05, 0.10, 0.20],
        "theta_low": [0.10, 0.20, 0.30, 0.40, 0.50],
        "base_beam": [2, 4, 6, 8],
    }

    os.makedirs(args.out_dir, exist_ok=True)
    all_results = {}

    for param, values in sweeps.items():
        print(f"\n=== Sweeping {param} ===", flush=True)
        all_results[param] = {}
        for val in values:
            cfg = {"name": "adaptive", "max_hops": 3, "base_beam": 4,
                   "delta": 0.05, "theta_low": 0.30}
            cfg[param] = val
            t0 = time.time()
            r = quick_eval(kg, examples, backend, cfg, args.limit, args.split)
            label = f"{param}={val}"
            all_results[param][str(val)] = r
            print(f"  {label:<18} F1={r['f1']:.3f} Hit@1={r['hit1']:.3f} "
                  f"toks={r['toks']:.0f} hops={r['hops']:.2f}  ({time.time()-t0:.0f}s)", flush=True)

    out_path = os.path.join(args.out_dir, f"sweep_{args.split}.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to {out_path}", flush=True)


if __name__ == "__main__":
    main()
