#!/usr/bin/env python3
"""Main runner: evaluate one or more controllers on a dataset, emit reports +
Pareto table. Designed so the baseline (FixedController) and our method
(AdaptiveController) run through the *same* engine -- the only difference is
the controller, which is exactly what a clean ablation needs.

Usage:
    # toy smoke test (no downloads, no GPU model load -> uses embed fallback)
    python run.py --dataset toy --controllers fixed adaptive

    # MetaQA 1-hop with a local HF generator
    python run.py --dataset metaqa --split 1-hop --data-dir ./data/MetaQA \
        --gen-model Qwen/Qwen2.5-1.5B-Instruct --limit 200 \
        --controllers fixed adaptive

Outputs:
    results/<dataset>_<controller>/reports.jsonl
    results/pareto.txt
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import asdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agr import (load_toy, load_metaqa, build_backend, build_controller,
                 GraphRAGRetriever, generate_answer, extract_answer_entities,
                 score_query, aggregate, pareto_table, save_reports, QueryReport)


def run_one(retriever, backend, examples, limit=None):
    reports = []
    n = len(examples) if limit is None else min(limit, len(examples))
    for i, ex in enumerate(examples[:n]):
        t0 = time.time()
        call0 = backend.usage.n_calls
        tok0 = backend.usage.n_input_tokens
        retrieval = retriever.retrieve(ex.question)
        ans = generate_answer(backend, ex.question, retrieval)
        pred = extract_answer_entities(ans.text)
        sc = score_query(pred, ex.answers)
        reports.append(QueryReport(
            qid=ex.qid, question=ex.question, gold=ex.answers, pred=pred,
            hit1=sc["hit1"], f1=sc["f1"], exact=sc["exact"],
            used_graph=retrieval.used_graph, n_hops=retrieval.n_hops_executed,
            n_llm_calls=backend.usage.n_calls - call0,
            n_input_tokens=(backend.usage.n_input_tokens - tok0),
            n_output_tokens=ans.n_output_tokens,
            stop_reasons=[],
        ))
        if (i + 1) % 25 == 0 or i == n - 1:
            agg = aggregate(reports)
            print(f"  [{i+1}/{n}] F1={agg.mean_f1:.3f} toks/q={agg.mean_n_input_tokens:.0f} "
                  f"calls/q={agg.mean_n_llm_calls:.2f} hops/q={agg.mean_n_hops:.2f} "
                  f"t={time.time()-t0:.1f}s", flush=True)
    return reports


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["toy", "metaqa"], default="toy")
    ap.add_argument("--split", default="1-hop")
    ap.add_argument("--data-dir", default="./data/MetaQA")
    ap.add_argument("--controllers", nargs="+", default=["fixed", "adaptive"])
    ap.add_argument("--max-hops", type=int, default=3)
    ap.add_argument("--beam", type=int, default=4, help="baseline fixed beam")
    ap.add_argument("--link-topk", type=int, default=5)
    ap.add_argument("--gen-model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--embed-model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no-llm-judge", action="store_true",
                    help="use embedding relevance instead of LLM judge (fast smoke test)")
    ap.add_argument("--no-gen-model", action="store_true",
                    help="skip loading the HF generator; answer/relevance via embeddings only")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out-dir", default="./results")
    args = ap.parse_args()

    # ---- data ----
    if args.dataset == "toy":
        kg, examples = load_toy()
    else:
        kg, examples = load_metaqa(args.data_dir, args.split)
    print(f"KG: {len(kg)} triples, {len(kg.entities)} entities | "
          f"{len(examples)} questions")

    # ---- backend ----
    if args.no_gen_model:
        # embed-only smoke path: build a tiny shim so nothing heavy loads
        from agr.llm_backend import HuggingFaceBackend
        backend = HuggingFaceBackend(gen_model=args.gen_model,
                                     embed_model=args.embed_model,
                                     device=args.device)
    else:
        backend = build_backend({"kind": "hf", "gen_model": args.gen_model,
                                 "embed_model": args.embed_model,
                                 "device": args.device, "dtype": "bfloat16"})
    print(f"Backend: {backend.__class__.__name__}  (judge_with_llm={not args.no_llm_judge})")

    # ---- run each controller ----
    system_aggs = {}
    for cname in args.controllers:
        print(f"\n=== Controller: {cname} ===")
        if cname == "fixed":
            controller = build_controller({"name": "fixed", "max_hops": args.max_hops,
                                           "beam": args.beam})
        else:
            controller = build_controller({"name": "adaptive", "max_hops": args.max_hops,
                                           "base_beam": args.beam})
        retriever = GraphRAGRetriever(
            kg, backend, controller, link_topk=args.link_topk,
            judge_with_llm=not args.no_llm_judge,
        )
        reports = run_one(retriever, backend, examples, limit=args.limit)
        agg = aggregate(reports)
        system_aggs[cname] = agg
        out_path = os.path.join(args.out_dir, f"{args.dataset}_{cname}", "reports.jsonl")
        save_reports(reports, out_path)
        print(f"  -> {out_path}")

    # ---- Pareto comparison ----
    print("\n" + "=" * 70)
    print("PARETO COMPARISON (sorted by mean input tokens/query)")
    print("=" * 70)
    print(pareto_table(system_aggs))
    with open(os.path.join(args.out_dir, "pareto.txt"), "w") as f:
        f.write(pareto_table(system_aggs) + "\n")
    print(f"\nPareto table saved to {os.path.join(args.out_dir, 'pareto.txt')}")


if __name__ == "__main__":
    main()
