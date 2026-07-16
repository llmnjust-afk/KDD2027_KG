#!/usr/bin/env python3
"""Main runner: evaluate one or more retrieval systems (controllers) on a
dataset, emit per-system reports + a Pareto table.

A "system" is a named controller configuration. The baseline (FixedController)
and our method (AdaptiveController) run through the *same* engine -- the only
difference is the controller, which is exactly what a clean ablation needs.

Systems are defined on the CLI via --systems NAME=spec; a "spec" is a short
control string. Recognised systems:
    fixed            naive ToG-style baseline (fixed beam, always graph, no early stop)
    adaptive         our full method (2a)+(2b)+(3)
    abl-nograph      ablation: adaptive with (2a) disabled
    abl-fixbeam      ablation: adaptive with (2b) disabled
    abl-nostop       ablation: adaptive with (3) disabled

Usage:
    # toy smoke test (embedding judge only, no GPU model load)
    python run.py --dataset toy --systems fixed adaptive --no-llm-judge --no-gen-model

    # MetaQA 1-hop, full method + baseline
    python run.py --dataset metaqa --split 1-hop --data-dir ./data/MetaQA \
        --gen-model Qwen/Qwen2.5-1.5B-Instruct --limit 500 \
        --systems fixed adaptive

    # full ablation suite
    python run.py --dataset metaqa --split 2-hop --data-dir ./data/MetaQA \
        --gen-model Qwen/Qwen2.5-1.5B-Instruct --limit 500 \
        --systems fixed adaptive abl-nograph abl-fixbeam abl-nostop

Outputs:
    results/<dataset>_<split>_<system>/reports.jsonl
    results/pareto_<dataset>_<split>.txt
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agr import (load_toy, load_metaqa, build_backend, build_controller,
                 GraphRAGRetriever, generate_answer, extract_answer_entities,
                 score_query, aggregate, pareto_table, save_reports, QueryReport)


SYSTEMS = {
    # name -> (controller cfg, human label for the Pareto table)
    "fixed":       ({"name": "fixed", "beam": 4}, "Fixed (ToG-style)"),
    "adaptive":    ({"name": "adaptive", "base_beam": 4}, "Adaptive (ours)"),
    "abl-nograph": ({"name": "adaptive", "base_beam": 4, "ablate_use_graph": True},    "Abl. no (2a)"),
    "abl-fixbeam": ({"name": "adaptive", "base_beam": 4, "ablate_adaptive_beam": True}, "Abl. no (2b)"),
    "abl-nostop":  ({"name": "adaptive", "base_beam": 4, "ablate_early_stop": True},   "Abl. no (3)"),
}


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
        if (i + 1) % 50 == 0 or i == n - 1:
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
    ap.add_argument("--systems", nargs="+", default=["fixed", "adaptive"])
    ap.add_argument("--max-hops", type=int, default=3)
    ap.add_argument("--beam", type=int, default=4, help="baseline fixed beam / adaptive base_beam")
    ap.add_argument("--link-topk", type=int, default=5)
    ap.add_argument("--gen-model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--embed-model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no-llm-judge", action="store_true",
                    help="use embedding relevance instead of LLM judge (fast smoke test)")
    ap.add_argument("--no-gen-model", action="store_true",
                    help="skip loading the HF generator; relevance via embeddings only")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out-dir", default="./results")
    args = ap.parse_args()

    # ---- data ----
    if args.dataset == "toy":
        kg, examples = load_toy()
    else:
        kg, examples = load_metaqa(args.data_dir, args.split)
    print(f"KG: {len(kg)} triples, {len(kg.entities)} entities | "
          f"{len(examples)} questions (using {min(len(examples), args.limit or len(examples))})",
          flush=True)

    # ---- backend ----
    backend = build_backend({"kind": "hf", "gen_model": args.gen_model,
                             "embed_model": args.embed_model,
                             "device": args.device, "dtype": "bfloat16"})
    print(f"Backend: {backend.__class__.__name__}  gen={args.gen_model}  "
          f"judge_with_llm={not args.no_llm_judge}", flush=True)

    # ---- run each system ----
    tag = f"{args.dataset}_{args.split}"
    system_aggs = {}
    for sname in args.systems:
        if sname not in SYSTEMS:
            print(f"  unknown system '{sname}', skipping", flush=True)
            continue
        cfg, label = SYSTEMS[sname]
        cfg = dict(cfg)
        cfg["max_hops"] = args.max_hops
        if "beam" in cfg:
            cfg["beam"] = args.beam
        if "base_beam" in cfg:
            cfg["base_beam"] = args.beam
        print(f"\n=== System: {sname}  [{label}] ===", flush=True)
        controller = build_controller(cfg)
        retriever = GraphRAGRetriever(
            kg, backend, controller, link_topk=args.link_topk,
            judge_with_llm=not args.no_llm_judge,
        )
        reports = run_one(retriever, backend, examples, limit=args.limit)
        agg = aggregate(reports)
        system_aggs[label] = agg
        out_path = os.path.join(args.out_dir, f"{tag}_{sname}", "reports.jsonl")
        save_reports(reports, out_path)
        print(f"  -> {out_path}", flush=True)

    # ---- Pareto comparison ----
    print("\n" + "=" * 72, flush=True)
    print(f"PARETO COMPARISON  [{tag}]  (sorted by mean input tokens/query)", flush=True)
    print("=" * 72, flush=True)
    tbl = pareto_table(system_aggs)
    print(tbl, flush=True)
    pareto_path = os.path.join(args.out_dir, f"pareto_{tag}.txt")
    with open(pareto_path, "w") as f:
        f.write(tbl + "\n")
    print(f"\nPareto table saved to {pareto_path}", flush=True)


if __name__ == "__main__":
    main()
