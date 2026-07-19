#!/usr/bin/env python3
"""P1.4: Natural-heterogeneous benchmark experiment (CWQ).

CWQ (Complex WebQuestions) has naturally varying question complexity
(1-4 hop compositions, comparatives, superlatives) — unlike MetaQA's
template-based artificial hop splits. This is the test of whether
per-query adaptation beats a tuned global budget when difficulty is
genuinely heterogeneous.

We run on CWQ with the SAME engine as MetaQA:
  - Per-query KG from RoG-cwq pre-extracted Freebase subgraphs
  - Fixed (default K=3) vs Adaptive vs tuned Fixed-K sweep
  - Report accuracy-cost Pareto frontier

Key hypothesis: on natural-heterogeneous data, no single K is optimal
for all queries, so per-query adaptation (Adaptive) can enter the
Pareto frontier — unlike MetaQA where K=2 is globally near-optimal.

Usage:
  python scripts/cwq_experiment.py --gen-model Qwen/Qwen2.5-7B-Instruct --limit 200
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agr import (build_backend, build_controller, GraphRAGRetriever,
                 generate_answer, extract_answer_entities, score_query,
                 aggregate, KnowledgeGraph, Triple, QAExample, QueryReport,
                 save_reports)


def load_rog_dataset(data_dir, split='test', limit=None):
    """Load a RoG-style dataset (WebQSP/CWQ) from parquet.

    Each row has: id, question, answer, q_entity, graph (list of [h,r,t]).
    Returns: list of (kg, example) tuples — one per query.
    """
    import pyarrow.parquet as pq
    import pandas as pd
    data_path = os.path.join(data_dir, 'data')
    parquet_files = sorted([f for f in os.listdir(data_path) if f.startswith(split)])
    all_rows = []
    for pf in parquet_files:
        df = pq.read_table(os.path.join(data_path, pf)).to_pandas()
        all_rows.append(df)
    df = pd.concat(all_rows, ignore_index=True)
    if limit:
        df = df.head(limit)

    items = []
    for _, row in df.iterrows():
        q = row['question']
        ans = list(row['answer']) if row['answer'] is not None else []
        q_ents = list(row['q_entity']) if row['q_entity'] is not None else []
        triples = []
        if row['graph'] is not None:
            for edge in row['graph']:
                if len(edge) >= 3:
                    h, r, t = str(edge[0]), str(edge[1]), str(edge[2])
                    triples.append(Triple(h, r, t))
        kg = KnowledgeGraph.from_triples(triples)
        ex = QAExample(
            qid=str(row['id']), question=q,
            answers=[str(a) for a in ans],
            n_hop=2,  # CWQ varies 1-4; we don't know per-query
        )
        items.append((kg, ex))
    return items


def run_system(kg, backend, examples, cfg, label, max_hops=3, beam=4,
               per_query_kg=None, limit=None):
    """Run one system config; return (agg, reports).

    IMPORTANT: when per_query_kg is set, we must build a fresh retriever per
    query, because the entity index is cached on the retriever and is KG-specific.
    """
    controller = build_controller(cfg)
    reports = []
    n = len(examples) if limit is None else min(limit, len(examples))
    for i, ex in enumerate(examples[:n]):
        # build a fresh retriever per query when using per-query KGs
        if per_query_kg is not None:
            qkg = per_query_kg[i]
            retriever = GraphRAGRetriever(qkg, backend, controller, link_topk=5,
                                          judge_with_llm=True)
        else:
            retriever = GraphRAGRetriever(kg, backend, controller, link_topk=5,
                                          judge_with_llm=True)
        t0 = time.time()
        call0 = backend.usage.n_calls
        tok0 = backend.usage.n_input_tokens
        ret = retriever.retrieve(ex.question)
        ans = generate_answer(backend, ex.question, ret)
        pred = extract_answer_entities(ans.text)
        sc = score_query(pred, ex.answers)
        reports.append(QueryReport(
            qid=ex.qid, question=ex.question, gold=ex.answers, pred=pred,
            hit1=sc["hit1"], f1=sc["f1"], exact=sc["exact"],
            used_graph=ret.used_graph, n_hops=ret.n_hops_executed,
            n_llm_calls=backend.usage.n_calls - call0,
            n_input_tokens=backend.usage.n_input_tokens - tok0,
            n_output_tokens=ans.n_output_tokens, stop_reasons=[],
        ))
        if (i + 1) % 25 == 0 or i == n - 1:
            a = aggregate(reports)
            print(f"  [{i+1}/{n}] {label:<28} F1={a.mean_f1:.3f} "
                  f"toks={a.mean_n_input_tokens:.0f} hops={a.mean_n_hops:.2f} "
                  f"({time.time()-t0:.1f}s)", flush=True)
    return (aggregate(reports) if reports else None, reports)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="./data/CWQ")
    ap.add_argument("--split", default="test")
    ap.add_argument("--gen-model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--max-hops", type=int, default=3)
    ap.add_argument("--out-dir", default="./results_cwq")
    ap.add_argument("--systems", nargs="+",
                    default=["adaptive", "fixed-k3", "fixed-k2-b6", "fixed-k2-b8"])
    args = ap.parse_args()

    print(f"Loading CWQ from {args.data_dir}...", flush=True)
    items = load_rog_dataset(args.data_dir, args.split, args.limit)
    print(f"Loaded {len(items)} CWQ questions "
          f"(avg KG: {np.mean([len(kg.triples) for kg, _ in items]):.0f} triples)", flush=True)

    # use a dummy shared KG for backend init; per-query KGs set in run_system
    dummy_kg = items[0][0]
    examples = [ex for _, ex in items]
    per_query_kg = [kg for kg, _ in items]

    backend = build_backend({"kind": "hf", "gen_model": args.gen_model,
                             "embed_model": "sentence-transformers/all-MiniLM-L6-v2",
                             "device": "cuda", "dtype": "bfloat16"})
    print(f"Backend: {args.gen_model}", flush=True)
    os.makedirs(args.out_dir, exist_ok=True)

    # system configs
    SYSTEMS = {
        "adaptive": ({"name": "adaptive", "base_beam": 4, "max_hops": args.max_hops,
                      "theta_low": 0.30, "delta": 0.05}, "Adaptive (ours)"),
        "fixed-k3": ({"name": "fixed", "beam": 4, "max_hops": 3}, "Fixed K=3 B=4"),
        "fixed-k2-b4": ({"name": "fixed", "beam": 4, "max_hops": 2}, "Fixed K=2 B=4"),
        "fixed-k2-b6": ({"name": "fixed", "beam": 6, "max_hops": 2}, "Fixed K=2 B=6"),
        "fixed-k2-b8": ({"name": "fixed", "beam": 8, "max_hops": 2}, "Fixed K=2 B=8"),
        "fixed-k1": ({"name": "fixed", "beam": 4, "max_hops": 1}, "Fixed K=1 B=4"),
        "patience": ({"name": "patience", "beam": 4, "patience": 1, "max_hops": args.max_hops}, "Patience p=1"),
    }

    results = {}
    for sname in args.systems:
        if sname not in SYSTEMS:
            print(f"  unknown system '{sname}', skipping", flush=True)
            continue
        cfg, label = SYSTEMS[sname]
        print(f"\n=== {label} ===", flush=True)
        agg, reps = run_system(dummy_kg, backend, examples, cfg, label,
                               max_hops=cfg.get("max_hops", args.max_hops),
                               beam=cfg.get("beam", 4),
                               per_query_kg=per_query_kg, limit=args.limit)
        if agg:
            results[label] = {
                "f1": agg.mean_f1, "hit1": agg.mean_hit1, "exact": agg.mean_exact,
                "toks": agg.mean_n_input_tokens, "hops": agg.mean_n_hops,
                "calls": agg.mean_n_llm_calls, "n": agg.n,
            }
            save_reports(reps, os.path.join(args.out_dir,
                         f"cwq_{sname}", "reports.jsonl"))

    # Pareto table
    print("\n" + "=" * 72, flush=True)
    print(f"CWQ PARETO COMPARISON (n={args.limit}, {args.gen_model})", flush=True)
    print("=" * 72, flush=True)
    rows = sorted(results.items(), key=lambda kv: kv[1]["toks"])
    print(f"{'system':<28} {'F1':>6} {'Hit@1':>6} {'toks':>6} {'hops':>5}", flush=True)
    print("-" * 55, flush=True)
    nondom = []
    for lab, r in rows:
        dom = any(r2["f1"] >= r["f1"] and r2["toks"] <= r["toks"] and
                  (r2["f1"] > r["f1"] or r2["toks"] < r["toks"])
                  for l2, r2 in results.items() if l2 != lab)
        nondom.append((lab, r, dom))
    for lab, r, dom in nondom:
        mark = " *" if dom else "  "
        print(f"{mark}{lab:<26} {r['f1']:>6.3f} {r['hit1']:>6.3f} "
              f"{r['toks']:>6.0f} {r['hops']:>5.2f}", flush=True)

    out = os.path.join(args.out_dir, "cwq_pareto.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out}", flush=True)


if __name__ == "__main__":
    main()
