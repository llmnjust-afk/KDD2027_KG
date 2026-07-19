#!/usr/bin/env python3
"""A4/A5: Full Pareto comparison on a FIXED subset.

Runs on the same query subset:
  - AdaptiveJudge (T=0.55): emb probe, skip LLM when confident
  - Fixed (LLM judge, K=1/2/3): the tuned frontier
  - Fixed (emb judge, K=3): the free lower bound
  - AdaptiveJudge (T=0.75): conservative operating point

All systems see identical queries, so the comparison is fair.

Usage:
  python scripts/full_pareto.py --dataset metaqa --mixed --gen-model Qwen/Qwen2.5-7B-Instruct --limit 240
  python scripts/full_pareto.py --dataset cwq --gen-model Qwen/Qwen2.5-7B-Instruct --limit 200
"""
from __future__ import annotations
import argparse, os, sys, json, time, random
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agr import (load_metaqa, build_backend, build_controller, GraphRAGRetriever,
                 generate_answer, extract_answer_entities, score_query, aggregate,
                 QueryReport, save_reports, KnowledgeGraph, Triple, QAExample)


def load_mixed_metaqa(data_dir, per, seed=0):
    _, ex1 = load_metaqa(data_dir, "1-hop")
    _, ex2 = load_metaqa(data_dir, "2-hop")
    _, ex3 = load_metaqa(data_dir, "3-hop")
    rng = random.Random(seed)
    rng.shuffle(ex1); rng.shuffle(ex2); rng.shuffle(ex3)
    examples = ex1[:per] + ex2[:per] + ex3[:per]
    rng.shuffle(examples)
    kg, _ = load_metaqa(data_dir, "3-hop")
    return kg, examples


def load_cwq(data_dir, limit, seed=0):
    import pyarrow.parquet as pq
    import pandas as pd
    data_path = os.path.join(data_dir, 'data')
    parquet_files = sorted([f for f in os.listdir(data_path) if f.startswith('test')])
    all_rows = []
    for pf in parquet_files:
        df = pq.read_table(os.path.join(data_path, pf)).to_pandas()
        all_rows.append(df)
    df = pd.concat(all_rows, ignore_index=True)
    rng = random.Random(seed)
    indices = list(range(len(df)))
    rng.shuffle(indices)
    items = []
    for i in indices[:limit]:
        row = df.iloc[i]
        q = row['question']
        ans = list(row['answer']) if row['answer'] is not None else []
        triples = []
        if row['graph'] is not None:
            for edge in row['graph']:
                if len(edge) >= 3:
                    triples.append(Triple(str(edge[0]), str(edge[1]), str(edge[2])))
        kg = KnowledgeGraph.from_triples(triples)
        ex = QAExample(qid=str(row['id']), question=q,
                       answers=[str(a) for a in ans], n_hop=2)
        items.append((kg, ex))
    return items


def run_system(backend, kg, examples, cfg, label, max_hops=3, beam=4,
               judge_with_llm=True, adaptive_judge=False, adj_threshold=0.55,
               per_query_kg=None, limit=None):
    controller = build_controller(cfg)
    reports = []
    n = len(examples) if limit is None else min(limit, len(examples))
    for i, ex in enumerate(examples[:n]):
        qkg = per_query_kg[i] if per_query_kg is not None else kg
        retriever = GraphRAGRetriever(qkg, backend, controller, link_topk=5,
                                      judge_with_llm=judge_with_llm,
                                      adaptive_judge=adaptive_judge,
                                      adaptive_judge_threshold=adj_threshold)
        t0 = time.time(); c0 = backend.usage.n_calls; tk0 = backend.usage.n_input_tokens
        ret = retriever.retrieve(ex.question)
        ans = generate_answer(backend, ex.question, ret)
        pred = extract_answer_entities(ans.text)
        sc = score_query(pred, ex.answers)
        reports.append(QueryReport(
            qid=ex.qid, question=ex.question, gold=ex.answers, pred=pred,
            hit1=sc["hit1"], f1=sc["f1"], exact=sc["exact"],
            used_graph=ret.used_graph, n_hops=ret.n_hops_executed,
            n_llm_calls=backend.usage.n_calls - c0,
            n_input_tokens=backend.usage.n_input_tokens - tk0,
            n_output_tokens=ans.n_output_tokens, stop_reasons=[]))
        if (i+1) % 50 == 0 or i == n-1:
            a = aggregate(reports)
            print(f"  [{i+1}/{n}] {label:<28} F1={a.mean_f1:.3f} toks={a.mean_n_input_tokens:.0f} "
                  f"calls={a.mean_n_llm_calls:.2f} ({time.time()-t0:.1f}s)", flush=True)
    agg = aggregate(reports) if reports else None
    stats = dict(getattr(retriever, '_judge_stats', {}))
    return agg, reports, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["metaqa", "cwq"], default="metaqa")
    ap.add_argument("--mixed", action="store_true")
    ap.add_argument("--data-dir", default="./data/MetaQA")
    ap.add_argument("--gen-model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--limit", type=int, default=240)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default="./results_full_pareto")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # load data ONCE (fixed subset for all systems)
    per_query_kg = None
    if args.dataset == "metaqa" and args.mixed:
        per = args.limit // 3
        kg, examples = load_mixed_metaqa(args.data_dir, per, seed=args.seed)
        tag = "metaqa_mixed"
    elif args.dataset == "metaqa":
        kg, examples = load_metaqa(args.data_dir, "2-hop")
        tag = "metaqa_2hop"
    elif args.dataset == "cwq":
        items = load_cwq(args.data_dir, args.limit, seed=args.seed)
        kg = items[0][0]
        examples = [ex for _, ex in items]
        per_query_kg = [k for k, _ in items]
        tag = "cwq"
    print(f"Dataset: {tag} | {len(examples)} queries | model: {args.gen_model}", flush=True)

    # system configs: (label, cfg, max_hops, beam, judge_with_llm, adaptive_judge, adj_threshold)
    SYSTEMS = [
        ("AdaptiveJudge T=0.55", {"name":"fixed","beam":4,"max_hops":3}, 3, 4, True, True, 0.55),
        ("AdaptiveJudge T=0.75", {"name":"fixed","beam":4,"max_hops":3}, 3, 4, True, True, 0.75),
        ("Fixed-LLM K=3 B=4", {"name":"fixed","beam":4,"max_hops":3}, 3, 4, True, False, 0.55),
        ("Fixed-LLM K=2 B=4", {"name":"fixed","beam":4,"max_hops":2}, 2, 4, True, False, 0.55),
        ("Fixed-LLM K=2 B=8", {"name":"fixed","beam":8,"max_hops":2}, 2, 8, True, False, 0.55),
        ("Fixed-LLM K=1 B=4", {"name":"fixed","beam":4,"max_hops":1}, 1, 4, True, False, 0.55),
        ("Fixed-emb K=3", {"name":"fixed","beam":4,"max_hops":3}, 3, 4, False, False, 0.55),
    ]

    results = {}
    for label, cfg, mh, bm, jllm, adj, at in SYSTEMS:
        print(f"\n=== {label} ===", flush=True)
        # fresh backend per system
        backend = build_backend({"kind":"hf","gen_model":args.gen_model,
                                 "embed_model":"sentence-transformers/all-MiniLM-L6-v2",
                                 "device":"cuda","dtype":"bfloat16"})
        agg, reps, stats = run_system(backend, kg, examples, cfg, label,
                                      max_hops=mh, beam=bm, judge_with_llm=jllm,
                                      adaptive_judge=adj, adj_threshold=at,
                                      per_query_kg=per_query_kg, limit=args.limit)
        if agg:
            results[label] = {"f1":agg.mean_f1, "hit1":agg.mean_hit1,
                              "toks":agg.mean_n_input_tokens, "hops":agg.mean_n_hops,
                              "calls":agg.mean_n_llm_calls, "judge_stats":stats}
            sname = label.replace(" ","_").replace("=","").replace("(","").replace(")","")
            save_reports(reps, os.path.join(args.out_dir, f"{tag}_{sname}", "reports.jsonl"))

    # Pareto table
    print(f"\n{'='*80}\nFULL PARETO [{tag}, n={args.limit}, {args.gen_model}]\n{'='*80}", flush=True)
    print(f"{'system':<28} {'F1':>6} {'Hit@1':>6} {'toks':>6} {'calls':>5} {'hops':>5} {'judge':>16}", flush=True)
    print("-"*70, flush=True)
    rows = sorted(results.items(), key=lambda kv: kv[1]["toks"])
    for lab, r in rows:
        js = r.get("judge_stats", {})
        js_str = f"emb:{js.get('emb_only',0)} llm:{js.get('llm',0)}" if js else "--"
        dom = any(r2["f1"]>=r["f1"] and r2["toks"]<=r["toks"] and
                  (r2["f1"]>r["f1"] or r2["toks"]<r["toks"])
                  for l2,r2 in results.items() if l2!=lab)
        mark = "*" if dom else " "
        print(f"{mark}{lab:<27} {r['f1']:>6.3f} {r['hit1']:>6.3f} {r['toks']:>6.0f} "
              f"{r['calls']:>5.2f} {r['hops']:>5.2f} {js_str:>16}", flush=True)
    print("\n(* = dominated)", flush=True)

    out = os.path.join(args.out_dir, f"full_pareto_{tag}.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out}", flush=True)


if __name__ == "__main__":
    main()
