#!/usr/bin/env python3
"""WebQSP pilot experiment (B3): test AquaRAG on a second KG.

Uses RoG-webqsp (Freebase subgraphs) to verify the cost saving transfers
beyond WikiMovies. Each question comes with a pre-extracted subgraph;
we treat that subgraph as the per-query KG and run the same engine.
"""
from __future__ import annotations
import argparse, os, sys, json, time, random
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agr import (build_backend, GraphRAGRetriever, generate_answer,
                 extract_answer_entities, score_query, aggregate,
                 pareto_table, save_reports, QueryReport, KnowledgeGraph, Triple,
                 FixedController, AdaptiveController, QAExample)


def load_webqsp(data_dir, split='test', limit=None):
    """Load WebQSP from parquet, build per-query KG + examples."""
    import pyarrow.parquet as pq
    # find parquet files
    parquet_files = sorted([f for f in os.listdir(os.path.join(data_dir, 'data'))
                           if f.startswith(split)])
    all_rows = []
    for pf in parquet_files:
        df = pq.read_table(os.path.join(data_dir, 'data', pf)).to_pandas()
        all_rows.append(df)
    import pandas as pd
    df = pd.concat(all_rows, ignore_index=True)
    if limit:
        df = df.head(limit)

    examples = []
    subgraphs = []
    for _, row in df.iterrows():
        q = row['question']
        ans = list(row['answer']) if row['answer'] is not None else []
        q_ents = list(row['q_entity']) if row['q_entity'] is not None else []

        # build KG from graph field (list of [head, relation, tail] arrays)
        triples = []
        if row['graph'] is not None:
            for edge in row['graph']:
                if len(edge) >= 3:
                    h, r, t = str(edge[0]), str(edge[1]), str(edge[2])
                    triples.append(Triple(h, r, t))

        # dedup entities for linking
        ent_set = set()
        for t in triples:
            ent_set.add(t.head)
            ent_set.add(t.tail)
        # add q_entity if not in graph
        for qe in q_ents:
            ent_set.add(str(qe))

        kg = KnowledgeGraph.from_triples(triples)
        examples.append(QAExample(
            qid=str(row['id']), question=q, answers=[str(a) for a in ans],
            n_hop=2  # WebQSP is mostly 1-2 hop
        ))
        subgraphs.append(kg)

    return subgraphs, examples


def run_webqsp_system(name, subgraphs, examples, backend, controller_cls, limit, **ctrl_kwargs):
    """Run a controller on WebQSP (per-query KG)."""
    reports = []
    n = min(limit, len(examples))
    for i in range(n):
        kg = subgraphs[i]
        ex = examples[i]
        # skip if KG is empty
        if len(kg) == 0:
            reports.append(QueryReport(qid=ex.qid, question=ex.question, gold=ex.answers,
                pred=[], hit1=0, f1=0, exact=0, used_graph=False, n_hops=0,
                n_llm_calls=0, n_input_tokens=0, n_output_tokens=0))
            continue

        ctrl = controller_cls(**ctrl_kwargs)
        ret = GraphRAGRetriever(kg, backend, ctrl, link_topk=3, judge_with_llm=True)

        t0 = time.time()
        c0 = backend.usage.n_calls; tk0 = backend.usage.n_input_tokens
        r = ret.retrieve(ex.question)
        a = generate_answer(backend, ex.question, r)
        p = extract_answer_entities(a.text)
        sc = score_query(p, ex.answers)
        reports.append(QueryReport(qid=ex.qid, question=ex.question, gold=ex.answers,
            pred=p, hit1=sc["hit1"], f1=sc["f1"], exact=sc["exact"],
            used_graph=r.used_graph, n_hops=r.n_hops_executed,
            n_llm_calls=backend.usage.n_calls-c0, n_input_tokens=backend.usage.n_input_tokens-tk0,
            n_output_tokens=a.n_output_tokens))
        if (i+1) % 25 == 0 or i == n-1:
            agg = aggregate(reports)
            print(f"  [{name} {i+1}/{n}] F1={agg.mean_f1:.3f} toks={agg.mean_n_input_tokens:.0f} "
                  f"hops={agg.mean_n_hops:.2f} t={time.time()-t0:.1f}s", flush=True)
    return reports


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="./data/WebQSP")
    ap.add_argument("--gen-model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--out-dir", default="./results_webqsp")
    args = ap.parse_args()

    # download if needed
    if not os.path.exists(args.data_dir):
        os.makedirs(args.data_dir, exist_ok=True)
        os.makedirs(os.path.join(args.data_dir, 'data'), exist_ok=True)
        from huggingface_hub import hf_hub_download
        import shutil
        for split in ['test']:
            for i in range(2):
                fname = f'data/{split}-{i:05d}-of-00002-' + ['9ee8d68f7d951e1f', '773a7b8213e159f5'][i] + '.parquet'
                try:
                    tmp = hf_hub_download(repo_id='rmanluo/RoG-webqsp', filename=fname, repo_type='dataset')
                    shutil.copy(tmp, os.path.join(args.data_dir, fname))
                    print(f"downloaded {fname}", flush=True)
                except Exception as e:
                    print(f"failed {fname}: {e}", flush=True)

    print("Loading WebQSP...", flush=True)
    subgraphs, examples = load_webqsp(args.data_dir, 'test', args.limit)
    print(f"Loaded {len(examples)} WebQSP questions (avg KG size: {np.mean([len(sg) for sg in subgraphs]):.0f} triples)", flush=True)

    backend = build_backend({"kind": "hf", "gen_model": args.gen_model,
        "embed_model": "sentence-transformers/all-MiniLM-L6-v2",
        "device": "cuda", "dtype": "bfloat16"})
    print(f"Backend: {args.gen_model}", flush=True)

    system_aggs = {}

    print("\n=== Fixed (K=3) ===", flush=True)
    r_fixed = run_webqsp_system("Fixed", subgraphs, examples, backend,
                                FixedController, args.limit, max_hops=3, beam=4)
    system_aggs["Fixed (K=3)"] = aggregate(r_fixed)

    print("\n=== Adaptive (ours) ===", flush=True)
    r_ada = run_webqsp_system("Adaptive", subgraphs, examples, backend,
                              AdaptiveController, args.limit, max_hops=3, base_beam=4)
    system_aggs["Adaptive (ours)"] = aggregate(r_ada)

    os.makedirs(args.out_dir, exist_ok=True)
    for name, reps in [("fixed", r_fixed), ("adaptive", r_ada)]:
        save_reports(reps, os.path.join(args.out_dir, f"webqsp_{name}", "reports.jsonl"))

    print("\n" + "=" * 70, flush=True)
    print(f"WebQSP COMPARISON [n={args.limit}]", flush=True)
    print("=" * 70, flush=True)
    print(pareto_table(system_aggs), flush=True)


if __name__ == "__main__":
    main()
