#!/usr/bin/env python3
"""P0.1: Does 'embedding judge >= LLM judge' survive on a STRONGER engine?

Our weak engine truncated candidates in arbitrary traversal order (cap=16),
which can hide the answer triple from the LLM judge on dense KGs (CWQ). A
stronger, ToG-2.0-style engine ranks ALL candidates by embedding similarity
first, then judges the top-cap. This script re-runs the emb-vs-LLM-judge
comparison WITH the strong prefilter enabled, on MetaQA 1/2/3-hop and CWQ.

If embedding judge still matches/beats LLM judge under the strong engine, the
paper's central claim is robust. If the LLM judge now clearly wins (because it
finally sees the answer candidates), the claim must be scoped to weak engines.

Usage:
  python scripts/strong_engine.py --gen-model Qwen/Qwen2.5-7B-Instruct --limit 200
"""
from __future__ import annotations
import argparse, os, sys, json, time, random
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agr import (load_metaqa, build_backend, build_controller, GraphRAGRetriever,
                 generate_answer, extract_answer_entities, score_query,
                 KnowledgeGraph, Triple, QAExample)


def load_cwq(data_dir, limit, seed=0):
    import pyarrow.parquet as pq, pandas as pd
    dp = os.path.join(data_dir, 'data')
    dfs = [pq.read_table(os.path.join(dp, f)).to_pandas()
           for f in sorted(os.listdir(dp)) if f.startswith('test')]
    df = pd.concat(dfs, ignore_index=True)
    idx = list(range(len(df))); random.Random(seed).shuffle(idx)
    items = []
    for i in idx[:limit]:
        row = df.iloc[i]
        triples = []
        if row['graph'] is not None:
            for e in row['graph']:
                if len(e) >= 3:
                    triples.append(Triple(str(e[0]), str(e[1]), str(e[2])))
        kg = KnowledgeGraph.from_triples(triples)
        ans = row['answer']
        ans = list(ans) if ans is not None else []
        ex = QAExample(qid=str(row['id']), question=row['question'],
                       answers=[str(a) for a in ans], n_hop=2)
        items.append((kg, ex))
    return items


def run(kg, backend, examples, judge_with_llm, strong, per_query_kg=None,
        max_hops=3, beam=4, limit=None):
    ctrl = build_controller({"name": "fixed", "beam": beam, "max_hops": max_hops})
    f1s, toks = [], []
    n = len(examples) if limit is None else min(limit, len(examples))
    for i, ex in enumerate(examples[:n]):
        qkg = per_query_kg[i] if per_query_kg is not None else kg
        ret = GraphRAGRetriever(qkg, backend, ctrl, link_topk=5,
                                judge_with_llm=judge_with_llm, strong_prefilter=strong)
        tk0 = backend.usage.n_input_tokens
        r = ret.retrieve(ex.question)
        a = generate_answer(backend, ex.question, r)
        sc = score_query(extract_answer_entities(a.text), ex.answers)
        f1s.append(sc["f1"]); toks.append(backend.usage.n_input_tokens - tk0)
        if (i+1) % 50 == 0:
            print(f"    {i+1}/{n} F1={np.mean(f1s):.3f}", flush=True)
    return float(np.mean(f1s)), float(np.mean(toks))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="./data/MetaQA")
    ap.add_argument("--cwq-dir", default="./data/CWQ")
    ap.add_argument("--gen-model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--out-dir", default="./results_strong_engine")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # build settings: (name, kg, examples, per_query_kg, max_hops)
    settings = []
    for split in ["1-hop", "2-hop", "3-hop"]:
        kg, exs = load_metaqa(args.data_dir, split)
        random.Random(0).shuffle(exs)
        settings.append((f"MetaQA {split}", kg, exs[:args.limit], None, 3))
    cwq = load_cwq(args.cwq_dir, args.limit)
    settings.append(("CWQ", cwq[0][0], [e for _, e in cwq], [k for k, _ in cwq], 3))

    results = {}
    for name, kg, exs, pqk, mh in settings:
        print(f"\n=== {name} (n={len(exs)}) ===", flush=True)
        backend = build_backend({"kind": "hf", "gen_model": args.gen_model,
                                 "embed_model": "sentence-transformers/all-MiniLM-L6-v2",
                                 "device": "cuda", "dtype": "bfloat16"})
        row = {}
        # weak (traversal-order cap) vs strong (embedding prefilter), emb & LLM judge
        for jname, jllm in [("emb", False), ("llm", True)]:
            for sname, strong in [("weak", False), ("strong", True)]:
                print(f"  -- {jname} judge / {sname} engine --", flush=True)
                f1, tk = run(kg, backend, exs, jllm, strong, per_query_kg=pqk,
                             max_hops=mh, limit=args.limit)
                row[f"{jname}_{sname}"] = {"f1": f1, "toks": tk}
                print(f"     F1={f1:.3f} toks={tk:.0f}", flush=True)
        results[name] = row

    print(f"\n{'='*78}\nSTRONG-ENGINE: emb vs LLM judge (weak vs strong prefilter)\n{'='*78}", flush=True)
    print(f"{'setting':<14} {'emb-weak':>10} {'emb-strong':>11} {'llm-weak':>10} {'llm-strong':>11}", flush=True)
    for name, row in results.items():
        def g(k): return f"{row[k]['f1']:.3f}"
        print(f"{name:<14} {g('emb_weak'):>10} {g('emb_strong'):>11} "
              f"{g('llm_weak'):>10} {g('llm_strong'):>11}", flush=True)
    # verdict: on strong engine, does LLM judge now beat emb?
    print(f"\n{'setting':<14} {'winner (strong engine)':>24}", flush=True)
    for name, row in results.items():
        e, l = row['emb_strong']['f1'], row['llm_strong']['f1']
        w = "LLM" if l > e + 0.005 else ("emb" if e > l + 0.005 else "tie")
        print(f"{name:<14} emb={e:.3f} llm={l:.3f} -> {w}", flush=True)

    with open(os.path.join(args.out_dir, "strong_engine.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {args.out_dir}/strong_engine.json", flush=True)


if __name__ == "__main__":
    main()
