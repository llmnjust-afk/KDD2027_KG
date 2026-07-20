#!/usr/bin/env python3
"""P2.6: Does the predictability ceiling hold on a THIRD benchmark (WebQSP)?

Our conclusions come from MetaQA (synthetic multi-hop) and CWQ. WebQSP
(RoG Freebase subgraphs, real web questions, mostly 1-2 hop) is a third,
independent testbed with a very different KG density and question style. We
re-run the SAME depth/judge comparison used for MetaQA/CWQ, under the strong
(ToG-2.0-style embedding-prefilter) engine, so the numbers are directly
comparable and not confounded by the weak-engine candidate truncation.

Policies (identical spec to the mixed-stream study):
  Fixed-emb  K=3            -- cheap embedding-only judge
  Fixed-LLM  K=2 B=8        -- tuned global winner elsewhere
  AdaptiveJudge T=0.55      -- per-query escalation to LLM judge
plus an oracle-depth upper bound (best of K in {1,2,3} per query).

If the tuned global point again dominates the adaptive policy and the oracle
again shows unrealizable headroom, the ceiling is benchmark-general.

Usage: python scripts/webqsp_strong.py --gen-model Qwen/Qwen2.5-7B-Instruct --limit 300
"""
from __future__ import annotations
import argparse, os, sys, json, random
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agr import (build_backend, build_controller, GraphRAGRetriever, generate_answer,
                 extract_answer_entities, score_query, KnowledgeGraph, Triple, QAExample)


def load_webqsp(data_dir, limit=None, seed=0):
    import pyarrow.parquet as pq
    dp = os.path.join(data_dir, 'data')
    pfs = sorted([f for f in os.listdir(dp) if f.startswith('test') and f.endswith('.parquet')])
    import pandas as pd
    df = pd.concat([pq.read_table(os.path.join(dp, f)).to_pandas() for f in pfs],
                   ignore_index=True)
    idx = list(range(len(df))); random.Random(seed).shuffle(idx)
    if limit: idx = idx[:limit]
    subgraphs, examples = [], []
    for i in idx:
        row = df.iloc[i]
        triples = []
        if row['graph'] is not None:
            for e in row['graph']:
                if len(e) >= 3:
                    triples.append(Triple(str(e[0]), str(e[1]), str(e[2])))
        kg = KnowledgeGraph.from_triples(triples)
        ans = list(row['answer']) if row['answer'] is not None else []
        examples.append(QAExample(qid=str(row['id']), question=row['question'],
                                  answers=[str(a) for a in ans], n_hop=2))
        subgraphs.append(kg)
    return subgraphs, examples


def run_policy(subgraphs, examples, backend, *, judge_llm, beam, max_hops,
               adaptive_judge=False, adaptive_thr=0.55):
    ctrl = build_controller({"name": "fixed", "beam": beam, "max_hops": max_hops})
    f1s = []
    for kg, ex in zip(subgraphs, examples):
        ret = GraphRAGRetriever(kg, backend, ctrl, link_topk=5,
                                judge_with_llm=judge_llm, strong_prefilter=True,
                                adaptive_judge=adaptive_judge,
                                adaptive_judge_threshold=adaptive_thr)
        r = ret.retrieve(ex.question)
        a = generate_answer(backend, ex.question, r)
        f1s.append(score_query(extract_answer_entities(a.text), ex.answers)["f1"])
    return float(np.mean(f1s)), f1s


def run_oracle_depth(subgraphs, examples, backend, beam=4):
    """Best F1 over K in {1,2,3} per query, LLM judge, strong engine."""
    per_k = {}
    for k in [1, 2, 3]:
        ctrl = build_controller({"name": "fixed", "beam": beam, "max_hops": k})
        f1s = []
        for kg, ex in zip(subgraphs, examples):
            ret = GraphRAGRetriever(kg, backend, ctrl, link_topk=5,
                                    judge_with_llm=True, strong_prefilter=True)
            r = ret.retrieve(ex.question)
            a = generate_answer(backend, ex.question, r)
            f1s.append(score_query(extract_answer_entities(a.text), ex.answers)["f1"])
        per_k[k] = f1s
        print(f"    K={k}: {np.mean(f1s):.3f}", flush=True)
    oracle = np.max(np.vstack([per_k[k] for k in [1, 2, 3]]), axis=0)
    return float(oracle.mean()), {f"K={k}": float(np.mean(v)) for k, v in per_k.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="./data/WebQSP")
    ap.add_argument("--gen-model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--out-dir", default="./results_webqsp_strong")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    subgraphs, examples = load_webqsp(args.data_dir, args.limit)
    print(f"WebQSP n={len(examples)} avg KG={np.mean([len(sg) for sg in subgraphs]):.0f} triples",
          flush=True)

    backend = build_backend({"kind": "hf", "gen_model": args.gen_model,
                             "embed_model": "sentence-transformers/all-MiniLM-L6-v2",
                             "device": "cuda", "dtype": "bfloat16"})

    results = {}
    policies = [
        ("Fixed-emb K=3",        dict(judge_llm=False, beam=4, max_hops=3)),
        ("Fixed-LLM K=2 B=8",    dict(judge_llm=True,  beam=8, max_hops=2)),
        ("AdaptiveJudge T=0.55", dict(judge_llm=True,  beam=8, max_hops=2,
                                      adaptive_judge=True, adaptive_thr=0.55)),
    ]
    for name, kw in policies:
        print(f"\n=== {name} ===", flush=True)
        f1, _ = run_policy(subgraphs, examples, backend, **kw)
        results[name] = f1
        print(f"  F1={f1:.3f}", flush=True)

    print(f"\n=== oracle-depth (best K per query) ===", flush=True)
    oracle_f1, per_k = run_oracle_depth(subgraphs, examples, backend)
    results["oracle-depth"] = oracle_f1
    results["per_k"] = per_k
    print(f"  oracle-depth F1={oracle_f1:.3f}", flush=True)

    print(f"\n{'='*64}\nP2.6 WebQSP (strong engine, n={len(examples)})\n{'='*64}", flush=True)
    for name in ["Fixed-emb K=3", "AdaptiveJudge T=0.55", "Fixed-LLM K=2 B=8", "oracle-depth"]:
        print(f"  {name:<22}{results[name]:.3f}", flush=True)
    tuned = results["Fixed-LLM K=2 B=8"]; adap = results["AdaptiveJudge T=0.55"]
    print(f"\ntuned-global vs adaptive: dF1={tuned-adap:+.3f}", flush=True)
    print(f"oracle-depth headroom over best fixed: "
          f"{oracle_f1 - max(results['Fixed-emb K=3'], tuned):+.3f}", flush=True)
    verdict = ("CEILING HOLDS ON WEBQSP: tuned global >= adaptive, oracle unrealized"
               if tuned >= adap - 0.005 else "adaptive wins on WebQSP -- investigate")
    print(f"VERDICT: {verdict}", flush=True)

    with open(os.path.join(args.out_dir, "webqsp_strong.json"), "w") as f:
        json.dump({"results": results, "verdict": verdict, "n": len(examples)}, f, indent=2)
    print(f"\nSaved to {args.out_dir}/webqsp_strong.json", flush=True)


if __name__ == "__main__":
    main()
