#!/usr/bin/env python3
"""P1.3: Larger n + multi-seed + bootstrap CIs for the main mixed-stream result.

Focuses on the decisive systems on MetaQA mixed:
  - AdaptiveJudge (best adaptive)  vs  Fixed-K2B8 (tuned global, non-dominated)
  - Fixed-K3 (default)  and  Fixed-emb (cheap non-dominated point)
Runs n=500 per seed over 3 independent query subsamples (seeds 0,1,2), reports
per-seed F1/toks, mean±std across seeds, and a paired bootstrap 95% CI for
(Fixed-K2B8 - AdaptiveJudge) F1 on the pooled queries.

Usage:
  python scripts/multiseed.py --gen-model Qwen/Qwen2.5-7B-Instruct --n 500 --seeds 0 1 2
"""
from __future__ import annotations
import argparse, os, sys, json, time, random
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agr import (load_metaqa, build_backend, build_controller, GraphRAGRetriever,
                 generate_answer, extract_answer_entities, score_query)


def build_mixed(data_dir, n, seed):
    _, e1 = load_metaqa(data_dir, "1-hop")
    _, e2 = load_metaqa(data_dir, "2-hop")
    _, e3 = load_metaqa(data_dir, "3-hop")
    per = n // 3
    rng = random.Random(seed)
    rng.shuffle(e1); rng.shuffle(e2); rng.shuffle(e3)
    exs = e1[:per] + e2[:per] + e3[:per]
    rng.shuffle(exs)
    kg, _ = load_metaqa(data_dir, "3-hop")
    return kg, exs


def run(kg, backend, exs, cfg, jllm, adj, mh, bm, strong=True):
    ctrl = build_controller(cfg)
    f1s, toks = [], []
    for ex in exs:
        ret = GraphRAGRetriever(kg, backend, ctrl, link_topk=5, judge_with_llm=jllm,
                                adaptive_judge=adj, adaptive_judge_threshold=0.55,
                                strong_prefilter=strong)
        tk0 = backend.usage.n_input_tokens
        r = ret.retrieve(ex.question)
        a = generate_answer(backend, ex.question, r)
        sc = score_query(extract_answer_entities(a.text), ex.answers)
        f1s.append(sc["f1"]); toks.append(backend.usage.n_input_tokens - tk0)
    return np.array(f1s), np.array(toks)


def paired_bootstrap(a, b, n=10000, seed=0):
    """95% CI for mean(b-a) via paired bootstrap; also p(b>a)."""
    rng = np.random.default_rng(seed)
    d = b - a; N = len(d)
    means = [d[rng.integers(0, N, N)].mean() for _ in range(n)]
    lo, hi = np.percentile(means, [2.5, 97.5])
    p = float(np.mean(np.array(means) <= 0))  # prob effect <= 0
    return float(d.mean()), float(lo), float(hi), p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="./data/MetaQA")
    ap.add_argument("--gen-model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--out-dir", default="./results_multiseed")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    SYS = [
        ("AdaptiveJudge", {"name":"fixed","beam":4,"max_hops":3}, True, True, 3, 4),
        ("Fixed-K2B8",    {"name":"fixed","beam":8,"max_hops":2}, True, False, 2, 8),
        ("Fixed-K3",      {"name":"fixed","beam":4,"max_hops":3}, True, False, 3, 4),
        ("Fixed-emb",     {"name":"fixed","beam":4,"max_hops":3}, False, False, 3, 4),
    ]

    per_seed = {s: {} for s in args.seeds}
    pooled = {name: {"f1": [], "toks": []} for name, *_ in SYS}
    for seed in args.seeds:
        kg, exs = build_mixed(args.data_dir, args.n, seed)
        print(f"\n=== seed {seed} (n={len(exs)}) ===", flush=True)
        backend = build_backend({"kind":"hf","gen_model":args.gen_model,
                                 "embed_model":"sentence-transformers/all-MiniLM-L6-v2",
                                 "device":"cuda","dtype":"bfloat16"})
        for name, cfg, jllm, adj, mh, bm in SYS:
            t0 = time.time()
            f1, tk = run(kg, backend, exs, cfg, jllm, adj, mh, bm)
            per_seed[seed][name] = {"f1": float(f1.mean()), "toks": float(tk.mean())}
            pooled[name]["f1"].extend(f1.tolist()); pooled[name]["toks"].extend(tk.tolist())
            print(f"  {name:<14} F1={f1.mean():.3f} toks={tk.mean():.0f} ({time.time()-t0:.0f}s)", flush=True)

    print(f"\n{'='*66}\nP1.3 MULTI-SEED (n={args.n}, seeds={args.seeds}, strong engine)\n{'='*66}", flush=True)
    print(f"{'system':<14} {'F1 mean±std':>16} {'toks':>7}", flush=True)
    for name, *_ in SYS:
        f1s = [per_seed[s][name]["f1"] for s in args.seeds]
        tks = [per_seed[s][name]["toks"] for s in args.seeds]
        print(f"{name:<14} {np.mean(f1s):.3f}±{np.std(f1s):.3f}      {np.mean(tks):>7.0f}", flush=True)

    # paired bootstrap: Fixed-K2B8 - AdaptiveJudge on pooled queries
    a = np.array(pooled["AdaptiveJudge"]["f1"]); b = np.array(pooled["Fixed-K2B8"]["f1"])
    m, lo, hi, p = paired_bootstrap(a, b)
    print(f"\nPaired bootstrap (Fixed-K2B8 - AdaptiveJudge) F1 on pooled n={len(a)}:", flush=True)
    print(f"  Delta={m:+.3f}  95% CI=[{lo:+.3f},{hi:+.3f}]  P(delta<=0)={p:.4f}", flush=True)
    print(f"  => tuned global is {'significantly' if lo>0 else 'not significantly'} better than best adaptive", flush=True)

    with open(os.path.join(args.out_dir, "multiseed.json"), "w") as f:
        json.dump({"per_seed": per_seed, "bootstrap": {"delta":m,"ci":[lo,hi],"p":p}}, f, indent=2)
    print(f"\nSaved to {args.out_dir}/multiseed.json", flush=True)


if __name__ == "__main__":
    main()
