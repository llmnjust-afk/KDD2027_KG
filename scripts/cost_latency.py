#!/usr/bin/env python3
"""P2.5: Does adaptation also cost more, not less?

The predictability-ceiling paper argues a tuned global budget dominates
training-free per-query adaptation on ACCURACY. A natural rebuttal is that
adaptation trades a little accuracy for large efficiency gains. This script
measures that trade-off directly: for representative policies on the mixed
MetaQA stream under the strong engine we record wall-clock latency per query,
the number of LLM relevance-judge calls (the dominant marginal cost), and F1.

Policies:
  Fixed-emb K=3        -- cheap lower bound, 0 LLM judge calls
  Fixed-LLM K=2 B=8    -- the tuned global winner
  AdaptiveJudge T=0.55 -- embedding probe, escalate to LLM only when uncertain

If AdaptiveJudge is neither more accurate NOR appreciably cheaper than the
tuned global point, the efficiency rebuttal fails too.

Usage: python scripts/cost_latency.py --gen-model Qwen/Qwen2.5-7B-Instruct --limit 120
"""
from __future__ import annotations
import argparse, os, sys, json, time, random
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agr import (load_metaqa, build_backend, build_controller, GraphRAGRetriever,
                 generate_answer, extract_answer_entities, score_query)


def build_mixed(data_dir, per_hop, seed=0):
    """Balanced mixed stream: per_hop questions from each of 1/2/3-hop."""
    stream = []
    for split in ["1-hop", "2-hop", "3-hop"]:
        kg, exs = load_metaqa(data_dir, split)
        random.Random(seed).shuffle(exs)
        for ex in exs[:per_hop]:
            stream.append((kg, ex))
    random.Random(seed + 1).shuffle(stream)
    return stream


def run_policy(name, stream, backend, *, judge_llm, beam, max_hops,
               adaptive_judge=False, adaptive_thr=0.55):
    ctrl = build_controller({"name": "fixed", "beam": beam, "max_hops": max_hops})
    f1s, lat, calls = [], [], []
    for kg, ex in stream:
        ret = GraphRAGRetriever(kg, backend, ctrl, link_topk=5,
                                judge_with_llm=judge_llm, strong_prefilter=True,
                                adaptive_judge=adaptive_judge,
                                adaptive_judge_threshold=adaptive_thr)
        t0 = time.perf_counter()
        r = ret.retrieve(ex.question)
        a = generate_answer(backend, ex.question, r)
        dt = time.perf_counter() - t0
        sc = score_query(extract_answer_entities(a.text), ex.answers)
        f1s.append(sc["f1"]); lat.append(dt); calls.append(r.n_llm_calls)
    return {
        "f1": float(np.mean(f1s)),
        "latency_s_mean": float(np.mean(lat)),
        "latency_s_p50": float(np.median(lat)),
        "llm_judge_calls_mean": float(np.mean(calls)),
        "llm_judge_calls_total": int(np.sum(calls)),
        "n": len(f1s),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="./data/MetaQA")
    ap.add_argument("--gen-model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--limit", type=int, default=120, help="total mixed queries")
    ap.add_argument("--out-dir", default="./results_cost")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    per_hop = max(1, args.limit // 3)
    stream = build_mixed(args.data_dir, per_hop)
    print(f"mixed stream n={len(stream)} (per_hop={per_hop})", flush=True)

    backend = build_backend({"kind": "hf", "gen_model": args.gen_model,
                             "embed_model": "sentence-transformers/all-MiniLM-L6-v2",
                             "device": "cuda", "dtype": "bfloat16"})

    policies = [
        ("Fixed-emb K=3",       dict(judge_llm=False, beam=4, max_hops=3)),
        ("Fixed-LLM K=2 B=8",   dict(judge_llm=True,  beam=8, max_hops=2)),
        ("AdaptiveJudge T=0.55", dict(judge_llm=True,  beam=8, max_hops=2,
                                      adaptive_judge=True, adaptive_thr=0.55)),
    ]
    results = {}
    for name, kw in policies:
        print(f"\n=== {name} ===", flush=True)
        # warm-up one query so the first timing isn't dominated by lazy init
        _ = run_policy(name, stream[:1], backend, **kw)
        results[name] = run_policy(name, stream, backend, **kw)
        r = results[name]
        print(f"  F1={r['f1']:.3f}  lat_mean={r['latency_s_mean']:.3f}s  "
              f"lat_p50={r['latency_s_p50']:.3f}s  "
              f"judge_calls/q={r['llm_judge_calls_mean']:.2f}", flush=True)

    print(f"\n{'='*72}\nP2.5 COST / LATENCY (mixed MetaQA, strong engine)\n{'='*72}", flush=True)
    print(f"{'policy':<22}{'F1':>7}{'lat_mean(s)':>13}{'lat_p50(s)':>12}{'judge/q':>10}", flush=True)
    for name, r in results.items():
        print(f"{name:<22}{r['f1']:>7.3f}{r['latency_s_mean']:>13.3f}"
              f"{r['latency_s_p50']:>12.3f}{r['llm_judge_calls_mean']:>10.2f}", flush=True)

    tuned = results["Fixed-LLM K=2 B=8"]; adap = results["AdaptiveJudge T=0.55"]
    print(f"\ntuned-vs-adaptive: dF1={tuned['f1']-adap['f1']:+.3f}  "
          f"dLatency={tuned['latency_s_mean']-adap['latency_s_mean']:+.3f}s  "
          f"dJudgeCalls/q={tuned['llm_judge_calls_mean']-adap['llm_judge_calls_mean']:+.2f}", flush=True)
    verdict = ("EFFICIENCY REBUTTAL FAILS: tuned global is at least as accurate "
               "and not appreciably more expensive than adaptation"
               if tuned["f1"] >= adap["f1"] - 0.005 else
               "adaptation buys accuracy-efficiency trade -- examine")
    print(f"VERDICT: {verdict}", flush=True)

    with open(os.path.join(args.out_dir, "cost_latency.json"), "w") as f:
        json.dump({"results": results, "verdict": verdict}, f, indent=2)
    print(f"\nSaved to {args.out_dir}/cost_latency.json", flush=True)


if __name__ == "__main__":
    main()
