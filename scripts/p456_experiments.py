#!/usr/bin/env python3
"""P4-P6 rebuttal experiments (submission #1972).

Modes:
  cap-sweep : candidate-cap sweep c_max in {16,32,64} (+ Fixed K=2 B=12) on the
              MetaQA mixed stream (n=240), strong engine. Answers V9mY-Q3 /
              f7vF-Q (tight caps; larger caps).
  tog       : ToG-style LLM-sufficiency early stopping (the published ToG
              early-termination mechanism, one extra Yes/No call per hop) on the
              same stream, strong engine, B in {4,8}. Answers V9mY-W3.
  features  : mid-retrieval feature logging (hop-0/hop-1 post-expansion stats +
              frontier overlap + expansion-size dynamics) and the joint outcome
              matrix over {K1B4,K2B4,K2B8,K3B4} x {LLM} + {K3 emb} on the
              verify_routing pool (300 queries, seed 1), default engine.
              Feeds scripts/p6_router_analysis.py (CPU). Answers 31Rd-Q4,
              xxtL-W4, and gives the joint router of 31Rd-Q1.

Usage:
  python3 scripts/p456_experiments.py --mode cap-sweep --limit 240
  python3 scripts/p456_experiments.py --mode tog --limit 240
  python3 scripts/p456_experiments.py --mode features
"""
from __future__ import annotations
import argparse, os, sys, json, time, random
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agr import (load_metaqa, build_backend, build_controller, GraphRAGRetriever,
                 generate_answer, extract_answer_entities, score_query, aggregate,
                 QueryReport, KnowledgeGraph, Triple, QAExample)

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


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


def load_routing_pool(data_dir, n=300):
    """Exactly the verify_routing.py pool: seed 1, n//3 per hop bucket."""
    kg, _ = load_metaqa(data_dir, "3-hop")
    _, ex1 = load_metaqa(data_dir, "1-hop")
    _, ex2 = load_metaqa(data_dir, "2-hop")
    _, ex3 = load_metaqa(data_dir, "3-hop")
    per = n // 3
    rng = random.Random(1)
    rng.shuffle(ex1); rng.shuffle(ex2); rng.shuffle(ex3)
    items = [(1, e) for e in ex1[:per]] + [(2, e) for e in ex2[:per]] + [(3, e) for e in ex3[:per]]
    rng.shuffle(items)
    return kg, items


def run_system_ex(backend, kg, examples, cfg, label, max_hops=3, beam=4,
                  judge_with_llm=True, adaptive_judge=False, adj_threshold=0.55,
                  strong=False, depth_aware=False, candidate_cap=None,
                  tog_stop=False, collect_signals=False):
    controller = build_controller(cfg)
    reports = []
    signals = []
    n = len(examples)
    for i, ex in enumerate(examples):
        if depth_aware:
            sd = getattr(controller, "set_depth", None)
            if callable(sd):
                hop = getattr(ex, "n_hop", None)
                if hop is None:
                    try:
                        hop = int(ex.qid.split("_")[1].split("-")[0])
                    except Exception:
                        hop = 3
                sd(int(hop))
        retriever = GraphRAGRetriever(kg, backend, controller, link_topk=5,
                                      judge_with_llm=judge_with_llm,
                                      adaptive_judge=adaptive_judge,
                                      adaptive_judge_threshold=adj_threshold,
                                      strong_prefilter=strong,
                                      candidate_cap=candidate_cap,
                                      tog_sufficiency_stop=tog_stop)
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
        if collect_signals:
            signals.append({"qid": ex.qid, "depth": getattr(ex, "n_hop", None),
                            "hops": ret.hop_signals})
        if (i + 1) % 25 == 0 or i == n - 1:
            a = aggregate(reports)
            print(f"  [{i+1}/{n}] {label:<30} F1={a.mean_f1:.3f} toks={a.mean_n_input_tokens:.0f} "
                  f"calls={a.mean_n_llm_calls:.2f} ({time.time()-t0:.1f}s)", flush=True)
    agg = aggregate(reports) if reports else None
    stats = dict(getattr(retriever, "_judge_stats", {}))
    return agg, reports, stats, signals


def fresh_backend(args):
    return build_backend({"kind": "hf", "gen_model": args.gen_model,
                          "embed_model": EMBED_MODEL,
                          "device": "cuda", "dtype": "bfloat16"})


def agg_dict(agg):
    return {"f1": agg.mean_f1, "hit1": agg.mean_hit1, "exact": agg.mean_exact,
            "toks": agg.mean_n_input_tokens, "calls": agg.mean_n_llm_calls,
            "hops": agg.mean_n_hops, "n": agg.n}


# --------------------------------------------------------------------------- P4
def mode_cap_sweep(args):
    os.makedirs(args.out_dir, exist_ok=True)
    kg, examples = load_mixed_metaqa(args.data_dir, per=args.limit // 3, seed=0)
    examples = examples[:args.limit]
    print(f"cap-sweep: {len(examples)} queries, strong engine", flush=True)

    def systems_for(cap):
        base = [
            ("Fixed-emb K=3",            {"name": "fixed", "beam": 4, "max_hops": 3}, 3, 4, False, False, 0.55),
            ("AdaptiveJudge T=0.55",     {"name": "fixed", "beam": 4, "max_hops": 3}, 3, 4, True, True, 0.55),
            ("AdaptiveJudge T=0.75",     {"name": "fixed", "beam": 4, "max_hops": 3}, 3, 4, True, True, 0.75),
            ("Oracle-depth B=4 (annot)", {"name": "oracle-depth", "beam": 4, "max_hops": 3}, 3, 4, True, False, 0.55),
            ("Fixed-LLM K=2 B=8",        {"name": "fixed", "beam": 8, "max_hops": 2}, 2, 8, True, False, 0.55),
        ]
        if cap >= 32:
            base.append(("Fixed-LLM K=2 B=12", {"name": "fixed", "beam": 12, "max_hops": 2}, 2, 12, True, False, 0.55))
        return base

    results = {}
    for cap in [16, 32, 64]:
        for label, cfg, mh, bm, jllm, adj, at in systems_for(cap):
            key = f"c{cap}/{label}"
            if key in results:
                continue
            backend = fresh_backend(args)
            print(f"\n=== c_max={cap} | {label} ===", flush=True)
            agg, _, stats, _ = run_system_ex(
                backend, kg, examples, cfg, f"[c{cap}] {label}",
                max_hops=mh, beam=bm, judge_with_llm=jllm,
                adaptive_judge=adj, adj_threshold=at,
                strong=True, depth_aware=("Oracle-depth" in label),
                candidate_cap=cap)
            results[key] = {**agg_dict(agg), "judge_stats": stats}
            with open(os.path.join(args.out_dir, "cap_sweep.json"), "w") as f:
                json.dump({"meta": {"gen_model": args.gen_model, "n": len(examples),
                                    "engine": "strong", "caps": [16, 32, 64]},
                           "results": results}, f, indent=2)
    print("\nCAP-SWEEP SUMMARY (F1 @ tokens)", flush=True)
    for cap in [16, 32, 64]:
        row = {k.split("/", 1)[1]: f"{v['f1']:.3f}@{v['toks']:.0f}"
               for k, v in results.items() if k.startswith(f"c{cap}/")}
        print(f"  c_max={cap}: " + json.dumps(row), flush=True)
    print("CAP_SWEEP_DONE", flush=True)


# --------------------------------------------------------------------------- P5
def mode_tog(args):
    os.makedirs(args.out_dir, exist_ok=True)
    kg, examples = load_mixed_metaqa(args.data_dir, per=args.limit // 3, seed=0)
    examples = examples[:args.limit]
    print(f"tog: {len(examples)} queries, strong engine", flush=True)
    systems = [
        ("ToG-Stop B=4", {"name": "fixed", "beam": 4, "max_hops": 3}, 4),
        ("ToG-Stop B=8", {"name": "fixed", "beam": 8, "max_hops": 3}, 8),
    ]
    results = {}
    for label, cfg, bm in systems:
        backend = fresh_backend(args)
        print(f"\n=== {label} ===", flush=True)
        agg, _, stats, _ = run_system_ex(
            backend, kg, examples, cfg, label, max_hops=3, beam=bm,
            judge_with_llm=True, strong=True, tog_stop=True)
        results[label] = {**agg_dict(agg), "judge_stats": stats}
        with open(os.path.join(args.out_dir, "tog.json"), "w") as f:
            json.dump({"meta": {"gen_model": args.gen_model, "n": len(examples),
                                "engine": "strong",
                                "mechanism": "LLM sufficiency check (Yes/No) per hop, ToG early termination"},
                       "results": results}, f, indent=2)
    print("\nTOG SUMMARY", flush=True)
    for k, v in results.items():
        print(f"  {k}: F1={v['f1']:.3f} toks={v['toks']:.0f} calls={v['calls']:.2f}", flush=True)
    print("TOG_DONE", flush=True)


# --------------------------------------------------------------------------- P6
def mode_features(args):
    os.makedirs(args.out_dir, exist_ok=True)
    kg, items = load_routing_pool(args.data_dir, n=args.pool_n)
    examples = [ex for _, ex in items]
    depths = [d for d, _ in items]
    out_path = os.path.join(args.out_dir, "features.json")
    rows = [{"qid": ex.qid, "depth": d} for ex, d in zip(examples, depths)]
    print(f"features: {len(examples)} pool queries, default engine", flush=True)
    backend = fresh_backend(args)

    # (1) probe arm: emb judge, full depth, logs per-hop signals
    print("\n=== probe (emb judge, K=3, B=4) ===", flush=True)
    agg, _, _, signals = run_system_ex(
        backend, kg, examples, {"name": "fixed", "beam": 4, "max_hops": 3},
        "probe/K3-emb", max_hops=3, beam=4, judge_with_llm=False,
        collect_signals=True)
    for r, s in zip(rows, signals):
        r["probe_hops"] = s["hops"]
    print("\n=== outcomes ===", flush=True)

    arms = [
        ("K3_emb",   {"name": "fixed", "beam": 4, "max_hops": 3}, 3, 4, False, False, 0.55),
        ("K1B4_llm", {"name": "fixed", "beam": 4, "max_hops": 1}, 1, 4, True, False, 0.55),
        ("K2B4_llm", {"name": "fixed", "beam": 4, "max_hops": 2}, 2, 4, True, False, 0.55),
        ("K2B8_llm", {"name": "fixed", "beam": 8, "max_hops": 2}, 2, 8, True, False, 0.55),
        ("K3B4_llm", {"name": "fixed", "beam": 4, "max_hops": 3}, 3, 4, True, False, 0.55),
    ]
    for label, cfg, mh, bm, jllm, adj, at in arms:
        agg, reports, _, _ = run_system_ex(
            backend, kg, examples, cfg, label, max_hops=mh, beam=bm,
            judge_with_llm=jllm, adaptive_judge=adj, adj_threshold=at)
        by_qid = {r.qid: r for r in reports}
        for r in rows:
            rep = by_qid[r["qid"]]
            r.setdefault("outcomes", {})[label] = {
                "f1": rep.f1, "hit1": rep.hit1, "toks": rep.n_input_tokens,
                "calls": rep.n_llm_calls, "hops": rep.n_hops}
        print(f"  arm {label}: F1={agg.mean_f1:.3f} toks={agg.mean_n_input_tokens:.0f}", flush=True)
        with open(out_path, "w") as f:
            json.dump({"meta": {"gen_model": args.gen_model, "n": len(rows),
                                "engine": "default (weak)",
                                "pool": "verify_routing seed-1 pool"},
                       "rows": rows}, f, indent=2)
    # question embeddings (384-d) for router comparability
    print("\n=== question embeddings ===", flush=True)
    qembs = backend.embed([r_qid_q for r_qid_q in [ex.question for ex in examples]])
    for r, qe in zip(rows, qembs):
        r["qemb"] = list(map(float, qe))
    with open(out_path, "w") as f:
        json.dump({"meta": {"gen_model": args.gen_model, "n": len(rows),
                            "engine": "default (weak)",
                            "pool": "verify_routing seed-1 pool"},
                   "rows": rows}, f, indent=2)
    print("FEATURES_DONE", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["cap-sweep", "tog", "features"])
    ap.add_argument("--data-dir", default="./data/MetaQA")
    ap.add_argument("--gen-model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--limit", type=int, default=240)
    ap.add_argument("--pool-n", type=int, default=300)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()
    if args.out_dir is None:
        args.out_dir = {"cap-sweep": "./results_cap_sweep",
                        "tog": "./results_tog",
                        "features": "./results_p6_features"}[args.mode]
    if args.mode == "cap-sweep":
        mode_cap_sweep(args)
    elif args.mode == "tog":
        mode_tog(args)
    else:
        mode_features(args)


if __name__ == "__main__":
    main()
