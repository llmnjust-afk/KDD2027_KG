#!/usr/bin/env python3
"""Mixed-stream Pareto sweep (reviewer-requested, addresses review #2/#3).

On the mixed 1/2/3-hop stream, sweep:
  - Fixed K in {1,2,3,4} x B in {2,4,6,8}   (the tuned frontier)
  - Simple stopping heuristics: patience, margin, conf-threshold, random-budget
  - Adaptive (ours)
  - Oracle-depth (upper bound: true per-query depth as max_hops)
  - LogReg-K and MLP-K trained routers (if available)

Then emit a full accuracy-vs-cost Pareto table + JSON for plotting. The key
question: is Adaptive non-dominated on the mixed stream, or does a tuned
Fixed-K=2 (which sacrifices some 3-hop accuracy) achieve a better trade-off?

Usage:
  python scripts/mixed_pareto.py --gen-model Qwen/Qwen2.5-7B-Instruct --limit 300
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import random

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agr import (load_metaqa, build_backend, build_controller,
                 GraphRAGRetriever, generate_answer, extract_answer_entities,
                 score_query, aggregate, QueryReport)


def build_mixed(data_dir, per, seed=0):
    """Build a mixed 1/2/3-hop stream with per-query depth labels."""
    _, ex1 = load_metaqa(data_dir, "1-hop")
    _, ex2 = load_metaqa(data_dir, "2-hop")
    _, ex3 = load_metaqa(data_dir, "3-hop")
    rng = random.Random(seed)
    rng.shuffle(ex1); rng.shuffle(ex2); rng.shuffle(ex3)
    examples = []
    for ex in ex1[:per]:
        examples.append((ex, 1))
    for ex in ex2[:per]:
        examples.append((ex, 2))
    for ex in ex3[:per]:
        examples.append((ex, 3))
    rng.shuffle(examples)
    return examples


def run_system(kg, backend, examples, cfg, label, max_hops=3, beam=4,
               depth_aware=False, limit=None):
    """Run one system config over the mixed stream; return (agg, per_query)."""
    controller = build_controller(cfg)
    retriever = GraphRAGRetriever(kg, backend, controller, link_topk=5,
                                  judge_with_llm=True)
    reports = []
    n = len(examples) if limit is None else min(limit, len(examples))
    for i, (ex, depth) in enumerate(examples[:n]):
        # per-query hooks for oracle / random controllers
        if depth_aware:
            sd = getattr(controller, "set_depth", None)
            if callable(sd):
                sd(depth)
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
        if (i + 1) % 50 == 0 or i == n - 1:
            a = aggregate(reports)
            print(f"  [{i+1}/{n}] {label:<28} F1={a.mean_f1:.3f} "
                  f"toks={a.mean_n_input_tokens:.0f} hops={a.mean_n_hops:.2f} "
                  f"({time.time()-t0:.1f}s)", flush=True)
    return (aggregate(reports) if reports else None, reports)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="./data/MetaQA")
    ap.add_argument("--gen-model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--embed-model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--limit", type=int, default=300, help="total mixed queries (3x this per depth)")
    ap.add_argument("--per", type=int, default=100, help="queries per depth")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default="./results_mixed_pareto")
    ap.add_argument("--skip-grid", action="store_true", help="skip the KxB fixed grid (longest)")
    ap.add_argument("--skip-routers", action="store_true", help="skip LogReg/MLP trained routers")
    args = ap.parse_args()

    per = args.per
    examples = build_mixed(args.data_dir, per, seed=args.seed)
    print(f"Mixed stream: {len(examples)} queries "
          f"({per} per depth, seed={args.seed})", flush=True)

    # load KG once (use 3-hop split to get the full KG; all share the same KG)
    kg, _ = load_metaqa(args.data_dir, "3-hop")
    print(f"KG: {len(kg)} triples, {len(kg.entities)} entities", flush=True)

    backend = build_backend({"kind": "hf", "gen_model": args.gen_model,
                             "embed_model": args.embed_model,
                             "device": "cuda", "dtype": "bfloat16"})
    print(f"Backend ready: gen={args.gen_model}", flush=True)

    os.makedirs(args.out_dir, exist_ok=True)
    results = {}  # label -> {f1, hit1, toks, hops, n}

    def record(label, agg, reports):
        if agg is None:
            return
        results[label] = {
            "f1": agg.mean_f1, "hit1": agg.mean_hit1, "exact": agg.mean_exact,
            "toks": agg.mean_n_input_tokens, "hops": agg.mean_n_hops,
            "calls": agg.mean_n_llm_calls, "n": agg.n,
        }
        # save per-query reports
        import json as _j
        sub = os.path.join(args.out_dir, label.replace(" ", "_").replace("(", "").replace(")", "").replace("=", ""))
        os.makedirs(sub, exist_ok=True)
        with open(os.path.join(sub, "reports.jsonl"), "w") as f:
            for r in reports:
                f.write(_j.dumps({"qid": r.qid, "f1": r.f1, "hit1": r.hit1,
                                  "exact": r.exact, "n_hops": r.n_hops,
                                  "n_input_tokens": r.n_input_tokens,
                                  "n_llm_calls": r.n_llm_calls,
                                  "used_graph": r.used_graph}) + "\n")

    # ---- 1. Adaptive (ours) FIRST — get the key result early ----
    print("\n=== Adaptive (ours) ===", flush=True)
    for B in [4]:
        cfg = {"name": "adaptive", "base_beam": B, "max_hops": 3,
               "theta_low": 0.30, "delta": 0.05}
        label = f"Adaptive B={B}"
        agg, reps = run_system(kg, backend, examples, cfg, label, max_hops=3, beam=B)
        record(label, agg, reps)

    # ---- 2. Oracle-depth (upper bound) ----
    print("\n=== Oracle-depth (upper bound) ===", flush=True)
    cfg = {"name": "oracle-depth", "beam": 4, "max_hops": 3}
    agg, reps = run_system(kg, backend, examples, cfg, "Oracle-depth",
                           max_hops=3, beam=4, depth_aware=True)
    record("Oracle-depth", agg, reps)

    # ---- 3. Fixed K x B grid (the tuned frontier). K in {1,2,3} only:
    #        no MetaQA question needs >3 hops, so K=4 just adds a noisy hop. ----
    if not args.skip_grid:
        print("\n=== Fixed K x B grid ===", flush=True)
        for K in [1, 2, 3]:
            for B in [2, 4, 6, 8]:
                label = f"Fixed K={K} B={B}"
                cfg = {"name": "fixed", "beam": B, "max_hops": K}
                agg, reps = run_system(kg, backend, examples, cfg, label,
                                       max_hops=K, beam=B)
                record(label, agg, reps)

    # ---- 4. Simple stopping heuristics (fixed beam=4, max_hops=3) ----
    print("\n=== Simple stopping heuristics ===", flush=True)
    for name, cfg, label in [
        ("patience", {"name": "patience", "beam": 4, "patience": 1, "max_hops": 3}, "Patience p=1"),
        ("patience2", {"name": "patience", "beam": 4, "patience": 2, "max_hops": 3}, "Patience p=2"),
        ("margin", {"name": "margin", "beam": 4, "threshold": 0.30, "max_hops": 3}, "Margin t=0.30"),
        ("margin5", {"name": "margin", "beam": 4, "threshold": 0.50, "max_hops": 3}, "Margin t=0.50"),
        ("conf", {"name": "conf-threshold", "beam": 4, "threshold": 0.70, "max_hops": 3}, "Conf t=0.70"),
        ("conf8", {"name": "conf-threshold", "beam": 4, "threshold": 0.85, "max_hops": 3}, "Conf t=0.85"),
        ("rand", {"name": "random-budget", "beam": 4, "seed": 0, "max_hops": 3}, "Random-budget"),
    ]:
        agg, reps = run_system(kg, backend, examples, cfg, label, max_hops=3, beam=4)
        record(label, agg, reps)

    # ---- 5. Trained routers (LogReg-K, MLP-K) ----
    if not args.skip_routers:
        print("\n=== Trained routers ===", flush=True)
        try:
            from scripts.trained_baseline import load_router, predict_depth
            for rname in ["logreg", "mlp"]:
                router = load_router(rname, args.data_dir)
                # run as fixed with per-query predicted K
                label = f"{rname.upper()}-K"
                reports = []
                controller = build_controller({"name": "fixed", "beam": 4, "max_hops": 3})
                retriever = GraphRAGRetriever(kg, backend, controller, link_topk=5,
                                              judge_with_llm=True)
                for i, (ex, depth) in enumerate(examples):
                    pred_K = predict_depth(router, rname, ex.question)
                    controller.max_hops = pred_K
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
                        n_output_tokens=ans.n_output_tokens, stop_reasons=[]))
                    if (i + 1) % 50 == 0:
                        a = aggregate(reports)
                        print(f"  [{i+1}/{len(examples)}] {label} F1={a.mean_f1:.3f}", flush=True)
                record(label, aggregate(reports), reports)
        except Exception as e:
            print(f"  trained routers skipped: {e}", flush=True)

    # ---- Pareto table + JSON ----
    print("\n" + "=" * 80, flush=True)
    print("MIXED-STREAM PARETO (sorted by toks)", flush=True)
    print("=" * 80, flush=True)
    rows = sorted(results.items(), key=lambda kv: kv[1]["toks"])
    print(f"{'system':<26} {'F1':>6} {'Hit@1':>6} {'toks':>6} {'hops':>5} {'calls':>5}", flush=True)
    print("-" * 60, flush=True)
    # find non-dominated set
    nondom = []
    for lab, r in rows:
        dom = False
        for lab2, r2 in results.items():
            if lab2 == lab:
                continue
            # r2 dominates r if r2 has >= F1 and <= toks (strictly better in one)
            if r2["f1"] >= r["f1"] and r2["toks"] <= r["toks"] and \
               (r2["f1"] > r["f1"] or r2["toks"] < r["toks"]):
                dom = True
                break
        nondom.append((lab, r, dom))
    for lab, r, dom in nondom:
        mark = " *" if dom else "  "
        print(f"{mark}{lab:<24} {r['f1']:>6.3f} {r['hit1']:>6.3f} "
              f"{r['toks']:>6.0f} {r['hops']:>5.2f} {r['calls']:>5.2f}", flush=True)
    print("\n(* = dominated by some other system on the mixed stream)", flush=True)

    out = os.path.join(args.out_dir, "mixed_pareto.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out}", flush=True)


if __name__ == "__main__":
    main()
