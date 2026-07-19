#!/usr/bin/env python3
"""A3: Validate AdaptiveJudge on MetaQA 1/2/3-hop + mixed stream.

Compares Fixed (LLM judge, K=3) vs AdaptiveJudge (emb probe + LLM when uncertain)
vs Fixed-emb (emb judge only, K=3) to isolate the F1 and cost effects.
"""
from __future__ import annotations
import argparse, os, sys, json, time, random
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agr import (load_metaqa, build_backend, build_controller, GraphRAGRetriever,
                 generate_answer, extract_answer_entities, score_query, aggregate,
                 QueryReport, save_reports)


def run_system(kg, backend, examples, cfg, label, max_hops=3, beam=4,
               judge_with_llm=True, adaptive_judge=False, adj_threshold=0.55,
               limit=None):
    controller = build_controller(cfg)
    retriever = GraphRAGRetriever(kg, backend, controller, link_topk=5,
                                  judge_with_llm=judge_with_llm,
                                  adaptive_judge=adaptive_judge,
                                  adaptive_judge_threshold=adj_threshold)
    reports = []
    n = len(examples) if limit is None else min(limit, len(examples))
    for i, ex in enumerate(examples[:n]):
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
            print(f"  [{i+1}/{n}] {label:<24} F1={a.mean_f1:.3f} toks={a.mean_n_input_tokens:.0f} "
                  f"calls={a.mean_n_llm_calls:.2f} hops={a.mean_n_hops:.2f} ({time.time()-t0:.1f}s)", flush=True)
    agg = aggregate(reports) if reports else None
    stats = dict(retriever._judge_stats) if hasattr(retriever, '_judge_stats') else {}
    return agg, reports, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="./data/MetaQA")
    ap.add_argument("--gen-model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--splits", nargs="+", default=["1-hop", "2-hop", "3-hop", "mixed"])
    ap.add_argument("--threshold", type=float, default=0.55)
    ap.add_argument("--out-dir", default="./results_adjudge")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    for split in args.splits:
        print(f"\n{'='*72}\n=== Split: {split} (n={args.limit}) ===\n{'='*72}", flush=True)
        if split == "mixed":
            _, ex1 = load_metaqa(args.data_dir, "1-hop")
            _, ex2 = load_metaqa(args.data_dir, "2-hop")
            _, ex3 = load_metaqa(args.data_dir, "3-hop")
            per = args.limit // 3
            rng = random.Random(0)
            rng.shuffle(ex1); rng.shuffle(ex2); rng.shuffle(ex3)
            examples = ex1[:per] + ex2[:per] + ex3[:per]
            rng.shuffle(examples)
            kg, _ = load_metaqa(args.data_dir, "3-hop")
        else:
            kg, examples = load_metaqa(args.data_dir, split)

        results = {}
        for label, kw in [
            ("Fixed (LLM judge)", {"judge_with_llm": True, "adaptive_judge": False}),
            ("AdaptiveJudge (ours)", {"judge_with_llm": True, "adaptive_judge": True}),
            ("Fixed (emb judge)", {"judge_with_llm": False, "adaptive_judge": False}),
        ]:
            print(f"\n--- {label} ---", flush=True)
            # fresh backend per system to avoid usage counter carry-over
            backend = build_backend({"kind": "hf", "gen_model": args.gen_model,
                                     "embed_model": "sentence-transformers/all-MiniLM-L6-v2",
                                     "device": "cuda", "dtype": "bfloat16"})
            agg, reps, stats = run_system(
                kg, backend, examples,
                {"name": "fixed", "beam": 4, "max_hops": 3}, label,
                max_hops=3, beam=4, limit=args.limit,
                adj_threshold=args.threshold, **kw)
            if agg:
                results[label] = {"f1": agg.mean_f1, "hit1": agg.mean_hit1,
                                  "toks": agg.mean_n_input_tokens,
                                  "hops": agg.mean_n_hops, "calls": agg.mean_n_llm_calls,
                                  "judge_stats": stats}
                save_reports(reps, os.path.join(args.out_dir,
                             f"metaqa_{split}_{label.split()[0].lower()}", "reports.jsonl"))

        print(f"\n{'='*72}\nPARETO [{split}, n={args.limit}]", flush=True)
        print(f"{'='*72}", flush=True)
        print(f"{'system':<26} {'F1':>6} {'Hit@1':>6} {'toks':>6} {'calls':>5} {'hops':>5} {'judge':>20}", flush=True)
        for lab, r in sorted(results.items(), key=lambda kv: kv[1]["toks"]):
            js = r.get("judge_stats", {})
            js_str = f"emb:{js.get('emb_only',0)} llm:{js.get('llm',0)}" if js else "--"
            print(f"{lab:<26} {r['f1']:>6.3f} {r['hit1']:>6.3f} {r['toks']:>6.0f} "
                  f"{r['calls']:>5.2f} {r['hops']:>5.2f} {js_str:>20}", flush=True)

        out = os.path.join(args.out_dir, f"adjudge_{split}.json")
        with open(out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved to {out}", flush=True)


if __name__ == "__main__":
    main()
