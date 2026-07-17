#!/usr/bin/env python3
"""Main runner: evaluate one or more retrieval systems (controllers) on a
dataset, emit per-system reports + a Pareto table.

A "system" is a named controller configuration. The baseline (FixedController)
and our method (AdaptiveController) run through the *same* engine -- the only
difference is the controller, which is exactly what a clean ablation needs.

Systems are defined on the CLI via --systems NAME=spec; a "spec" is a short
control string. Recognised systems:
    fixed            naive ToG-style baseline (fixed beam, always graph, no early stop)
    adaptive         our full method (2a)+(2b)+(3)
    abl-nograph      ablation: adaptive with (2a) disabled
    abl-fixbeam      ablation: adaptive with (2b) disabled
    abl-nostop       ablation: adaptive with (3) disabled

Usage:
    # toy smoke test (embedding judge only, no GPU model load)
    python run.py --dataset toy --systems fixed adaptive --no-llm-judge --no-gen-model

    # MetaQA 1-hop, full method + baseline
    python run.py --dataset metaqa --split 1-hop --data-dir ./data/MetaQA \
        --gen-model Qwen/Qwen2.5-1.5B-Instruct --limit 500 \
        --systems fixed adaptive

    # full ablation suite
    python run.py --dataset metaqa --split 2-hop --data-dir ./data/MetaQA \
        --gen-model Qwen/Qwen2.5-1.5B-Instruct --limit 500 \
        --systems fixed adaptive abl-nograph abl-fixbeam abl-nostop

Outputs:
    results/<dataset>_<split>_<system>/reports.jsonl
    results/pareto_<dataset>_<split>.txt
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agr import (load_toy, load_metaqa, build_backend, build_controller,
                 GraphRAGRetriever, generate_answer, extract_answer_entities,
                 score_query, aggregate, pareto_table, save_reports, QueryReport)


SYSTEMS = {
    # name -> (controller cfg, human label for the Pareto table)
    "fixed":       ({"name": "fixed", "beam": 4}, "Fixed (ToG-style)"),
    "adaptive":    ({"name": "adaptive", "base_beam": 4}, "Adaptive (ours)"),
    "abl-nograph": ({"name": "adaptive", "base_beam": 4, "ablate_use_graph": True},    "Abl. no (2a)"),
    "abl-fixbeam": ({"name": "adaptive", "base_beam": 4, "ablate_adaptive_beam": True}, "Abl. no (2b)"),
    "abl-nostop":  ({"name": "adaptive", "base_beam": 4, "ablate_early_stop": True},   "Abl. no (3)"),
    "nograph":     ({"name": "nograph"}, "No-Graph (pure LLM)"),
    "vector-rag":  ({"name": "vector-rag"}, "Vector-RAG (no traversal)"),
}


def _corrupt_question(question, rng, rate):
    """Corrupt [Entity] mentions to simulate noisy entity linking.

    With probability `rate`, each bracketed entity is corrupted by truncating
    to its first word or dropping a random character. This creates genuinely
    uncertain links where embedding matching may pick the wrong entity.
    """
    import re
    def _corrupt(m):
        if rng.random() > rate:
            return m.group(0)
        ent = m.group(1)
        words = ent.split()
        if len(words) > 1:
            return f"[{rng.choice(words)}]"  # partial name
        if len(ent) > 3:
            idx = rng.randint(1, len(ent) - 2)
            return f"[{ent[:idx]}{ent[idx+1:]}]"  # drop a char (typo)
        return m.group(0)
    return re.sub(r"\[([^\]]+)\]", _corrupt, question)


def run_one(retriever, backend, examples, limit=None, corrupt_rate=0.0):
    reports = []
    n = len(examples) if limit is None else min(limit, len(examples))
    import random
    rng = random.Random(42)
    for i, ex in enumerate(examples[:n]):
        t0 = time.time()
        call0 = backend.usage.n_calls
        tok0 = backend.usage.n_input_tokens
        # optionally corrupt entity mentions to simulate noisy linking
        question = ex.question
        if corrupt_rate > 0:
            question = _corrupt_question(ex.question, rng, corrupt_rate)
        retrieval = retriever.retrieve(question)
        ans = generate_answer(backend, question, retrieval)
        pred = extract_answer_entities(ans.text)
        sc = score_query(pred, ex.answers)
        reports.append(QueryReport(
            qid=ex.qid, question=ex.question, gold=ex.answers, pred=pred,
            hit1=sc["hit1"], f1=sc["f1"], exact=sc["exact"],
            used_graph=retrieval.used_graph, n_hops=retrieval.n_hops_executed,
            n_llm_calls=backend.usage.n_calls - call0,
            n_input_tokens=(backend.usage.n_input_tokens - tok0),
            n_output_tokens=ans.n_output_tokens,
            stop_reasons=[],
        ))
        if (i + 1) % 50 == 0 or i == n - 1:
            agg = aggregate(reports)
            print(f"  [{i+1}/{n}] F1={agg.mean_f1:.3f} toks/q={agg.mean_n_input_tokens:.0f} "
                  f"calls/q={agg.mean_n_llm_calls:.2f} hops/q={agg.mean_n_hops:.2f} "
                  f"t={time.time()-t0:.1f}s", flush=True)
    return reports


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["toy", "metaqa"], default="toy")
    ap.add_argument("--split", default="1-hop")
    ap.add_argument("--data-dir", default="./data/MetaQA")
    ap.add_argument("--systems", nargs="+", default=["fixed", "adaptive"])
    ap.add_argument("--max-hops", type=int, default=3)
    ap.add_argument("--beam", type=int, default=4, help="baseline fixed beam / adaptive base_beam")
    ap.add_argument("--link-topk", type=int, default=5)
    ap.add_argument("--gen-model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--embed-model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no-llm-judge", action="store_true",
                    help="use embedding relevance instead of LLM judge (fast smoke test)")
    ap.add_argument("--no-gen-model", action="store_true",
                    help="skip loading the HF generator; relevance via embeddings only")
    ap.add_argument("--noisy-linking", action="store_true",
                    help="strip [Entity] brackets -> embedding linking (activates 2a)")
    ap.add_argument("--theta-low", type=float, default=0.30,
                    help="use-graph threshold (raise to activate 2a under noisy linking)")
    ap.add_argument("--corrupt-rate", type=float, default=0.0,
                    help="probability of corrupting entity mentions in questions (simulates noisy linking)")
    ap.add_argument("--vector-rag-topk", type=int, default=0,
                    help=">0 enables vector-RAG mode (retrieve top-k triples, no traversal)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0,
                    help="random seed: shuffles the test-split order before taking --limit; "
                         "also sets torch/numpy seeds. Enables multi-seed variance reporting.")
    ap.add_argument("--mixed", action="store_true",
                    help="build a mixed 1/2/3-hop query stream (equal parts) instead of a single split")
    ap.add_argument("--out-dir", default="./results")
    args = ap.parse_args()

    # ---- seeds ----
    import random as _random
    import numpy as _np
    _random.seed(args.seed)
    _np.random.seed(args.seed)
    try:
        import torch as _torch
        _torch.manual_seed(args.seed)
        if _torch.cuda.is_available():
            _torch.cuda.manual_seed_all(args.seed)
    except Exception:
        pass

    # ---- data ----
    if args.dataset == "toy":
        kg, examples = load_toy()
    elif args.mixed:
        # mixed query stream: equal parts 1/2/3-hop, sharing the same KG
        kg, ex1 = load_metaqa(args.data_dir, "1-hop")
        _, ex2 = load_metaqa(args.data_dir, "2-hop")
        _, ex3 = load_metaqa(args.data_dir, "3-hop")
        per = (args.limit or 300) // 3
        rng_m = _random.Random(args.seed)
        rng_m.shuffle(ex1); rng_m.shuffle(ex2); rng_m.shuffle(ex3)
        examples = ex1[:per] + ex2[:per] + ex3[:per]
        rng_m.shuffle(examples)
        args.split = "mixed"
    else:
        kg, examples = load_metaqa(args.data_dir, args.split)
        # shuffle before taking --limit so different seeds see different subsets
        if args.seed != 0:
            _random.Random(args.seed).shuffle(examples)
    print(f"KG: {len(kg)} triples, {len(kg.entities)} entities | "
          f"{len(examples)} questions (using {min(len(examples), args.limit or len(examples))})",
          flush=True)

    # ---- backend ----
    backend = build_backend({"kind": "hf", "gen_model": args.gen_model,
                             "embed_model": args.embed_model,
                             "device": args.device, "dtype": "bfloat16"})
    print(f"Backend: {backend.__class__.__name__}  gen={args.gen_model}  "
          f"judge_with_llm={not args.no_llm_judge}", flush=True)

    # ---- run each system ----
    tag = f"{args.dataset}_{args.split}"
    if args.seed != 0:
        tag += f"_s{args.seed}"
    system_aggs = {}
    for sname in args.systems:
        if sname not in SYSTEMS:
            print(f"  unknown system '{sname}', skipping", flush=True)
            continue
        cfg, label = SYSTEMS[sname]
        cfg = dict(cfg)
        cfg["max_hops"] = args.max_hops
        if "beam" in cfg:
            cfg["beam"] = args.beam
        if "base_beam" in cfg:
            cfg["base_beam"] = args.beam
        # pass theta_low for adaptive variants
        if cfg.get("name") == "adaptive":
            cfg["theta_low"] = args.theta_low
        print(f"\n=== System: {sname}  [{label}] ===", flush=True)
        controller = build_controller(cfg)
        retriever = GraphRAGRetriever(
            kg, backend, controller, link_topk=args.link_topk,
            judge_with_llm=not args.no_llm_judge,
            noisy_linking=args.noisy_linking,
            vector_rag_topk=args.vector_rag_topk if sname == "vector-rag" else 0,
        )
        reports = run_one(retriever, backend, examples, limit=args.limit,
                          corrupt_rate=args.corrupt_rate)
        agg = aggregate(reports)
        system_aggs[label] = agg
        out_path = os.path.join(args.out_dir, f"{tag}_{sname}", "reports.jsonl")
        save_reports(reports, out_path)
        print(f"  -> {out_path}", flush=True)

    # ---- Pareto comparison ----
    print("\n" + "=" * 72, flush=True)
    print(f"PARETO COMPARISON  [{tag}]  (sorted by mean input tokens/query)", flush=True)
    print("=" * 72, flush=True)
    tbl = pareto_table(system_aggs)
    print(tbl, flush=True)
    pareto_path = os.path.join(args.out_dir, f"pareto_{tag}.txt")
    with open(pareto_path, "w") as f:
        f.write(tbl + "\n")
    print(f"\nPareto table saved to {pareto_path}", flush=True)


if __name__ == "__main__":
    main()
