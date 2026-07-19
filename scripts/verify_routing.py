#!/usr/bin/env python3
"""Verify judge-routing: can gap-based rule or trained router approach oracle?

For each query we compute:
  - hop0 emb signals (top1, top2, gap, n_cand, mean, entropy)
  - question embedding (384-dim)
  - emb-judge F1 and LLM-judge F1 (full retrieval, K=3)
  - oracle label: which judge is better (or tie)

Then evaluate routing strategies on a held-out split:
  - Always-emb, Always-LLM (baselines)
  - Oracle (upper bound: pick best per query)
  - Gap-routing: small gap -> LLM, large gap -> emb (training-free), sweep threshold
  - Trained router: logistic regression on [question_emb + hop0 signals] -> emb/LLM
    (train on cal split, eval on test split)

Reports F1 and judge-call rate (cost proxy) for each, on a fixed test set.

Usage:
  python scripts/verify_routing.py --gen-model Qwen/Qwen2.5-7B-Instruct --n 200
"""
from __future__ import annotations
import argparse, os, sys, json, time, random
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agr import (load_metaqa, build_backend, build_controller, GraphRAGRetriever,
                 generate_answer, extract_answer_entities, score_query)


def collect(kg, backend, items, max_hops=3):
    """For each (depth, ex), collect hop0 signals + question emb + emb/LLM F1."""
    rows = []
    for idx, (depth, ex) in enumerate(items):
        ctrl = build_controller({"name": "fixed", "beam": 4, "max_hops": max_hops})
        # hop0 emb signals
        ret = GraphRAGRetriever(kg, backend, ctrl, link_topk=5, judge_with_llm=False)
        hits = ret.link_entities(ex.question)
        frontier = [e for e, _ in hits if e in kg.entities] or [e for e, _ in hits]
        cand = []
        for e in frontier:
            for (rel, nb, d) in kg.all_neighbors(e):
                cand.append((e, rel, nb) if d == "out" else (nb, rel, e))
        seen = set(); cand = [c for c in cand if not (c in seen or seen.add(c))][:16]
        emb_s = sorted(ret._embedding_scores(ex.question, cand), reverse=True) if cand else [0.0]
        top1 = emb_s[0]; top2 = emb_s[1] if len(emb_s) > 1 else 0.0
        gap = top1 - top2
        mean_s = float(np.mean(emb_s)); n_cand = len(cand)
        qemb = np.asarray(backend.embed([ex.question]), dtype=np.float32)[0]
        # emb F1
        re = ret.retrieve(ex.question); ae = generate_answer(backend, ex.question, re)
        ef1 = score_query(extract_answer_entities(ae.text), ex.answers)["f1"]
        # LLM F1
        retl = GraphRAGRetriever(kg, backend, ctrl, link_topk=5, judge_with_llm=True)
        rl = retl.retrieve(ex.question); al = generate_answer(backend, ex.question, rl)
        lf1 = score_query(extract_answer_entities(al.text), ex.answers)["f1"]
        rows.append({"depth": depth, "top1": top1, "top2": top2, "gap": gap,
                     "mean": mean_s, "n_cand": n_cand, "qemb": qemb.tolist(),
                     "emb_f1": ef1, "llm_f1": lf1})
        if (idx+1) % 25 == 0:
            print(f"  collected {idx+1}/{len(items)}", flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="./data/MetaQA")
    ap.add_argument("--gen-model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--n", type=int, default=200, help="total queries (split cal/test 50/50)")
    ap.add_argument("--out-dir", default="./results_routing")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    kg, _ = load_metaqa(args.data_dir, "3-hop")
    _, ex1 = load_metaqa(args.data_dir, "1-hop")
    _, ex2 = load_metaqa(args.data_dir, "2-hop")
    _, ex3 = load_metaqa(args.data_dir, "3-hop")
    per = args.n // 3
    rng = random.Random(1)
    rng.shuffle(ex1); rng.shuffle(ex2); rng.shuffle(ex3)
    items = [(1, e) for e in ex1[:per]] + [(2, e) for e in ex2[:per]] + [(3, e) for e in ex3[:per]]
    rng.shuffle(items)

    backend = build_backend({"kind": "hf", "gen_model": args.gen_model,
                             "embed_model": "sentence-transformers/all-MiniLM-L6-v2",
                             "device": "cuda", "dtype": "bfloat16"})
    print(f"Collecting signals + emb/LLM F1 for {len(items)} queries...", flush=True)
    rows = collect(kg, backend, items)
    with open(os.path.join(args.out_dir, "routing_data.json"), "w") as f:
        json.dump([{k: v for k, v in r.items() if k != "qemb"} for r in rows], f, indent=2)

    # split cal/test
    n = len(rows); half = n // 2
    cal, test = rows[:half], rows[half:]

    def f1_of(subset, choice_fn):
        """choice_fn(row) -> 'emb' or 'llm'. Returns mean F1 + llm-call rate."""
        f1s = []; llm_calls = 0
        for r in subset:
            c = choice_fn(r)
            f1s.append(r["llm_f1"] if c == "llm" else r["emb_f1"])
            if c == "llm": llm_calls += 1
        return float(np.mean(f1s)), llm_calls / len(subset)

    print(f"\n{'='*72}\nROUTING VERIFICATION [test n={len(test)}]\n{'='*72}", flush=True)
    print(f"{'strategy':<28} {'F1':>7} {'LLM-rate':>9}", flush=True)
    print("-"*48, flush=True)
    # baselines
    f,_ = f1_of(test, lambda r: "emb"); print(f"{'Always-emb':<28} {f:>7.3f} {0.0:>9.1%}", flush=True)
    f,_ = f1_of(test, lambda r: "llm"); print(f"{'Always-LLM':<28} {f:>7.3f} {1.0:>9.1%}", flush=True)
    # oracle
    f,rate = f1_of(test, lambda r: "llm" if r["llm_f1"] > r["emb_f1"] else "emb")
    print(f"{'Oracle (upper bound)':<28} {f:>7.3f} {rate:>9.1%}", flush=True)

    # gap-routing: small gap -> LLM. sweep threshold on cal, eval on test
    best_g, best_calf1 = 0.15, -1
    for g in np.arange(0.0, 0.5, 0.02):
        cf,_ = f1_of(cal, lambda r: "llm" if r["gap"] < g else "emb")
        if cf > best_calf1: best_calf1 = cf; best_g = g
    f,rate = f1_of(test, lambda r: "llm" if r["gap"] < best_g else "emb")
    print(f"{'Gap-routing (g<'+f'{best_g:.2f})':<28} {f:>7.3f} {rate:>9.1%}", flush=True)

    # trained router: logreg on [qemb + signals] -> better judge
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    def feats(r): return r["qemb"] + [r["top1"], r["top2"], r["gap"], r["mean"], r["n_cand"]]
    Xc = np.array([feats(r) for r in cal]); yc = np.array([1 if r["llm_f1"] > r["emb_f1"] else 0 for r in cal])
    Xt = np.array([feats(r) for r in test])
    if len(set(yc.tolist())) >= 2:
        sc = StandardScaler(); Xcs = sc.fit_transform(Xc); Xts = sc.transform(Xt)
        clf = LogisticRegression(max_iter=1000, C=1.0).fit(Xcs, yc)
        preds = clf.predict(Xts)
        f1s = [test[i]["llm_f1"] if preds[i] == 1 else test[i]["emb_f1"] for i in range(len(test))]
        rate = float(np.mean(preds))
        print(f"{'Trained router (logreg)':<28} {np.mean(f1s):>7.3f} {rate:>9.1%}", flush=True)
        # router with only cheap signals (no qemb)
        def feats2(r): return [r["top1"], r["top2"], r["gap"], r["mean"], r["n_cand"], r["depth"]]
        Xc2 = np.array([feats2(r) for r in cal]); Xt2 = np.array([feats2(r) for r in test])
        sc2 = StandardScaler(); clf2 = LogisticRegression(max_iter=1000).fit(sc2.fit_transform(Xc2), yc)
        preds2 = clf2.predict(sc2.transform(Xt2))
        f1s2 = [test[i]["llm_f1"] if preds2[i]==1 else test[i]["emb_f1"] for i in range(len(test))]
        print(f"{'Trained router (signals)':<28} {np.mean(f1s2):>7.3f} {np.mean(preds2):>9.1%}", flush=True)

    print(f"\nSaved to {args.out_dir}/routing_data.json", flush=True)


if __name__ == "__main__":
    main()
