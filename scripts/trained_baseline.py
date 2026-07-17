#!/usr/bin/env python3
"""Trained difficulty classifier baseline (B1).

Trains a logistic regression on question embeddings -> predict hop depth (1/2/3).
At inference, the predicted K sets the per-query fixed budget (a "trained router").
Runs through the SAME retrieval engine as AquaRAG and Fixed, so the comparison
isolates "learned adaptation" vs "training-free adaptation" vs "no adaptation".
"""
from __future__ import annotations
import argparse, json, os, sys, time, re
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agr import (load_metaqa, build_backend, GraphRAGRetriever, generate_answer,
                 extract_answer_entities, score_query, aggregate, pareto_table,
                 save_reports, QueryReport, FixedController, AdaptiveController)
from dataclasses import dataclass


@dataclass
class TrainedKController:
    classifier: object; scaler: object; embedder: object
    beam: int = 4; max_hops: int = 3
    _predicted_k: int = 3
    def _predict_k(self, question):
        emb = self.embedder.embed([question])[0]
        emb = np.asarray(emb, dtype=np.float32).reshape(1, -1)
        emb_s = self.scaler.transform(emb)
        self._predicted_k = int(self.classifier.predict(emb_s)[0])
        return self._predicted_k
    def decide_use_graph(self, q, eh, ctx):
        self._predict_k(q)  # set per-query K before retrieval starts
        return len(eh) > 0
    def beam_for_hop(self, q, hi, fs, sc, ctx): return self.beam
    def should_stop(self, q, hi, fr, pf, ss, ctx): return hi + 1 >= self._predicted_k


def train_classifier(data_dir, backend):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    questions, labels = [], []
    for hop_idx, split in enumerate(['1-hop','2-hop','3-hop'], 1):
        p = os.path.join(data_dir, split, 'train.txt')
        if not os.path.exists(p): raise FileNotFoundError(p)
        for line in open(p, encoding='utf-8'):
            line = line.strip()
            if not line: continue
            q = line.rsplit('\t',1)[0] if '\t' in line else line
            q = re.sub(r'\[([^\]]+)\]', r'\1', q)
            questions.append(q); labels.append(hop_idx)
    rng = np.random.default_rng(42)
    sub_q, sub_l = [], []
    for hop in [1,2,3]:
        idx = [i for i,l in enumerate(labels) if l==hop]
        rng.shuffle(idx)
        for i in idx[:min(10000,len(idx))]:
            sub_q.append(questions[i]); sub_l.append(labels[i])
    print(f"Embedding {len(sub_q)} train questions...", flush=True)
    X = np.asarray(backend.embed(sub_q), dtype=np.float32)
    y = np.array(sub_l)
    scaler = StandardScaler(); X_s = scaler.fit_transform(X)
    clf = LogisticRegression(max_iter=1000, C=1.0, solver='lbfgs')
    clf.fit(X_s, y)
    print(f"Train accuracy: {clf.score(X_s,y):.3f}", flush=True)
    return clf, scaler


def run_system(name, kg, backend, controller, examples, limit):
    ret = GraphRAGRetriever(kg, backend, controller, link_topk=5, judge_with_llm=True)
    reports = []
    for i, ex in enumerate(examples[:limit]):
        t0 = time.time(); c0 = backend.usage.n_calls; tk0 = backend.usage.n_input_tokens
        r = ret.retrieve(ex.question)
        a = generate_answer(backend, ex.question, r)
        p = extract_answer_entities(a.text)
        sc = score_query(p, ex.answers)
        reports.append(QueryReport(qid=ex.qid, question=ex.question, gold=ex.answers,
            pred=p, hit1=sc["hit1"], f1=sc["f1"], exact=sc["exact"],
            used_graph=r.used_graph, n_hops=r.n_hops_executed,
            n_llm_calls=backend.usage.n_calls-c0, n_input_tokens=backend.usage.n_input_tokens-tk0,
            n_output_tokens=a.n_output_tokens))
        if (i+1)%50==0 or i==limit-1:
            agg=aggregate(reports)
            print(f"  [{name} {i+1}/{limit}] F1={agg.mean_f1:.3f} toks={agg.mean_n_input_tokens:.0f} t={time.time()-t0:.1f}s",flush=True)
    return reports


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="./data/MetaQA")
    ap.add_argument("--gen-model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--out-dir", default="./results_trained")
    ap.add_argument("--split", default="2-hop")
    args = ap.parse_args()

    kg, examples = load_metaqa(args.data_dir, args.split)
    print(f"KG: {len(kg)} triples | {len(examples)} test Q (using {args.limit})", flush=True)
    backend = build_backend({"kind":"hf","gen_model":args.gen_model,
        "embed_model":"sentence-transformers/all-MiniLM-L6-v2","device":"cuda","dtype":"bfloat16"})
    print(f"Backend: {args.gen_model}", flush=True)

    print("\n=== Training classifier ===", flush=True)
    clf, scaler = train_classifier(args.data_dir, backend)

    system_aggs = {}
    print("\n=== Fixed (K=3) ===", flush=True)
    r_fixed = run_system("Fixed", kg, backend, FixedController(max_hops=3,beam=4), examples, args.limit)
    system_aggs["Fixed (K=3)"] = aggregate(r_fixed)

    print("\n=== Adaptive (ours) ===", flush=True)
    r_ada = run_system("Adaptive", kg, backend, AdaptiveController(max_hops=3,base_beam=4), examples, args.limit)
    system_aggs["Adaptive (ours)"] = aggregate(r_ada)

    print("\n=== Trained-K ===", flush=True)
    ctrl = TrainedKController(classifier=clf, scaler=scaler, embedder=backend, beam=4, max_hops=3)
    r_trained = run_system("Trained-K", kg, backend, ctrl, examples, args.limit)
    system_aggs["Trained-K (logreg)"] = aggregate(r_trained)

    os.makedirs(args.out_dir, exist_ok=True)
    for name, reps in [("fixed",r_fixed),("adaptive",r_ada),("trained",r_trained)]:
        save_reports(reps, os.path.join(args.out_dir, f"metaqa_{args.split}_{name}", "reports.jsonl"))

    print("\n"+"="*70, flush=True)
    print(f"COMPARISON [{args.split}, n={args.limit}]", flush=True)
    print("="*70, flush=True)
    print(pareto_table(system_aggs), flush=True)


if __name__ == "__main__":
    main()
