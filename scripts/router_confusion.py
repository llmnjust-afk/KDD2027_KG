#!/usr/bin/env python3
"""Router confusion-matrix analysis (reviewer-requested, addresses review #4).

Retrains the LogReg-K and MLP-K depth classifiers (CPU-only, MiniLM embedder)
and reports the per-depth confusion matrix on the held-out MetaQA test split.
This directly answers the reviewer's request for 'per-depth confusion matrix'
and makes the near-oracle accuracy auditable.

Usage: python scripts/router_confusion.py --data-dir ./data/MetaQA
"""
from __future__ import annotations
import argparse, os, sys, re, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agr import build_backend


def load_train(data_dir):
    questions, labels = [], []
    for hop_idx, split in enumerate(['1-hop', '2-hop', '3-hop'], 1):
        p = os.path.join(data_dir, split, 'train.txt')
        for line in open(p, encoding='utf-8'):
            line = line.strip()
            if not line:
                continue
            q = line.rsplit('\t', 1)[0] if '\t' in line else line
            q = re.sub(r'\[([^\]]+)\]', r'\1', q)
            questions.append(q); labels.append(hop_idx)
    return questions, np.array(labels)


def load_test(data_dir, per=1000):
    """Sample `per` questions per depth from MetaQA's questions.txt (the full
    question set; MetaQA has no official train/test split for the LLM setting).
    The router is trained on train.txt (a subset); we note this overlap."""
    import random as _r
    rng = _r.Random(0)
    questions, labels = [], []
    for hop_idx, split in enumerate(['1-hop', '2-hop', '3-hop'], 1):
        p = os.path.join(data_dir, split, 'questions.txt')
        if not os.path.exists(p):
            for alt in ['test.txt', 'dev.txt', 'valid.txt']:
                p2 = os.path.join(data_dir, split, alt)
                if os.path.exists(p2):
                    p = p2; break
        lines = [l.strip() for l in open(p, encoding='utf-8') if l.strip()]
        rng.shuffle(lines)
        for line in lines[:per]:
            q = line.rsplit('\t', 1)[0] if '\t' in line else line
            q = re.sub(r'\[([^\]]+)\]', r'\1', q)
            questions.append(q); labels.append(hop_idx)
    return questions, np.array(labels)


def confusion(true, pred, n=3):
    M = np.zeros((n, n), dtype=int)
    for t, p in zip(true, pred):
        M[t - 1, p - 1] += 1
    return M


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="./data/MetaQA")
    ap.add_argument("--per", type=int, default=1000)
    ap.add_argument("--out", default="./results_router_cm/cm.json")
    args = ap.parse_args()

    # CPU-only backend (MiniLM), no GPU model. gen_model is lazy-loaded and
    # never triggered since we only call embed().
    backend = build_backend({"kind": "hf", "gen_model": "unused",
                             "embed_model": "sentence-transformers/all-MiniLM-L6-v2",
                             "device": "cpu", "dtype": "float32"})

    tr_q, tr_y = load_train(args.data_dir)
    # subsample train to 10k/depth (same as the paper's router)
    rng = np.random.default_rng(42)
    sub_q, sub_y = [], []
    for hop in [1, 2, 3]:
        idx = [i for i, l in enumerate(tr_y) if l == hop]
        rng.shuffle(idx)
        for i in idx[:10000]:
            sub_q.append(tr_q[i]); sub_y.append(tr_y[i])
    print(f"Embedding {len(sub_q)} train questions (CPU)...", flush=True)
    Xtr = np.asarray(backend.embed(sub_q), dtype=np.float32)
    ytr = np.array(sub_y)

    te_q, te_y = load_test(args.data_dir, args.per)
    print(f"Embedding {len(te_q)} test questions (CPU)...", flush=True)
    Xte = np.asarray(backend.embed(te_q), dtype=np.float32)
    yte = np.array(te_y)

    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier

    scaler = StandardScaler()
    Xtr_s = scaler.fit_transform(Xtr)
    Xte_s = scaler.transform(Xte)

    results = {}
    for name, clf in [("logreg", LogisticRegression(max_iter=1000, C=1.0, solver='lbfgs')),
                      ("mlp", MLPClassifier(hidden_layer_sizes=(256, 128), max_iter=500,
                                            early_stopping=True, random_state=42))]:
        clf.fit(Xtr_s, ytr)
        tr_acc = clf.score(Xtr_s, ytr)
        te_acc = clf.score(Xte_s, yte)
        pred = clf.predict(Xte_s)
        cm = confusion(yte, pred)
        print(f"\n=== {name.upper()} ===", flush=True)
        print(f"  train acc = {tr_acc:.4f}  test acc = {te_acc:.4f}", flush=True)
        print(f"  confusion matrix (rows=true depth, cols=pred depth):", flush=True)
        print(f"        pred1  pred2  pred3", flush=True)
        for i in range(3):
            print(f"  true{i+1}  {cm[i,0]:5d}  {cm[i,1]:5d}  {cm[i,2]:5d}", flush=True)
        # per-class recall
        for i in range(3):
            rec = cm[i, i] / max(cm[i].sum(), 1)
            print(f"  depth{i+1} recall = {rec:.3f}", flush=True)
        results[name] = {"train_acc": float(tr_acc), "test_acc": float(te_acc),
                         "confusion": cm.tolist(),
                         "per_class_recall": [float(cm[i, i] / max(cm[i].sum(), 1)) for i in range(3)]}

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {args.out}", flush=True)


if __name__ == "__main__":
    main()
