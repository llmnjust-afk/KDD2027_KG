#!/usr/bin/env python3
"""P1.4: Can a RICHER router beat the predictability ceiling?

Loads the routing_data.json collected earlier (per query: question embedding +
hop-0 signals + emb_f1 + llm_f1), builds progressively richer feature sets and
stronger models, and measures the test AUROC for predicting the oracle label
"LLM judge better than embedding judge for this query", plus the routed F1 that
results. If even the richest features + strongest models give AUROC ~0.5 and
routed-F1 below always-LLM, the predictability ceiling is a property of the
information, not of the policy class.

Usage: python scripts/richer_router.py --data results_routing/routing_data.json
"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np


def auroc(scores, labels):
    scores = np.asarray(scores, float); labels = np.asarray(labels, int)
    if len(set(labels.tolist())) < 2:
        return float("nan")
    order = np.argsort(-scores); ranks = np.empty_like(order, float)
    ranks[order] = np.arange(1, len(scores) + 1)
    for s in np.unique(scores):
        m = scores == s; ranks[m] = np.mean(np.where(m)[0] + 1)
    npos = labels.sum(); nneg = len(labels) - npos
    if npos == 0 or nneg == 0:
        return float("nan")
    return float((ranks[labels == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="results_routing/routing_data.json")
    ap.add_argument("--qemb-data", default="results_routing/routing_qemb.json",
                    help="optional file with question embeddings per query")
    ap.add_argument("--out", default="results_routing/richer_router.json")
    args = ap.parse_args()

    rows = json.load(open(args.data))
    # rows have: depth, top1, top2, gap, mean, n_cand, emb_f1, llm_f1 (+ maybe qemb)
    n = len(rows)
    print(f"loaded {n} rows", flush=True)

    # label: 1 if LLM strictly better than emb for this query
    y = np.array([1 if r["llm_f1"] > r["emb_f1"] + 1e-9 else 0 for r in rows])
    emb_f1 = np.array([r["emb_f1"] for r in rows])
    llm_f1 = np.array([r["llm_f1"] for r in rows])
    print(f"positives (LLM better): {int(y.sum())}/{n} = {y.mean():.2%}", flush=True)
    print(f"always-emb F1={emb_f1.mean():.3f}  always-LLM F1={llm_f1.mean():.3f}  "
          f"oracle F1={np.maximum(emb_f1,llm_f1).mean():.3f}", flush=True)

    # feature sets of increasing richness
    def feat_signals(r):
        return [r["top1"], r["top2"], r["gap"], r["mean"], r["n_cand"]]
    def feat_signals_depth(r):
        return feat_signals(r) + [r.get("depth", 0)]
    has_qemb = "qemb" in rows[0]
    feat_sets = {"signals(5)": feat_signals, "signals+depth(6)": feat_signals_depth}
    if has_qemb:
        feat_sets["qemb(384)"] = lambda r: list(r["qemb"])
        feat_sets["qemb+signals+depth"] = lambda r: list(r["qemb"]) + feat_signals_depth(r)

    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import cross_val_predict

    models = {
        "LogReg": LogisticRegression(max_iter=2000, C=1.0),
        "MLP(256,128)": MLPClassifier(hidden_layer_sizes=(256, 128), max_iter=800,
                                      early_stopping=True, random_state=0),
        "RandomForest": RandomForestClassifier(n_estimators=300, random_state=0),
    }

    print(f"\n{'features':<22}{'model':<16}{'CV-AUROC':>9}{'routed-F1':>11}", flush=True)
    print("-" * 60, flush=True)
    results = {}
    for fname, ffn in feat_sets.items():
        X = np.array([ffn(r) for r in rows], float)
        Xs = StandardScaler().fit_transform(X)
        for mname, model in models.items():
            try:
                # 5-fold cross-validated predictions (proper held-out)
                proba = cross_val_predict(model, Xs, y, cv=5, method="predict_proba")[:, 1]
                a = auroc(proba, y)
                # routed F1: pick LLM when proba>0.5 else emb
                routed = np.where(proba > 0.5, llm_f1, emb_f1).mean()
                results[f"{fname}|{mname}"] = {"auroc": a, "routed_f1": float(routed)}
                print(f"{fname:<22}{mname:<16}{a:>9.3f}{routed:>11.3f}", flush=True)
            except Exception as e:
                print(f"{fname:<22}{mname:<16}  failed: {str(e)[:30]}", flush=True)

    best_auroc = max(v["auroc"] for v in results.values() if not np.isnan(v["auroc"]))
    best_routed = max(v["routed_f1"] for v in results.values())
    print(f"\nBest CV-AUROC across all richer routers: {best_auroc:.3f}", flush=True)
    print(f"Best routed-F1: {best_routed:.3f}  vs  always-LLM {llm_f1.mean():.3f}  "
          f"oracle {np.maximum(emb_f1,llm_f1).mean():.3f}", flush=True)
    verdict = ("CEILING CONFIRMED: even the richest router is near-random and "
               "cannot beat always-LLM" if best_auroc < 0.6 and best_routed <= llm_f1.mean() + 0.005
               else "router shows signal -- revisit ceiling claim")
    print(f"VERDICT: {verdict}", flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump({"results": results, "best_auroc": best_auroc, "best_routed_f1": best_routed,
               "always_llm": float(llm_f1.mean()), "oracle": float(np.maximum(emb_f1,llm_f1).mean()),
               "verdict": verdict}, open(args.out, "w"), indent=2)
    print(f"\nSaved to {args.out}", flush=True)


if __name__ == "__main__":
    main()
