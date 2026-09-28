#!/usr/bin/env python3
"""P6 CPU analysis: mid-retrieval-feature routers + joint (K,B,judge) router.

Input : results_p6_features/features.json (from scripts/p456_experiments.py --mode features)
Output: results_p6_analysis/p6_analysis.json

Task A: judge-type router (K3B4 LLM vs K3 emb) on mid-retrieval features
        (hop-1 post-expansion stats, frontier overlap, expansion dynamics)
        vs the old hop-0-only feature set. Cross-validated AUROC + routed F1.
Task B: joint router over 5 actions {K1B4,K2B4,K2B8,K3B4} x {LLM} + {K3 emb}:
        out-of-fold routed F1 vs always-best-single-arm, joint oracle, and
        rate-matched mixture references; paired bootstrap CI.

Usage: python3 scripts/p6_router_analysis.py --features results_p6_features/features.json
"""
from __future__ import annotations
import argparse, json, os
import numpy as np

HOP0_KEYS = ["top1", "top2", "gap", "mean", "n_cand"]


def hop_feats(hops, hop_idx):
    if hop_idx >= len(hops):
        return {f"h{hop_idx}_{k}": 0.0 for k in HOP0_KEYS} | \
               {f"h{hop_idx}_overlap": 0.0, f"h{hop_idx}_missing": 1.0}
    h = hops[hop_idx]
    tk = h.get("topk_scores") or []
    top1 = float(h.get("top_score", tk[0] if tk else 0.0))
    top2 = float(tk[1]) if len(tk) > 1 else 0.0
    return {f"h{hop_idx}_top1": top1, f"h{hop_idx}_top2": top2,
            f"h{hop_idx}_gap": top1 - top2, f"h{hop_idx}_mean": float(h.get("mean_topk", 0.0)),
            f"h{hop_idx}_n_cand": float(h.get("n_candidates", 0)),
            f"h{hop_idx}_overlap": float(h.get("frontier_overlap", 0.0)),
            f"h{hop_idx}_missing": 0.0}


def feature_rows(rows):
    X_mid, X_hop0, depths = [], [], []
    for r in rows:
        hops = r.get("probe_hops") or []
        f0 = hop_feats(hops, 0)
        f1 = hop_feats(hops, 1)
        dyn = {"n_cand_ratio": (f1["h1_n_cand"] / f0["h0_n_cand"]) if f0["h0_n_cand"] else 0.0,
               "top1_drop": f0["h0_top1"] - f1["h1_top1"],
               "depth": float(r.get("depth") or 0)}
        X_mid.append({**f0, **f1, **dyn})
        X_hop0.append({**f0, **dyn})
        depths.append(float(r.get("depth") or 0))
    return X_hop0, X_mid, depths


def to_matrix(rows, dicts, use_qemb):
    cols = sorted({k for d in dicts for k in d})
    X = np.array([[d.get(c, 0.0) for c in cols] for d in dicts], dtype=np.float32)
    if use_qemb:
        Q = np.array([r["qemb"] for r in rows], dtype=np.float32)
        X = np.hstack([X, Q])
    return X


def cv_auroc_f1(y, X, model_kind, n_folds=5, seed=0):
    from sklearn.model_selection import StratifiedKFold
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    aurocs, f1s, rates = [], [], []
    for tr, te in skf.split(X, y):
        if len(set(y[tr].tolist())) < 2:
            aurocs.append(0.5); f1s.append(float(np.mean(y[te]))); rates.append(0.0); continue
        sc = StandardScaler()
        Xtr = sc.fit_transform(X[tr]); Xte = sc.transform(X[te])
        if model_kind == "logreg":
            clf = LogisticRegression(max_iter=2000, C=1.0)
        else:
            clf = RandomForestClassifier(n_estimators=300, min_samples_leaf=3, random_state=seed)
        clf.fit(Xtr, y[tr])
        p = clf.predict_proba(Xte)[:, 1]
        aurocs.append(roc_auc_score(y[te], p))
        f1s.append(p); rates.append(te)
    return aurocs, f1s, rates


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="./results_p6_features/features.json")
    ap.add_argument("--out-dir", default="./results_p6_analysis")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    data = json.load(open(args.features))
    rows = data["rows"]
    n = len(rows)
    arms = sorted({a for r in rows for a in r.get("outcomes", {})})
    print(f"n={n} arms={arms}")

    f_emb = np.array([r["outcomes"]["K3_emb"]["f1"] for r in rows], dtype=np.float32)
    f_llm = np.array([r["outcomes"]["K3B4_llm"]["f1"] for r in rows], dtype=np.float32)
    y = (f_llm > f_emb).astype(np.int8)

    X_hop0, X_mid, _ = feature_rows(rows)
    results = {"n": n, "arms": arms, "base_rate_llm_better": float(y.mean())}

    # Task A: judge-type router, hop0-only vs mid-retrieval features
    for name, dicts, use_qemb in [("hop0_signals", X_hop0, False),
                                  ("hop0_plus_qemb", X_hop0, True),
                                  ("mid_retrieval", X_mid, False),
                                  ("mid_retrieval_plus_qemb", X_mid, True)]:
        X = to_matrix(rows, dicts, use_qemb)
        for kind in ["logreg", "rf"]:
            aurocs, _, _ = cv_auroc_f1(y, X, kind)
            results.setdefault("taskA_judge_router", {})[f"{name}/{kind}"] = {
                "cv_auroc_mean": float(np.mean(aurocs)),
                "cv_auroc_folds": [float(a) for a in aurocs]}
    # routed F1 for the best mid-retrieval router (out-of-fold, threshold 0.5)
    from sklearn.model_selection import StratifiedKFold
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    X = to_matrix(rows, X_mid, False)
    oof = np.zeros(n, dtype=np.float32)
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(X, y):
        sc = StandardScaler(); Xtr = sc.fit_transform(X[tr]); Xte = sc.transform(X[te])
        clf = RandomForestClassifier(n_estimators=300, min_samples_leaf=3, random_state=0)
        clf.fit(Xtr, y[tr]); oof[te] = clf.predict_proba(Xte)[:, 1]
    route_llm = oof >= 0.5
    f1_routed = np.where(route_llm, f_llm, f_emb)
    llm_rate = float(route_llm.mean())
    # rate-matched mixture reference: choose arm by coin flip with p = llm_rate
    rng = np.random.default_rng(0)
    mix_f1 = np.where(rng.random(n) < llm_rate, f_llm, f_emb)
    def paired_boot(a, b, B=4000, seed=1):
        d = a - b; rng2 = np.random.default_rng(seed)
        idx = rng2.integers(0, n, size=(B, n))
        means = d[idx].mean(axis=1)
        lo, hi = np.percentile(means, [2.5, 97.5])
        return float(d.mean()), float(lo), float(hi), float((means <= 0).mean() * 2 if d.mean() > 0 else (means >= 0).mean() * 2)
    dm, lo, hi, p = paired_boot(f1_routed, mix_f1)
    results["taskA_judge_router"]["mid_router_vs_rate_matched_mixture"] = {
        "router_f1": float(f1_routed.mean()), "mixture_f1": float(mix_f1.mean()),
        "llm_rate": llm_rate, "delta": dm, "ci": [lo, hi], "p_two_sided_approx": p}

    # Task B: joint router over arms
    F = np.array([[r["outcomes"][a]["f1"] for a in arms] for r in rows], dtype=np.float32)
    T = np.array([[r["outcomes"][a]["toks"] for a in arms] for r in rows], dtype=np.float32)
    best = F.argmax(axis=1)
    oracle_f1 = F.max(axis=1)
    always_best = float(F[:, int(np.argmax(F.mean(axis=0)))].mean())
    # out-of-fold joint router (RF, mid features + qemb)
    Xj = to_matrix(rows, X_mid, True)
    oof_arm = np.zeros(n, dtype=np.int64)
    skf = StratifiedKFold(5, shuffle=True, random_state=0)
    for tr, te in skf.split(Xj, best):
        sc = StandardScaler(); Xtr = sc.fit_transform(Xj[tr]); Xte = sc.transform(Xj[te])
        clf = RandomForestClassifier(n_estimators=400, min_samples_leaf=2, random_state=0)
        clf.fit(Xtr, best[tr]); oof_arm[te] = clf.predict(Xte)
    routed = F[np.arange(n), oof_arm]
    usage = {arms[a]: float((oof_arm == a).mean()) for a in range(len(arms))}
    dm, lo, hi, p = paired_boot(routed, np.full(n, always_best))
    results["taskB_joint_router"] = {
        "always_best_arm": arms[int(np.argmax(F.mean(axis=0)))],
        "always_best_f1": always_best,
        "always_best_toks": float(T[:, int(np.argmax(F.mean(axis=0)))].mean()),
        "joint_oracle_f1": float(oracle_f1.mean()),
        "joint_oracle_toks": float(T[np.arange(n), best].mean()),
        "router_f1": float(routed.mean()), "router_arm_usage": usage,
        "router_vs_always_best": {"delta": dm, "ci": [lo, hi], "p_two_sided_approx": p}}

    # per-arm pool summary
    results["arm_summary"] = {arms[a]: {"f1": float(F[:, a].mean()), "toks": float(T[:, a].mean())}
                              for a in range(len(arms))}

    with open(os.path.join(args.out_dir, "p6_analysis.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))
    print("P6_ANALYSIS_DONE")


if __name__ == "__main__":
    main()
