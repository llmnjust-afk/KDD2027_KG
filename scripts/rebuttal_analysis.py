#!/usr/bin/env python3
"""Rebuttal analyses for KDD2027 submission 1972 (all CPU, from logged per-query data).

Blocks:
  A. Table-2 dominance re-audit: discrete dominance, paired bootstrap CIs,
     randomized-mixture frontier (f7vF-W1/Q, xxtL-W2).
  B. Held-out validation/test re-selection protocol (f7vF-W4, V9mY-Q1, xxtL-W5).
  C. Router learning curves + utility-aware routers: gain regression, U-lambda C,
     gain-weighted classification, matched-cost (routing-rate) comparison (f7vF-W3, V9mY-W1).
  D. Joint (K,B,judge) outcome oracle from logs (31Rd-Q1, partial; router pending GPU).
  E. Table-9 latency chord verification (f7vF-W1).
Outputs -> results_rebuttal/*.json
"""
from __future__ import annotations
import json, os, sys, itertools
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results_rebuttal")
os.makedirs(OUT, exist_ok=True)
rng_global = np.random.default_rng(0)


def load_reports(d):
    p = os.path.join(ROOT, d, "reports.jsonl")
    return {json.loads(l)["qid"]: json.loads(l) for l in open(p) if l.strip()}


def agg(rows):
    f1 = np.array([r["f1"] for r in rows])
    tok = np.array([r["n_input_tokens"] for r in rows], dtype=float)
    return dict(f1=float(f1.mean()), toks=float(tok.mean()), n=len(rows), f1_per=f1)


def paired_bootstrap(a, b, B=10000, seed=0):
    rng = np.random.default_rng(seed)
    n = len(a); d = a - b
    idx = rng.integers(0, n, size=(B, n))
    deltas = d[idx].mean(axis=1)
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return dict(delta=float(d.mean()), ci=[float(lo), float(hi)],
                p_two_sided=float(2 * min((deltas <= 0).mean(), (deltas >= 0).mean())))


# ---------------------------------------------------------------- Block A
print("=" * 30, "\nBLOCK A: Table-2 dominance audit (results_full_pareto, n=240)")
FP = {
    "Fixed-emb K=3":        "results_full_pareto/metaqa_mixed_Fixed-emb_K3",
    "Fixed-LLM K=1 B=4":    "results_full_pareto/metaqa_mixed_Fixed-LLM_K1_B4",
    "AdaptiveJudge T=0.55": "results_full_pareto/metaqa_mixed_AdaptiveJudge_T0.55",
    "Fixed-LLM K=2 B=4":    "results_full_pareto/metaqa_mixed_Fixed-LLM_K2_B4",
    "Fixed-LLM K=2 B=8":    "results_full_pareto/metaqa_mixed_Fixed-LLM_K2_B8",
    "AdaptiveJudge T=0.75": "results_full_pareto/metaqa_mixed_AdaptiveJudge_T0.75",
    "Fixed-LLM K=3 B=4":    "results_full_pareto/metaqa_mixed_Fixed-LLM_K3_B4",
}
data = {k: load_reports(v) for k, v in FP.items()}
qids = sorted(set.intersection(*[set(v) for v in data.values()]))
print("aligned qids:", len(qids))
per = {k: np.array([data[k][q]["f1"] for q in qids]) for k in data}
toks = {k: float(np.mean([data[k][q]["n_input_tokens"] for q in qids])) for k in data}
aggA = {k: dict(f1=float(per[k].mean()), toks=toks[k]) for k in data}
print(json.dumps(aggA, indent=1))

# discrete dominance
points = {k: (toks[k], float(per[k].mean())) for k in data}
dom = {}
for k, (t, f) in points.items():
    dominators = [j for j, (t2, f2) in points.items()
                  if j != k and f2 >= f and t2 <= t and (f2 > f or t2 < t)]
    dom[k] = dominators
    print(f"{k:26s} ({t:7.1f}, {f:.4f})  dominated-by: {dominators}")

# paired bootstrap for key contrasts
bootA = {}
for a, b in [("Fixed-LLM K=2 B=8", "AdaptiveJudge T=0.75"),
             ("Fixed-LLM K=2 B=8", "AdaptiveJudge T=0.55"),
             ("Fixed-emb K=3", "AdaptiveJudge T=0.55"),
             ("Fixed-LLM K=2 B=8", "Fixed-LLM K=2 B=4"),
             ("Fixed-LLM K=2 B=8", "Fixed-LLM K=3 B=4"),
             ("Fixed-emb K=3", "Fixed-LLM K=2 B=4")]:
    bootA[f"{a} vs {b}"] = paired_bootstrap(per[a], per[b])
    print(f"boot {a} vs {b}: {bootA[f'{a} vs {b}']}")

# mixture frontier between the two frontier static endpoints
# NOTE: points[k] = (tokens, f1)
emb, llm = "Fixed-emb K=3", "Fixed-LLM K=2 B=8"
t0, f0 = points[emb]; t1, f1_ = points[llm]
def mixture(p):
    return ((1 - p) * f0 + p * f1_, (1 - p) * t0 + p * t1)  # (F1, tokens)
mixA = {}
for target in ["AdaptiveJudge T=0.55", "AdaptiveJudge T=0.75"]:
    tt, tf = points[target]
    p = (tf - f0) / (f1_ - f0)
    fm, tm = mixture(min(max(p, 0.0), 1.0))
    mixA[target] = dict(
        target_tokens=float(tt), target_f1=float(tf),
        p_for_same_f1=float(min(max(p, 0.0), 1.0)),
        mixture_tokens_for_same_f1=float(tm),
        mixture_f1_at_target_tokens=float(f0 + (tt - t0) / (t1 - t0) * (f1_ - f0)),
        discretely_dominated=bool(dom[target]),
    )
    print(f"mixture {target}: p={mixA[target]['p_for_same_f1']:.3f} -> tokens={tm:.1f} "
          f"(target {tt:.1f}); chord F1 at target tokens = {mixA[target]['mixture_f1_at_target_tokens']:.4f}")
# annotated-hop as reported in Table 2 (0.444 @ 999) -> mixture check
tf, tt = 0.444, 999.0
p = (tf - f0) / (f1_ - f0)
mixA["Annotated-hop (Table 2 coords)"] = dict(
    target_tokens=tt, target_f1=tf, p_for_same_f1=float(p),
    mixture_tokens_for_same_f1=float((1 - p) * t0 + p * t1),
    mixture_f1_at_target_tokens=float(f0 + (tt - t0) / (t1 - t0) * (f1_ - f0)))
print("mixture Annotated-hop:", json.dumps(mixA["Annotated-hop (Table 2 coords)"], indent=1))

# complete (K,B) LLM grid from the same-240-qid mixed_pareto run (different run/engine caveat)
MP = {}
mp_base = "results_mixed_pareto"
for d in sorted(os.listdir(mp_base)):
    p_ = os.path.join(mp_base, d, "reports.jsonl")
    if os.path.exists(p_):
        r = load_reports(os.path.join(mp_base, d))
        if len(set(r) & set(qids)) == 240:
            MP[d] = r
print("mixed_pareto configs aligned to same 240 qids:", sorted(MP))
grid = {}
for k, r in MP.items():
    rows = [r[q] for q in qids]
    grid[k] = dict(f1=float(np.mean([x["f1"] for x in rows])),
                   toks=float(np.mean([x["n_input_tokens"] for x in rows])))
# discrete dominance over the union grid (12-config LLM grid + full_pareto 7)
union = {}
union.update({f"[FP] {k}": points[k] for k in points})
union.update({f"[MP] {k}": (v["toks"], v["f1"]) for k, v in grid.items()})
dom2 = {}
for k, (t, f) in union.items():
    dom2[k] = [j for j, (t2, f2) in union.items()
               if j.split("]")[0] == k.split("]")[0]   # within-run only (FP vs MP are different runs)
               and j != k and f2 >= f and t2 <= t and (f2 > f or t2 < t)]
annotated_mp = union.get("[MP] Oracle-depth")
print("MP Oracle-depth (annotated-hop, naive engine):", annotated_mp)
print("Annotated within-MP dominators:", dom2.get("[MP] Oracle-depth", []))
json.dump(dict(agg=aggA, dominance=dom, bootstrap=bootA, mixture=mixA,
               mp_grid=grid, union_dominance=dom2,
               mp_oracle_depth=annotated_mp),
          open(os.path.join(OUT, "A_pareto_audit.json"), "w"), indent=1)

# ---------------------------------------------------------------- Block B
print("=" * 30, "\nBLOCK B: held-out validation/test re-selection")
def stratified_split(qids, seed=0):
    groups = {}
    for q in qids:
        g = q.split("_")[1]  # 1-hop/2-hop/3-hop
        groups.setdefault(g, []).append(q)
    rng = np.random.default_rng(seed)
    val, test = [], []
    for g, qs in sorted(groups.items()):
        qs = sorted(qs); rng.shuffle(qs)
        h = len(qs) // 2
        val += qs[:h]; test += qs[h:]
    return val, test

def select_eval(perq, val, test):
    scores = {k: float(np.mean([perq[k][q] for q in val])) for k in perq}
    best = max(scores, key=scores.get)
    testF = {k: float(np.mean([perq[k][q] for q in test])) for k in perq}
    return scores, best, testF

val, test = stratified_split(qids, seed=0)
print(f"val n={len(val)} test n={len(test)} (stratified by depth, seed 0)")
# global grid on full_pareto fixed configs only
fixed = {k: per[k] for k in data if k.startswith("Fixed")}
perq_fixed = {k: {q: data[k][q]["f1"] for q in qids} for k in fixed}
scores, best, testF = select_eval(perq_fixed, val, test)
print("val scores:", {k: round(v, 4) for k, v in sorted(scores.items(), key=lambda x: -x[1])})
print(f"val-selected global: {best}  test F1={testF[best]:.4f} (val F1={scores[best]:.4f})")
# val-selected tau
perq_tau = {k: {q: data[k][q]["f1"] for q in qids} for k in data if "AdaptiveJudge" in k}
s2, best_tau, testF2 = select_eval(perq_tau, val, test)
print(f"val-selected tau: {best_tau}  test F1={testF2[best_tau]:.4f}")
gapboot = paired_bootstrap(np.array([perq_fixed[best][q] for q in test]),
                           np.array([perq_tau[best_tau][q] for q in test]))
print("global vs adaptive on test half:", gapboot)
# same on the 12-config LLM grid (mixed_pareto run, same qids)
perq_mp = {k: {q: r[q]["f1"] for q in qids} for k, r in MP.items() if k.startswith("Fixed")}
s3, best3, testF3 = select_eval(perq_mp, val, test)
print("MP val scores:", {k: round(v, 4) for k, v in sorted(s3.items(), key=lambda x: -x[1])})
print(f"MP val-selected: {best3} test F1={testF3[best3]:.4f}")
# adaptive on MP
perq_mp_ada = {k: {q: r[q]["f1"] for q in qids} for k, r in MP.items() if not k.startswith("Fixed")}
s4, best4, testF4 = select_eval(perq_mp_ada, val, test)
print(f"MP adaptive val-selected: {best4} test F1={testF4[best4]:.4f}")
gapboot2 = paired_bootstrap(np.array([perq_mp[best3][q] for q in test]),
                            np.array([perq_mp_ada[best4][q] for q in test]))
print("MP global vs adaptive on test half:", gapboot2)
# selection-stability across validation draws
wins = {}
for seed in range(20):
    v, t = stratified_split(qids, seed=seed)
    _, b, _ = select_eval(perq_fixed, v, t)
    wins[b] = wins.get(b, 0) + 1
    _, b3, _ = select_eval(perq_mp, v, t)
    wins["MP:" + b3] = wins.get("MP:" + b3, 0) + 1
print("selection stability over 20 val draws:", wins)
json.dump(dict(val_n=len(val), test_n=len(test), fp_scores=scores, fp_best=best,
               fp_best_test=testF[best], fp_tau_best=best_tau,
               fp_tau_best_test=testF2[best_tau], fp_gap_boot=gapboot,
               mp_scores=s3, mp_best=best3, mp_best_test=testF3[best3],
               mp_adaptive_best=best4, mp_adaptive_test=testF4[best4],
               mp_gap_boot=gapboot2, stability=wins),
          open(os.path.join(OUT, "B_valtest_protocol.json"), "w"), indent=1)

# ---------------------------------------------------------------- Block C
print("=" * 30, "\nBLOCK C: routers on routing_data.json (n=300)")
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score

rows = json.load(open(os.path.join(ROOT, "results_routing", "routing_data.json")))
X = np.array([[r["depth"], r["top1"], r["top2"], r["gap"], r["mean"], r["n_cand"]] for r in rows])
y = np.array([1 if r["llm_f1"] > r["emb_f1"] + 1e-9 else 0 for r in rows])
embf = np.array([r["emb_f1"] for r in rows]); llmf = np.array([r["llm_f1"] for r in rows])
delta = llmf - embf
n = len(rows)
print(f"n={n} base rate={y.mean():.3f} always-emb={embf.mean():.4f} always-LLM={llmf.mean():.4f} oracle={np.maximum(embf,llmf).mean():.4f}")

def cv_auroc_f1(model, X, y, seeds=(0, 1, 2), k=5):
    aurocs, routeds = [], []
    for s in seeds:
        skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=s)
        for tr, te in skf.split(X, y):
            m = make_pipeline(StandardScaler(), model)
            m.fit(X[tr], y[tr])
            p = m.predict_proba(X[te])[:, 1]
            aurocs.append(roc_auc_score(y[te], p))
            routeds.append(np.where(p > 0.5, llmf[te], embf[te]).mean())
    return float(np.mean(aurocs)), float(np.std(aurocs)), float(np.mean(routeds))

models = {
    "LogReg": LogisticRegression(max_iter=2000),
    "MLP(256,128)": MLPClassifier(hidden_layer_sizes=(256, 128), max_iter=800, random_state=0),
    "RandomForest": RandomForestClassifier(n_estimators=300, random_state=0),
}
cvC = {}
for name, m in models.items():
    a, s, rf = cv_auroc_f1(m, X, y)
    cvC[name] = dict(auroc=a, auroc_std=s, routed_f1=rf)
    print(f"{name:14s} AUROC={a:.4f}±{s:.3f} routedF1={rf:.4f}")

# learning curves: fixed stratified test half, varying train size
def strat_split_idx(y, seed=0, frac=0.5):
    rng = np.random.default_rng(seed)
    idx_pos = np.where(y == 1)[0]; idx_neg = np.where(y == 0)[0]
    rng.shuffle(idx_pos); rng.shuffle(idx_neg)
    hp = int(len(idx_pos) * frac); hn = int(len(idx_neg) * frac)
    te = np.concatenate([idx_pos[:hp], idx_neg[:hn]])
    tr_pool = np.concatenate([idx_pos[hp:], idx_neg[hn:]])
    rng.shuffle(tr_pool)
    return tr_pool, te

curves = {}
for mname, m in models.items():
    pts = []
    for size in [25, 50, 75, 100, 150]:
        aus, rfs = [], []
        for s in range(20):
            tr_pool, te = strat_split_idx(y, seed=s)
            rng = np.random.default_rng(1000 + s)
            tr = rng.choice(tr_pool, size=min(size, len(tr_pool)), replace=False)
            mm = make_pipeline(StandardScaler(), m)
            mm.fit(X[tr], y[tr])
            p = mm.predict_proba(X[te])[:, 1]
            aus.append(roc_auc_score(y[te], p))
            rfs.append(np.where(p > 0.5, llmf[te], embf[te]).mean())
        pts.append(dict(n_train=size,
                        auroc=float(np.mean(aus)), auroc_ci=[float(np.percentile(aus, 2.5)), float(np.percentile(aus, 97.5))],
                        routed_f1=float(np.mean(rfs)), routed_ci=[float(np.percentile(rfs, 2.5)), float(np.percentile(rfs, 97.5))]))
        print(f"curve {mname} n={size}: AUROC={np.mean(aus):.4f} CI[{np.percentile(aus,2.5):.3f},{np.percentile(aus,97.5):.3f}] routedF1={np.mean(rfs):.4f}")
    curves[mname] = pts

# gain regression (utility-aware): predict delta, route if >0
def cv_reg_route(model, seeds=(0, 1, 2), k=5):
    rfs, rates = [], []
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=0)
    for tr, te in skf.split(X, y):
        mm = make_pipeline(StandardScaler(), model)
        mm.fit(X[tr], delta[tr])
        d_hat = mm.predict(X[te])
        rfs.append(np.where(d_hat > 0, llmf[te], embf[te]).mean())
        rates.append(float((d_hat > 0).mean()))
    return float(np.mean(rfs)), float(np.mean(rates))

regC = {}
for name, m in {"GBDT-delta": GradientBoostingRegressor(random_state=0),
                "RF-delta": RandomForestRegressor(n_estimators=300, random_state=0)}.items():
    rf, rr = cv_reg_route(m)
    regC[name] = dict(routed_f1=rf, llm_rate=rr)
    print(f"{name}: routedF1={rf:.4f} llm_rate={rr:.3f}")

# gain-weighted classification: label noise scaled by |delta| via sample_weight
def cv_gw_route(seeds=(0,), k=5):
    rfs, aus = [], []
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=0)
    for tr, te in skf.split(X, y):
        m = make_pipeline(StandardScaler(), RandomForestClassifier(n_estimators=300, random_state=0))
        m.fit(X[tr], y[tr], randomforestclassifier__sample_weight=np.abs(delta[tr]) + 1e-3)
        p = m.predict_proba(X[te])[:, 1]
        aus.append(roc_auc_score(y[te], p))
        rfs.append(np.where(p > 0.5, llmf[te], embf[te]).mean())
    return float(np.mean(aus)), float(np.mean(rfs))

a_gw, rf_gw = cv_gw_route()
regC["gain-weighted-RF"] = dict(auroc=a_gw, routed_f1=rf_gw)
print(f"gain-weighted RF: AUROC={a_gw:.4f} routedF1={rf_gw:.4f}")

# U-lambda C: threshold sweep on GBDT delta-hat -> (rate, routedF1) vs mixture frontier
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
d_hat_oof = np.zeros(n)
for tr, te in skf.split(X, y):
    mm = make_pipeline(StandardScaler(), GradientBoostingRegressor(random_state=0))
    mm.fit(X[tr], delta[tr])
    d_hat_oof[te] = mm.predict(X[te])
mixtureC = []
for r in np.arange(0.05, 1.0, 0.05):
    mixtureC.append(dict(rate=float(r), f1=float((1 - r) * embf.mean() + r * llmf.mean())))
ulamC = []
for thr in np.quantile(d_hat_oof, np.linspace(0.02, 0.98, 25)):
    sel = d_hat_oof > thr
    ulamC.append(dict(threshold=float(thr), rate=float(sel.mean()),
                      f1=float(np.where(sel, llmf, embf).mean())))
best = max(ulamC, key=lambda x: x["f1"])
mix_at_best = (1 - best["rate"]) * embf.mean() + best["rate"] * llmf.mean()
# router-vs-mixture-at-matched-rate paired bootstrap (per-query expected mixture value)
sel_best = d_hat_oof > best["threshold"]
router_val = np.where(sel_best, llmf, embf)
mix_val = (1 - sel_best.mean()) * embf + sel_best.mean() * llmf
boot_router = paired_bootstrap(router_val, mix_val, seed=1)
# max gain over the mixture frontier across the router rate curve
gains = [(u["f1"] - ((1 - u["rate"]) * embf.mean() + u["rate"] * llmf.mean()), u) for u in ulamC]
gmax, gmax_u = max(gains, key=lambda t: t[0])
sel_g = d_hat_oof > gmax_u["threshold"]
boot_gmax = paired_bootstrap(np.where(sel_g, llmf, embf),
                             (1 - gmax_u["rate"]) * embf + gmax_u["rate"] * llmf, seed=2)
# RF-delta router (its own operating point) vs mixture at its rate
skf2 = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
rf_oof = np.zeros(n)
for tr, te in skf2.split(X, y):
    mm = make_pipeline(StandardScaler(), RandomForestRegressor(n_estimators=300, random_state=0))
    mm.fit(X[tr], delta[tr])
    rf_oof[te] = mm.predict(X[te])
sel_rf = rf_oof > 0
rf_rate = float(sel_rf.mean())
boot_rf = paired_bootstrap(np.where(sel_rf, llmf, embf),
                           (1 - rf_rate) * embf + rf_rate * llmf, seed=3)
print(f"U-lambdaC best routed F1={best['f1']:.4f} at rate={best['rate']:.3f}; "
      f"mixture frontier at same rate={mix_at_best:.4f}; gain={best['f1']-mix_at_best:+.4f} "
      f"CI={boot_router['ci']}")
print(f"max gain over mixture across U-lambdaC curve: {gmax:+.4f} at rate {gmax_u['rate']:.3f} CI={boot_gmax['ci']}")
print(f"RF-delta router: rate={rf_rate:.3f} routedF1={np.where(sel_rf, llmf, embf).mean():.4f} "
      f"vs mixture-at-rate boot={boot_rf}")
json.dump(dict(n=n, base_rate=float(y.mean()), always_emb=float(embf.mean()),
               always_llm=float(llmf.mean()), oracle=float(np.maximum(embf, llmf).mean()),
               cv=cvC, learning_curves=curves, regression=regC,
               ulam_curve=ulamC, mixture_frontier=mixtureC,
               ulam_best=dict(**best, mixture_at_same_rate=float(mix_at_best),
                              gain=float(best["f1"] - mix_at_best),
                              boot_vs_mixture=boot_router),
               max_gain_over_mixture=dict(gain=float(gmax), rate=float(gmax_u["rate"]),
                                          boot=boot_gmax),
               rf_delta_router=dict(rate=rf_rate,
                                    routed_f1=float(np.where(sel_rf, llmf, embf).mean()),
                                    boot_vs_mixture=boot_rf)),
          open(os.path.join(OUT, "C_routers.json"), "w"), indent=1)

# ---------------------------------------------------------------- Block D
print("=" * 30, "\nBLOCK D: joint (K,B,judge) outcome oracle from logs")
# full_pareto action set: LLM judge x {K1B4, K2B4, K2B8, K3B4} + emb judge K3
acts = ["Fixed-LLM K=1 B=4", "Fixed-LLM K=2 B=4", "Fixed-LLM K=2 B=8",
        "Fixed-LLM K=3 B=4", "Fixed-emb K=3"]
M = np.array([[data[a][q]["f1"] for a in acts] for q in qids])
T = np.array([[data[a][q]["n_input_tokens"] for a in acts] for q in qids])
choice = M.argmax(axis=1)
joint_f1 = M[np.arange(len(qids)), choice].mean()
joint_toks = T[np.arange(len(qids)), choice].mean()
rate_llm = np.mean([acts[c] != "Fixed-emb K=3" for c in choice])
from collections import Counter
dist = Counter(acts[c] for c in choice)
print(f"joint oracle over {acts}: F1={joint_f1:.4f} tokens={joint_toks:.1f} LLM-rate={rate_llm:.3f}")
print("choice distribution:", dict(dist))
tuned_f1 = per["Fixed-LLM K=2 B=8"].mean()
print(f"tuned global K2B8 F1={tuned_f1:.4f}; joint headroom=+{joint_f1-tuned_f1:.4f}")
# joint oracle on the 12-config LLM grid (mixed_pareto run)
acts2 = sorted([k for k in MP if k.startswith("Fixed")])
M2 = np.array([[MP[a][q]["f1"] for a in acts2] for q in qids])
T2 = np.array([[MP[a][q]["n_input_tokens"] for a in acts2] for q in qids])
c2 = M2.argmax(axis=1)
j2_f1 = M2[np.arange(len(qids)), c2].mean(); j2_t = T2[np.arange(len(qids)), c2].mean()
print(f"joint oracle over 12-config LLM grid: F1={j2_f1:.4f} tokens={j2_t:.1f} "
      f"(tuned K2B8 on this run={grid.get('Fixed_K2_B8',{}).get('f1')})")
json.dump(dict(action_set=acts, joint_f1=float(joint_f1), joint_toks=float(joint_toks),
               llm_rate=float(rate_llm), choice_dist=dict(dist),
               tuned_global_f1=float(tuned_f1),
               headroom=float(joint_f1 - tuned_f1),
               action_set_12=acts2, joint12_f1=float(j2_f1), joint12_toks=float(j2_t)),
          open(os.path.join(OUT, "D_joint_oracle.json"), "w"), indent=1)

# ---------------------------------------------------------------- Block E
print("=" * 30, "\nBLOCK E: Table-9 latency chord check")
cl = json.load(open(os.path.join(ROOT, "results_cost", "cost_latency.json")))["results"]
e = cl["Fixed-emb K=3"]; a = cl["AdaptiveJudge T=0.55"]; l = cl["Fixed-LLM K=2 B=8"]
chord = e["f1"] + (a["latency_s_mean"] - e["latency_s_mean"]) / (l["latency_s_mean"] - e["latency_s_mean"]) * (l["f1"] - e["f1"])
print(f"chord F1 at adaptive latency {a['latency_s_mean']:.2f}s = {chord:.4f}; "
      f"adaptive F1={a['f1']:.4f}; gain over chord={a['f1']-chord:+.4f}")
rate_calls = a["llm_judge_calls_mean"] / l["llm_judge_calls_mean"]
mix_calls = (1 - rate_calls) * e["f1"] + rate_calls * l["f1"]
print(f"mixture at matched judge-calls ({a['llm_judge_calls_mean']:.2f} of {l['llm_judge_calls_mean']:.2f}): F1={mix_calls:.4f}; adaptive gain={a['f1']-mix_calls:+.4f}")
json.dump(dict(chord_f1_at_adaptive_latency=float(chord), adaptive_f1=a["f1"],
               gain_over_chord=float(a["f1"] - chord),
               mixture_f1_at_matched_calls=float(mix_calls),
               gain_over_matched_mixture=float(a["f1"] - mix_calls)),
          open(os.path.join(OUT, "E_latency_chord.json"), "w"), indent=1)

# multiseed bootstrap exact CI
ms = json.load(open(os.path.join(ROOT, "results_multiseed", "multiseed.json")))
print("multiseed bootstrap:", ms["bootstrap"])
print("ALL DONE -> results_rebuttal/")
