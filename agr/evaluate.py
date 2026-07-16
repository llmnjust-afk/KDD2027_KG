"""Evaluation: accuracy metrics + cost accounting + Pareto aggregation.

Metrics reported per query and aggregated:
  - Hit@1            : whether the top extracted answer entity is a gold answer
  - F1 (entity-level): token-overlap F1 between predicted & gold entity sets
                       (the standard MetaQA metric, identical to ToG/GNN-RAG)
  - Exact            : set equality of predicted vs gold answer sets
Cost (the axis our method improves):
  - n_llm_calls      : relevance-judge + generation calls
  - n_input_tokens   : prompt tokens (dominates cost on frontier APIs)
  - n_output_tokens  : generated tokens
Pareto:
  - aggregate (mean_cost, mean_f1) per (controller, config) and emit a table
    + the points needed to plot the accuracy-vs-cost Pareto frontier.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Sequence

import numpy as np


@dataclass
class QueryReport:
    qid: str
    question: str
    gold: List[str]
    pred: List[str]
    hit1: float
    f1: float
    exact: float
    used_graph: bool
    n_hops: int
    n_llm_calls: int
    n_input_tokens: int
    n_output_tokens: int
    stop_reasons: List[str] = field(default_factory=list)


def _normalize_set(items: Sequence[str]) -> set:
    return {" ".join(s.lower().strip().split()) for s in items if s.strip()}


def entity_f1(pred: Sequence[str], gold: Sequence[str]) -> float:
    pset, gset = _normalize_set(pred), _normalize_set(gold)
    if not pset and not gset:
        return 1.0
    if not pset or not gset:
        return 0.0
    tp = len(pset & gset)
    prec = tp / len(pset)
    rec = tp / len(gset)
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)


def score_query(pred: List[str], gold: List[str]) -> Dict[str, float]:
    pset, gset = _normalize_set(pred), _normalize_set(gold)
    hit1 = 1.0 if (pred and _normalize_set([pred[0]]) & gset) else 0.0
    return {"hit1": hit1, "f1": entity_f1(pred, gold),
            "exact": 1.0 if pset == gset else 0.0}


@dataclass
class Aggregate:
    n: int
    mean_hit1: float
    mean_f1: float
    mean_exact: float
    mean_n_llm_calls: float
    mean_n_input_tokens: float
    mean_n_output_tokens: float
    mean_n_hops: float
    frac_used_graph: float

    def as_dict(self) -> Dict:
        return asdict(self)


def aggregate(reports: List[QueryReport]) -> Aggregate:
    if not reports:
        return Aggregate(0, 0, 0, 0, 0, 0, 0, 0, 0)
    n = len(reports)
    return Aggregate(
        n=n,
        mean_hit1=float(np.mean([r.hit1 for r in reports])),
        mean_f1=float(np.mean([r.f1 for r in reports])),
        mean_exact=float(np.mean([r.exact for r in reports])),
        mean_n_llm_calls=float(np.mean([r.n_llm_calls for r in reports])),
        mean_n_input_tokens=float(np.mean([r.n_input_tokens for r in reports])),
        mean_n_output_tokens=float(np.mean([r.n_output_tokens for r in reports])),
        mean_n_hops=float(np.mean([r.n_hops for r in reports])),
        frac_used_graph=float(np.mean([1.0 if r.used_graph else 0.0 for r in reports])),
    )


def pareto_table(system_aggs: Dict[str, Aggregate]) -> str:
    """Pretty-print a Pareto comparison: system rows sorted by cost."""
    rows = sorted(system_aggs.items(), key=lambda kv: kv[1].mean_n_input_tokens)
    header = (f"{'system':<28}{'F1':>8}{'Hit@1':>8}{'Exact':>8}"
              f"{'toks/q':>10}{'calls/q':>9}{'hops/q':>8}{'%graph':>8}")
    lines = [header, "-" * len(header)]
    for name, a in rows:
        lines.append(
            f"{name:<28}{a.mean_f1:>8.3f}{a.mean_hit1:>8.3f}{a.mean_exact:>8.3f}"
            f"{a.mean_n_input_tokens:>10.0f}{a.mean_n_llm_calls:>9.2f}"
            f"{a.mean_n_hops:>8.2f}{a.frac_used_graph*100:>7.1f}%"
        )
    return "\n".join(lines)


def save_reports(reports: List[QueryReport], path: str) -> None:
    import os
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in reports:
            f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")
