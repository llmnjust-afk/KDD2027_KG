#!/usr/bin/env python3
"""Smoke test with a mock backend -- zero downloads, zero GPU.

Validates that the baseline (FixedController) and our method
(AdaptiveController) both run end-to-end through the shared engine on the toy
KG, produce well-formed reports, and that the cost meter + Pareto table work.
Run:  python tests/test_smoke.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from agr import (load_toy, GraphRAGRetriever, FixedController, AdaptiveController,
                 generate_answer, extract_answer_entities, score_query,
                 aggregate, pareto_table, Usage)


class MockBackend:
    """Deterministic mock: fixed embeddings (hash-based) + canned generation.

    Embeddings: stable pseudo-vectors so cosine similarity is meaningful enough
    for entity linking to find the right seed entity. Generation: returns the
    first gold-ish token so the answer pipeline is exercised without a model.
    """

    def __init__(self):
        self.usage = Usage()
        self._cache = {}

    def _vec(self, text: str) -> list:
        if text in self._cache:
            return self._cache[text]
        rng = np.random.default_rng(abs(hash(text)) % (2**32))
        v = rng.standard_normal(64).astype(np.float32)
        v = v / (np.linalg.norm(v) + 1e-9)
        # boost overlap: add the token-mean so shared words raise cosine
        for tok in text.lower().split():
            seed = abs(hash(tok)) % (2**32)
            r = np.random.default_rng(seed).standard_normal(64).astype(np.float32)
            v += 0.5 * r / (np.linalg.norm(r) + 1e-9)
        v = v / (np.linalg.norm(v) + 1e-9)
        self._cache[text] = v.tolist()
        return v.tolist()

    def embed(self, texts):
        return [self._vec(t) for t in texts]

    def generate(self, prompt, max_new_tokens=64, temperature=0.0, stop=None):
        # crude: echo a plausible answer token from the prompt
        n_in = len(prompt.split())
        # try to surface an entity-ish word after "Answer:"
        text = "Christopher Nolan"
        if "genre" in prompt.lower():
            text = "Science Fiction"
        if "born" in prompt.lower():
            text = "UK"
        if "actors" in prompt.lower() or "star" in prompt.lower():
            text = "Leonardo DiCaprio"
        n_out = len(text.split())
        self.usage.add(n_in, n_out)
        return {"text": text, "usage": {"n_input_tokens": n_in, "n_output_tokens": n_out}}


def run_controller(controller, name, kg, examples):
    backend = MockBackend()
    retriever = GraphRAGRetriever(kg, backend, controller, link_topk=5,
                                  judge_with_llm=False)  # embedding judge -> no model
    reports = []
    for ex in examples:
        ret = retriever.retrieve(ex.question)
        ans = generate_answer(backend, ex.question, ret)
        pred = extract_answer_entities(ans.text)
        sc = score_query(pred, ex.answers)
        reports.append({
            "qid": ex.qid, "gold": ex.answers, "pred": pred,
            "f1": sc["f1"], "hit1": sc["hit1"], "used_graph": ret.used_graph,
            "n_hops": ret.n_hops_executed,
            "n_llm_calls": backend.usage.n_calls,
            "n_input_tokens": backend.usage.n_input_tokens,
        })
    # build proper QueryReport objects for the cost/Pareto aggregator
    from agr import QueryReport
    qrs = [QueryReport(qid=r["qid"], question="", gold=r["gold"], pred=r["pred"],
                       hit1=r["hit1"], f1=r["f1"], exact=0.0, used_graph=r["used_graph"],
                       n_hops=r["n_hops"], n_llm_calls=r["n_llm_calls"],
                       n_input_tokens=r["n_input_tokens"], n_output_tokens=0) for r in reports]
    agg = aggregate(qrs)
    print(f"  [{name}] F1={agg.mean_f1:.3f} Hit@1={agg.mean_hit1:.3f} "
          f"toks/q={agg.mean_n_input_tokens:.0f} calls/q={agg.mean_n_llm_calls:.2f} "
          f"hops/q={agg.mean_n_hops:.2f} %graph={agg.frac_used_graph*100:.0f}%")
    return agg


def main():
    kg, examples = load_toy()
    print(f"toy KG: {len(kg)} triples, {len(examples)} questions\n")

    aggs = {}
    print("Baseline (FixedController, ToG-style):")
    aggs["fixed"] = run_controller(FixedController(max_hops=3, beam=4), "fixed", kg, examples)

    print("\nOur method (AdaptiveController, training-free):")
    aggs["adaptive"] = run_controller(AdaptiveController(max_hops=3), "adaptive", kg, examples)

    print("\n" + "=" * 64)
    print(pareto_table(aggs))
    print("\nSMOKE TEST PASSED" if all(a.n > 0 for a in aggs.values()) else "SMOKE TEST FAILED")


if __name__ == "__main__":
    main()
