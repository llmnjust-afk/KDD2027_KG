"""Answer generation from a retrieved subgraph (the final LLM call).

For graph-skipped queries (controller routed to direct answer), we generate
from the question alone (optionally with the linked entity name), which is the
cheap path that motivates Innovation 2a: a query that doesn't need the graph
costs zero relevance-judge LLM calls and one short generation call.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

from .kg import KnowledgeGraph
from .llm_backend import LLMBackend
from .retriever import RetrievalResult, Subgraph


@dataclass
class AnswerResult:
    text: str
    n_input_tokens: int
    n_output_tokens: int


def _normalize(s: str) -> str:
    return " ".join(s.lower().strip().split())


def generate_answer(
    backend: LLMBackend,
    question: str,
    retrieval: RetrievalResult,
    max_new_tokens: int = 32,
) -> AnswerResult:
    before = (backend.usage.n_input_tokens, backend.usage.n_output_tokens)

    if retrieval.used_graph and retrieval.subgraph.triples:
        facts = retrieval.subgraph.as_text()
        prompt = (
            "Use ONLY the following facts to answer the question. "
            "Answer with a short phrase or a pipe-separated list of entities. "
            "If the facts are insufficient, answer 'unknown'.\n\n"
            f"Facts:\n{facts}\n\nQuestion: {question}\nAnswer:"
        )
    else:
        # graph-free path (Innovation 2a): the controller judged the seed-entity
        # anchor too unreliable for multi-hop expansion, so we answer directly
        # from the question. This is the cheap route -- zero relevance-judge
        # LLM calls, one short generation call.
        prompt = (
            "Answer the question briefly. "
            "Give a short phrase or a pipe-separated list.\n\n"
            f"Question: {question}\nAnswer:"
        )

    out = backend.generate(prompt, max_new_tokens=max_new_tokens, temperature=0.0,
                           stop=["\n", "Question:"])
    text = out["text"]
    after = (backend.usage.n_input_tokens, backend.usage.n_output_tokens)
    return AnswerResult(
        text=text,
        n_input_tokens=after[0] - before[0],
        n_output_tokens=after[1] - before[1],
    )


def extract_answer_entities(text: str) -> List[str]:
    """Split a model answer into candidate entity strings."""
    if not text:
        return []
    # handle pipe-separated or comma-separated or newline-separated answers
    parts = text.replace("|", "\n").replace(",", "\n").split("\n")
    out = []
    for p in parts:
        p = p.strip().strip(".")
        if p and p.lower() not in {"unknown", "none", "n/a", ""}:
            out.append(_normalize(p))
    return out
