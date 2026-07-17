"""Adaptive GraphRAG -- training-free query-adaptive retrieval for GraphRAG.

Package layout:
    kg             - knowledge graph data structure + MetaQA / toy loaders
    llm_backend    - pluggable LLM (local HF default, optional API) + cost meter
    retriever      - one controller-driven multi-hop retrieval engine
    controller     - FixedController (baseline) + AdaptiveController (our method)
    answer         - answer generation from retrieved subgraph
    evaluate       - F1/Hit@1/exact + cost accounting + Pareto table
"""
from .kg import KnowledgeGraph, Triple, QAExample, load_toy, load_metaqa
from .llm_backend import LLMBackend, HuggingFaceBackend, APIBackend, Usage, build_backend
from .retriever import GraphRAGRetriever, RetrievalResult, Subgraph
from .controller import FixedController, AdaptiveController, NoGraphController, build_controller
from .answer import generate_answer, extract_answer_entities, AnswerResult
from .evaluate import (QueryReport, Aggregate, score_query, aggregate,
                       pareto_table, save_reports)

__all__ = [
    "KnowledgeGraph", "Triple", "QAExample", "load_toy", "load_metaqa",
    "LLMBackend", "HuggingFaceBackend", "APIBackend", "Usage", "build_backend",
    "GraphRAGRetriever", "RetrievalResult", "Subgraph",
    "FixedController", "AdaptiveController", "NoGraphController", "build_controller",
    "generate_answer", "extract_answer_entities", "AnswerResult",
    "QueryReport", "Aggregate", "score_query", "aggregate", "pareto_table",
    "save_reports",
]
