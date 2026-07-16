"""Knowledge graph data structure and loaders.

Self-contained KG over which GraphRAG retrieval runs. Designed to mirror the
WikiMovies KG used by MetaQA (the standard multi-hop QA benchmark of ToG /
GNN-RAG / NSM), but also ships a tiny synthetic KG so the full pipeline
(including baselines and the adaptive controller) can be smoke-tested with
zero downloads and zero network access.
"""
from __future__ import annotations

import json
import os
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Set, Tuple

import networkx as nx


@dataclass
class Triple:
    head: str
    relation: str
    tail: str

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return f"({self.head}, {self.relation}, {self.tail})"


@dataclass
class QAExample:
    qid: str
    question: str
    answers: List[str]
    n_hop: int = 1  # reasoning depth required (MetaQA: 1/2/3)


@dataclass
class KnowledgeGraph:
    """In-memory KG with bidirectional adjacency + relation index.

    The graph is intentionally small and CPU-resident; the whole point of the
    paper is that the *retrieval controller* (not the index) is the expensive
    part, so a full in-memory graph is the realistic setting.
    """

    triples: List[Triple] = field(default_factory=list)
    entities: Set[str] = field(default_factory=set)
    relations: Set[str] = field(default_factory=set)
    # adjacency[h][rel] -> set(tails);  adjacency_rev[t][rel] -> set(heads)
    adj: Dict[str, Dict[str, Set[str]]] = field(default_factory=lambda: defaultdict(lambda: defaultdict(set)))
    adj_rev: Dict[str, Dict[str, Set[str]]] = field(default_factory=lambda: defaultdict(lambda: defaultdict(set)))
    nx_graph: nx.MultiDiGraph = field(default_factory=nx.MultiDiGraph)

    @classmethod
    def from_triples(cls, triples: Iterable[Triple]) -> "KnowledgeGraph":
        kg = cls()
        for t in triples:
            kg._add(t)
        return kg

    def _add(self, t: Triple) -> None:
        self.triples.append(t)
        self.entities.add(t.head)
        self.entities.add(t.tail)
        self.relations.add(t.relation)
        self.adj[t.head][t.relation].add(t.tail)
        self.adj_rev[t.tail][t.relation].add(t.head)
        self.nx_graph.add_edge(t.head, t.tail, key=t.relation, relation=t.relation)

    # ------------------------------------------------------------------ API
    def neighbors(self, entity: str, relation: str | None = None) -> List[Tuple[str, str]]:
        """Outgoing (relation, tail) edges from `entity`."""
        if entity not in self.adj:
            return []
        if relation is None:
            out: List[Tuple[str, str]] = []
            for rel, tails in self.adj[entity].items():
                out.extend((rel, t) for t in tails)
            return out
        return [(relation, t) for t in self.adj[entity].get(relation, set())]

    def reverse_neighbors(self, entity: str, relation: str | None = None) -> List[Tuple[str, str]]:
        if entity not in self.adj_rev:
            return []
        if relation is None:
            out: List[Tuple[str, str]] = []
            for rel, heads in self.adj_rev[entity].items():
                out.extend((rel, h) for h in heads)
            return out
        return [(relation, h) for h in self.adj_rev[entity].get(relation, set())]

    def all_neighbors(self, entity: str) -> List[Tuple[str, str, str]]:
        """All (relation, neighbor, direction) edges, both directions."""
        out: List[Tuple[str, str, str]] = []
        for rel, tails in self.adj.get(entity, {}).items():
            for t in tails:
                out.append((rel, t, "out"))
        for rel, heads in self.adj_rev.get(entity, {}).items():
            for h in heads:
                out.append((rel, h, "in"))
        return out

    def __len__(self) -> int:
        return len(self.triples)


# --------------------------------------------------------------------------- loaders
def load_toy() -> Tuple[KnowledgeGraph, List[QAExample]]:
    """A tiny hand-built KG (movie domain, mirrors WikiMovies schema).

    Lets the entire pipeline run end-to-end in <1s with no downloads. The real
    experiments swap in MetaQA via `load_metaqa`.
    """
    triples = [
        Triple("Inception", "directed_by", "Christopher Nolan"),
        Triple("The Dark Knight", "directed_by", "Christopher Nolan"),
        Triple("Interstellar", "directed_by", "Christopher Nolan"),
        Triple("Inception", "has_genre", "Science Fiction"),
        Triple("Interstellar", "has_genre", "Science Fiction"),
        Triple("The Dark Knight", "has_genre", "Action"),
        Triple("Inception", "starred_actors", "Leonardo DiCaprio"),
        Triple("The Dark Knight", "starred_actors", "Christian Bale"),
        Triple("The Revenant", "starred_actors", "Leonardo DiCaprio"),
        Triple("The Revenant", "directed_by", "Alejandro Gonzalez Inarritu"),
        Triple("The Revenant", "has_genre", "Adventure"),
        Triple("Titanic", "starred_actors", "Leonardo DiCaprio"),
        Triple("Titanics", "directed_by", "James Cameron"),  # alias noise
        Triple("Titanic", "directed_by", "James Cameron"),
        Triple("Avatar", "directed_by", "James Cameron"),
        Triple("Avatar", "has_genre", "Science Fiction"),
        Triple("Christian Bale", "born_in", "Wales"),
        Triple("Leonardo DiCaprio", "born_in", "USA"),
        Triple("Christopher Nolan", "born_in", "UK"),
    ]
    kg = KnowledgeGraph.from_triples(triples)
    examples = [
        QAExample("t1", "who directed Inception?", ["Christopher Nolan"], n_hop=1),
        QAExample("t2", "what movies were directed by Christopher Nolan?",
                  ["Inception", "The Dark Knight", "Interstellar"], n_hop=1),
        QAExample("t3", "what genre are the movies directed by Christopher Nolan?",
                  ["Science Fiction", "Action"], n_hop=2),
        QAExample("t4", "where was the director of Inception born?", ["UK"], n_hop=2),
        QAExample("t5", "what actors star in movies of the Science Fiction genre?",
                  ["Leonardo DiCaprio"], n_hop=2),
    ]
    return kg, examples


def load_metaqa(data_dir: str, split: str = "1-hop") -> Tuple[KnowledgeGraph, List[QAExample]]:
    """Load MetaQA (WikiMovies KG + questions).

    Expects the standard MetaQA layout:
        data_dir/kb.txt                -> 'entity\\trelation\\tentity' per line
        data_dir/{split}/questions.txt -> '{question}\\t{answer_list|sep=|}'
    Falls back to a helper that downloads MetaQA if missing (see scripts/).
    """
    kb_path = os.path.join(data_dir, "kb.txt")
    q_path = os.path.join(data_dir, split, "questions.txt")
    if not (os.path.exists(kb_path) and os.path.exists(q_path)):
        raise FileNotFoundError(
            f"MetaQA not found under {data_dir}. Run scripts/download_metaqa.py first."
        )
    triples: List[Triple] = []
    with open(kb_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = [p.strip() for p in line.strip().split("\t") if p.strip()]
            if len(parts) == 3:
                triples.append(Triple(parts[0], parts[1], parts[2]))
    kg = KnowledgeGraph.from_triples(triples)
    examples: List[QAExample] = []
    n_hop = int(split.split("-")[0]) if split[0].isdigit() else 1
    with open(q_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            if "\t" in line:
                q, ans = line.rsplit("\t", 1)
            else:
                q, ans = line, ""
            answers = [a.strip() for a in ans.split("|") if a.strip()]
            examples.append(QAExample(f"metaqa_{split}_{i}", q, answers, n_hop=n_hop))
    return kg, examples


def save_jsonl(objs: Iterable[dict], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for o in objs:
            f.write(json.dumps(o, ensure_ascii=False) + "\n")
