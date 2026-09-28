"""GraphRAG retrieval engine + the naive strong baseline it defaults to.

Design decision (the spine of the paper): one retrieval engine parameterised by
a *controller*. The controller decides, per query and per hop:

    1. whether to use the graph at all             (Innovation 2a)
    2. how wide the beam is at this hop            (Innovation 2b)
    3. whether to stop expanding now               (Innovation 3)

The *baseline* is simply a FixedController that always uses the graph, always
uses the same beam width, and only stops at a hard max_hops. This mirrors the
canonical ToG-style multi-hop GraphRAG retrieval that every recent adaptive
method (Use-Graph-When-It-Needs, GraphRAG-Router) tries to beat -- but those
methods *train* a router/RL policy, while our controller is training-free.

Because baseline and method share one engine, the *only* thing that changes
between them is the controller, which is exactly what a clean ablation needs.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Protocol, Tuple

import numpy as np

from .kg import KnowledgeGraph
from .llm_backend import LLMBackend


@dataclass
class Subgraph:
    """A retrieved reasoning subgraph: list of (head, rel, tail) with scores."""
    triples: List[Tuple[str, str, str]] = field(default_factory=list)
    entities: List[str] = field(default_factory=list)

    def as_text(self) -> str:
        if not self.triples:
            return "(no facts retrieved)"
        lines = [f"- {h} | {r} | {t}" for (h, r, t) in self.triples]
        return "\n".join(lines)


@dataclass
class RetrievalResult:
    subgraph: Subgraph
    used_graph: bool
    n_hops_executed: int
    n_llm_calls: int           # relevance-judge calls
    n_input_tokens: int
    n_output_tokens: int
    entity_link_scores: List[float] = field(default_factory=list)
    hop_signals: List[Dict] = field(default_factory=list)  # diagnostics per hop


class Controller(Protocol):
    max_hops: int

    def decide_use_graph(self, question: str, entity_hits: List[Tuple[str, float]],
                         ctx: Dict) -> bool: ...

    def beam_for_hop(self, question: str, hop_idx: int, frontier_size: int,
                     scores: List[float], ctx: Dict) -> int: ...

    def should_stop(self, question: str, hop_idx: int, frontier: List[str],
                    prev_frontier: List[str], score_stats: Dict, ctx: Dict) -> bool: ...


# --------------------------------------------------------------------------- engine
class GraphRAGRetriever:
    """Faithful ToG-style multi-hop retrieval, controller-driven.

    Pipeline per query:
      (A) entity linking  : embed question, nearest KG entities by cosine.
      (B) for hop in 0..max_hops:
            - expand frontier one hop over the KG (all relations, both dirs)
            - score each candidate (head,rel,tail) for relevance to the question
              using the LLM as a relevance judge  <-- the expensive call
            - controller decides beam width + whether to stop
      (C) collect visited triples into the reasoning subgraph.
    """

    def __init__(
        self,
        kg: KnowledgeGraph,
        backend: LLMBackend,
        controller: Controller,
        link_topk: int = 5,
        score_topk_per_hop: int = 8,
        judge_with_llm: bool = True,
        noisy_linking: bool = False,
        vector_rag_topk: int = 0,
        adaptive_judge: bool = False,
        adaptive_judge_threshold: float = 0.55,
        strong_prefilter: bool = False,
        candidate_cap: Optional[int] = None,
        tog_sufficiency_stop: bool = False,
    ):
        self.kg = kg
        self.backend = backend
        self.controller = controller
        self.link_topk = link_topk
        self.score_topk_per_hop = score_topk_per_hop
        self.judge_with_llm = judge_with_llm
        self.noisy_linking = noisy_linking  # strip [Entity] brackets -> test 2a
        self.vector_rag_topk = vector_rag_topk  # >0 => vector-RAG mode (no graph traversal)
        self.adaptive_judge = adaptive_judge  # adaptive judge: emb probe then maybe LLM
        self.adaptive_judge_threshold = adaptive_judge_threshold  # emb top-1 above this -> skip LLM
        self.strong_prefilter = strong_prefilter  # ToG-2.0-style embedding prefilter
        self.candidate_cap = candidate_cap  # explicit cap override (P4 sweep); None = default rule
        self.tog_sufficiency_stop = tog_sufficiency_stop  # ToG-style LLM sufficiency check per hop
        self._judge_stats = {"emb_only": 0, "llm": 0}  # diagnostics
        self._ent_index: Optional[List[str]] = None
        self._ent_emb: Optional[np.ndarray] = None
        self._triple_index: Optional[List[Tuple[str, str, str]]] = None
        self._triple_emb: Optional[np.ndarray] = None

    # ---- entity linking ----
    def _build_entity_index(self):
        if self._ent_index is not None:
            return
        ents = sorted(self.kg.entities)
        self._ent_index = ents
        emb = np.asarray(self.backend.embed(ents), dtype=np.float32)
        # normalise (embedder already normalises; keep for safety)
        norms = np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9
        self._ent_emb = emb / norms

    def link_entities(self, question: str) -> List[Tuple[str, float]]:
        """Link seed entities.

        MetaQA questions mark the seed entity in [brackets]; we use that exact
        mention when it matches a KG entity (perfect-link path), which is the
        standard setup used by ToG/GNN-RAG on MetaQA. Embedding nearest-neighbour
        over all KG entities serves as a fair fallback for queries without a
        bracketed mention (and for the toy KG).

        With `noisy_linking=True`, brackets are stripped so linking falls back to
        embedding matching -- simulating real-world noisy entity linking where no
        ground-truth entity annotation is available. This is the setting that
        activates Innovation 2a (use-graph decision).
        """
        import re
        bracketed = [] if self.noisy_linking else re.findall(r"\[([^\]]+)\]", question)
        hits: List[Tuple[str, float]] = []
        if bracketed:
            ent_set = self.kg.entities
            for m in bracketed:
                # exact match preferred; else embedding match of the mention
                if m in ent_set:
                    hits.append((m, 1.0))
                else:
                    hits.extend(self._embed_link(m, topk=1))
        if not hits:
            hits = self._embed_link(question, topk=self.link_topk)
        # de-dup, cap to link_topk
        seen = set()
        out = []
        for e, s in hits:
            if e not in seen:
                seen.add(e)
                out.append((e, s))
            if len(out) >= self.link_topk:
                break
        return out

    def _embed_link(self, text: str, topk: int) -> List[Tuple[str, float]]:
        self._build_entity_index()
        q = np.asarray(self.backend.embed([text]), dtype=np.float32)[0]
        q = q / (np.linalg.norm(q) + 1e-9)
        sims = self._ent_emb @ q
        k = min(topk, len(self._ent_index))
        idx = np.argpartition(-sims, k - 1)[:k]
        idx = idx[np.argsort(-sims[idx])]
        return [(self._ent_index[i], float(sims[i])) for i in idx]

    # ---- relevance judge (expensive LLM call; the cost we target) ----
    def _relevance_scores(self, question: str,
                          candidates: List[Tuple[str, str, str]]) -> List[float]:
        if not candidates:
            return []
        # Adaptive judge: first compute free embedding scores; if the top-1
        # embedding score is high (confident match), skip the LLM call entirely
        # and use embedding scores — this is free and, as our diagnosis shows,
        # embedding judge is actually *better* than LLM judge on simple 1-hop
        # lookups. Only invoke the expensive LLM judge when the embedding signal
        # is uncertain (top-1 below threshold).
        if self.adaptive_judge:
            emb_scores = self._embedding_scores(question, candidates)
            top1 = max(emb_scores) if emb_scores else 0.0
            if top1 >= self.adaptive_judge_threshold:
                self._judge_stats["emb_only"] += 1
                return emb_scores
            # uncertain: fall through to LLM judge
            self._judge_stats["llm"] += 1
        if self.judge_with_llm:
            # Few-shot judge prompt: 2 examples calibrate the scoring scale,
            # improving judge accuracy and thus downstream QA F1.
            cand_lines = "\n".join(
                f"{i}. {h} | {r} | {t}" for i, (h, r, t) in enumerate(candidates)
            )
            prompt = (
                "You are a strict relevance judge. For each numbered fact, output an integer 0-100 "
                "for how relevant it is to answering the question. 100=directly answers it, 50=partially "
                "relevant, 0=irrelevant. Output ONLY a comma-separated list of integers.\n\n"
                "Example 1:\n"
                "Question: who directed Inception?\n"
                "Facts:\n1. Inception | directed_by | Christopher Nolan\n2. Titanic | directed_by | James Cameron\n3. Inception | has_genre | Science Fiction\n\n"
                "Scores: 100, 0, 20\n\n"
                "Example 2:\n"
                "Question: what movies star Leonardo DiCaprio?\n"
                "Facts:\n1. Inception | starred_actors | Leonardo DiCaprio\n2. The Dark Knight | starred_actors | Christian Bale\n\n"
                "Scores: 100, 0\n\n"
                f"Question: {question}\nFacts:\n{cand_lines}\n\nScores:"
            )
            out = self.backend.generate(prompt, max_new_tokens=64, temperature=0.0)
            scores = _parse_int_list(out["text"], len(candidates))
            return [s / 100.0 for s in scores]
        # cheap fallback: embedding similarity of the fact text to the question
        return self._embedding_scores(question, candidates)

    def _embedding_scores(self, question: str,
                          candidates: List[Tuple[str, str, str]]) -> List[float]:
        """Free embedding-similarity relevance scores (no LLM call)."""
        texts = [f"{h} {r} {t}" for (h, r, t) in candidates]
        emb = np.asarray(self.backend.embed([question] + texts), dtype=np.float32)
        q = emb[0] / (np.linalg.norm(emb[0]) + 1e-9)
        e = emb[1:] / (np.linalg.norm(emb[1:], axis=1, keepdims=True) + 1e-9)
        return (e @ q).tolist()

    # ---- vector-RAG: retrieve top-k triples by embedding similarity (no graph traversal) ----
    def _vector_rag(self, question: str, topk: int) -> RetrievalResult:
        if self._triple_index is None:
            self._triple_index = [(t.head, t.relation, t.tail) for t in self.kg.triples]
            texts = [f"{h} {r} {t}" for (h, r, t) in self._triple_index]
            emb = np.asarray(self.backend.embed(texts), dtype=np.float32)
            norms = np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9
            self._triple_emb = emb / norms
        q = np.asarray(self.backend.embed([question]), dtype=np.float32)[0]
        q = q / (np.linalg.norm(q) + 1e-9)
        sims = self._triple_emb @ q
        k = min(topk, len(self._triple_index))
        idx = np.argpartition(-sims, k - 1)[:k]
        idx = idx[np.argsort(-sims[idx])]
        triples = [tuple(self._triple_index[i]) for i in idx]
        before = (self.backend.usage.n_input_tokens, self.backend.usage.n_output_tokens,
                  self.backend.usage.n_calls)
        sub = Subgraph(triples=triples, entities=sorted({e for t in triples for e in (t[0], t[2])}))
        after = self.backend.usage
        return RetrievalResult(
            subgraph=sub, used_graph=True, n_hops_executed=1,  # single retrieval pass
            n_llm_calls=after.n_calls - before[2],
            n_input_tokens=after.n_input_tokens - before[0],
            n_output_tokens=after.n_output_tokens - before[1],
            entity_link_scores=[float(sims[i]) for i in idx],
        )

    # ---- main retrieval ----
    def retrieve(self, question: str) -> RetrievalResult:
        # vector-RAG shortcut: no entity linking, no multi-hop, no LLM judge
        if self.vector_rag_topk > 0:
            return self._vector_rag(question, self.vector_rag_topk)

        hits = self.link_entities(question)
        link_scores = [s for _, s in hits]
        ctx: Dict = {"link_scores": link_scores,
                     "link_entities": [e for e, _ in hits]}

        # Innovation 2a: maybe skip the graph entirely (handled by controller)
        use_graph = self.controller.decide_use_graph(question, hits, ctx)
        if not use_graph or not hits:
            return RetrievalResult(
                subgraph=Subgraph(), used_graph=False, n_hops_executed=0,
                n_llm_calls=self.backend.usage.n_calls, n_input_tokens=0,
                n_output_tokens=0, entity_link_scores=link_scores,
            )

        before = (self.backend.usage.n_input_tokens, self.backend.usage.n_output_tokens,
                  self.backend.usage.n_calls)

        frontier: List[str] = [e for e, _ in hits if e in self.kg.entities] or \
                              [e for e, _ in hits]
        # reset per-query rolling state on the controller (marginal-gain memory,
        # patience counters, random budgets, etc.)
        _reset = getattr(self.controller, "reset", None)
        if callable(_reset):
            _reset()
        visited_entities = set(frontier)
        collected: List[Tuple[str, str, str]] = []
        hop_signals: List[Dict] = []
        n_hops = 0
        prev_frontier: List[str] = []

        for hop in range(self.controller.max_hops):
            # expand one hop
            cand: List[Tuple[str, str, str]] = []
            for e in frontier:
                for (rel, nb, direction) in self.kg.all_neighbors(e):
                    triple = (e, rel, nb) if direction == "out" else (nb, rel, e)
                    cand.append(triple)
            # dedup
            seen = set()
            cand = [c for c in cand if not (c in seen or seen.add(c))]
            if not cand:
                break
            cap = self.candidate_cap if self.candidate_cap is not None else max(self.score_topk_per_hop * 4, 16)
            if self.strong_prefilter and len(cand) > cap:
                # ToG-2.0-style two-stage retrieval: instead of truncating in
                # arbitrary traversal order, rank ALL candidates by cheap
                # embedding similarity and keep the top-`cap` for the (expensive)
                # judge. This removes the traversal-order truncation bias that
                # otherwise hides relevant triples from the judge on dense KGs.
                pre = self._embedding_scores(question, cand)
                pre_order = np.argsort([-s for s in pre])[:cap]
                cand = [cand[i] for i in pre_order]
            else:
                # cap candidates before judging to keep prompts bounded
                cand = cand[:cap]
            scores = self._relevance_scores(question, cand)
            order = np.argsort([-s for s in scores])
            cand = [cand[i] for i in order]
            scores = [scores[i] for i in order]

            topk = self.score_topk_per_hop
            beam = self.controller.beam_for_hop(question, hop, len(frontier), scores[:topk], ctx)
            beam = max(1, int(beam))
            keep = cand[:beam]
            keep_scores = scores[:beam]

            collected.extend(keep)
            new_frontier = []
            for (h, r, t) in keep:
                for ent in (h, t):
                    if ent not in visited_entities:
                        visited_entities.add(ent)
                        new_frontier.append(ent)

            stats = {
                "hop": hop,
                "n_candidates": len(cand),
                "top_score": float(keep_scores[0]) if keep_scores else 0.0,
                "mean_topk": float(np.mean(keep_scores)) if keep_scores else 0.0,
                "topk_scores": [float(s) for s in keep_scores],
                "beam": beam,
                "frontier_overlap": (len(set(new_frontier) & set(frontier)) / max(1, len(new_frontier))) if new_frontier else 0.0,
            }
            hop_signals.append(stats)
            n_hops = hop + 1

            # ToG-style sufficiency check: after judging each hop, ask the LLM
            # whether the collected facts are already sufficient (published
            # early-termination mechanism of ToG; costs one extra short call).
            if self.tog_sufficiency_stop and collected:
                facts = "\n".join(f"- {h} | {r} | {t}" for (h, r, t) in collected)
                suff_prompt = (
                    f"Question: {question}\n\n"
                    f"Retrieved facts so far:\n{facts}\n\n"
                    "Do you already have sufficient information to answer the question? "
                    "Answer with exactly one word, Yes or No."
                )
                resp = self.backend.generate(suff_prompt, max_new_tokens=8)
                if "yes" in str(resp.get("text", "")).strip().lower()[:8]:
                    frontier = new_frontier
                    prev_frontier = frontier
                    break

            # Innovation 3: early stop
            if self.controller.should_stop(question, hop, new_frontier, frontier, stats, ctx):
                frontier = new_frontier
                prev_frontier = frontier
                break
            prev_frontier = frontier
            frontier = new_frontier
            if not frontier:
                break

        # dedup collected triples preserving order
        seen = set()
        uniq = []
        for t in collected:
            if t not in seen:
                seen.add(t)
                uniq.append(t)
        sub = Subgraph(triples=uniq, entities=sorted(visited_entities))

        after = self.backend.usage
        return RetrievalResult(
            subgraph=sub, used_graph=True, n_hops_executed=n_hops,
            n_llm_calls=after.n_calls - before[2],
            n_input_tokens=after.n_input_tokens - before[0],
            n_output_tokens=after.n_output_tokens - before[1],
            entity_link_scores=link_scores, hop_signals=hop_signals,
        )


def _parse_int_list(text: str, expected: int) -> List[int]:
    """Robustly parse '12, 88, 3' style judge output."""
    import re
    nums = re.findall(r"\d+", text)
    out = [int(n) for n in nums[:expected]]
    if len(out) < expected:  # pad with 0 (irrelevant) if model underproduced
        out += [0] * (expected - len(out))
    return out
