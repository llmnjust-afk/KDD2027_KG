"""Controllers = the paper's contribution.

Two controllers share the engine in retriever.py:

  FixedController     : the naive strong baseline (ToG-style). Always uses the
                        graph, fixed beam, stops only at max_hops. This is what
                        every adaptive GraphRAG method is measured against.

  AdaptiveController  : training-free, query-adaptive budgeting. Implements the
                        three innovations:

      (2a) decide_use_graph   -- skip the graph when the seed-entity anchor is
                                  unreliable (training-free signal from the link
                                  score distribution); route to direct answer.
      (2b) beam_for_hop        -- allocate beam width to the *spread* of hop
                                  relevance scores: flat scores -> wider beam,
                                  concentrated scores -> narrower beam.
      (3)  should_stop         -- marginal-relevance-gain early stopping: stop
                                  expanding once the best per-hop relevance
                                  score stops improving by more than delta.

All signals are computed from distributions that already exist in the retrieval
pipeline (link cosines, LLM relevance scores). No training, no learned router,
no extra parameters -- exactly the gap left open by Use-Graph-When-It-Needs
(trains a detector) and GraphRAG-Router (trains an RL policy).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np


# --------------------------------------------------------------------------- baseline
@dataclass
class FixedController:
    """Naive strong baseline: fixed budget, no adaptation, no early stop.

    This is the canonical GraphRAG retrieval policy: use the graph for every
    query, expand a fixed beam every hop, stop only at max_hops. Tuning
    `max_hops` and `beam` recovers the ToG / GNN-RAG operating points.
    """
    max_hops: int = 3
    beam: int = 4

    def decide_use_graph(self, question: str, entity_hits: List[Tuple[str, float]],
                         ctx: Dict) -> bool:
        # baseline: always use the graph if any entity was linked
        return len(entity_hits) > 0

    def beam_for_hop(self, question: str, hop_idx: int, frontier_size: int,
                     scores: List[float], ctx: Dict) -> int:
        return self.beam

    def should_stop(self, question: str, hop_idx: int, frontier: List[str],
                    prev_frontier: List[str], score_stats: Dict, ctx: Dict) -> bool:
        # baseline: never stop early (engine handles max_hops cap)
        return False


# --------------------------------------------------------------------------- method
@dataclass
class AdaptiveController:
    """Training-free query-adaptive retrieval controller (our method)."""
    max_hops: int = 3

    # (2a) use-graph threshold on top-1 link cosine
    theta_low: float = 0.30          # below this, seed entity is unreliable -> skip graph
    theta_skip_entropy: float = 0.15 # if link-score entropy extremely low AND top1 high,
                                     # query is a trivial lookup -> still do graph but the
                                     # early-stop will cut it to 1 hop (handled in (3))

    # (2b) beam allocation
    base_beam: int = 4
    min_beam: int = 1
    max_beam: int = 8

    # (3) marginal-gain stopping
    delta: float = 0.05              # min per-hop top-score improvement to continue
    abs_floor: float = 0.10          # if best hop score below this, nothing useful here

    # ---- ablation toggles (each disables exactly one innovation) ----
    ablate_use_graph: bool = False     # True => always use graph (disables 2a)
    ablate_adaptive_beam: bool = False # True => fixed base_beam every hop (disables 2b)
    ablate_early_stop: bool = False    # True => never stop early (disables 3)

    # rolling state across hops (reset per query by the engine creating a fresh
    # controller instance per query is NOT required; we track via ctx instead)
    def __post_init__(self):
        self._prev_top: float | None = None

    # ---- helpers on distributions ----
    @staticmethod
    def _norm_entropy(scores: List[float]) -> float:
        """Normalised entropy in [0,1]. High => flat (uncertain) distribution."""
        s = np.asarray(scores, dtype=np.float64)
        s = np.clip(s, 1e-6, None)
        s = s / s.sum()
        h = -float(np.sum(s * np.log(s)))
        return h / math.log(len(s)) if len(s) > 1 else 0.0

    @staticmethod
    def _top1(scores: List[float]) -> float:
        return float(max(scores)) if scores else 0.0

    # ---- (2a) ----
    def decide_use_graph(self, question: str, entity_hits: List[Tuple[str, float]],
                         ctx: Dict) -> bool:
        if not entity_hits:
            return False
        # ablation: force graph use (behaves like baseline on this axis)
        if self.ablate_use_graph:
            return True
        scores = [s for _, s in entity_hits]
        top1 = self._top1(scores)
        # Innovation 2a: if the best entity link is too weak, the seed anchor is
        # unreliable and multi-hop expansion will amplify noise rather than
        # evidence -> route to a direct (graph-free) answer, saving all hop
        # relevance-judge calls for this query.
        if top1 < self.theta_low:
            ctx["skip_reason"] = "low_link_confidence"
            return False
        return True

    # ---- (2b) ----
    def beam_for_hop(self, question: str, hop_idx: int, frontier_size: int,
                     scores: List[float], ctx: Dict) -> int:
        """Allocate beam to score spread.

        Concentrated top-k (low entropy) => one or few triples clearly dominate,
        so a narrow beam suffices. Flat top-k (high entropy) => many candidates
        look equally relevant, so widen the beam to avoid pruning the true path.
        """
        # ablation: fixed beam (behaves like baseline on this axis)
        if self.ablate_adaptive_beam:
            return self.base_beam
        if not scores:
            return self.base_beam
        H = self._norm_entropy(scores)
        # map entropy [0,1] -> beam factor [0.5, 1.5]
        factor = 0.5 + H  # 0.5..1.5
        beam = int(round(self.base_beam * factor))
        # also grow beam with frontier size so a large frontier isn't over-pruned
        beam = max(beam, min(self.max_beam, max(self.min_beam, frontier_size // 2)))
        return int(np.clip(beam, self.min_beam, self.max_beam))

    # ---- (3) ----
    def should_stop(self, question: str, hop_idx: int, frontier: List[str],
                    prev_frontier: List[str], score_stats: Dict, ctx: Dict) -> bool:
        top = float(score_stats.get("top_score", 0.0))
        # frontier collapse is a structural stop, kept even under ablation
        if not frontier:
            ctx.setdefault("stop_reasons", []).append("frontier_collapse")
            return True
        # ablation: disable early stopping (only structural stops remain)
        if self.ablate_early_stop:
            return False
        # absolute floor: nothing retrieved at this hop is worth carrying
        if top < self.abs_floor and hop_idx >= 1:
            ctx.setdefault("stop_reasons", []).append("abs_floor")
            return True
        # marginal-relevance-gain stopping
        if self._prev_top is not None:
            gain = top - self._prev_top
            if hop_idx >= 1 and gain <= self.delta:
                ctx.setdefault("stop_reasons", []).append(
                    f"marginal_gain={gain:.3f}<=delta={self.delta}"
                )
                return True
        self._prev_top = top
        return False


# --------------------------------------------------------------------------- factory
def build_controller(cfg: Dict) -> object:
    name = cfg.get("name", "adaptive")
    common = {"max_hops": cfg.get("max_hops", 3)}
    if name == "fixed":
        return FixedController(max_hops=common["max_hops"],
                               beam=cfg.get("beam", 4))
    return AdaptiveController(
        max_hops=common["max_hops"],
        theta_low=cfg.get("theta_low", 0.30),
        base_beam=cfg.get("base_beam", 4),
        min_beam=cfg.get("min_beam", 1),
        max_beam=cfg.get("max_beam", 8),
        delta=cfg.get("delta", 0.05),
        abs_floor=cfg.get("abs_floor", 0.10),
        ablate_use_graph=cfg.get("ablate_use_graph", False),
        ablate_adaptive_beam=cfg.get("ablate_adaptive_beam", False),
        ablate_early_stop=cfg.get("ablate_early_stop", False),
    )
