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
        return False


# --------------------------------------------------------------------------- no-graph baseline
@dataclass
class NoGraphController:
    """Pure-LLM baseline: never use the graph, answer directly from the question.

    Establishes the lower bound -- shows that the graph itself contributes value
    (the motivation for GraphRAG rather than plain LLM QA).
    """
    max_hops: int = 0

    def decide_use_graph(self, question: str, entity_hits: List[Tuple[str, float]],
                         ctx: Dict) -> bool:
        return False

    def beam_for_hop(self, question: str, hop_idx: int, frontier_size: int,
                     scores: List[float], ctx: Dict) -> int:
        return 0

    def should_stop(self, question: str, hop_idx: int, frontier: List[str],
                    prev_frontier: List[str], score_stats: Dict, ctx: Dict) -> bool:
        return True


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

    def reset(self):
        """Reset per-query rolling state. Called by the engine at retrieval start."""
        self._prev_top = None

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


# --------------------------------------------------------------------------- simple stopping heuristics (reviewer-requested baselines)
@dataclass
class PatienceStopController:
    """Stop if the top relevance score has not improved for `patience` hops.

    A classic early-stopping heuristic. Shares the fixed beam of the baseline;
    only the stop rule differs -- isolating 'marginal-gain' (our D3) from a
    generic patience rule.
    """
    max_hops: int = 3
    beam: int = 4
    patience: int = 1

    def __post_init__(self):
        self._best: float | None = None
        self._since_improve: int = 0

    def reset(self):
        self._best = None
        self._since_improve = 0

    def decide_use_graph(self, question, entity_hits, ctx):
        return len(entity_hits) > 0

    def beam_for_hop(self, question, hop_idx, frontier_size, scores, ctx):
        return self.beam

    def should_stop(self, question, hop_idx, frontier, prev_frontier, score_stats, ctx):
        if not frontier:
            return True
        top = float(score_stats.get("top_score", 0.0))
        if self._best is None or top > self._best:
            self._best = top
            self._since_improve = 0
        else:
            self._since_improve += 1
        return hop_idx >= 1 and self._since_improve >= self.patience


@dataclass
class MarginStopController:
    """Stop when the top-1/top-2 score margin exceeds `threshold` (confident).

    A confidence-threshold stopping heuristic: a large margin means one
    candidate clearly dominates, so further hops are unlikely to help.
    """
    max_hops: int = 3
    beam: int = 4
    threshold: float = 0.30

    def decide_use_graph(self, question, entity_hits, ctx):
        return len(entity_hits) > 0

    def beam_for_hop(self, question, hop_idx, frontier_size, scores, ctx):
        return self.beam

    def should_stop(self, question, hop_idx, frontier, prev_frontier, score_stats, ctx):
        if not frontier:
            return True
        topk = score_stats.get("topk_scores") or []
        if len(topk) >= 2 and hop_idx >= 1:
            margin = float(topk[0] - topk[1])
            if margin >= self.threshold:
                return True
        return False


@dataclass
class ConfThresholdController:
    """Stop when the top relevance score exceeds `threshold` (good enough).

    A confidence-threshold stopping heuristic: once any hop retrieves a
    highly-relevant fact, stop. Complements MarginStop (absolute vs relative).
    """
    max_hops: int = 3
    beam: int = 4
    threshold: float = 0.70

    def decide_use_graph(self, question, entity_hits, ctx):
        return len(entity_hits) > 0

    def beam_for_hop(self, question, hop_idx, frontier_size, scores, ctx):
        return self.beam

    def should_stop(self, question, hop_idx, frontier, prev_frontier, score_stats, ctx):
        if not frontier:
            return True
        top = float(score_stats.get("top_score", 0.0))
        return hop_idx >= 1 and top >= self.threshold


@dataclass
class RandomBudgetController:
    """Random per-query max-hops (uniform over 1..max_hops). A naive baseline
    that adapts the budget without any signal -- controls for 'any adaptation
    helps' vs 'signal-driven adaptation helps'."""
    max_hops: int = 3
    beam: int = 4
    seed: int = 0

    def __post_init__(self):
        import random as _r
        self._rng = _r.Random(self.seed)
        self._cur_budget: int = self.max_hops

    def reset(self):
        # draw a fresh random budget for this query
        self._cur_budget = self._rng.randint(1, self.max_hops)

    def set_query(self):
        self._cur_budget = self._rng.randint(1, self.max_hops)

    def decide_use_graph(self, question, entity_hits, ctx):
        return len(entity_hits) > 0

    def beam_for_hop(self, question, hop_idx, frontier_size, scores, ctx):
        return self.beam

    def should_stop(self, question, hop_idx, frontier, prev_frontier, score_stats, ctx):
        if not frontier:
            return True
        return hop_idx + 1 >= self._cur_budget


@dataclass
class OracleDepthController:
    """Upper bound: use the TRUE hop depth as max_hops (cheating). Shows the
    ceiling of any depth-predicting router, including trained ones."""
    max_hops: int = 3
    beam: int = 4
    cur_depth: int = 1

    def set_depth(self, d: int):
        self.cur_depth = max(1, d)

    def decide_use_graph(self, question, entity_hits, ctx):
        return len(entity_hits) > 0

    def beam_for_hop(self, question, hop_idx, frontier_size, scores, ctx):
        return self.beam

    def should_stop(self, question, hop_idx, frontier, prev_frontier, score_stats, ctx):
        if not frontier:
            return True
        return hop_idx + 1 >= self.cur_depth


# --------------------------------------------------------------------------- Retrieve-then-Decide (RtD) — redesigned method
@dataclass
class RtDController:
    """Retrieve-then-Decide: probe with 1 hop, then allocate the remaining budget.

    Diagnosis of why the original AdaptiveController fails: its D3 uses the
    top-score *gain* (t_k - t_{k-1}) as the stop signal, but this signal is
    unreliable because (a) scores are not comparable across hops with different
    candidate sets, and (b) a flat gain does not imply the next hop is useless.
    On MetaQA mixed streams, the optimal K varies with true depth (1-hop→K=1-2,
    2-hop→K=2, 3-hop→K=3), but D3 stops at the wrong point, losing to a tuned
    global K=2.

    RtD redesigns the decision rule around three observations from the per-hop
    F1 diagnosis:
      1. After 1 hop, the *candidate count* and *score distribution* reveal
         query difficulty: a query with many high-scoring candidates is likely
         multi-hop (needs more exploration); a query with one dominant candidate
         is likely a simple lookup (K=1 suffices).
      2. The *absolute* top score after hop 0 is a better confidence signal
         than the *gain* (which is undefined at hop 0 and noisy afterwards).
      3. Once we decide the budget, we run to that budget without further
         stopping (avoiding the false-stop problem).

    Decision rule (training-free):
      - Always run hop 0 (probe).
      - After hop 0, compute a difficulty score d in [0,1] from:
          d = w1 * (1 - top_score_0) + w2 * H_0 + w3 * min(n_cand / cap, 1)
        where top_score_0 is the absolute top score at hop 0, H_0 the entropy,
        n_cand the candidate count, cap a normalizer.
        High d (low top score, high entropy, many candidates) → hard query → more hops.
      - Map d to a per-query budget K*(d):
          K* = 1 if d < tau_easy
          K* = 2 if tau_easy <= d < tau_hard
          K* = 3 if d >= tau_hard
      - Run to K* without early stopping.

    This avoids the false-stop problem entirely (no mid-retrieval stopping) and
    uses the hop-0 probe signal (which is reliable) rather than cross-hop gains
    (which are not). The cost is one mandatory hop-0 judge call for every query,
    but this is cheaper than always running K=3.
    """
    max_hops: int = 3
    beam: int = 4
    # difficulty weights
    w_confidence: float = 0.5    # weight on (1 - top_score)
    w_entropy: float = 0.3       # weight on normalized entropy
    w_candidates: float = 0.2    # weight on candidate count
    # difficulty thresholds
    tau_easy: float = 0.3        # d < tau_easy → K*=1
    tau_hard: float = 0.6        # d >= tau_hard → K*=3
    # cap for normalizing candidate count
    cand_cap: int = 32
    # internal: per-query budget decided after hop 0
    _decided_budget: int = 3
    _probed: bool = False

    @staticmethod
    def _norm_entropy(scores):
        s = np.asarray(scores, dtype=np.float64)
        s = np.clip(s, 1e-6, None)
        s = s / s.sum()
        h = -float(np.sum(s * np.log(s)))
        return h / math.log(len(s)) if len(s) > 1 else 0.0

    def reset(self):
        self._decided_budget = self.max_hops
        self._probed = False

    def decide_use_graph(self, question, entity_hits, ctx):
        return len(entity_hits) > 0

    def beam_for_hop(self, question, hop_idx, frontier_size, scores, ctx):
        return self.beam

    def should_stop(self, question, hop_idx, frontier, prev_frontier, score_stats, ctx):
        if not frontier:
            return True
        # probe phase: never stop at hop 0
        if hop_idx == 0:
            # decide the per-query budget based on hop-0 signals
            top_score = float(score_stats.get("top_score", 0.0))
            topk = score_stats.get("topk_scores") or []
            n_cand = int(score_stats.get("n_candidates", 0))
            H = self._norm_entropy(topk) if topk else 0.0
            d = (self.w_confidence * (1.0 - top_score)
                 + self.w_entropy * H
                 + self.w_candidates * min(n_cand / self.cand_cap, 1.0))
            if d < self.tau_easy:
                self._decided_budget = 1
            elif d < self.tau_hard:
                self._decided_budget = 2
            else:
                self._decided_budget = 3
            self._probed = True
            ctx["rtd_difficulty"] = d
            ctx["rtd_budget"] = self._decided_budget
            # at hop 0, stop only if budget is 1
            return self._decided_budget <= 1
        # after hop 0: stop when we reach the decided budget
        return hop_idx + 1 >= self._decided_budget


# --------------------------------------------------------------------------- factory
def build_controller(cfg: Dict) -> object:
    name = cfg.get("name", "adaptive")
    common = {"max_hops": cfg.get("max_hops", 3)}
    if name == "fixed":
        return FixedController(max_hops=common["max_hops"],
                               beam=cfg.get("beam", 4))
    if name == "nograph":
        return NoGraphController()
    if name == "vector-rag":
        return NoGraphController()  # vector-RAG handled in retriever, not controller
    if name == "patience":
        return PatienceStopController(max_hops=common["max_hops"],
                                      beam=cfg.get("beam", 4),
                                      patience=cfg.get("patience", 1))
    if name == "margin":
        return MarginStopController(max_hops=common["max_hops"],
                                    beam=cfg.get("beam", 4),
                                    threshold=cfg.get("threshold", 0.30))
    if name == "conf-threshold":
        return ConfThresholdController(max_hops=common["max_hops"],
                                       beam=cfg.get("beam", 4),
                                       threshold=cfg.get("threshold", 0.70))
    if name == "random-budget":
        return RandomBudgetController(max_hops=common["max_hops"],
                                      beam=cfg.get("beam", 4),
                                      seed=cfg.get("seed", 0))
    if name == "oracle-depth":
        return OracleDepthController(max_hops=common["max_hops"],
                                     beam=cfg.get("beam", 4))
    if name == "rtd":
        return RtDController(max_hops=common["max_hops"],
                             beam=cfg.get("beam", 4),
                             w_confidence=cfg.get("w_confidence", 0.5),
                             w_entropy=cfg.get("w_entropy", 0.3),
                             w_candidates=cfg.get("w_candidates", 0.2),
                             tau_easy=cfg.get("tau_easy", 0.3),
                             tau_hard=cfg.get("tau_hard", 0.6),
                             cand_cap=cfg.get("cand_cap", 32))
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
