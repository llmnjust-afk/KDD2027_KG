# Adaptive GraphRAG

Training-free, query-adaptive retrieval budgeting for GraphRAG. Targets KDD 2027 Research Track (Modern AI and Big Data).

## Idea in one paragraph
Existing GraphRAG applies the same expensive multi-hop retrieval to *every* query. Recent adaptive methods (`Use-Graph-When-It-Needs`, `GraphRAG-Router`) fix this but **train** a router / RL policy, adding compute and hurting transfer. We propose a **training-free** controller that, per query and per hop, decides (a) whether to use the graph at all, (b) how wide the beam is, and (c) when to stop expanding — using only distributions already computed by the pipeline (entity-link cosines, LLM relevance scores). The baseline and our method share one retrieval engine; the *only* difference is the controller, so ablations are clean by construction.

## Innovations (all training-free)
1. **(2a) Use-graph decision** — skip the graph when the seed-entity anchor is unreliable (top-1 link cosine < `theta_low`); route to a direct answer, saving all relevance-judge LLM calls for that query.
2. **(2b) Adaptive beam** — allocate beam width to the *spread* of hop relevance scores: flat (uncertain) scores → wider beam; concentrated scores → narrower beam. (Normalized-entropy signal.)
3. **(3) Marginal-gain early stopping** — stop expanding once the best per-hop relevance score stops improving by more than `delta`, with an absolute floor and frontier-collapse check.

## Repo layout
```
agr/
  kg.py            knowledge graph + MetaQA/toy loaders
  llm_backend.py   pluggable LLM (local HF default, optional API) + cost meter
  retriever.py     one controller-driven multi-hop retrieval engine (baseline lives here as FixedController via the same engine)
  controller.py    FixedController (baseline) + AdaptiveController (our method)
  answer.py        answer generation from retrieved subgraph
  evaluate.py      F1/Hit@1/exact + cost accounting + Pareto table
run.py             main runner: baseline vs. method, emits reports + Pareto
configs/default.yaml
```

## Quick start (smoke test, no GPU model load)
```bash
python run.py --dataset toy --no-llm-judge --no-gen-model
```

## Real run (MetaQA, local HF model on a single 32GB GPU)
```bash
# 1. get MetaQA (WikiMovies KG) -- see scripts/download_metaqa.py
python scripts/download_metaqa.py --out ./data/MetaQA
# 2. evaluate baseline vs. our method
python run.py --dataset metaqa --split 1-hop --data-dir ./data/MetaQA \
    --gen-model Qwen/Qwen2.5-1.5B-Instruct --limit 200 \
    --controllers fixed adaptive
```

## What the output looks like
A Pareto table comparing systems on accuracy (F1/Hit@1) vs. cost (tokens/query, LLM calls/query, hops/query, % queries that used the graph). Our method should sit on or above the baseline's Pareto frontier at markedly lower cost.

## Compute
Single GPU (tested on RTX 5090, 32GB). No training. Default generator is a 1.5B open model; swap to 7-8B via `--gen-model` for stronger results.

## Status
Skeleton + baseline + all three innovations implemented. Pending: MetaQA download helper, full-scale runs, plots, paper.
