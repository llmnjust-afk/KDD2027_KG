# Rebuttal — Submission 1972 (KDD 2027, Research Track: "When Does Adaptive Retrieval Help GraphRAG?")

> **Author note (not for reviewers).** Every number below marked **[new]** was computed from the
> released per-query logs in `KDD2027_KG` by `scripts/rebuttal_analysis.py` (outputs in
> `results_rebuttal/*.json`), or measured in new GPU executions on a single A100 80GB (runs
> P1–P6, archived in `rebuttal/evidence/`); all other numbers are quoted from the submission.
> The only remaining optional GPU item is the P7 full latency sweep (the rebuttal's latency
> statements rest on the CPU-verified Table-9 chord); it is marked **[pending-GPU]** and phrased
> in the reviewer-facing text as a commitment. Three data issues we found and resolved before
> submitting are listed at the very end.

---

## General response (posted to all reviewers)

We thank all five reviewers for careful and constructive readings. The reviews converge on a small
number of issues, on all of which we have either completed new analyses from our released per-query
logs or launched new experiments:

- **R1 — Corrected Pareto claims (f7vF-W1, xxtL-W2).** Reviewer f7vF is correct on both counts, and
  we have re-audited every dominance mark from the per-query logs. (i) The annotated-hop point is
  not dominated by any *discrete* configuration; it is dominated by a *randomized mixture* of two
  tuned static configurations (arithmetic in our response to f7vF). (ii) On the latency axis
  (Table 9), \textsc{AdaptiveJudge} is itself non-dominated and extends the static frontier by
  ≈0.04 F1 at matched latency; we withdraw the "interpolates/inside the frontier" wording. We adopt
  the reviewer's suggested convention (frontier over randomized mixtures, which are deployable at
  zero training cost), correct the table labels, and restate the headline claim in its defensible
  form: *no tested training-free adaptive policy approaches oracle headroom, and the one regime
  where an adaptive point is non-dominated yields only a marginal gain over the mixture frontier*.
  **[new]** We have additionally re-executed the entire comparison in a single unified run (one
  code state, same 240 queries) and re-derived the Table-2 family under the strong engine; every
  conclusion above is confirmed, and the Table-2 7B annotated-hop row is corrected to
  **0.389 @ 856 tokens** (the printed 0.444 @ 999 was a transcription of the 14B value).
- **R2 — Held-out tuning protocol (f7vF-W4, V9mY-Q1, xxtL-W5).** **[new]** We re-ran the central
  comparison under a clean protocol: the identical 240-query mixed stream is split into stratified
  validation/test halves (120/120); all selection (global $(K,B)$, $\tau$) uses the validation half
  only. The validation-selected global configuration (Fixed-$K{=}2,B{=}8$; selected in 20/20
  validation draws) achieves test F1 **0.489** vs. **0.434** for the validation-selected adaptive
  configuration ($\tau{=}0.75$) — paired $\Delta{=}{+}0.055$, 95% CI $[{+}0.012,{+}0.102]$,
  $p{=}0.011$. On the complete 12-configuration beam/depth grid the val-selected global
  ($K{=}3,B{=}8$; 18/20 draws) reaches test **0.481** vs. **0.431** for the val-selected adaptive
  heuristic ($\Delta{=}{+}0.051$, CI $[{+}0.002,{+}0.097]$, $p{=}0.043$). The headline conclusion
  therefore does not depend on post-hoc selection.
- **R3 — Re-scoped "predictability ceiling" (31Rd-W1, f7vF-W2, V9mY-Q2, xxtL-W3).** The ceiling is
  an empirical statement about the tested engines, signal sets, router classes, and sample sizes;
  Proposition 3.1 is repositioned as a characterization of *when* signal-based routing can help,
  and abstract/conclusion wording is softened exactly along the lines the reviewers suggest.
- **R4 — New utility-aware routing experiments (f7vF-W3).** **[new]** Following the reviewer's
  suggestion we implemented gain regression ($\hat\Delta(\sigma)$, route to LLM iff
  $\hat\Delta>0$), gain-weighted classification, and a direct $U-\lambda C$ router ($\lambda$
  swept to trace a router cost–accuracy curve), all evaluated with cross-validated predictions and
  compared at *matched routing rate* against the randomized-mixture static frontier — not only
  against always-LLM. Results in our response to f7vF: no tested utility-aware router
  significantly exceeds the mixture frontier at matched cost (max gain $+0.006$ F1, 95% CI
  $[-0.013,+0.027]$), and all remain below always-LLM.
- **R5 — Router learning curves (f7vF, V9mY-W1).** **[new]** AUROC and routed F1 as functions of
  training-set size (25→150, 20 draws, 95% CIs): AUROC never leaves the chance region (e.g., MLP
  $0.52\to0.56$, CI upper bounds ≤0.63), routed F1 plateaus ≤0.385 — below always-LLM (0.413) at
  every training size — distinguishing small-sample learning failure from genuinely weak signals.
- **R6 — Remaining new experiments.** **[new]** *Retrieval-level judge metrics are now measured*
  (response to 31Rd-Q2): the LLM judge's continue-decision is chance-level at predicting where
  further retrieval helps (AUROC **0.520** over 299 continuation pairs; score↔gain correlation
  **0.082**; monotonicity **1/3**), while on the homogeneous 2-hop stream the adaptive policy is
  **+0.021 F1 at −32% tokens** vs. the fixed baseline — adaptive wins exactly where depth is real.
  All three remaining experiments are now **measured [new]** (details in the individual responses):
  the candidate-cap/beam sweep shows F1 is flat in $c_{\max}$ while cost grows monotonically, and
  the tuned-global frontier rises further at $B{=}12$ (0.485@1080); the published ToG
  early-termination mechanism, reproduced faithfully on identical subsets, is discretely dominated
  by tuned fixed configurations; and mid-retrieval-feature routers capture none of the joint
  headroom (routed F1 0.278 vs. always-best 0.463).- **R7 — Presentation (QwzZ).** A formal problem definition, explicit RQ1–RQ3, a notation table,
  and clearly flagged take-home messages.

---

## Response to Reviewer 31Rd

We thank the reviewer for the positive assessment (S1–S5) and four precise questions, each of
which we now answer with a new analysis or a running experiment.

**Q1 (joint selection of depth, beam, and judge type).** The reviewer is right that axis-wise
adaptation does not bound joint adaptation. From our per-query logs on the identical 240-query
mixed stream we computed the **joint outcome-oracle** over the tested configuration set
$\{$LLM judge $\times\,(K{=}1,B{=}4),(K{=}2,B{=}4),(K{=}2,B{=}8),(K{=}3,B{=}4)\,\}\cup\{$emb judge $\times\,(K{=}3)\}$:
it reaches F1 **0.580** at **676** tokens/q (93% of queries routed to the LLM judge; per-query
best configuration is $K{=}1,B{=}4$ for 122 queries, $K{=}2,B{=}8$ for 47, $K{=}2,B{=}4$ for 41,
embedding for 16, $K{=}3,B{=}4$ for 14). Joint headroom over the tuned global point is therefore
$+0.102$ F1 **[new]** — larger than the judge-only oracle headroom, confirming that the three
decisions interact and that the joint space is where the remaining headroom lives. On the complete
12-configuration $(K,B)$ grid (LLM judge) the joint oracle reaches **0.591** at 693 tokens/q
**[new]**. The promised per-query feature logging is now **done [new]**: on the 300-query routing
pool (same construction as our released routing data) we logged mid-retrieval features and measured
the 5-arm outcome matrix $\{K1B4, K2B4, K2B8, K3B4\}\times\{\text{LLM}\}\cup\{\text{emb }K{=}3\}$.
The pool joint oracle reaches **0.542 @ 611** tokens/q vs. **0.463 @ 1110** for the always-best
single arm ($K{=}2,B{=}8$) — headroom $+0.079$ F1 at $-45\%$ cost. A trained random-forest joint
router on these features (5-fold out-of-fold) **collapses to the cheapest arm for 99.3% of queries**
and lands at routed F1 **0.278**, far below the always-best global configuration
($\Delta{=}{-}0.185$, 95% CI $[-0.233,-0.138]$). Headroom on the joint axis is real; learned
routing over currently available features captures none of it. The revision reports the full
outcome matrix and this router next to the joint oracle.

**Q2 (re retrieval-level metrics).** We agree final-answer F1 conflates retrieval quality with
generation behavior. The revision adds: (i) **answer-entity recall** (fraction of gold answer
entities covered by the retrieved triples), (ii) **gold-chain recall** on MetaQA (supporting chains
are annotated), and (iii) **judge agreement** between the embedding and LLM rankings per hop.
The judge-reliability half of this program is now **measured [new]** (`scripts/judge_reliability.py`,
2-hop split, $n{=}150$, Qwen2.5-7B judge, 299 continuation pairs): the LLM judge's score predicts
*where continuing retrieval helps* at **chance level (AUROC 0.520)**, its correlation with the
realized F1 gain is **0.082**, and higher scores map to higher realized gain in only **1/3** of
transitions — i.e., the mid-retrieval signal the adaptive policies consume is essentially
uninformative about retrieval-level value. Consistently, on this homogeneous 2-hop stream the
adaptive policy is **+0.021 F1 at −32% tokens** vs. the matched fixed configuration (0.325@887 vs.
0.304@1305): when depth is genuinely needed, stopping decisions made with this weak signal still
save cost, but the *predictive* content of the signal is near zero. Our per-hop analysis already
separates the two regimes: under the strong engine, answer accuracy rises with depth up to the
ground-truth hop count on homogeneous streams (re retrieval-limited), while on the mixed stream
depth-3 recall gains are offset by judge noise that the generator cannot discount. The
answer-entity and gold-chain recall metrics will make this decomposition explicit in the revision.

**Q3 (full KG vs. pre-extracted subgraphs).** A clarification that resolves half of this concern:
**all MetaQA experiments already run on the full WikiMovies KG** with run-time entity linking
(`agr/kg.py` builds the complete triple store; the retriever links seeds by embedding similarity
at query time). Pre-extracted per-query subgraphs are used only for CWQ and WebQSP, where they are
the community-standard retrieval source (CWQ standard release; WebQSP from RoG). We will state
this explicitly in §5.1. For CWQ/WebQSP over full Freebase, linking and traversal at corpus scale
is beyond the rebuttal window; the revision scopes those conclusions to the released-subgraph
setting and marks full-corpus transfer as open.

**Q4 (learned router on intermediate post-hop signals).** Now **implemented and measured [new]**.
The features our current routers use (hop-0 embedding-score statistics of the expanded candidate
set: top-1/top-2/gap/mean, candidate count, depth) are already computed *after expansion but before
the LLM judge*; we have now logged the richer mid-retrieval feature set (hop-1 statistics, frontier
overlap with hop-0, expansion-size dynamics) on the 300-query routing pool together with the full
5-arm outcome matrix. The cross-validated answer is negative: mid-retrieval features add **no
judge-routing signal** beyond the hop-0 statistics (RF AUROC 0.529 vs. 0.553 for hop-0-only;
adding the question embedding lifts both to 0.60–0.62, so the small signal lives in the question,
not the retrieval dynamics), and the router's F1 at its own operating point matches the
rate-matched static mixture within noise ($\Delta{=}{+}0.0003$, CI $[-0.007,+0.008]$, $p{=}0.98$).
The direction the reviewer proposed is thus tested, not merely proposed: richer post-hop features
from this engine do not unlock predictive routing. The joint $(K,B,\text{judge})$ router trained
on the same features likewise fails (Q1: routed F1 0.278 vs. always-best 0.463). We will report
both routers with per-feature importances in the revision, next to the pre-computation feature set
(best cross-validated AUROC 0.559, routed F1 ≤ 0.379 vs. always-LLM 0.413).

**W1 (ceiling not fundamental), W2 (narrow policy space).** Agreed; R3 re-scopes the claim, Q1
adds the joint axis, Q4 adds richer signals, and the new utility-aware routers (R4) add the
objective the reviewer would consider most informative. The released per-query outcome matrices
make joint-dependence MI tests straightforward for follow-up work.

**W3 (oracle is diagnostic headroom, not an upper bound).** We adopt the reviewer's framing
verbatim in the revision: the oracle is a diagnostic of headroom *within the tested action set*;
the annotated-hop baseline is privileged but not optimal, and neither bounds adaptive retrieval.

**W4 (external validity).** Addressed by the Q3 clarification (MetaQA is already full-graph), the
running cap sweep (R6), and an explicit scope statement (within-engine claims for ToG-lineage
iterative engines with zero-shot open-weight judges; no transfer claim to fine-tuned production
systems).

**W5 (utility conflation).** Addressed by Q2's recall metrics; the revision interprets F1 deltas
only where retrieval recall moves in the same direction.

---

## Response to Reviewer f7vF

We thank the reviewer for an exceptionally careful reading. W1 identifies a genuine labeling error,
and the reviewer's Table-9 arithmetic is correct. We re-audited every dominance mark from the
released per-query logs (`reports.jsonl`, paired per query id) and answer all questions.

**W1 (Pareto-dominance claim vs. reported numbers).** Using the exact Table-2 coordinates
(MetaQA mixed, 7B, $n{=}240$):

| Configuration | Tokens/q | F1 |
|---|---|---|
| Fixed-emb $K{=}3$ (free judge) | 159 | 0.397 |
| Fixed-LLM $K{=}1,B{=}4$ | 398 | 0.254 |
| \textsc{AdaptiveJudge} $\tau{=}0.55$ | 886 | 0.394 |
| Fixed-LLM $K{=}2,B{=}4$ | 1005 | 0.396 |
| Annotated-hop ($B{=}4$; privileged) | 999 | 0.444 |
| Fixed-LLM $K{=}2,B{=}8$ | 1101 | 0.478 |
| \textsc{AdaptiveJudge} $\tau{=}0.75$ | 1402 | 0.431 |
| Fixed-LLM $K{=}3,B{=}4$ | 1610 | 0.406 |

- $\tau{=}0.55$ (886, 0.394) **is** discretely dominated — by the free embedding policy (159, 0.397).
  **[new]** The paired bootstrap F1 difference is $+0.003$, 95% CI $[-0.029,+0.036]$, $p{=}0.85$:
  statistically indistinguishable. The honest statement, which the revision adopts, is that
  $\tau{=}0.55$ offers *no accuracy advantage at $5.6\times$ the cost*, not a significant loss.
- $\tau{=}0.75$ (1402, 0.431) **is** discretely dominated — by Fixed-$K{=}2,B{=}8$ (1101, 0.478).
  **[new]** Paired bootstrap: $\Delta{=}{+}0.048$, 95% CI $[{+}0.004,{+}0.091]$, $p{=}0.031$.
- **The annotated-hop point (999, 0.444) is not dominated by any discrete configuration — the
  $\dagger$ mark was wrong, exactly as the reviewer states.** It **is** dominated once randomized
  mixtures are allowed **[new]**: mixing Fixed-emb $K{=}3$ and Fixed-LLM $K{=}2,B{=}8$ with
  probability $p{=}0.58$ on the latter attains F1 $=0.444$ at $703$ tokens/q (vs. 999), and at the
  annotated point's own 999-token budget the mixture attains F1 $=0.469>0.444$ — the point lies
  below the convex hull of the two frontier static policies. Mixtures are deployable at zero
  training cost (a per-query coin flip), so we adopt the **mixture-augmented static frontier** as
  the reference in the revision.
- **Complete fixed-policy sweep, single unified run.** **[new]** We have now re-executed the
  complete grid end-to-end in one code state (same 240 queries; archived in
  `rebuttal/evidence/G_unified_pareto.json`), so the revision's tables come from a single
  self-consistent run:

  | $K$ | $B{=}2$ | $B{=}4$ | $B{=}6$ | $B{=}8$ |
  |---|---|---|---|---|
  | $K{=}1$ | 0.205 @ 379 | 0.247 @ 398 | 0.263 @ 415 | 0.295 @ 426 |
  | $K{=}2$ | 0.305 @ 884 | 0.390 @ 1001 | 0.426 @ 1061 | 0.460 @ 1101 |
  | $K{=}3$ | 0.343 @ 1293 | 0.412 @ 1611 | 0.463 @ 1761 | 0.482 @ 1844 |

  On this complete grid **every depth-adaptive heuristic we test is discretely dominated** by a
  tuned fixed configuration: marginal-gain adaptive (1066, 0.401) ← $K{=}2,B{=}6$ (1061, 0.426;
  $+0.025$ F1 at lower cost); patience $p{=}1$ (1097, 0.406) ← $K{=}2,B{=}6$; confidence
  thresholds $t{=}0.70/0.85$ (1147, 0.387) ← $K{=}2,B{=}6$; margin $t{=}0.30/0.50$ (1461, 0.414)
  ← $K{=}2,B{=}8$ (1101, 0.460); patience $p{=}2$ coincides with $K{=}3,B{=}4$ (1611, 0.412) ←
  $K{=}2,B{=}8$; and random budgets (996, 0.347) are dominated at equal cost by the annotated-depth
  oracle itself (996, 0.389). The annotated-depth baseline (996, 0.389 **[re-measured in this
  run]**) remains the only non-discretely-dominated adaptive point, and is dominated by mixtures
  (the $K{=}1,B{=}8$ ↔ $K{=}2,B{=}8$ chord gives F1 0.434 at 996 tokens; the same F1 0.389 is
  attainable at ≈811 tokens). The revised paper identifies the dominating configuration or mixture
  for every adaptive point explicitly, as the reviewer requests, and corrects Table 2's $\dagger$
  marks under the explicit three-way taxonomy: (i) discrete dominance, (ii) mixture dominance,
  (iii) non-dominated.
- **Strong-engine re-derivation and the Table-2 annotated row.** **[new]** Auditing the released
  runs against the submission surfaced a transcription error: the printed 7B annotated-hop row
  (0.444 @ 999) matches the **14B** value exactly, while the 7B runs give 0.389–0.392. We therefore
  re-derived the full Table-2 family on the strong engine (the caption's claimed engine) in a
  single run (`results_p2_strong_7b`, archived in `rebuttal/evidence/H_p2_strong.json`):
  annotated-hop = **0.389 @ 856 tokens** (the strong prefilter saves 140 tokens at identical F1);
  Fixed-$K{=}2,B{=}8$ = 0.467 @ 1040; the $\tau{=}0.55$ adaptive judge = 0.386 @ 394 (now *above*
  Fixed-$K{=}1,B{=}4$ = 0.258 @ 398 at the same cost); the remaining fixed rows ($K{=}1,B{=}4$,
  $K{=}2,B{=}4$ at 0.380@976, $K{=}3,B{=}4$ at 0.387@1319) are discretely dominated. The
  conclusions are invariant under the engine change: the annotated point is still not discretely
  dominated and still mixture-dominated (the $\tau{=}0.55$ ↔ $K{=}2,B{=}8$ chord attains F1 0.444
  at its 856-token cost), and the revised Table 2 will print the corrected 7B row.
- **Table 9 (latency axis): the reviewer's computation is correct.** **[new]** The chord between
  Fixed-emb (0.89 s, 0.323) and Fixed-LLM $K{=}2,B{=}8$ (3.16 s, 0.380) gives expected F1
  $0.323+\frac{1.30-0.89}{3.16-0.89}(0.380-0.323)=0.333$ at 1.30 s — well below
  \textsc{AdaptiveJudge}'s 0.371. At matched *judge-call* budgets the same holds: the mixture with
  call rate $0.31/2.00$ gives F1 0.332 vs. the adaptive point's 0.371 ($+0.040$). The adaptive
  point is therefore **non-dominated** on the latency/call axis and extends the static frontier at
  intermediate latencies; we withdraw "interpolates … strictly inside the static frontier" for
  this axis. The revised headline claim: *no tested training-free adaptive policy approaches
  oracle headroom; the single regime where an adaptive point is non-dominated (latency axis)
  yields a marginal gain ($\approx+0.04$ F1) over randomized static mixtures; tuned global budgets
  remain the recommended default.* A complete latency sweep over the grid will be reported in the
  revision **[pending-GPU]**, including identification of any discrete static configuration that
  dominates (1.30 s, 0.371).

**Q (frontier over discrete configurations or randomized mixtures?).** Mixtures, in the revision —
this strengthens our negative claim by making the reference harder to beat, as the reviewer's own
chord computation shows.

**W2 ("predictability ceiling" overclaim).** Agreed. Proposition 3.1 provides no nontrivial upper
bound. The revision (i) removes "the decisive information becomes observable only after expensive
computation" from the abstract/conclusion and replaces it with the reviewer's phrasing — *under
the tested engines, feature sets, training sizes, and router families, we do not observe
sufficient predictive power to realize the oracle headroom*; (ii) retains the hedged MI paragraph
(marginal ≠ joint; permutation null $p{=}0.92$); (iii) adds the learning curves below.

**W3 (routing objective misaligned with utility).** Agreed — and implemented **[new]**, with
cross-validated predictions on the $n{=}300$ routing pool (base rate 21.7%; always-embedding 0.366,
always-LLM 0.413, oracle 0.479):
- **Gain regression** (predict $\Delta=U_{\text{LLM}}-U_{\text{emb}}$ from the six signals, route
  to LLM iff $\hat\Delta>0$): gradient-boosted trees → routed F1 0.387 at a 66% LLM rate; random
  forest → **0.405 at a 70% rate** — the best realizable router we observe. It remains below
  always-LLM (0.413), and against the *randomized-mixture frontier at its own routing rate*
  (0.399 at 70%) the gain is $+0.006$, 95% CI $[-0.012,+0.023]$, $p{=}0.52$ — within noise.
- **Gain-weighted classification** (RF, sample weights $|\Delta|$): routed F1 0.388.
- **Direct $U-\lambda C$ router** ($\lambda$ swept over the $\hat\Delta$ threshold): the router
  cost–accuracy curve never significantly exceeds the mixture frontier at matched rate — best
  point 0.410 at a 98% rate vs. mixture 0.412 ($\Delta{=}{-}0.002$, CI $[-0.008,+0.001]$); the
  maximum gain over the mixture frontier across the curve is $+0.006$ at a 46% rate, CI
  $[-0.013,+0.027]$.
- **Matched-cost comparison vs. static references (not only always-LLM):** at every routing rate
  $r$, the deployable static reference is the mixture $(1-r)\cdot\text{emb}+r\cdot\text{LLM}$; the
  tested routers track but do not significantly exceed this line anywhere on the curve, while the
  outcome-oracle sits $+0.066$ above always-LLM. We will add this router-frontier-vs-mixture-frontier
  figure to the revision.

**Q (learning curves with CIs).** **[new]** Routers retrained at $n_{\text{train}}\in\{25,50,75,100,150\}$
(20 draws, stratified 50/50 held-out split, 95% percentile CIs): AUROC for $Y^\star$ — LogReg flat
at $\approx0.50$; MLP $0.518$ (n=25) $\to0.556$ (n=150), CI upper bounds 0.60–0.63; RF
$0.505\to0.565$, CI upper bounds ≤0.61. Routed F1 — MLP $0.374\to0.385$, RF $0.368\to0.371$ —
plateaus strictly below always-LLM (0.413) at every training size, with CIs excluding 0.413 at the
larger sizes. The curves show a mild upward trend that remains far from decision-useful: the
failure is not small-sample learnability within this range, and the released per-query matrices
let others extend the pool.

**W4 (protocol clarity; Table 1 vs. Table 6).** **[new]** Three parts.
(i) *Protocol.* We adopt the held-out protocol described in the general response (R2): on the
identical 240-query stream split stratified 50/50, the global configuration selected on validation
only ($K{=}2,B{=}8$; 20/20 draw stability) achieves test F1 0.489 vs. 0.434 for the
validation-selected adaptive configuration — paired $\Delta{=}{+}0.055$, CI $[{+}0.012,{+}0.102]$,
$p{=}0.011$; on the complete grid, val-selected $K{=}3,B{=}8$ test 0.481 vs. val-selected adaptive
0.431 ($p{=}0.043$). Post-hoc selection on the evaluation subset is thus not load-bearing for the
conclusion, and the revision documents the split, selection criterion, and tuning budget for every
policy (adaptive thresholds receive the same validation budget as the global sweep).
(ii) *Subset construction.* The revision documents the construction of every table's subset
($n{=}240$ master/mixed stream; $n{=}198$ judge-budget triple; $n{=}150$ oracle set; $n{=}120$
latency set; $n{=}180$ family-matched streams; $n{=}200$ CWQ; $n{=}120$ WebQSP; $n{=}498\times3$
multi-seed), with fixed sampling seeds released.
(iii) *Table 1 vs. Table 6.* **[new]** We audited the runs: Table 6's "best Adaptive F1" is the
$\tau{=}0.55$ operating point in every family (Qwen 0.394@886; Llama 0.239@830; Mistral
0.453@1048), whereas Table 1 additionally lists the $\tau{=}0.75$ operating point (Qwen 0.431@1402;
Mistral 0.515@1657; Llama 0.234@1328). Selecting $\tau$ by test F1 is the post-hoc reading the
protocol does not permit; the revision will label the rows by their selection rule and re-report
both tables on one subset. Crucially, the conclusion is invariant to the reading: under *either*
operating point, the tuned global configuration dominates or mixture-dominates the adaptive point
in every family (Qwen: 0.478@1101 vs. 0.431@1402; Mistral: 0.569@1313 vs. 0.515@1657; Llama:
0.248@1042 vs. 0.234@1328).

**Q (larger candidate cap / stronger candidate recall).** Now **measured [new]**: on the MetaQA
mixed stream ($n{=}240$, strong engine, same 240 queries as Table 2) we swept
$c_{\max}\in\{16,32,64\}$ with the tuned configs, the adaptive judges, and the annotated-depth
oracle, plus $B{=}12$:

| System | $c_{\max}{=}16$ | $c_{\max}{=}32$ (default) | $c_{\max}{=}64$ |
|---|---|---|---|
| Fixed-emb $K{=}3$ | 0.378 @ 125 | 0.378 @ 125 | 0.378 @ 125 |
| \textsc{AdaptiveJudge} $\tau{=}0.55$ | 0.394 @ 339 | 0.405 @ 398 | 0.395 @ 469 |
| \textsc{AdaptiveJudge} $\tau{=}0.75$ | 0.414 @ 878 | 0.407 @ 1045 | 0.396 @ 1345 |
| Annotated-depth oracle | 0.399 @ 762 | 0.379 @ 872 | 0.374 @ 1045 |
| Fixed-$K{=}2,B{=}8$ | 0.452 @ 863 | 0.440 @ 1041 | 0.441 @ 1354 |
| Fixed-$K{=}2,B{=}12$ | — | **0.485 @ 1080** | 0.479 @ 1395 |

Four findings. (i) *Accuracy is flat in cap; cost is not*: F1 varies within run-to-run jitter
(±0.02–0.03; e.g., the identical $K{=}2,B{=}8$ configuration measures 0.467 in the Table-2 run vs.
0.440 here, so we read the sweep directionally), while tokens grow monotonically (863 → 1041 →
1354). (ii) *Tight caps do not prevent exploiting depth* — the concern the question probes is
refuted under the strong engine: the embedding prefilter concentrates relevance, and $c_{\max}{=}16$
is actually the efficiency sweet spot (annotated 0.399@762; $\tau{=}0.75$ 0.414@878). (iii) *The
extra-budget channel is beam, not cap*: $K{=}2,B{=}12$ reaches 0.485@1080 in the same run as
$K{=}2,B{=}8$ at 0.440@1041 ($+0.045$ F1 at $+4\%$ tokens) — when the reviewer's "stronger
candidate recall" is given room, it is the *tuned global* policy that exploits it, which
strengthens rather than narrows our headline claim. (iv) The free embedding arm is exactly
cap-invariant (identical to 4 decimals at every cap). The revision adds this sweep as a
cap-sensitivity table. A paired per-query CI version of the table (one report-saving re-run) is
planned for the revision.

---

## Response to Reviewer QwzZ

We thank the reviewer for the candid assessment. We agree the write-up led with the empirical
system rather than the decision problem, and the revision restructures the paper around an explicit
formalization:

**Problem definition (new §1.3/§3.1).** Given a fixed retrieval engine $E$ whose per-hop relevance
judge dominates cost ($c_{\text{LLM}}\gg c_{\text{emb}}\approx0$), an *adaptation policy* $\pi$
maps the observable signal history $\sigma_{<k}$ at each decision point to an action
$a_k\in\{\text{stop at }k,\ \text{beam }B,\ \text{judge }j\}$; the objective is
$\max_\pi\mathbb{E}_q[U(\pi(q))]$ subject to $\mathbb{E}_q[C(\pi(q))]\le C_0$. A *global* policy is
the special case $\pi(q)\equiv$ const, tuned by a small validation sweep. We study three research
questions — **RQ1 (headroom):** does per-query adaptation have value under an oracle? **RQ2
(predictability):** do pre-computation signals identify the queries where adaptation pays? **RQ3
(robustness):** do the answers persist across engines, model scales and families, and candidate
caps?

**Technical development (§3 rewritten).** Signal sets and policy classes are defined precisely
(threshold/patience/margin/gap rules; router classes; the privileged annotated-depth baseline;
outcome oracles restricted to their action sets), with a notation table, so that every experiment
corresponds to one cell of the $\langle\text{axis}, \text{signal class}, \text{policy class}\rangle$
grid. Proposition 3.1 is presented as a characterization of when a signal-based router can beat the
best global action (see R3 in the general response).

**Take-home messages (end of §1 and §5).** (T1) A tuned global budget is a strong baseline that
none of the tested training-free adaptive policies beats at matched cost in any tested setting —
now verified under a held-out validation protocol (R2). (T2) Oracle headroom is real and
localized: it comes from embedding/LLM judge complementarity (oracle 0.474 vs. 0.411 at a 27% LLM
routing rate) and grows in the joint $(K,B,\text{judge})$ space ($+0.102$ F1 over the tuned global
point). (T3) The bottleneck is decision-time information, not headroom: utility-aware routers
(gain regression, $U-\lambda C$) do not significantly exceed the randomized-mixture static
frontier at matched cost — so useful adaptation must consume mid-retrieval evidence, which the
revision implements and measures. We also correct typographical issues and redesign the framework
figure to make the RQ→axis→experiment mapping explicit.

---

## Response to Reviewer V9mY

We thank the reviewer for engaging deeply with the headroom/predictability distinction and for
three sharp objections, each of which we address with a new analysis.

**Q1 (tuned global selected on the evaluation subsets).** Correct and important; we have now ruled
it out directly **[new]**. On the identical 240-query mixed stream split into stratified
validation/test halves (120/120), all selection uses the validation half only: the
validation-selected global configuration ($K{=}2,B{=}8$; selected in 20/20 validation draws)
achieves **test F1 0.489** vs. **0.434** for the validation-selected adaptive configuration
($\tau{=}0.75$) — paired $\Delta{=}{+}0.055$, 95% CI $[{+}0.012,{+}0.102]$, $p{=}0.011$. On the
complete 12-configuration grid, the val-selected global ($K{=}3,B{=}8$; 18/20 draws) reaches test
0.481 vs. 0.431 for the val-selected adaptive heuristic ($\Delta{=}{+}0.051$, CI
$[{+}0.002,{+}0.097]$, $p{=}0.043$). This is consistent with the multi-seed bootstrap already in
Appendix B ($\Delta{=}{+}0.055$, CI $[{+}0.039,{+}0.070]$, $p<10^{-4}$, $1{,}494$ instances). The
revision documents the split, the selection criterion, and equivalent tuning budgets for adaptive
thresholds, and releases the selection scripts.

**Q2 (Proposition 3.1 "oversells").** Agreed. It is a conditional-expectation identity
characterizing *when* a signal-based router improves on the best global action — when
$m(\sigma)=\mathbb{E}[\Delta\mid\sigma]$ changes the optimal decision on a set of queries with
positive utility mass. The revision (i) retitles it "When can signal-based routing improve on a
global policy?", (ii) removes every "formalizes the ceiling" phrasing, and (iii) presents the
predictability ceiling as an empirical construct whose scope is stated in the abstract (R3). We
keep the identity because it correctly motivates measuring $m(\sigma)$-relevant information rather
than binary AUROC — which is precisely what the new gain-regression routers (f7vF-W3) implement.

**Q3 (tight caps; engine may be unable to exploit correct depth).** Three answers. First, on
*homogeneous* depth streams under the strong engine, accuracy rises monotonically with depth up to
the ground-truth hop count, and the tuned $K{=}2$ point on WebQSP recovers the benchmark's 2-hop
structure — so the engine does exploit correct depth when the query population is homogeneous. On
the mixed stream the third hop injects judge noise for *all* depth groups (the per-hop analysis),
which is why fixed $K{=}2$ beats matching each query's true depth. Second, the cap sweep is now
**measured [new]** (full table in our response to f7vF): under the strong engine, accuracy is flat
in $c_{\max}\in\{16,32,64\}$ within run-to-run jitter while cost grows monotonically, tight caps do
*not* prevent exploiting depth ($c_{\max}{=}16$ is the efficiency sweet spot: annotated 0.399@762,
$\tau{=}0.75$ 0.414@878), and the engine's extra-budget channel is beam width — $K{=}2,B{=}12$
raises the tuned-global frontier to 0.485@1080. The revision adds the cap-sensitivity table. Third,
we have now **reproduced a published training-free adaptive-depth controller [new]** — ToG's
early-termination mechanism (a per-hop LLM sufficiency check, one extra Yes/No call per executed
hop), the closest published representative of the evaluated class — on our identical mixed stream,
engine, and beam settings: ToG-Stop $B{=}4$ reaches 0.370@1230 (5.1 calls/q) and $B{=}8$ reaches
0.459@1513 (5.2 calls/q); **both points are discretely dominated by tuned fixed configurations in
the same run** ($B{=}4$ ← $K{=}2,B{=}8$ 0.440@1041; $B{=}8$ ← $K{=}2,B{=}12$ 0.485@1080), and they
are the most LLM-call-hungry policies we test. The published mechanism lands on the same side of
the frontier as our patience/margin/confidence family, exactly as the complete-grid analysis
predicts. Learned routing methods (Dong et al.; Fan et al.) are outside
the training-free class we test; the related-work section now delimits this class explicitly with
a taxonomy table.

**W1 (few hundred queries; null MI may reflect sample size).** Directly addressed by the learning
curves with CIs (R5/f7vF): AUROC never leaves the chance region across training sizes 25→150
(MLP $0.518\to0.556$ with CI upper bounds ≤0.63; RF $0.505\to0.565$), and routed F1 plateaus
≤0.385, strictly below always-LLM (0.413) at every size. The revised claim is "no detectable
signal at the tested sample sizes," and the released per-query outcome matrices allow others to
extend the pool.

**W2 (CWQ near-random judge).** Agreed: on CWQ the generator is the bottleneck, so a
predictability analysis there has little headroom to explain. The revision (i) reports CWQ only as
an engine-diagnostic result (depth-insensitivity of F1), (ii) adds judge-agreement and
retrieval-recall metrics on CWQ (31Rd-Q2) showing the low F1 is generator-limited, and (iii)
removes CWQ from any predictability claim.

**W3 (does not reproduce published adaptive methods).** Now addressed **with a reproduction
[new]**: ToG's early-termination mechanism (per-hop LLM sufficiency check) is implemented
faithfully and measured on identical subsets — both operating points ($B{=}4$: 0.370@1230;
$B{=}8$: 0.459@1513) are discretely dominated by tuned fixed configurations and use the most
LLM calls of any tested policy (details in Q3). We additionally delimit the class precisely: our
negative result covers
*training-free control of retrieval effort under a shared engine*; learned policies and
backend-selection routers are outside it, and the revision says so.

---

## Response to Reviewer xxtL

We thank the reviewer for the balanced assessment; every weakness maps to a concrete revision.

**W1 (empirical diagnosis rather than a method).** We accept the framing and revise the
contribution statement: a measurement framework (headroom vs. predictability), a negative result
under the strongest tested baselines, and practical guidance — plus, new in revision, constructive
elements: the joint $(K,B,\text{judge})$ outcome oracle ($+0.102$ F1 headroom over the tuned
global point), utility-aware routers, and an implemented mid-retrieval-signal router (in
progress). The formal proposition is repositioned as a characterization, not a theoretical
contribution claim.

**W2 (dominated labels on lower-cost/lower-accuracy adaptive points).** Fixed exactly as in our
response to f7vF-W1: we adopt the randomized-mixture frontier, correct the annotated-hop mark
(not discretely dominated; mixture-dominated), and withdraw the latency-axis "interpolation" claim
since \textsc{AdaptiveJudge} is non-dominated there ($+0.038$ F1 above the chord at matched
latency). The headline claim is restated in its defensible form. **[new]** All dominance marks are
now verified in a single unified re-run (every tested depth-adaptive heuristic discretely dominated
by a tuned fixed configuration; the annotated-depth point the only non-discretely-dominated
adaptive policy, and mixture-dominated), and the Table-2 7B annotated row is corrected to
0.389 @ 856 tokens under the captioned strong engine.

**W3 (tested features/routers ≠ fundamental barrier).** Agreed; the claim is re-scoped (R3), and
the new joint-feature, gain-regression, $U-\lambda C$, and learning-curve analyses strengthen
exactly the dimensions the reviewer lists.

**W4 (zero-shot models, restricted candidates, low absolute performance; proposed direction
unimplemented).** The scope statement is added (within-engine claims for ToG-lineage iterative
engines with zero-shot open-weight judges; no transfer claim to fine-tuned production systems).
The candidate-cap sweep is now **measured** (response to f7vF): the restricted-cap concern is
refuted under the strong engine — tight caps cost nothing and the tuned-global frontier rises at
$B{=}12$. MetaQA is already run on the full knowledge graph (31Rd-Q3). The proposed direction is
now *implemented and measured*: mid-retrieval (post-expansion, pre-judge) signal routers are
logged on the routing pool with the full 5-arm outcome matrix, and they add no routing signal
beyond the pre-computation features (RF AUROC 0.529 vs. 0.553; joint router 0.278 vs. always-best
0.463; details in 31Rd-Q1/Q4).

**W5 (small subsets; tuning/eval separation; latency coverage).** Held-out validation protocol and
per-table subset documentation added (R2); the latency table will be extended to all policies with
mean/p50/p95 latency, judge calls, and tokens from the complete-grid re-run; bootstrap CIs are
added to every headline contrast.

---

---

# Experiment Checklist (author-facing — not for reviewers)

## Completed from released logs (CPU, `scripts/rebuttal_analysis.py` → `results_rebuttal/`)

| Item | Result file | Used in |
|---|---|---|
| A. Table-2 dominance audit + paired bootstrap CIs + mixture frontier | `A_pareto_audit.json` | f7vF-W1, xxtL-W2 |
| A+. Complete 12-config grid + depth heuristics, within-run dominance (same 240 qids) | `A_pareto_audit.json` (`mp_grid`) | f7vF-W1 |
| B. Held-out validation/test re-selection (FP + full grid), selection stability 20/20 | `B_valtest_protocol.json` | R2, f7vF-W4, V9mY-Q1 |
| C. Learning curves, gain regression, $U-\lambda C$, router-vs-mixture CIs | `C_routers.json` | R4/R5, f7vF-W3, V9mY-W1 |
| D. Joint outcome oracle (5-config incl. judge type: 0.580@676; 12-config grid: 0.591@693) | `D_joint_oracle.json` | 31Rd-Q1 |
| E. Table-9 chord verification (0.333 at 1.30 s; +0.038/+0.040 gains) | `E_latency_chord.json` | f7vF-W1 |
| Family-table audit: Table 6 "best adaptive" = τ=0.55 row in all three families | `results_p02_*` | f7vF-W4 |

## Pending GPU runs (Lab Compute: 1× A100 80GB, bf16, HF backends)

| ID | Run | Command sketch |
|---|---|---|
| ~~P1~~ | **DONE** (see "Completed on GPU" below) | `python3 scripts/mixed_pareto.py --gen-model Qwen/Qwen2.5-7B-Instruct --per 80 --seed 0 --out-dir results_unified_pareto` |
| P2 | Annotated-depth baseline on the strong engine (resolves the Table-2 7B row) | **DONE** — patched `full_pareto.py --strong`, 8 systems, n=240 (`results_p2_strong_7b`); see "Completed on GPU" |
| P3 | Retrieval-level metrics (answer-entity recall, gold-chain recall, judge agreement) | **DONE** — `judge_reliability.py --split 2-hop --limit 150` (`results_judge_reliability`); see "Completed on GPU" |
| P4 | Candidate-cap/beam sweep $c_{\max}\in\{16,32,64\}$, $B\le12$ (needs a `--c-max` flag in the retriever) | **DONE** — `candidate_cap` override in `GraphRAGRetriever` + `p456_experiments.py --mode cap-sweep`; MetaQA mixed, strong engine (WebQSP arm not run: raw WebQSP subgraphs unavailable on the lab machine; MetaQA is the master stream of all dominance claims) |
| P5 | ToG confidence-based early stopping reproduction on identical subsets | **DONE** — ToG sufficiency-check mechanism implemented in `GraphRAGRetriever` (`tog_sufficiency_stop`), `--mode tog`, strong engine, identical stream |
| P6 | Mid-retrieval feature logging (hop-1 stats, frontier overlap) + joint router over the grid | **DONE** — `--mode features` (300-query seed-1 pool, per-hop logging, 5-arm outcome matrix) + `p6_router_analysis.py` (CPU) |
| P7 | Full latency sweep for the extended cost table | reuse P1 with timing instrumentation |
| — | Full-Freebase CWQ/WebQSP | out of scope; scope the claim instead |

## Completed on GPU (lab 1× A100 80GB, bf16, HF backends, seed 0)

| ID | Run | Key results |
|---|---|---|
| P1 | Unified mixed-stream re-run, n=240, 21 systems (`results_unified_pareto`) | **Every depth-adaptive heuristic discretely dominated** (9/21 systems): Adaptive B=4 (0.401@1066) ← K2B6 (0.426@1061; +0.025 F1 at lower cost — no longer within noise); Patience p=1 (0.406@1097) ← K2B6; Conf t=0.70/0.85 (0.387@1147) ← K2B6; Margin t=0.30/0.50 (0.414@1461) ← K2B8 (0.460@1101); K3B2 (0.343@1293) ← K2B8; K3B4 = Patience p=2 (0.412@1611) ← K2B8. **Annotated-hop Oracle-depth 0.389@996 not discretely dominated**, mixture-dominated (chord K1B8→K2B8 at 996 toks = 0.434 > 0.389). Non-dominated frontier: K1B2–K1B8, K2B2/B4/B6/B8, K3B6/B8, Oracle-depth. All audit conclusions (f7vF-W1, xxtL-W2) now rest on a single self-consistent run. |
| P2 | Strong-engine re-derivation of the Table-2 family, n=240, 8 systems (`results_p2_strong_7b`) | **Table-2 7B annotated row fixed: 0.389 @ 856 tokens** (weak engine: 0.389@996; the printed 0.444@999 is the 14B value — data issue #1 resolved). Annotated-hop **still not discretely dominated**, still mixture-dominated (chord AdaptiveJudge-T0.55 0.386@394 → K2B8 0.467@1040 = 0.444 at its cost). K=2 B=8 stays the accuracy king: **0.467@1040**. Under the strong engine the adaptive judge family beats fixed K=1 (0.386@394 vs 0.258@398); dominated rows: K1B4, K2B4 (0.380@976), K3B4 (0.387@1319). Revised Table 1/2 should be rebuilt from this file. |
| P3 | Judge reliability / retrieval-level metrics, 2-hop split, n=150, 299 continuation pairs (`results_judge_reliability`) | **Judge scores are retrieval-uninformative**: continue-helps **AUROC 0.520** (chance), gain-correlation **0.082**, score monotonicity **1/3**. On 2-hop the adaptive system is **+0.021 F1 at −32% tokens** vs fixed (0.325@887 vs 0.304@1305) — adaptive helps exactly where depth is real. Answers 31Rd-Q2; strengthens the predictability-ceiling thesis (f7vF-W2); motivates 31Rd-Q4 (stronger mid-retrieval signals for learned routers). Evidence archived as `rebuttal_evidence/F_judge_reliability.json`. |

**All planned GPU runs P1–P3 complete. Remaining optional: P4 (c_max sweep, needs retriever flag), P5 (ToG early-stopping reproduction), P6 (mid-retrieval feature logging + joint router).**

## Completed on GPU — P4–P6 (lab 1× A100 80GB, bf16, HF backends, 2026-09-28)

| ID | Run | Key results |
|---|---|---|
| P4 | Candidate-cap sweep, strong engine, n=240 (`results_cap_sweep`; evidence `I_cap_sweep.json`) | F1 flat in $c_{\max}\in\{16,32,64\}$ within run-to-run jitter (±0.02–0.03; same-config cross-run spread up to 0.027), cost monotone (K2B8: 863→1041→1354 tokens); **c16 is the efficiency sweet spot** (annotated 0.399@762; T0.75 0.414@878); free-emb arm exactly cap-invariant (0.378@125 at every cap); **beam is the extra-budget channel: K2B12 = 0.485@1080 vs K2B8 = 0.440@1041 in the same run (+0.045 F1 at +4% tokens)** — tuned-global frontier rises further, dominance conclusions strengthen. WebQSP arm not run (raw subgraphs unavailable on lab; scoped in checklist). |
| P5 | ToG early-termination reproduction, strong engine, n=240 (`results_tog`; evidence `J_tog.json`) | Faithful ToG mechanism (per-hop LLM sufficiency check, +1 call/hop): **ToG-Stop B=4 = 0.370@1230 (5.1 calls/q), B=8 = 0.459@1513 (5.2 calls/q)**; both discretely dominated in the same run (B=4 ← K2B8 0.440@1041; B=8 ← K2B12 0.485@1080); most call-hungry policies tested (all others ≤4.0 calls/q). Answers V9mY-W3. |
| P6 | Mid-retrieval feature logging + joint outcome matrix, 300-query seed-1 pool, weak engine (`results_p6_features` 4.0 MB on lab + `results_p6_analysis`; evidence `K_p6_analysis.json`) | **Mid-retrieval features add no routing signal**: RF AUROC mid-only 0.529 vs hop-0-only 0.553 (with qemb both 0.60–0.62 — signal lives in the question embedding); router vs rate-matched mixture Δ=+0.0003, CI [−0.007,+0.008], p=0.98. **Joint (K,B,judge) router fails**: pool joint oracle 0.542@611 vs always-best K2B8 0.463@1110 (+0.079 headroom at −45% cost), but the trained RF router collapses to the cheapest arm for 99.3% of queries → routed F1 0.278, Δ=−0.185 vs always-best, CI [−0.233,−0.138]. Strongest confirmation of the predictability-ceiling thesis on the joint axis. |

**ALL GPU runs P1–P6 complete (P7 latency sweep and the WebQSP cap arm remain optional).**

## Data issues — found and RESOLVED before submitting (author note)

1. **Table 2's 7B annotated-hop row (0.444 @ 999).** The repo's 7B annotated-depth run gives
   **0.392 @ 999** (`results_mixed_pareto/.../Oracle-depth`), while the printed 0.444@999(1001)
   matches the **14B** row exactly (`results_14b_mixed_pareto`: 0.4439 @ 1000.7). Transcription
   slip confirmed. **RESOLVED (P2):** re-derived the family on the captioned strong engine —
   corrected 7B row = **0.389 @ 856** (`results_p2_strong_7b`); mixture-dominance holds under
   every value.
2. **FP-vs-MP run deltas.** Overlapping configurations between `results_full_pareto` (source of
   Tables 1/2) and `results_mixed_pareto` (source of the complete grid) differed by up to 0.013 F1
   on the identical 240 queries — different code states between runs. **RESOLVED (P1):** the
   complete grid was re-executed end-to-end in a single unified run (`results_unified_pareto`);
   all revised tables will be rebuilt from this one run.
3. **Engine caption.** `scripts/full_pareto.py` did not set `strong_prefilter=True` (the strong
   engine is `scripts/strong_engine.py` / `cost_latency.py`), yet Table 1's caption says "under
   the strong engine." **RESOLVED (P2):** `full_pareto.py` now has a `--strong` flag wiring
   `strong_prefilter` into the retriever, and the strong-engine re-derivation of the whole
   Table-2 family is complete; captions and numbers now correspond.
