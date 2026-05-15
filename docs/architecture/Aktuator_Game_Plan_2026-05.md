# Aktuator Game Plan — May 2026

**Date:** 2026-05-13
**Horizon:** 16 weeks (through early September 2026)
**Status:** Strategic source-of-truth. Supersedes prior MVV strategy framing for the validation track.
**Companion docs:** `CORAL_MVV_Fundraise_Strategy.md` (the raise narrative remains valid; this doc updates what's being raised against). `CORAL_v3_ARC_Codebook_Design.md` becomes archival upon ARC Phase 0 close. `CORAL_DAG_Codebook_Design.md` (to be drafted) is this doc's technical companion.

**Out of scope for this doc:** Series A pitch deck (downstream artifact, draws from this), hiring plan specifics, technical implementation details (covered in codebook + adapter docs), product roadmap beyond the 16-week validation horizon, legal/IP strategy.

---

## §0 — Strategic premise

Two sets of incumbents define the games Aktuator could play, and both games are losing positions.

On the model side, HRM, TRM, and GRAM (and to a lesser extent Augmented HRM, Kona/Aleph) have established Sudoku and ARC as the benchmarks for reasoning architectures. Each has spent years optimizing for their chosen ground. Showing up on that field as the new entrant means winning benchmarks that don't tell a commercial story — research artifacts with no deployment path. The ARC window in particular is closing fast as LLMs (o3-class and successors) close the abstraction gap that ARC was designed to expose.

On the application side, Aera, Celonis, and SKAN.ai have established "decision intelligence" and "process intelligence" as the categories enterprise buyers know how to budget for. Each owns their category. Showing up as a competitor inside their categories means losing to their distribution and brand. Both layers — model and application — are categories someone else already won.

The win is to define new ground. **Aktuator's new ground is full-stack reasoning automation: a category that doesn't exist yet, built on a model architecture (CRA) that doesn't exist as a peer category yet, sold against a buyer pain (LLM-failure-modes in deterministic enterprise workflows) that doesn't have a named winner yet.** This document specifies how the next 16 weeks of work produce the evidence that this category, this architecture, and this product fit together — and that Aktuator is the only company that does the full stack.

---

## §1 — Positioning

### 1.1 The two-name structure

Aktuator owns two names, both of which deserve dedicated naming work but should be working-titled now so subsequent communication compounds onto consistent terms.

**Architecture: Cortical Reasoning Architecture (CRA).** Established in prior strategic work. CORAL is the first instance of a CRA. Other future architectures may be CRAs (active inference variants, different crystallization mechanisms, different modalities) — the category is not CORAL-specific.

**Product category: Cognitive Workflow Automation (CWA) — working title.** This is the buyer-facing category. Distinguishes from:
- *RPA* (mechanical, brittle, no reasoning)
- *IPA / Intelligent Process Automation* (already a saturated analyst term meaning RPA-plus-some-ML, doesn't claim reasoning)
- *LLM agents* (probabilistic, unreliable, expensive)
- *Decision Intelligence* (Aera's category, oriented around decision recommendations, not workflow execution)
- *Process Intelligence* (Celonis's category, oriented around process mining, not reasoning)

CWA's anchor concepts: *cognitive* (genuine reasoning, not pattern matching), *workflow* (where buyer budget lives), *automation* (operational outcome, not advisory). The category claim is that LLM-based reasoning automation has failed on faithfulness and cost; CWA is the deterministic, compounding alternative.

**Naming session deadline: Week 2.** This document uses CWA as the working term throughout; a final naming decision is one of the open decisions in §8.

### 1.2 Headline positioning sentence

> Aktuator is the first Cognitive Workflow Automation platform, built on Cortical Reasoning Architecture — the only end-to-end reasoning system that executes enterprise workflows faithfully, deterministically, and at a fraction of LLM cost, with deployment-time compounding.

This sentence has to do four things and must be revised if it stops doing any of them:
1. Name the category (CWA)
2. Name the architecture (CRA)
3. Anchor the value claim (faithfulness, determinism, cost)
4. Differentiate (deployment-time compounding — the Recognition Cortex thesis)

### 1.3 Competitive map

Where each adjacent category sits and what each one doesn't do:

| Category | Example players | What they do | What they don't do |
|---|---|---|---|
| LLM-only reasoning | GPT/Claude direct API use | Generate reasoning text | Faithful to structured knowledge; deterministic; cheap at scale |
| LLM + orchestration | LangChain, LlamaIndex, AutoGen | Chain LLM calls and tools | Make the LLM in the chain reliable; the unreliability compounds across hops |
| RPA / IPA | UiPath, Automation Anywhere, Blue Prism | Execute pre-scripted workflows | Reason; adapt to workflow variants; handle non-deterministic inputs |
| Rule engines | Drools, IBM ODM | Execute hand-authored logic | Learn; generalize; adapt |
| Graph foundation models | GFM-RAG, NodePiece | Encode graphs; support retrieval | Reason iteratively; execute multi-step workflows |
| Neuro-symbolic | Kona/Aleph | LLM + symbolic engine | Eliminate LLM cost/dependency; compound at deployment |
| Decision Intelligence | Aera | Recommend decisions from data | Execute deterministic multi-step workflows |
| Process Intelligence | Celonis, SKAN.ai | Mine processes from event logs | Execute reasoning during workflow runtime |
| Reasoning architectures (research) | HRM, TRM, GRAM | Reason on academic benchmarks | Deploy commercially; serve enterprise workflows |

**The gap:** end-to-end ownership of the reasoning stack — workflow encoding (DAG), faithful reasoning execution (CORAL), deployment-time compounding (Recognition Cortex), interpretable execution traces. Each competitor owns part of this stack; none own all four. Aktuator's defensibility claim is that the stack only works as a whole — partial implementations leak failure modes at the seams (LLM hallucinating in LangChain, RPA breaking on variant workflows, rule engines collapsing under maintenance load).

### 1.4 Defensibility

The moat has three layers:

1. **Architecture moat (CRA).** The cortical-faithful reasoning substrate. Mechanism-level differentiation: predictive coding, crystallization, amortization, precision gating, hierarchical abstraction composed into a working reasoning core. Patentable. Hard to replicate without re-running the architecture-design process from scratch.
2. **Deployment moat (Recognition Cortex).** Compounding at inference: the system gets cheaper and faster on each customer's recurring workflows. Unique to CRA. Not present in any competitor at any layer.
3. **Full-stack moat.** Owning the encoding-to-execution path means Aktuator captures the value at every layer rather than being a component vendor to someone else's stack. Customer lock-in via workflow library + crystallized state.

Any one of these is defensible. The combination is category-defining.

---

## §2 — Validation track

Three layers of evidence, in order. Each layer answers a specific question that the next layer doesn't.

### 2.1 Layer A — Synthetic DAG execution benchmark

**Aktuator builds this.** Controllable parameters: graph topology (chains, trees, lattices, sparse vs dense, varying node counts 10–1000), branch logic (conditional routing, aggregation, constraint checking, lookup, comparison), execution semantics, distribution of recurring workflow patterns.

**Validates the mechanisms underlying every commercial claim:**
- Faithful multi-step state tracking (no drift, no hallucinated intermediate states)
- Halt-at-completion (Q-halt fires at correct DAG terminus)
- Generalization across topologies (train on one topology distribution, test on another)
- Crystallization on recurring sub-DAGs (the deployment-compounding claim)
- Compositional generalization (novel compositions of known primitives)

**Pre-registered claims for Layer A:**
- *A1 — Faithfulness:* execution-trace edit distance vs. ground truth ≤ ε on test DAGs (ε to be set during synthetic benchmark design)
- *A2 — Halt precision:* halt-step matches ground-truth DAG terminus on ≥ 95% of test DAGs
- *A3 — Compositional generalization:* test on novel compositions of training primitives ≥ 80% accuracy
- *A4 — Crystallization (existential):* workflows seen ≥ N times during training halt ≥ K steps earlier on repeat encounters than first-encounter baseline. If A4 fails the Recognition Cortex thesis breaks.

**Secondary value:** the synthetic generator doubles as a sales tool. "We can simulate your DAG class and show CORAL works on it" is a powerful pre-engagement motion for design partner conversations.

**Timeline:** Weeks 3–6 build and first training runs. Weeks 6–8 results.

### 2.2 Layer B — One public benchmark for academic anchor

**Recommended: CLRS-30, single algorithm.** Pure deterministic multi-step graph execution. Test split is OOD by design (larger graphs than train) — exactly the LLM weakness. One result is sufficient; we are not chasing the leaderboard. The point is a peer-reviewable, citable number, not a SOTA crown.

**Why CLRS over MetaQA/WebQSP:** MetaQA introduces language and entity-recognition confounds that distract from the core reasoning claim. CLRS is pure graph reasoning — the result reads as "CORAL solves algorithmic reasoning that LLMs lose precision on at OOD scale" with no ambiguity.

**Pre-registered claims for Layer B:**
- *B1:* OOD generalization (test on graphs ≥ 2× training size) ≥ baseline graph transformer (GPS / Graphormer published numbers)
- *B2:* Comparable performance to specialized algorithmic-reasoning architectures (CLRS leaderboard mid-tier or above)

**Timeline:** Weeks 6–8 in parallel with synthetic results writeup.

### 2.3 Layer C — Design partner workflow

**The commercial proof.** One named customer's real DAG, with measurable improvement over their current LLM-based or rule-based system. The validation that closes Series A.

**Pre-registered claims for Layer C:**
- *C1 — Accuracy:* CORAL's reasoning output matches expert/ground-truth on ≥ X% of workflow instances (X set per vertical; typically ≥ 95% for high-stakes verticals, ≥ 90% for medium-stakes)
- *C2 — Cost:* compute cost per workflow execution ≤ 1/10× incumbent LLM-based solution
- *C3 — Latency:* p99 latency at production volume ≤ Y ms (Y set per vertical)
- *C4 — Interpretability:* every prediction comes with an inspectable execution trace; failure modes are localized to specific DAG nodes

**Timeline:** Weeks 8–12 scoping and integration. Weeks 12–16 results.

### 2.4 Dependencies and parallelism

- Layer A is prerequisite for Layer B (B uses the same training infrastructure)
- Layers A + B reinforce Layer C but don't gate it externally — design partner outreach starts in Week 1
- Layer C scoping (data access, workflow understanding) can happen in parallel with Layer A build
- Layer C *delivery* requires A complete (we don't ship to a customer without synthetic validation)

---

## §3 — Commercial wedge: vertical selection

### 3.1 Selection criteria

The first vertical must satisfy all five:

1. **High-volume deterministic reasoning workflows.** The workflow recurs at scale; the compounding-at-deployment claim is measurable.
2. **Currently LLM-deployed with visible failure modes.** Buyer has seen LLM-based reasoning fail in production; pain is articulable.
3. **Regulatory or cost pressure on accuracy.** Determinism is a feature, not a curiosity. Failure has measurable downstream cost.
4. **Accessible to warmest network.** First customer conversations happen via warm intro, not cold outreach.
5. **Willing to engage as design partner.** Data access, willingness to compare outputs against existing system, willingness to be named (even if quietly) in fundraise materials.

### 3.2 Universe sketch

Verticals that potentially satisfy the criteria, in no specific order:
- Insurance claims processing (high volume, regulatory, LLM-deployed)
- Financial reconciliation (deterministic by design, regulatory, accuracy critical)
- Regulatory compliance — KYC/AML (high regulatory pressure, multi-step decision graphs)
- Healthcare coding — ICD-10 traversal (deterministic, high volume, LLM accuracy poor)
- Tax/audit workflow (deterministic, regulatory, high stakes)
- Legal document workflow — contract review, due diligence (volume + accuracy)
- Supply chain decision routing (high volume, cost-sensitive)

### 3.3 Decision deadline

**Vertical decision: Week 2.** The synthetic DAG benchmark design depends on the vertical — synthetic DAGs should mimic the target vertical's actual structural distribution (typical fan-out, branching depth, primitive mix, recurrence patterns). A generic synthetic is wasted effort.

### 3.4 Alignment principle

The synthetic benchmark is designed for the vertical, not adapted to it later. Concretely:
- Sample 10–20 real workflow DAGs from the target vertical (anonymized, sketched, or analogous public examples)
- Extract structural statistics: node count distribution, fan-out, branching depth, primitive mix
- Generate synthetic DAGs matching these statistics
- This ensures Layer A → Layer C transfer holds

---

## §4 — Fundraise narrative (MVV update)

### 4.1 Narrative shift

**Old narrative** (pre-pivot): "We've built a brain-inspired reasoning core. Help us validate the architecture on ARC and beyond, with eventual edge deployment."

**New narrative:** "We've built the first Cortical Reasoning Architecture — substrate generality is established (Sudoku closed, ARC supporting). We're funding commercial validation: synthetic DAG execution benchmark + one public algorithmic-reasoning benchmark + one named design partner in [vertical], targeting [specific operational metric]. The product category we're defining — Cognitive Workflow Automation — has Aera and Celonis as the closest analogs, both built into multi-billion-dollar businesses by owning newly-named categories."

### 4.2 What changes for the investor

- Substrate generality stops being the *goal* and becomes the *foundation*. The goal becomes commercial validation.
- The raise underwrites a *named milestone with a named partner*, not abstract architectural research.
- The comparables shift from research papers (HRM/TRM) to category-defining companies (Aera, Celonis) — much friendlier comp for a Series A investor evaluating commercial trajectory.
- The "what's the product" question — historically the weakest part of the CORAL pitch — now has a clean answer: Cognitive Workflow Automation, deployed via design partner integration.

### 4.3 Target raise

**$8–12M anchor round, unchanged.** Use of proceeds shifts from "validation + edge compression research" to "commercial validation + first design partner + productization." Hiring plan downstream of this doc.

### 4.4 Pitch evidence at 16-week mark

What the deck shows at the end of Week 16:
- Synthetic DAG benchmark results with pre-registered probes met (A1–A4)
- CLRS-30 single-algorithm result with OOD generalization claim
- Design partner workflow results with C1–C4 metrics
- Architecture papers / patents in flight
- Category articulation: CWA defined, competitive map, defensibility
- Recognition Cortex compounding demonstrated (the unique-in-market mechanism)

---

## §5 — Sequencing: 16-week plan

### 5.1 Phase-by-phase

| Weeks | Primary work | Parallel work | Gating decisions |
|---|---|---|---|
| 1–2 | ARC Phase 0 close: eval-wedge fix, completion, probes filed | Vertical decision; design-partner outreach starts | Category naming; vertical pick |
| 3–6 | Synthetic DAG benchmark built; first training runs | Design partner scoping conversations | Synthetic scope; primitive vocabulary |
| 6–8 | Synthetic results; CLRS-30 single-algorithm run | Design partner data access negotiated | Layer A claims pass/fail; Layer B target |
| 8–12 | Design partner workflow scoped, data ingested, first integration runs | Architecture paper draft begins | Layer C metric targets confirmed |
| 12–16 | Design partner results; fundraise materials assembled; deck rebuilt | Patent filings; advisor outreach (Hawkins/Friston) | Fundraise launch decision |

### 5.2 Critical path

The critical path runs through synthetic benchmark → design partner workflow. Layer B (CLRS) is parallel-credibility, not on the critical path; if Layer B slips, fundraise can proceed without it. If Layer A slips, everything slips. If Layer C slips, fundraise narrative weakens but doesn't break.

### 5.3 Discipline notes for the 16-week window

- ARC Phase 0 work consumes minimal GPU/time after Week 2; the eval-wedge diagnostic is small remaining work.
- Synthetic benchmark build is the GPU-bound work for Weeks 3–6.
- Design partner integration in Weeks 8–12 may require dedicated compute; plan Vast.ai budget accordingly.
- Anwar is the only person on technical work; CC handles implementation via prompt files. The 16-week plan assumes that workflow continues unchanged.

---

## §6 — Discipline rules

These rules exist to prevent the failure modes that have hurt CORAL before (overcorrection mid-flight, sunk-cost continuation, scope creep).

1. **ARC closes at Phase 0 verdict, regardless of outcome.** No reopening. If Phase 0 produces tantalizing results, the impulse to launch Phase 1 is the impulse to play the wrong game. Resist.
2. **One validation track at a time post-Phase-0.** Layers A, B, C run in their defined sequence/parallelism. No interleaving of academic and commercial threads beyond what §5 specifies.
3. **Vertical decision uses explicit criteria (§3.1), not what's easiest.** Default-by-convenience is the failure mode to avoid.
4. **Vertical precedes synthetic design.** Synthetic mimics target vertical's structural distribution. No generic synthetic followed by retrofit.
5. **Probes before training.** Workflow-crystallization probe (the existential safeguard from §7) must be implemented and tested before Layer A training launches.
6. **Pre-register kill criteria.** Each layer's claims have pass/fail thresholds set in the design doc, not adjusted after results come in.
7. **No fragmentation between academic and commercial tracks.** Layer B is one benchmark, one algorithm, one result. Not a side-project that swallows attention.
8. **Test fixtures match real data shapes.** Lesson from Phase 0: synthetic exact-fill counts can hide bugs that real variable-length data exposes. Layer A synthetic must reflect real-vertical-DAG variability from day one.

---

## §7 — Risks

### 7.1 Existential: workflow crystallization fails

**The risk.** Layer A trains PC-only DAG-execution CORAL successfully (faithful execution, halt precision, compositional generalization) but the carry state doesn't acquire workflow-type structure. Codebook can't compound on what isn't there. Same failure shape as Sudoku R0.5.

**Why this is existential.** The Recognition Cortex deployment-compounding claim is the unique-in-market differentiator. If it fails on synthetic — where supervision is clean and recurrence is controllable — it will fail on customer workflows. The full-stack moat collapses to "another reasoning architecture," competing with HRM/TRM on benchmark numbers we've already decided not to play for.

**Mitigation:**
- Workflow-type clustering probe runs against PC-only Layer A checkpoint before codebook engages
- AUC threshold ≥ 0.6 required for codebook engagement
- If AUC < 0.6, apply supervised auxiliary loss (synthetic labels are clean) to inject workflow-type structure before adding codebook
- If even supervised injection fails, fundamental architectural review — Layer A pauses, Layer B pauses, Layer C pauses
- The synthetic benchmark's deployment-phase split is designed to test compounding online from the start (Claim A4)

### 7.2 Wrong vertical (recoverable but slow)

**The risk.** Vertical selected satisfies criteria on paper but in practice (a) data access proves harder than expected, (b) the customer's actual DAG isn't representative, or (c) the LLM-failure-mode story doesn't resonate with the actual buyer persona in that vertical.

**Mitigation:**
- Maintain a shortlist of 2–3 backup verticals from §3.2
- First 3 weeks of design-partner outreach validate the vertical thesis before deep technical commitment
- If signals are weak by Week 6 (no warm-intro response, no buyer interest), pivot vertical before Week 8 integration starts

### 7.3 Synthetic doesn't transfer to real DAGs

**The risk.** Layer A passes all probes; Layer C real-customer integration reveals that real workflows have structural properties the synthetic didn't capture, and CORAL fails on them.

**Mitigation:**
- Vertical-driven synthetic design (§3.4) reduces this risk substantially
- Layer C scoping (Weeks 8–10) includes structural audit of real workflows vs. synthetic distribution
- If structural gap is found, augment synthetic and continue Layer A training in parallel with Layer C

### 7.4 LLM time-pressure

**The risk.** LLMs improve on enterprise reasoning faster than expected (o3-class models matched by open-weight equivalents; better tool use; cheaper inference). The "LLM-alternative" pitch loses its time-window.

**Mitigation:**
- The cost differential (≥ 10× per C2) is durable even if LLM accuracy closes — operational cost at scale matters
- The interpretability differential (C4) is structural — LLMs cannot produce inspectable execution traces by nature
- The deployment-compounding differential is structural — LLMs do not get cheaper on recurring workflows
- Pitch hedge: even if LLMs match accuracy, the cost + interpretability + compounding gap remains

### 7.5 Operational risks

- **Compute budget overrun** if synthetic-benchmark training is more expensive than projected. Vast.ai budget tracked weekly.
- **Solo founder bandwidth.** No mitigation other than discipline; the 16-week plan is aggressive but feasible at current cadence.
- **CC implementation bottleneck** if prompt-file workflow doesn't scale to graph-substrate complexity. Watch for this in Weeks 3–4.

---

## §8 — Open decisions (must resolve in Week 1–2)

| # | Decision | Owner | Deadline | Notes |
|---|---|---|---|---|
| OD1 | Vertical selection | Anwar | Week 2 | Use §3.1 criteria; warm network priority |
| OD2 | Category naming (CWA or alternative) | Anwar | Week 2 | Consider dedicated naming consultant |
| OD3 | Design partner shortlist (3 named candidates) | Anwar | Week 2 | Conditional on OD1 |
| OD4 | Public benchmark confirmation (CLRS-30 vs. alternative) | Anwar + Claude | Week 3 | CLRS-30 recommended; revisit if vertical strongly suggests KGQA |
| OD5 | Synthetic benchmark scope (node-count range, primitive vocabulary, workflow library size) | Anwar + Claude | Week 3 | Conditional on OD1; flows into codebook design doc |
| OD6 | ConceptARC labeling pass for archived ARC work | Anwar | Week 2 | Skip — ARC is closed; labels not needed |

OD6 closed by recommendation: skip the manual labeling pass; ARC is closed at Phase 0 verdict and the doc was archival before this game plan. No reason to invest further.

---

## §9 — Success criteria for the 16-week plan

The plan succeeds if, by end of Week 16, all of the following hold:

1. **Synthetic DAG benchmark** with all pre-registered Layer A claims met (A1 faithfulness, A2 halt precision, A3 compositional generalization, A4 crystallization compounding).
2. **One public benchmark result** in hand (CLRS-30 single algorithm with OOD generalization claim B1, or equivalent).
3. **One design partner** with measurable workflow improvement against incumbent solution, all Layer C claims met (C1–C4).
4. **Fundraise deck rebuilt** around the new narrative (§4), with §1 positioning, §2 evidence, §3 vertical, §4 milestones explicit.
5. **Category name decided** (CWA or alternative) and externally used in at least one piece of public-facing communication (blog post, conference talk, technical paper).

Partial success (e.g., 3 of 5) does not invalidate the plan — it triggers a re-scoping conversation at Week 16, not a continuation under the same assumptions.

Failure of (1) — synthetic benchmark fails Layer A claims — triggers existential review per §7.1.

---

## §10 — Document maintenance

This is the source-of-truth doc for the 16-week window. Updates:

- Every two weeks: progress notes appended to a dated section at the end (Week 2 update, Week 4 update, etc.)
- Decisions in §8 marked closed as they resolve, with the resolution documented
- Risks in §7 updated if new risks emerge or existing risks materialize
- Major scope changes (e.g., vertical pivot under §7.2) trigger a new document version rather than in-place edits

The doc is not the deck. The deck draws from this doc and is rebuilt at Week 14–16.

---

*End of game plan. Status: active for execution.*
