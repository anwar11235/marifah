# CORAL Build Plan — May 2026

**Date:** 2026-05-13
**Horizon:** 16 weeks (through early September 2026)
**Status:** Active execution document. Living artifact — updates as components ship and decisions resolve.
**Companion docs:** `Aktuator_Game_Plan_2026-05.md` (strategic), `CORAL_DAG_Codebook_Design.md` (codebook architecture), `CORAL_Synthetic_DAG_Benchmark_Spec.md` (synthetic benchmark spec).

---

## §0 — Why this document exists

The game plan (2026-05-13) specifies the 16-week strategic horizon at the level of phases and gates. This document operationalizes that at week-by-week granularity, with component-level build sheets, dependencies made explicit, gates pre-registered, and parallel tracks coordinated.

This is the document Anwar reads on Monday morning to know what ships this week. It is updated every two weeks at minimum (more often during high-integration phases), with status notes appended rather than the doc rewritten.

The build plan is operational, not strategic. Strategy lives in the game plan; architecture lives in the codebook + synthetic benchmark docs. This doc answers: given those decisions, what gets done, when, by whom, and what gates the next step.

---

## §1 — Scope and non-scope

**In scope:**
- Week-by-week sequencing of technical and commercial work
- Component build sheets (description, dependencies, effort, deliverable, verification)
- Parallel-track coordination (technical vs. commercial)
- Compute budget tracking
- Dependencies and gates explicitly documented
- Operational risk register (distinct from game plan strategic risks)

**Not in scope:**
- Strategic decisions (game plan)
- Technical architecture specifications (codebook + synthetic docs)
- Product positioning (game plan §1)
- Long-horizon hiring or org-design decisions (post-Series A territory)

---

## §2 — Phase structure

Three phases over 16 weeks. Phases overlap in calendar weeks (commercial work in Phase 2 starts in Week 8, while Phase 1 technical work is still landing) — the phase distinction is about work *content*, not strict calendar boundaries.

### Phase 0 — Foundation (Weeks 1–6)

**Goal:** Close ARC. Build all infrastructure for DAG validation. Reach Phase 0 decision gate with PC-only DAG training results.

**Exit criterion:** Workflow-type clustering AUC ≥ 0.6 on Phase 0 PC-only checkpoint (codebook design §7 gate), OR Path B (supervised injection) triggered, OR Path C (existential review) triggered.

### Phase 1 — Mechanism validation (Weeks 6–10)

**Goal:** HMSC engages. Four pre-registered Layer A claims (A1 faithfulness, A2 halt precision, A3 compositional generalization, A4 training-time compounding) measured. CLRS-30 single-algorithm result as academic anchor.

**Exit criterion:** A1, A3, A4 all passing pre-registered thresholds. (A2 desirable, B1 from CLRS desirable but not gating.)

### Phase 2 — Commercial validation (Weeks 8–16)

**Goal:** Design partner workflow integrated. Layer C results in hand. Fundraise materials rebuilt around new narrative.

**Exit criterion:** At least one design partner with measurable workflow improvement against incumbent; deck rebuilt; ready for fundraise launch.

---

## §3 — Week-by-week breakdown

### Week 1 (now — w/e May 17)

**Primary technical work:**
- ARC Phase 0 eval-wedge fix: `DISABLE_COMPILE=1` A/B run at B=384 with eval_interval=5 epochs and 20 total epochs. Pre-registered outcomes per the diagnostic plan.
- If wedge fixed → let Phase 0 ARC run to natural completion in background; collect probes for archival.
- If wedge persists → py-spy dump diagnostic; the issue is in eval code path, not compile.

**Parallel commercial work:**
- Vertical criteria conversation (game plan §3.1) — internalize criteria, start narrowing the network.
- Begin scoping warmest-network candidates in 3–4 of the §3.2 universe verticals.

**Decisions due:** None hard-required this week; vertical thinking begins.

**Gates / checkpoints:** ARC eval-wedge diagnostic verdict.

**Deliverables:**
- ARC Phase 0 either running to completion or in py-spy-diagnostic mode
- Initial vertical shortlist (3–4 candidates, internal)

**Compute use:** ~24–48 A100 SXM4 80GB hours (one disable_compile diagnostic run + completion of ARC Phase 0 if unwedged).

---

### Week 2 (w/e May 24)

**Primary technical work:**
- ARC Phase 0 closes (per game plan §6 discipline rule — hard cutoff at verdict regardless of outcome). Probes filed; results documented as archival.
- Synthetic benchmark spec → CC prompt drafting begins. Implementation can start without vertical decision (per spec §14.2).
- Graph adapter design doc drafted (sibling to ARC adapter design — DAG-specific encoding, attention mask, positional encoding).

**Parallel commercial work:**
- **Vertical decision finalized.** Use game plan §3.1 criteria. Document the choice with rationale.
- Category naming session (CWA or alternative) — game plan §1.1, open decision OD2.
- First design-partner outreach calls (warm intros to 2–3 named candidates within the vertical).

**Decisions due:**
- Vertical pick (game plan OD1 + synthetic spec OD5-S + OD6-S + codebook OD8)
- Category name (game plan OD2)
- Design partner shortlist (3 named candidates, game plan OD3)
- ConceptARC labeling pass closed (game plan OD6, recommended: skip)

**Gates / checkpoints:** ARC formally closed. Vertical pick locks downstream synthetic config and cyclic handling decisions.

**Deliverables:**
- ARC Phase 0 results filed (`docs/ARC_Phase0_Results_2026-05.md` or equivalent)
- CC prompt for synthetic generator (default config; vertical augmentations added in Week 3)
- Graph adapter design doc
- Vertical decision document with rationale
- Category name decided

**Compute use:** ~12–24 A100 hours (ARC tail + small experiment cycles).

---

### Week 3 (w/e May 31)

**Primary technical work:**
- Synthetic generator implementation begins (CC executes against the spec). ~5-day build per spec §12.1.
- Graph adapter implementation begins in parallel (different codebase area, no conflict).
- Vertical-specific config tuned: sample 10–20 real workflows from selected vertical, extract structural statistics, fit generator config (spec §9.2).

**Parallel commercial work:**
- Design-partner outreach continues; aim for 2–3 first-call meetings this week.
- Begin scoping technical conversations with most promising design-partner candidate (workflow profile, data access posture).

**Decisions due:**
- Vertical-specific primitive augmentations (synthetic OD5-S)
- Cyclic-or-acyclic for chosen vertical (synthetic OD6-S → codebook OD8)
- Novel-composition holdout fraction (synthetic OD4-S, default 15% unless real-workflow analysis suggests otherwise)
- Public benchmark confirmation (game plan OD4 — CLRS-30 unless vertical strongly suggests KGQA)

**Gates / checkpoints:** Generator config validated against real-workflow statistics before full dataset generation.

**Deliverables:**
- Synthetic generator: primitives + executor (foundation components) complete
- Graph adapter: encoding format spec'd, attention-mask logic implemented
- Vertical config file (`configs/vertical_{name}.yaml`)
- Real-workflow structural statistics document

**Compute use:** ~12 A100 hours (warm-start comparison run on tiny dataset begins late week).

---

### Week 4 (w/e Jun 7)

**Primary technical work:**
- Synthetic generator completion: pattern templates, workflow assembler, generator pipeline, validation logic, splits. End-of-week: tiny dataset (~1K DAGs) generated.
- Graph adapter: tokenization tested against tiny dataset. Smoke test through training loop.
- HMSC codebook implementation begins. Three codebook classes (G, R, P), routing, composition, auxiliary loss heads.
- **Warm-start comparison run** (codebook OD7): Sudoku Phase 3c checkpoint vs. from-scratch on tiny synthetic DAG benchmark. 5K steps each, measure workflow-type AUC and faithfulness.

**Parallel commercial work:**
- Design-partner conversations deepen; at least one candidate moving toward NDA/data-access conversation.
- Begin drafting design-partner one-pager (vertical-specific, references CWA category).

**Decisions due:**
- Warm-start commit (OD7): use Sudoku-warm if matches/exceeds from-scratch on metrics; else use from-scratch.
- HMSC composition method (codebook OD2): default sum unless probe data suggests gating early.
- Tap frequency (codebook OD5): default every ACT step unless compute-bound.

**Gates / checkpoints:** Tiny dataset usable through training loop end-to-end. Warm-start verdict.

**Deliverables:**
- Synthetic generator: complete, tiny dataset emitted
- Graph adapter: working end-to-end on tiny dataset
- HMSC codebook: ~50% implemented (G + R; P-codebook with cross-attention slated for Week 5)
- Warm-start verdict documented
- Design-partner one-pager v1

**Compute use:** ~36 A100 hours (warm-start comparison + integration testing).

---

### Week 5 (w/e Jun 14)

**Primary technical work:**
- HMSC codebook completion: P-codebook (cross-attention), composition module, auxiliary loss heads.
- Probe scripts: A1 (faithfulness) and workflow-type AUC probe implemented and tested on tiny dataset.
- **Full synthetic dataset generation** for the chosen vertical (~1M DAGs, ~1 hour generation per spec §5.3).
- Phase 0 PC-only DAG training launches mid-to-late week. Initial config: HMSC disabled, auxiliary losses zero, standard CORAL training otherwise.

**Parallel commercial work:**
- Design-partner data access negotiations.
- Begin CLRS-30 benchmark setup (lower priority — runs in spare GPU time).

**Decisions due:**
- P-codebook discreteness (codebook OD3): default soft training, hard top-1 eval.
- Supervised vs. unsupervised default (codebook OD4): default supervised; parallel unsupervised ablation scheduled for Week 6 if compute permits.

**Gates / checkpoints:**
- HMSC code reviewed and signed off before integration with training.
- Phase 0 DAG training reaches early checkpoint (~5K steps) by end of week — apply codebook design §8.2 early-checkpoint gate.

**Deliverables:**
- HMSC codebook: complete
- Probe scripts: A1 + workflow-type AUC ready
- Full synthetic dataset (~1M DAGs) generated, archived, manifest documented
- Phase 0 DAG training underway with early-checkpoint probe data

**Compute use:** ~96 A100 hours (Phase 0 DAG training is GPU-intensive).

---

### Week 6 (w/e Jun 21)

**Primary technical work:**
- Phase 0 DAG training reaches midpoint checkpoint (~20K steps) and final checkpoint (~50K+ steps).
- Apply codebook design §7 decision gate: workflow-type AUC ≥ 0.6 → Path A, AUC 0.4–0.6 → Path B (supervised injection), AUC < 0.4 → Path C (existential review).
- Probe scripts for A3 (compositional), A4 (compounding), A2 (halt precision) implemented during training tail.
- **Phase 0 verdict.** Phase 1 launch preparation depending on path.

**Parallel commercial work:**
- Design-partner: aim for data access agreement signed.
- Layer C metric targets confirmed with design-partner candidate.

**Decisions due:**
- Phase 0 → Phase 1 path (codebook §7 Path A / B / C).
- Phase 1 hyperparameter strategy (lambda values for auxiliary losses).

**Gates / checkpoints:**
- **Critical gate:** Phase 0 decision per codebook §7.
- If Path C triggered: pause game plan §5 sequencing, schedule architectural review.

**Deliverables:**
- Phase 0 results document (all probe outputs, decision gate verdict, path commitment)
- Phase 1 training config (HMSC engaged, auxiliary losses active, schedule per §7 path)
- Design-partner data access agreement (commercial milestone)

**Compute use:** ~96 A100 hours (Phase 0 completion + Phase 1 launch).

---

### Week 7 (w/e Jun 28)

**Primary technical work:**
- Phase 1 training: HMSC engaged, auxiliary supervision active per chosen path.
- Mid-training probe checkpoints every 10K steps.
- CLRS-30 setup: adapter modifications for CLRS data format, single-algorithm pipeline (default: shortest-path or BFS).

**Parallel commercial work:**
- Design-partner: real workflow data starts flowing; first structural-statistics extraction on real customer data (compare to synthetic distribution).
- Architecture paper draft begins (CRA category, codebook mechanism, results-to-date).

**Decisions due:**
- CLRS-30 algorithm selection (game plan OD4 secondary): shortest-path recommended for cleanest result.
- Real-workflow vs. synthetic distribution match: any structural gaps surface this week; corrective synthetic regeneration triggered if needed.

**Gates / checkpoints:** Phase 1 training health checks (loss trajectory, codebook utilization, auxiliary loss balance).

**Deliverables:**
- Phase 1 training in progress with health metrics tracked
- CLRS-30 setup complete; first runs underway
- Real-workflow analysis vs. synthetic distribution

**Compute use:** ~120 A100 hours (Phase 1 training + CLRS-30 in parallel).

---

### Week 8 (w/e Jul 5)

**Primary technical work:**
- Phase 1 training completes. Full probe suite runs against final checkpoint.
- **Claim verdicts:** A1 (faithfulness), A2 (halt precision), A3 (compositional generalization), A4 (training-time compounding). Pre-registered thresholds applied.
- CLRS-30 single-algorithm result.
- Design-partner Layer C scoping begins: data ingestion pipeline for customer DAGs, initial integration runs.

**Parallel commercial work:**
- Phase 1 results writeup begins (technical document, draft for paper).
- Design-partner: ingest customer DAGs, run preliminary inference, identify gaps.

**Decisions due:**
- Phase 1 → Phase 2 commit: do Layer A claims pass? If yes, proceed to Layer C integration with confidence. If some claims fail, identify which and decide whether to retrain, redesign, or proceed with caveats.
- Customer DAG integration strategy: transfer codebook frozen-from-synthetic (preferred), partial supervised (fallback), or fully unsupervised (lowest-confidence).

**Gates / checkpoints:**
- **Critical gate:** Phase 1 claim verdicts. A1 + A3 + A4 passing = green light for Phase 2. A1 or A4 failing = red light, requires diagnostic.

**Deliverables:**
- Phase 1 results document with all four claim verdicts and CLRS-30 result
- Customer DAG data ingestion pipeline (functional, not production-grade)
- First customer DAG inference runs (results may be poor; baselining)

**Compute use:** ~60 A100 hours (Phase 1 tail + Layer C initial runs).

---

### Week 9 (w/e Jul 12)

**Primary technical work:**
- Layer C integration deepens. Production-grade data pipeline for customer DAGs.
- Codebook adaptation strategy applied (synthetic-trained codebook frozen on customer data; or supervised fine-tuning if labels exist).
- Customer DAG probe suite: same probes A1, A2, plus Layer C claims C1 (accuracy vs. incumbent), C2 (cost), C3 (latency), C4 (interpretability).

**Parallel commercial work:**
- Architecture paper draft progresses.
- Patent filings prepared (CRA architecture + HMSC + Recognition Cortex compounding).
- Advisor outreach begins: Hawkins (Numenta) first per memory note.

**Decisions due:**
- Layer C metric calibration: are pre-registered C1–C4 thresholds appropriate against actual incumbent system?
- Patent scope: provisional vs. full filing pace.

**Gates / checkpoints:** Customer DAG inference reaches at least incumbent-comparable accuracy on initial test (not full claim validation; pre-claim sanity check).

**Deliverables:**
- Production-grade Layer C pipeline
- Customer DAG initial accuracy results
- Advisor outreach (first cold or warm message to Hawkins)

**Compute use:** ~60 A100 hours.

---

### Week 10 (w/e Jul 19)

**Primary technical work:**
- Layer C full validation runs. C1–C4 measured against pre-registered thresholds (game plan §2.3).
- Workflow library analysis: how many distinct workflow types in customer data? How well does the synthetic-trained codebook cover them?
- If coverage gaps: targeted retraining with customer-DAG-augmented training data.

**Parallel commercial work:**
- Architecture paper draft v1 complete.
- Patent provisional filings.
- Advisor outreach: response from Hawkins (or follow-up if no response in 1 week). Friston outreach prepared.

**Decisions due:**
- Layer C verdict: do C1–C4 pass against incumbent? If yes, design partner is ready to be named in fundraise materials. If no, identify gaps and whether they're patchable in remaining time.
- Customer-data-augmented retraining: trigger or skip.

**Gates / checkpoints:** Layer C verdict.

**Deliverables:**
- Layer C results document
- Architecture paper v1 draft
- Provisional patent applications
- First advisor conversation (if Hawkins responsive) or pivot to next candidate

**Compute use:** ~48 A100 hours.

---

### Week 11 (w/e Jul 26)

**Primary technical work:**
- Targeted retraining if Layer C verdict required it.
- Layer C re-validation if retraining occurred.
- Begin productization scope: what does the demo look like? What's the minimal product surface for the deck?

**Parallel commercial work:**
- Deck rebuild begins. Source material: this build plan, game plan, codebook results, Layer C results.
- Advisor relationships develop (assuming Hawkins or Friston engaged).
- Series A investor preliminary conversations (selective, low-key).

**Decisions due:**
- Deck structure: lead with category (CWA), architecture (CRA), or design partner (Layer C result)?
- Series A timing: launch in Week 14, 15, or 16?

**Gates / checkpoints:** Layer C results stable (no further retraining needed).

**Deliverables:**
- Deck outline v1
- Layer C results final
- Advisor commitments (if any)

**Compute use:** ~36 A100 hours.

---

### Week 12 (w/e Aug 2)

**Primary technical work:**
- Productization MVP: minimal demo that an investor can interact with (or watch in a 5-minute video).
- Demo workflows from design partner's vertical, executed by CORAL, traces inspectable.

**Parallel commercial work:**
- Deck rebuild v2.
- Architecture paper revision; submission target if academic angle is valuable.
- Design-partner contract / LOI for fundraise narrative use.

**Decisions due:**
- Demo scope: how many workflows? Static vs. interactive? Customer-named or generic?
- Paper submission venue (ICLR, NeurIPS, AAAI, or skip in favor of arxiv-only).

**Gates / checkpoints:** Demo MVP functional end-to-end.

**Deliverables:**
- Demo MVP
- Deck v2
- Design-partner LOI or similar formal artifact

**Compute use:** ~24 A100 hours (mostly demo refinement, not training).

---

### Weeks 13–14 (w/e Aug 9, Aug 16)

**Primary technical work:**
- Demo polish.
- Documentation hardening (technical brief for due diligence, architecture writeup).

**Parallel commercial work:**
- Deck rebuild v3 → final.
- Investor outreach intensifies. Warm intros activated.
- Advisor talks if scheduled.

**Decisions due:**
- Final deck sign-off.
- Investor target list finalized.

**Gates / checkpoints:** Deck ready for fundraise launch.

**Deliverables:**
- Deck final
- Demo final
- Technical due-diligence package
- Investor target list

**Compute use:** ~12 A100 hours/week.

---

### Weeks 15–16 (w/e Aug 23, Aug 30)

**Primary work:** Fundraise launch. Pitch meetings. Term sheet conversations.

**Compute use:** Minimal new training. Existing infrastructure runs for any live demos.

---

## §4 — Component build sheets

For each major technical component:

### 4.1 ARC eval-wedge fix

- **Description:** Diagnose and fix the eval hang in Phase 0 ARC training. Compile-disable A/B test first; deeper diagnosis if needed.
- **Owner:** Anwar (diagnostic), CC (any code changes).
- **Dependencies:** None (existing branch is live).
- **Effort:** 1–3 days.
- **Deliverable:** Eval completes within 3 min at boundary, OR py-spy diagnostic identifies eval-code-path bug for separate fix.
- **Verification:** Phase 0 ARC training reaches eval boundary and completes without hang.

### 4.2 Synthetic benchmark generator

- **Description:** Implements `CORAL_Synthetic_DAG_Benchmark_Spec.md`. ~2200 LOC.
- **Owner:** Anwar (design sign-off on CC output), CC (implementation).
- **Dependencies:** Spec finalized (done). Vertical config not required for implementation; required for full-dataset generation.
- **Effort:** 4–5 days.
- **Deliverable:** Generator emits valid datasets with default config; passes unit tests; emits tiny + full datasets on demand.
- **Verification:** Unit tests pass. Tiny dataset (1K DAGs) generates in < 1 min. Full dataset (1M DAGs) generates in ~1 hour. Manifest consistent with output.

### 4.3 Graph adapter

- **Description:** Encodes DAGs as CORAL input (node embeddings, edge-attention mask, Laplacian or RWPE positional encoding). Analog of ARC adapter.
- **Owner:** Anwar (design), CC (implementation).
- **Dependencies:** Synthetic generator emits well-formed DAGs.
- **Effort:** 3–4 days.
- **Deliverable:** Adapter converts a parquet-stored DAG to model input batch.
- **Verification:** End-to-end smoke test: tiny dataset → adapter → CORAL forward pass produces output without error.

### 4.4 HMSC codebook module

- **Description:** Hierarchical Multi-Scale Codebook per `CORAL_DAG_Codebook_Design.md`. Three codebook classes (G, R, P), routing, composition, output projection.
- **Owner:** Anwar (design sign-off), CC (implementation).
- **Dependencies:** Codebook design doc (done), graph adapter (for testing in training loop context).
- **Effort:** 4–5 days.
- **Deliverable:** HMSC module: G + R + P codebooks, routing logic, composition module, parameter counts within budget.
- **Verification:** Forward pass through HMSC with tiny dataset produces outputs of correct shape. Codebook utilization measurable. Backward pass produces gradients on codebook entries.

### 4.5 Auxiliary loss heads

- **Description:** Three classifier heads per codebook scale, loss aggregation, lambda scheduling.
- **Owner:** CC.
- **Dependencies:** HMSC complete; labels available from generator.
- **Effort:** 1–2 days.
- **Deliverable:** Per-head losses computed and added to main loss with configurable lambda.
- **Verification:** Per-head loss values reasonable on tiny dataset; lambda scheduling configurable.

### 4.6 Probe scripts

- **Description:** Six probes total. Four primary (A1 faithfulness, A2 halt precision, A3 compositional generalization, A4 training-time compounding) plus two carry-forward (workflow-type AUC, codebook content MI).
- **Owner:** Anwar (probe design sign-off), CC (implementation).
- **Dependencies:** Generator (for labels and splits), codebook (for codebook-content probes).
- **Effort:** 3–4 days.
- **Deliverable:** Probe scripts that read parquet splits, run probe, emit results JSON.
- **Verification:** Probes run on tiny dataset; results consistent with hand-computed values on small examples.

### 4.7 Training pipeline integration

- **Description:** Wire HMSC + adapter + generator + probes into the HRM-derived training loop. Logging, checkpointing, W&B integration.
- **Owner:** Anwar (integration review), CC (implementation).
- **Dependencies:** All upstream components complete.
- **Effort:** 2–3 days.
- **Deliverable:** End-to-end training run launches on tiny dataset, runs without error, logs to W&B.
- **Verification:** 1K-step smoke training run completes with sane loss trajectory.

### 4.8 CLRS-30 setup

- **Description:** Adapter modifications for CLRS-30 input format, single-algorithm pipeline.
- **Owner:** CC.
- **Dependencies:** Graph adapter (modified, not rebuilt).
- **Effort:** 2–3 days.
- **Deliverable:** CLRS-30 single-algorithm training runs end-to-end.
- **Verification:** Loss decreases on training set; OOD test produces measurable accuracy.

### 4.9 Design-partner integration layer

- **Description:** Data ingestion for customer DAGs (anonymization, format conversion, validation). Inference pipeline. Result reporting.
- **Owner:** Anwar (design + customer interface), CC (implementation).
- **Dependencies:** Customer data access; production HMSC checkpoint.
- **Effort:** 5–7 days spread across Weeks 8–10.
- **Deliverable:** Customer DAGs run through CORAL, results comparable to incumbent.
- **Verification:** End-to-end run on customer data with documented comparison metrics.

### 4.10 Demo MVP

- **Description:** Investor-facing demo. Either live interactive (workflow input → CORAL execution → inspectable trace) or 5-min recorded walkthrough.
- **Owner:** Anwar (design), CC (implementation if interactive).
- **Dependencies:** Phase 1 results, Layer C integration.
- **Effort:** 5–7 days across Weeks 11–13.
- **Deliverable:** Demo asset deployable to a meeting.
- **Verification:** Internal walkthrough; tested against ELI5-level investor questions.

### 4.11 Fundraise deck

- **Description:** Series A pitch deck rebuilt around CWA category, CRA architecture, Layer C result. Sources: game plan §1, §4; this doc results.
- **Owner:** Anwar (writing + design); Claude (drafting input).
- **Dependencies:** Phase 1 + Layer C results.
- **Effort:** 2–3 weeks (Weeks 11–14, interleaved with technical work).
- **Deliverable:** Final deck with all claims supported by results.
- **Verification:** Tested against 2–3 advisor / friendly-investor reads.

---

## §5 — Parallel tracks

### 5.1 Technical track

Linear-ish: ARC close → synthetic generator → graph adapter → HMSC → training → Phase 0 verdict → Phase 1 training → probes → Layer C integration → demo.

Some parallelism possible: graph adapter and synthetic generator can be built simultaneously (different code areas). HMSC and probe scripts can be built in parallel. CLRS-30 setup runs alongside Phase 1 main training.

### 5.2 Commercial track

Vertical decision (Week 2) → outreach (Weeks 2–6) → scoping (Weeks 4–7) → data access (Weeks 6–8) → integration (Weeks 8–10) → results (Weeks 10–11) → LOI / contract (Weeks 11–12) → deck use (Weeks 12+).

Parallel sub-tracks: advisor outreach (Weeks 9+), architecture paper (Weeks 7+), patent filings (Weeks 9–10).

### 5.3 Interlock points

Critical interlocks where the two tracks must coordinate:

- **Week 2:** Vertical pick locks synthetic config. Commercial track informs technical track.
- **Week 3:** Real-workflow statistics extracted (commercial) → generator config tuned (technical).
- **Weeks 6–7:** Customer data access (commercial) → Layer C integration begins (technical).
- **Weeks 8–10:** Layer C metrics (technical) → design-partner LOI conversation (commercial).
- **Weeks 12+:** Layer C results (technical) → deck content (commercial).

If commercial track lags (e.g., no design partner by Week 8), the technical track must produce a contingency: synthesized "design-partner-equivalent" results using realistic synthetic data, or pivot to using a public benchmark workflow with documented LLM failure modes (e.g., from SOP-Bench, found in earlier search).

---

## §6 — Critical path

Critical path runs through:

1. ARC eval-wedge fix (Week 1)
2. Vertical pick (Week 2)
3. Synthetic generator complete (Week 4)
4. HMSC complete (Week 5)
5. Phase 0 DAG training complete (Week 6)
6. Phase 0 decision gate passed (Week 6)
7. Phase 1 training complete (Week 8)
8. Layer A claims passed (Week 8)
9. Layer C integration (Weeks 8–10)
10. Layer C verdict (Week 10)
11. Deck rebuilt (Weeks 11–14)
12. Fundraise launch (Weeks 15–16)

Any slip in items 1–10 slips the fundraise window proportionally. Items 11–12 have buffer.

**Off-critical-path items** (can slip without blocking fundraise launch):
- CLRS-30 (Weeks 7–9) — nice-to-have, not required
- Advisor outreach (Weeks 9+) — desirable for fundraise but doesn't gate it
- Architecture paper (Weeks 7+) — academic credential, not commercial gate
- Patent filings (Weeks 9–10) — important IP work but not on fundraise critical path

If anything must slip, slip off-critical items first. Slipping critical items requires re-baselining the 16-week horizon.

---

## §7 — Dependencies and gates

### 7.1 Component dependency table

| Component | Depends on | Blocks |
|---|---|---|
| ARC eval-wedge fix | (nothing) | ARC closure |
| Synthetic generator | Spec (done) | All downstream |
| Graph adapter | Synthetic generator (for testing) | HMSC integration |
| HMSC codebook | Codebook design doc (done) | Training integration |
| Auxiliary loss heads | HMSC, generator labels | Phase 1 training |
| Probe scripts | Generator splits, HMSC | Phase 0 / Phase 1 verdicts |
| Training integration | All technical components | Phase 0 launch |
| CLRS-30 setup | Graph adapter | Layer B result |
| Layer C integration | HMSC trained, customer data | Layer C result |
| Demo MVP | Phase 1 + Layer C results | Deck final |
| Deck | All technical results | Fundraise launch |

### 7.2 Phase gates

| Gate | Condition | Triggers |
|---|---|---|
| Week 2 — ARC close | Phase 0 ARC completes (verdict, regardless of outcome) | Synthetic track gets full attention |
| Week 2 — Vertical pick | Vertical decision documented with rationale | Synthetic config tuning, design-partner outreach focused |
| Week 4 — Warm-start verdict | Sudoku-warm vs from-scratch comparison run completes | Phase 0 init strategy locked |
| Week 6 — Phase 0 decision gate | Workflow-type AUC ≥ 0.6 (Path A) / Path B / Path C | Phase 1 launches; or supervised injection; or architectural review |
| Week 8 — Phase 1 claim verdicts | A1 + A3 + A4 thresholds | Layer C integration confidence; deck preparation begins |
| Week 10 — Layer C verdict | C1–C4 thresholds | Design-partner LOI conversation; deck content locked |
| Week 14 — Deck final | All material results assembled | Fundraise launch authorized |

### 7.3 Halt conditions

Per game plan §7.1 and codebook design §8.4, certain results trigger project halt:

- **Path C at Phase 0** (workflow-type AUC < 0.4 even after supervised injection): architectural review, game plan §5 sequencing paused.
- **A1 failure at Phase 1** (execution faithfulness < 70%): substrate broken; not a codebook issue; major investigation needed.
- **A4 failure at Phase 1** (no significant compounding): Recognition Cortex thesis broken; fundraise narrative needs major revision.
- **Layer C failure at customer integration** (C1 < pre-registered threshold by significant margin): re-baseline, possibly pivot vertical.

Halt conditions are pre-committed. They are not negotiable when they trigger — pausing for honest assessment is the only valid response.

---

## §8 — Compute and budget

### 8.1 Per-phase compute estimate

| Phase | Weeks | A100 SXM4 80GB hours (estimate) | Notes |
|---|---|---|---|
| Phase 0 (Foundation) | 1–6 | ~250 hours | ARC tail + synthetic dataset experiments + Phase 0 DAG training |
| Phase 1 (Validation) | 6–10 | ~350 hours | Phase 1 main training + CLRS-30 + Layer C initial runs |
| Phase 2 (Commercial) | 8–16 | ~150 hours | Layer C integration + targeted retraining + demo |

Total: ~750 A100 SXM4 80GB hours over 16 weeks.

### 8.2 Cost ceiling

At Vast.ai A100 SXM4 80GB rates (~$1.50–2.50/hour depending on availability):
- Low estimate: 750 × $1.50 = ~$1,125
- High estimate: 750 × $2.50 = ~$1,875
- Plus storage, networking: ~$200

**Total estimated GPU spend: $1,300–2,100 across 16 weeks.**

This is well within reasonable pre-seed/MVV bridge spend.

### 8.3 Compute risk

- **Availability:** A100 SXM4 80GB has spotty availability. If unavailable, fall back to PCIe 40GB at reduced batch size (training takes longer; same wall-clock plan slips).
- **Spot interruption:** Vast.ai spot instances can be preempted. Use checkpointing; budget time for re-launches.
- **Budget overrun trigger:** if cumulative spend exceeds $2,500 by Week 10, escalate — either budget revision or compute optimization (smaller batch, fewer probe runs).

### 8.4 Weekly compute log

Tracked in this doc (§10 update sections). Each two-week update appends a row to a per-week compute table with actual spend.

---

## §9 — Operational risk register

Distinct from game plan §7 strategic risks. Operational failure modes:

### 9.1 CC implementation bottleneck

- **Description:** Claude Code prompt-file workflow doesn't scale to HMSC complexity. Debugging cycles slow.
- **Probability:** Medium. HMSC is more complex than prior CC-implemented components.
- **Impact:** High. Bottleneck on critical path.
- **Mitigation:** Break HMSC into smaller CC tasks (G-codebook first, then R, then P, then composition); each task independently verifiable. Pre-write CC prompts in Week 3, refine before Week 4 implementation start.
- **Trigger condition:** If HMSC implementation hasn't completed by end of Week 5, treat as red flag and intervene.

### 9.2 Vast.ai SXM4 80GB availability

- **Description:** A100 SXM4 80GB unavailable at training launch.
- **Probability:** Medium.
- **Impact:** Medium. Training slips by 1–2 days per availability gap.
- **Mitigation:** PCIe 40GB fallback for non-critical runs. SXM4 reserved for Phase 0 / Phase 1 main training. Check availability mid-week before launching each major run.

### 9.3 Synthetic generator slowness

- **Description:** Generator produces DAGs at < 100/sec, making 1M dataset generation > 3 hours and iteration costly.
- **Probability:** Low.
- **Impact:** Medium.
- **Mitigation:** Parallel generation across CPU cores. If still slow, profile and optimize executor (current bottleneck). Reduce dataset size if necessary (500K is workable if 1M is too slow).

### 9.4 Regression in existing infrastructure

- **Description:** Code changes break Sudoku-trained checkpoint compatibility (relevant for warm-start) or ARC infrastructure.
- **Probability:** Medium.
- **Impact:** Medium.
- **Mitigation:** Warm-start regression test required before merging codebook changes to main branch (per memory discipline rule). Sudoku Phase 3c checkpoint preserved untouched.

### 9.5 Cost overrun

- **Description:** GPU spend exceeds budget.
- **Probability:** Low–medium.
- **Impact:** Low (absolute amounts are small).
- **Mitigation:** Weekly compute tracking. $2,500 cumulative trigger for review.

### 9.6 Solo-founder bandwidth

- **Description:** Anwar is single point of attention for technical design + commercial outreach + advisor relationships + deck work + integration debugging.
- **Probability:** High during integration weeks (4–6 and 8–10).
- **Impact:** High. No mitigation other than priority discipline.
- **Mitigation:** Use critical path (§6) to make priority calls under pressure. Off-critical items (CLRS-30, paper draft, patent) accept slips. Critical items get attention regardless.

### 9.7 Design-partner timeline slip

- **Description:** Design partner data access takes longer than Week 6–8 window allows.
- **Probability:** High (this is rarely faster than expected).
- **Impact:** High if Layer C runs late.
- **Mitigation:** Multiple design partner conversations in parallel (game plan §3); fall back to "best of 3" rather than "the one." Synthetic-only contingency results documented as deck-backup if Layer C slips beyond Week 12.

### 9.8 Vertical change mid-stream

- **Description:** Selected vertical reveals problems mid-stream (Week 4–7); need to pivot.
- **Probability:** Low–medium.
- **Impact:** High. Synthetic dataset regeneration, codebook K-sizing potentially affected, commercial outreach restart.
- **Mitigation:** Strong vertical decision in Week 2 with explicit criteria (game plan §3.1). If pivot needed, accept timeline slip; do not pretend continuity.

---

## §10 — Tracking and reporting

### 10.1 Update cadence

- **Default:** Two-week status notes appended to this doc.
- **High-integration weeks (Weeks 4–6, 8–10):** Weekly updates.
- **Critical-gate weeks (Weeks 2, 6, 8, 10):** Update immediately after gate verdict.

### 10.2 Update format

Each update is a new dated section appended to this doc (not in-place edit), containing:

- **Week N update — YYYY-MM-DD**
- *Completed:* components shipped, decisions resolved, results in hand
- *In flight:* what's being worked on
- *Blocked:* what's stuck and why
- *Gates passed/failed:* specific gate names and verdicts
- *Compute spent:* A100 hours used this period, cumulative
- *Risk register changes:* new risks identified, existing risks materialized/closed
- *Next 2 weeks:* preview of upcoming work

### 10.3 Decisions tracking

A separate section maintained as decisions resolve. Each decision:
- Decision name (matching the OD identifier in the relevant doc)
- Resolution
- Date resolved
- Rationale (one line)
- Impact (what this enables / closes)

### 10.4 Component status tracking

§4 component build sheets get a status column updated as work progresses:
- *Not started*
- *In design*
- *In implementation*
- *In testing*
- *Complete*
- *Blocked* (with reason)

### 10.5 Critical-path health indicator

Top of doc gets a one-line status: "Critical path health: GREEN / YELLOW / RED." Updated weekly. Green = all critical-path items on schedule. Yellow = one item slipping by < 1 week. Red = critical-path item slipping ≥ 1 week or a halt condition triggered.

---

## §11 — Document maintenance

- Updates per §10 cadence.
- Major scope changes (e.g., vertical pivot, halt-condition triggered) trigger new doc version, not in-place edits.
- Versioned alongside game plan; if game plan moves to v2, this doc moves to v2 simultaneously.
- Component build sheets in §4 are the most-frequently-updated section; rest of doc is stable unless scope changes.

This doc is the operational source-of-truth. The deck draws from it (at fundraise time); investors don't see it. Internal artifact.

---

## §12 — Status as of 2026-05-13

**Critical path health: GREEN.**

**Completed:**
- Strategic pivot decision (ARC closed, DAG validation track committed)
- Game plan drafted and signed off
- Codebook design doc drafted and signed off
- Synthetic benchmark spec drafted and signed off
- Build plan drafted (this doc)

**In flight:**
- ARC Phase 0 eval-wedge diagnostic (Week 1 work, in progress)

**Blocked:** None.

**Decisions pending Week 1–2:**
- Vertical pick (game plan OD1)
- Category name (game plan OD2)
- Design partner shortlist (game plan OD3)
- Vertical config tuning (synthetic OD5-S, OD6-S)
- Cyclic handling (codebook OD8)

**Next 2 weeks:** Week 1 eval-wedge fix + vertical narrowing; Week 2 vertical decision + synthetic CC prompt drafting + graph adapter design doc.

---

*End of build plan. Status: active execution.*
