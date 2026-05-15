# CORAL DAG Codebook Design

**Date:** 2026-05-13
**Status:** Pre-implementation design. Decision menu in §10; commitments in §3 and §7. Architecture decision (Hierarchical Multi-Scale Codebook) is committed; tuning parameters are open.
**Companion docs:** `Aktuator_Game_Plan_2026-05.md` (strategic source-of-truth; this doc is its technical companion). `CORAL_v3_ARC_Codebook_Design.md` is archived as of game plan §6.1; this doc replaces it as the primary architectural plan for the validation track.
**Supersedes:** ARC-specific codebook design. The architectural decisions there (carry-state tap, capacity 900) are preserved; the codebook *form* is redesigned for DAG execution.

---

## §0 — Why this document exists

The game plan (2026-05-13) closes the ARC validation track at Phase 0 verdict and commits the next 16 weeks to commercial validation on DAG execution. The codebook is the architectural component that does the commercial work — workflow crystallization is the unique-in-market mechanism, and it lives in the codebook design. Without a codebook architecture that handles the compositional structure of DAG reasoning, the Recognition Cortex thesis collapses to "another reasoning architecture," and Aktuator becomes a competitor inside HRM/TRM/GRAM's category.

This document specifies the codebook architecture for the validation track: what it does, where it taps the substrate, how it's supervised, what probes validate it, and what decisions remain open.

**Pre-implementation design.** No code is written until the synthetic benchmark spec lands (Week 2–3 per game plan §5) and the CC prompt drafted from this doc plus that spec is reviewed and signed off.

---

## §1 — Scope and non-scope

**In scope:**
- Codebook architecture for DAG execution: Hierarchical Multi-Scale Codebook (HMSC)
- Three-scale structure (global / regional / per-position)
- Tap points and tap topology on the carry state
- Supervision design (auxiliary loss heads, label sources)
- Probe suite for the four pre-registered claims in game plan §2.1
- Phase 0 decision gates and §7.1 existential-risk safeguards

**Not in scope:**
- Synthetic benchmark generator specification — separate document, drafted in Week 2–3 per game plan §5
- Graph adapter (DAG encoding, attention mask, positional encoding) — separate adapter design doc, analog of `CORAL_v3_ARC_Adapter_Design.md`
- Training infrastructure changes — handled in adapter doc
- Product positioning — game plan §1
- SOP-to-DAG encoder pipeline — separate product layer above CORAL, LLM-dependent at encoding time only, runtime path is CORAL-only (see §11 encoder dependency note)
- Recognition Cortex online-compounding layer — stacks on top of this codebook at deployment, separate spec, post-Phase-1

---

## §2 — Crystallization units (the foundational shift)

ARC asked the codebook to crystallize abstract rule types *inferred* from few-shot demos — the unit was emergent, the supervision was indirect, the failure mode was rule-induction collapse.

DAG execution asks the codebook to crystallize *what's structurally present in the data*. The rule is given (in the graph); the codebook's job is to recognize structure that exists at three scales:

### 2.1 The three units

**Workflow signatures (global scale).** The whole-DAG pattern. Two DAG instances of "insurance claim routing — bodily injury subtype" share the same workflow signature even if specific node values differ. Recurring workflow signatures are the *primary* compounding-at-deployment claim — the second time CORAL sees the same workflow shape, it should execute faster. This is the unit investors care about.

**Sub-DAG patterns (regional scale).** Recurring motifs that appear *across* workflows. Branch-and-merge structures, lookup-then-aggregate patterns, constraint-then-route motifs, validation loops. A regional pattern is a sub-graph that appears as a building block in multiple workflows. Crystallizing regional patterns produces compositional generalization — novel workflows built from familiar building blocks execute correctly because the building blocks are recognized.

**Reasoning primitives (per-position scale).** Atomic operations performed at individual nodes: conditional branch, aggregate, lookup, compare, validate, transform. The vocabulary of primitives is small (~10–20 per vertical), but the codebook needs per-node specialization because different nodes require different primitives at the same step of execution.

### 2.2 Why this decomposition

Three arguments converge on this three-scale structure:

*Data shape.* DAGs decompose compositionally — a workflow is built from sub-patterns, which are built from primitives. The structure exists at three scales in the data itself; the codebook architecture should mirror it.

*Commercial relevance.* Each scale carries a distinct commercial claim. Workflow signatures → deployment compounding. Sub-DAG patterns → cross-workflow knowledge transfer. Primitives → faithful execution of novel reasoning steps. Collapse the scales and one or more of these claims becomes unprovable.

*Cortical analog.* Cortical microcircuits encode at multiple scales — V1 has local detectors, IT cortex has object-level abstractions, association cortex has task-level representations. CRA-as-category means the architecture should map to cortical structure, not collapse to a single-scale flat codebook for engineering convenience.

### 2.3 Cyclic-vs-acyclic note

The architecture as designed assumes acyclic DAGs. Some real-world workflows include cycles — retry loops, escalation cycles, iterative review. Three handling options:

1. **Unroll to bounded-depth DAG.** Cycle becomes a chain of length K (max iterations). Loses true loop semantics but works for bounded retries.
2. **ACT halt mechanism captures the cycle.** Treat the workflow as a single sub-DAG that the ACT loop repeats until convergence. This is closer to how Sudoku constraint propagation works.
3. **Explicit cycle nodes.** Add edge types for back-edges, codebook learns to recognize and handle. Most architecturally clean, most expensive to implement.

**Decision deferred to OD8.** Resolution depends on which vertical is selected — if the vertical's workflows are predominantly acyclic (e.g., insurance claims routing, ICD-10 coding), defer cyclic handling to a later milestone. If the vertical has fundamental cyclic structure (e.g., audit/review workflows), commit to ACT-based cycle handling (option 2) before the synthetic benchmark locks.

---

## §3 — Architecture: Hierarchical Multi-Scale Codebook (HMSC)

Committed design. Not a menu. The data-shape argument in §2 picks it; no alternative is being evaluated.

### 3.1 Structure

Three codebooks, each with its own routing mechanism and output composition:

**Global codebook (G).**
- *Capacity:* K_G ~ 16–32 entries. Initial value depends on synthetic workflow library size (OD1).
- *Dimensionality:* d_G = 512 (matches carry state dim).
- *Input:* mean-pooled carry state across all DAG nodes — a single vector per DAG per ACT step.
- *Routing:* softmax over K_G entries. Returns weighted mixture, optionally sparsified to top-1 for interpretability probes.
- *Output:* mode vector of dim d_G, broadcast identically to all nodes in the DAG.
- *Captures:* workflow signature — what kind of DAG this is.

**Regional codebook (R).**
- *Capacity:* K_R ~ 32–64 entries.
- *Dimensionality:* d_R = 256.
- *Input:* regional pooling of carry state. Regions defined by learned attention pooling (graph attention over node clusters) or by graph-cluster pre-segmentation. Default: learned attention with a small number of region tokens (8–16 per DAG).
- *Routing:* per-region softmax over K_R entries.
- *Output:* per-region mode vectors of dim d_R, each tiled to nodes within its region.
- *Captures:* sub-DAG pattern — what kind of structural motif this region implements.

**Per-position codebook (P).**
- *Capacity:* K_P ~ 8–16 entries (small — primitives are atomic).
- *Dimensionality:* d_P = 128.
- *Input:* per-node carry state.
- *Routing:* cross-attention from each node to the K_P codebook entries. Soft attention during training; hard top-1 at eval for interpretability probes (OD3).
- *Output:* per-node mode vector of dim d_P (mixture over codebook entries weighted by attention).
- *Captures:* reasoning primitive — what operation this node performs.

### 3.2 Composition

Per-node downstream input is composed from the three codebook outputs:

`node_output = α_G × G_mode + α_R × R_mode[region(node)] + α_P × P_mode[node]`

**Default composition:** α_G = α_R = α_P = 1 (simple sum, projected to common dim if dimensions differ via small learned projections).

**Alternative (OD2):** learned gating — small MLP per node decides the weights based on the carry state at that node. Adds parameters but allows the model to learn when each scale matters.

**Recommendation:** ship default sum for v1; gating as ablation if sum underperforms on probe results.

### 3.3 Memory and compute budget

At A100 80GB with DAG sizes 10–1000 nodes:
- Codebook parameters: K_G × d_G + K_R × d_R + K_P × d_P ≈ 16K + 16K + 2K = ~34K parameters (negligible)
- Routing cost per ACT step: O(N × (K_G + K_R + K_P)) where N is node count
- Regional pooling: O(N × num_regions) for attention pooling
- Per-position cross-attention to P-codebook: O(N × K_P × d_P)

For N=1000, K_total≈100, the per-step routing cost is ~10⁵ ops — negligible compared to the GNN/attention backbone.

Memory is dominated by carry state (B × N × d_model), not codebooks. HMSC adds <1% to total memory footprint.

---

## §4 — Tap topology

### 4.1 Where the codebook reads

**Intermediate carry states, every ACT step.** Not just the final state.

Rationale: DAG execution is multi-step by nature; reasoning trajectory matters, not just final outcome. Per-step crystallization captures "what reasoning is happening here" — the codebook entries that fire at step k vs. step k+1 encode the reasoning trajectory. This trajectory is what compounds at deployment: recurring workflows produce recurring trajectories, and the codebook learns to recognize trajectories cheaply.

Per-puzzle tap (ARC's approach) loses the trajectory — the carry state collapses to a single "rule hypothesis" with no temporal information. On DAGs, the temporal structure of reasoning is the structure to crystallize.

### 4.2 Three-scale reading at each step

At each ACT step t:
1. Pool carry_t globally → route through G-codebook → produce G_mode_t
2. Pool carry_t regionally (per region) → route through R-codebook → produce R_mode_{r,t} for each region r
3. Per-node carry_t → cross-attend to P-codebook → produce P_mode_{n,t} for each node n
4. Compose per-node output per §3.2

Downstream architecture (the rest of CORAL) sees the composed output at every step.

### 4.3 Difference from ARC's Option B tap

ARC tap (Option B from the adapter design):
- After processing all puzzle demos
- Carry state = rule hypothesis
- Single tap per puzzle

DAG tap (this design):
- At every ACT step during DAG execution
- Carry state = current reasoning state
- Repeated tap during execution

The repeated tap means the codebook learns to recognize not just "what kind of DAG" but "what kind of reasoning step within this DAG." This is the per-step crystallization claim — and it's what enables the deployment-compounding mechanism (a familiar reasoning trajectory halts faster).

### 4.4 Tap frequency (OD5)

Default: every ACT step. Reduction to every-N-steps only if compute-bound. Initial estimate: every-step is affordable at the parameter/compute budget in §3.3. Confirm during first training run.

---

## §5 — Supervision design

### 5.1 Label sources

Synthetic benchmark provides clean labels at all three scales — the simplification that makes this tractable.

For each synthetic DAG:
- *Global label:* workflow type ID (which workflow signature this DAG instantiates)
- *Regional labels:* sub-DAG pattern ID per region (which motif each region implements)
- *Per-position labels:* primitive type ID per node (which operation each node performs)

These labels are generated by the synthetic DAG generator, so coverage is exhaustive and noise-free. This is a substantial improvement over ARC's ConceptARC label situation (sparse, partial, manually annotated).

### 5.2 Auxiliary loss heads

One classifier head per codebook scale, trained jointly with the main DAG-execution loss:

- **Global head:** linear classifier on G_mode_t, predicts workflow type label. Loss: cross-entropy.
- **Regional head:** per-region linear classifier on R_mode_{r,t}, predicts sub-DAG pattern label. Loss: cross-entropy averaged over regions.
- **Per-position head:** per-node linear classifier on P_mode_{n,t}, predicts primitive type label. Loss: cross-entropy averaged over nodes.

Combined loss:

`L_total = L_main + λ_G × L_G + λ_R × L_R + λ_P × L_P`

where L_main is the standard CORAL execution loss (per-node prediction accuracy + Q-halt loss).

### 5.3 Lambda values and schedule

Initial values: λ_G = λ_R = λ_P = 0.1 (each auxiliary loss contributes ~10% of L_main magnitude).

**Sweep schedule (OD-implicit):**
- First synthetic training run: all λ at 0.1, observe codebook utilization and main-task accuracy.
- If aux losses dominate (main accuracy degrades): reduce λ values to 0.05 or 0.01.
- If aux losses don't engage (codebook utilization stays uniform): increase λ values to 0.3.

Lambda schedule: constant for v1. Warmup or annealing as fallback if utilization is unstable.

### 5.4 Default: supervised. Ablation: unsupervised.

**Default training:** supervised with all three λ at non-zero values. This produces fastest convergence to the right codebook structure and the strongest probe results.

**Ablation for strongest Claim 2 evidence:** run a parallel training with all λ = 0. Check whether codebook entries still cluster by workflow/pattern/primitive. If unsupervised crystallization holds, the claim becomes "the codebook *discovered* structure without being told" — the strongest possible evidence for the Recognition Cortex thesis.

The ablation is essential for the fundraise narrative. Supervised crystallization could be dismissed as "you classified the workflows; of course they cluster." Unsupervised crystallization isn't dismissible — the codebook had to find the structure on its own.

### 5.5 Transfer to design partner (Layer C)

Real customer data won't have clean labels at all three scales. Three transfer strategies, in order of preference:

1. *Synthetic-trained codebook, frozen on customer.* The codebook structure learned on synthetic transfers directly; CORAL fine-tunes on customer workflows without modifying the codebook. Cleanest transfer, requires synthetic to match customer's vertical structure (game plan §3.4 alignment principle).
2. *Partial supervised on customer.* Where customer has labels (e.g., workflow types are tracked in their existing system), use them as auxiliary signal for the global head. Regional and per-position remain frozen-from-synthetic.
3. *Pure unsupervised on customer.* Most permissive; depends on the unsupervised ablation in §5.4 holding up.

The transfer strategy is a Layer C scoping decision (game plan §5, Weeks 8–10). It depends on what labels the design partner has available.

---

## §6 — Probe suite

Four primary probes mapped to the four pre-registered claims in game plan §2.1, plus carry-forward probes for completeness.

### 6.1 Execution faithfulness (Claim A1)

*Question:* Does CORAL's traversal sequence match ground-truth DAG execution?

*Method:* For each test DAG, compute per-step edit distance between predicted execution trace and ground-truth execution trace. Trace = sequence of (node, state) tuples.

*Metrics:*
- Mean per-step edit distance
- Failure rate (DAGs with any non-zero edit distance)
- Catastrophic-failure rate (DAGs with edit distance > 50% of trace length)

*Threshold (pre-registered):* Mean edit distance ≤ ε (ε set during synthetic benchmark design, target ε ≤ 0.1 normalized to trace length). Catastrophic-failure rate ≤ 1%.

*Failure mode interpretation:* High edit distance indicates the substrate isn't executing faithfully — base mechanism failure, not codebook failure. If A1 fails, codebook isn't the problem; revisit adapter or PC mechanism.

### 6.2 Primitive crystallization (Claim 2 — primary)

*Question:* Do specific codebook entries fire when specific primitives are needed?

*Method:* Mutual information between per-position codebook activation (P-codebook attention distribution) and primitive labels. Computed per primitive type:
- For each primitive p, which P-codebook entries have high attention weight on nodes labeled p?
- MI(P_attention, primitive_label) over the test set

*Metrics:*
- Per-primitive top-1 code purity: fraction of nodes of primitive p where the top-attended code is the same
- Overall MI(P_attention, primitive_label) in bits
- Codebook utilization: how many codes have non-trivial usage

*Threshold (pre-registered):* MI ≥ 1.5 bits. Top-1 purity per primitive ≥ 70% (most primitives have a dominant code).

*Failure mode interpretation:* Low MI means the codebook didn't separate primitives — the per-position scale isn't doing its job. Could mean K_P too small (sweep), or the per-position carry state doesn't differ enough across nodes (substrate issue).

### 6.3 Compositional generalization (Claim A3)

*Question:* Does CORAL execute novel compositions of trained primitives correctly?

*Method:* Train on primitive set {A, B, C} in compositions {AB, AC, ABA}. Test on novel compositions {BC, CB, BCA, ACB, etc.} held out from training.

*Metrics:*
- Accuracy on training compositions (sanity check)
- Accuracy on novel compositions (the LLM-failure-mode probe)
- Generalization gap: training-composition accuracy minus novel-composition accuracy

*Threshold (pre-registered):* Novel-composition accuracy ≥ 80%. Generalization gap ≤ 10pp.

*Failure mode interpretation:* Large generalization gap = CORAL memorized training compositions rather than learning to compose primitives. The per-position codebook should be enabling compositionality; if A3 fails badly, the per-position scale isn't doing the work it's supposed to.

This is the rhetorically most important probe. LLMs fail compositional generalization as a published result; a clean A3 pass is the cleanest "CORAL does what LLMs can't" claim in the deck.

### 6.4 Training-time compounding (Claim A4 — existential)

*Question:* Do workflows seen more often during training halt faster on test?

*Method:* Construct training distribution with workflow types at varying frequencies (e.g., type 1 seen 1000×, type 2 seen 500×, type 3 seen 100×, type 4 seen 10×). At test, measure halt step per workflow type.

*Metrics:*
- Halt step vs. log(train frequency): linear regression
- Slope: negative slope indicates compounding (more-seen workflows halt earlier)
- Statistical significance (p-value on slope)

*Threshold (pre-registered):* Negative slope with p < 0.01. Magnitude: halt step decreases by ≥ 2 steps per 10× increase in training frequency.

*Failure mode interpretation:* If A4 fails — workflows don't halt earlier despite repeated exposure — the Recognition Cortex compounding mechanism doesn't engage. This is the existential failure per game plan §7.1. Codebook is being used but isn't doing the compounding work.

### 6.5 Carry-forward probes

From the ARC codebook design, retained with DAG-appropriate labels:

- **Workflow-type clustering AUC on carry state.** Run before codebook engages (Phase 0 PC-only checkpoint). Pre-registered threshold ≥ 0.6 per §7 decision gates.
- **Codebook content MI at each scale.** MI(G_activation, workflow_label), MI(R_activation, pattern_label), MI(P_activation, primitive_label). Complement to §6.2.
- **Halt-by-workflow-frequency on training distribution.** Within-training-distribution version of A4 — useful for diagnosing whether the compounding effect is training-time or generalization.

### 6.6 Probe implementation order

Implementation priority for CC prompt:
1. A1 (execution faithfulness) — first, because A1 failure invalidates everything downstream
2. Workflow-type AUC — second, because it gates §7 decision
3. A4 (compounding) — third, because it's existential
4. A3 (compositional generalization) — fourth, because it's the rhetorically strongest claim
5. A2 / primitive crystallization — fifth, complementary to A3

---

## §7 — Decision gates

Phase 0 PC-only DAG-execution training produces a checkpoint. The decision gates determine what happens next, based on probe data from that checkpoint.

### 7.1 The gating probe

**Workflow-type clustering AUC** on the Phase 0 PC-only carry state. Computed with three readouts:
- Pooled (mean over nodes)
- Per-region (with learned region attention)
- Per-position (per-node features)

Take the maximum AUC across readouts as the gating metric.

### 7.2 Three paths from Phase 0

**Path A — AUC ≥ 0.6 on at least one readout.**
- Codebook engages at the scale where AUC passes.
- If only global readout passes (AUC pooled ≥ 0.6): train G-codebook initially; add R and P after G converges.
- If per-position readout passes: train full HMSC simultaneously.
- Phase 1 launches with HMSC + auxiliary supervision per §5.

**Path B — All readouts AUC < 0.6 but ≥ 0.4.**
- Substrate didn't acquire workflow-type structure on its own.
- Codebook can't compound on what isn't there (Sudoku R0.5 analog).
- Apply Phase 0' — same training with auxiliary supervised loss (workflow-type classifier head on carry state) at λ = 0.3 for ~5K steps to inject structure.
- Re-run probes. If AUC ≥ 0.6 after Phase 0', proceed to Path A. If not, Path C.

**Path C — All readouts AUC < 0.4, or Phase 0' fails to lift above 0.6.**
- Existential review per game plan §7.1.
- Architecture-level investigation: is carry state too small? Is PC mechanism appropriate? Is the synthetic benchmark too uniform (no structure to learn)?
- May trigger architecture redesign. Game plan §5 sequencing pauses until review concludes.

### 7.3 The 0.6 threshold

The threshold is pre-registered, not adjusted post-hoc. Rationale:
- AUC 0.5 = random baseline. 0.6 = meaningful-but-weak signal.
- ARC R0.5 found Sudoku-trained CORAL had AUC ~0.50 on input-layout features — the failure mode this gate is designed to catch.
- 0.6 is high enough to be non-trivial, low enough that synthetic supervision can usually push above it.

The threshold is intentionally lower than what a strong classifier would achieve. The codebook will lift this further during training; the gate just confirms there's signal to lift.

---

## §8 — Existential-risk safeguards

### 8.1 The failure mode

The Sudoku R0.5 analog: substrate trains successfully — execution faithfulness passes, halt mechanism fires correctly — but the carry state doesn't acquire workflow-type structure. The codebook has nothing to crystallize on, and the compounding claim breaks.

This is the existential risk identified in game plan §7.1.

### 8.2 Probe sequence (catch it early)

Three checkpoints during Phase 0:

1. **Early (~5K steps):** basic workflow-type clustering probe. If AUC < 0.3, stop early — substrate isn't trending the right way.
2. **Midpoint (~20K steps):** full carry-forward probe suite minus codebook-dependent probes (the codebook isn't trained yet). Check workflow-type AUC and execution faithfulness.
3. **Final (~50K+ steps):** decision gate per §7.

The early checkpoint matters most. R0.5 on Sudoku revealed the failure *late* in training when retraining is expensive. Catching the same failure at 5K steps means a few hours of wasted GPU, not weeks.

### 8.3 Mitigation paths

Pre-registered responses to specific probe outcomes:

| Midpoint observation | Response |
|---|---|
| Workflow-type AUC < 0.6, faithfulness > 80% | Pause Phase 0, run Phase 0' with auxiliary supervision |
| Workflow-type AUC < 0.4 at midpoint | Stop Phase 0, redesign before continuing |
| Faithfulness < 70% at midpoint | Investigate execution mechanism — separate problem from clustering |
| Faithfulness < 60% at midpoint | Halt: substrate is broken, not just under-trained |
| All passing at midpoint | Continue to final checkpoint |

### 8.4 Halt vs. patch criteria

**"Patch and continue"** (auxiliary supervision injected, training continues):
- Workflow-type AUC ≥ 0.4 but < 0.6
- Execution faithfulness ≥ 80%
- Substrate is partially acquiring structure; supervision can lift it

**"Halt and review"** (architectural session triggered, game plan §5 sequencing paused):
- Workflow-type AUC < 0.4 even after Phase 0'
- Or execution faithfulness < 60%
- Substrate is wrong for the task; patching won't fix it

### 8.5 What "halt" means operationally

- Stop synthetic benchmark training runs
- Schedule architectural review session (Anwar + Claude)
- Pause game plan §5 Weeks 6+ until review concludes
- Communicate timeline impact to anyone tracking validation milestones (advisors, design-partner conversations in flight)

The halt branch is unlikely but must be pre-committed. The discipline rule "Two consecutive same-signature failures = diagnose mode, not another knob turn" applies here directly.

---

## §9 — Implementation pathway

This doc doesn't generate CC prompts. The implementation pathway:

1. **Synthetic benchmark spec drafted** (game plan §5 Week 2–3). Specifies DAG generator parameters, primitive vocabulary, workflow library, recurrence schedule, label format.
2. **CC prompt drafted** from this doc + synthetic spec. Includes HMSC module spec, auxiliary loss heads, probe scripts.
3. **HMSC module implemented** (~600–800 LOC for the three codebook classes + routing + composition logic). Includes G-codebook, R-codebook (with regional pooling attention), P-codebook (with cross-attention), composition module.
4. **Auxiliary loss heads implemented** (~200 LOC). Three classifier heads, loss aggregation, lambda scheduling.
5. **Probe scripts implemented** (~400 LOC for the four new probes plus carry-forward probes).
6. **First training run launches.** Phase 0 PC-only DAG execution on synthetic, no codebook engaged yet.
7. **Early checkpoint probes run** (~5K steps). Gate per §8.2.
8. **Midpoint checkpoint probes** (~20K steps). Gate per §8.2.
9. **Final checkpoint probes** (~50K+ steps). §7 decision gate. Path A/B/C.
10. **Phase 1 launches** with HMSC engaged per the §7 path.

### 9.1 Timeline alignment with game plan §5

- Week 2–3: synthetic benchmark spec + CC prompt drafting
- Week 3–4: HMSC implementation
- Week 4–5: first training runs (Phase 0 PC-only)
- Week 5–6: early and midpoint checkpoint probes
- Week 6: final checkpoint, §7 decision
- Week 6–8: Phase 1 launches with codebook; results landing

This fits the game plan's Weeks 3–6 build window and Weeks 6–8 results window with a few days of buffer.

---

## §10 — Open decisions

| # | Decision | Resolution path | Deadline |
|---|---|---|---|
| OD1 | Initial K values (K_G, K_R, K_P) | Synthetic vocabulary size from benchmark spec drives initial values; sweep after Phase 0 if utilization is poor | Week 3 (synthetic spec) |
| OD2 | Composition method (sum vs. learned gating) | Default sum; gated as ablation if sum probes underperform | Week 5 (during impl) |
| OD3 | P-codebook attention discreteness (soft vs. top-k vs. hard) | Default soft during training; hard top-1 at eval for interpretability | Week 5 |
| OD4 | Supervised vs. unsupervised default | Default supervised; unsupervised parallel run as Claim 2 evidence | Week 4 |
| OD5 | Tap frequency (every ACT step vs. every N) | Default every step; reduce only if compute-bound | Week 5 |
| OD6 | Online-compounding boundary (Reasoning vs. Recognition Cortex) | Out of scope this doc; Recognition Cortex spec post-Phase-1 | Post-Phase-1 |
| **OD7** | **Warm-start from Sudoku Phase 3c checkpoint vs. from-scratch** | **Small comparison run on tiny synthetic DAG (5K steps each); commit to warm-start if matches or exceeds from-scratch** | **Week 3** |
| OD8 | Cyclic workflow handling | Depends on vertical selection (game plan §3.3, Week 2); commit before synthetic locks | Week 2–3 |

### 10.1 Note on OD7 (warm-start)

The transfer argument from the conversation preceding this doc: Sudoku-trained CORAL has the same base reasoning mechanisms (constraint propagation, iterative refinement, halt-at-fixed-point, faithful step-by-step execution) that DAG execution requires. ARC's hard capability (few-shot rule induction) is *not* needed for DAG execution — the rule is given. So Sudoku is a closer prior than ARC, and possibly closer than from-scratch.

Risk: Sudoku surface-form bias (per R0.5) might persist after substrate change. Test cheaply on a small synthetic DAG benchmark (~5K steps each, warm vs. cold), measure workflow-type AUC and faithfulness, commit to the winner. If warm-start matches or exceeds from-scratch on both metrics, use warm-start for full Phase 0 — saves significant training time.

If warm-start lags from-scratch on workflow-type AUC: from-scratch is preserved as default. The R0.5 surface-form bias may be doing more damage than expected.

### 10.2 Note on OD8 (cyclic handling)

Three options per §2.3. Decision depends on vertical:
- Insurance claims routing, ICD-10 coding, KYC/AML: predominantly acyclic. Defer cyclic handling.
- Audit/review workflows, financial reconciliation with reopened-account flows: cyclic structure is fundamental. Commit to ACT-based cycle handling before synthetic locks.

This is a synthetic-benchmark-design dependency, not just a codebook concern. Must close before Week 3.

---

## §11 — Coupling with v2 metric-shape and game plan

### 11.1 v2 Run B verdict transfer

The v2 metric-shape mechanism on Sudoku is pending verdict at time of writing. Three outcomes:

**v2 PASSES on Sudoku (all four pre-registered criteria).** The InfoNCE-on-augmentation-pairs recipe is validated. Less necessary on DAG because synthetic labels are clean and exhaustive, but available as fallback if supervised auxiliary loss fails to lift workflow-type AUC. Keep in reserve.

**v2 FAILS on Sudoku.** More concerning for the DAG track in principle, but synthetic supervision substitutes for v2's role here (see §11.2). Failure on Sudoku doesn't block DAG progression; it does affect the broader claim about CRA's mechanism for putting structure into geometry. Treat as a research-credibility issue, not a validation-track blocker.

**v2 PARTIALLY passes (criteria 1–3 pass, criterion 4 — accuracy gap — fails).** Recipe engages but at cost to main-task accuracy. Same caution applies on DAG: lambda values for supervised auxiliary loss need careful tuning to avoid main-task degradation. Treat λ as a sensitive hyperparameter from the start.

### 11.2 Substitution argument

v2 on Sudoku: put input-layout structure into H-state geometry via training-time InfoNCE loss.

Synthetic labels on DAG: put workflow-type/pattern/primitive structure into carry state via supervised auxiliary loss heads.

These are different mechanisms achieving the same architectural goal — ensuring the carry state encodes the structure the codebook needs to crystallize on. Synthetic labels are cleaner (no augmentation pair design, no negative sampling, no temperature tuning), easier to ablate (set λ = 0), and more interpretable (each scale's structure is directly supervised).

The substitution doesn't invalidate v2 — it's still the right mechanism for tasks without clean labels. But for the DAG validation track, where synthetic labels are exhaustive, supervised auxiliary loss is the preferred path.

### 11.3 Connection to game plan §7.1

Game plan §7.1 identifies workflow crystallization failure as the existential risk for the 16-week validation window. This doc operationalizes that risk:

- §7 decision gates implement the explicit AUC ≥ 0.6 threshold referenced in game plan §7.1
- §8 existential-risk safeguards implement the probe sequence and mitigation paths
- §6.4 Claim A4 (training-time compounding) is the probe that directly tests the existential claim

If §6.4 fails — workflows don't halt faster despite repeated exposure — the Recognition Cortex thesis is broken, and game plan §7.1's "halt the project" branch triggers.

### 11.4 Encoder dependency note

The DAG input to CORAL is assumed valid: well-formed, semantically meaningful, with correct primitive labels at each node and correct edge structure. In production, this DAG is produced by an upstream encoder — LLM-based, separate product layer above CORAL.

The encoder is not in scope for this doc (§1 non-scope), but its existence has two implications:

1. *Failure-mode propagation.* Encoder failures (bad DAG extraction from SOP) propagate as CORAL failures (faithful execution of an invalid DAG). The codebook can't compensate for an upstream encoding error. Validation that CORAL executes DAGs faithfully (Claim A1) is silent about whether the input DAG was correct in the first place. This is a customer-side validation question, not a codebook-design question, but worth flagging in the design partner workflow (game plan §2.3 / Layer C).

2. *"No LLM in the runtime execution path" claim.* The architectural commitment is that no LLM is invoked during workflow execution. Encoding-time LLM use is amortized across all execution instances (one DAG, many executions). The codebook design doesn't depend on encoding mechanism — it assumes the DAG is given. The runtime-path claim is preserved regardless of how the encoder is built.

The encoder is a separate product/IP layer. The codebook design is execution-side. The two compose to produce the full Aktuator stack.

---

## §12 — Document maintenance

This is the technical source-of-truth doc for the codebook architecture during the 16-week validation window.

Update protocol:
- §10 decisions resolved as they close; resolution documented in-line in the table
- Probe outputs from Phase 0 and Phase 1 land as appendix data after they complete
- Major architecture changes (e.g., §7 Path C triggering an architecture redesign) require a new document version, not in-place edits
- Updates every two weeks during active validation phase (Weeks 3–8 are the high-update period)

The doc is not the CC prompt. The CC prompt draws from this doc + synthetic benchmark spec and is rebuilt when either changes.

---

*End of codebook design. Status: active, gated on synthetic benchmark spec for implementation.*
