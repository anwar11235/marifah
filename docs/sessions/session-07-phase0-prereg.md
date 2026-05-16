# Phase 0 Verdict Pre-Registration

**Date locked:** 2026-05-16 (pre-launch, before any verdict-quality probe run)
**Substrate:** marifah-core CORAL Reasoning Cortex
**Verdict deliverable:** `docs/sessions/session-07-phase0-verdict.md` (Session 7)
**Pre-reg authority:** Anwar Haq, Founder, Aktuator AI

This document fixes the decision criteria for Phase 0 *before* results are seen. Per discipline rule #8 (pre-register kill criteria before runs). Any post-result adjustment to these thresholds is a discipline violation and must be flagged in the verdict doc with explicit acknowledgment and rationale.

---

## What Phase 0 is

Phase 0 is the **substrate gate**: does PC-only training on graph DAGs produce H-states that are organized by workflow type? If yes, HMSC + Recognition Cortex have a working foundation to build on (Phase 1+). If no, the substrate needs rework.

It is **not** measuring final task performance. Main loss is structurally trivial (predicting input primitives from inputs that contain them). The Phase 0 question is whether the *carry states* (`z_H` after mean-pooling over real nodes) develop workflow-discriminating structure under PC + ACT, without any explicit supervision toward workflow identity.

## Substrate configuration (locked)

```yaml
model:
  d_model: 512
  num_heads: 8
  H_cycles: 1
  L_cycles: 1
  H_layers: 4
  L_layers: 4
  use_hmsc: false
  hmsc: null
  vocab_size: 10
  max_nodes: 100
  halt_max_steps: 4
  halt_exploration_prob: 0.1
  forward_dtype: "bfloat16"

training:
  lambda_G: 0.0
  lambda_R: 0.0
  lambda_P: 0.0
  main_loss_weight: 1.0
  halt_loss_weight: 0.1
```

~28.8M params. From-scratch initialization for the verdict-establishing Phase 0 run. Warmstart cold/warm comparisons run separately as supporting evidence.

## Data configuration (locked)

- Generator config: `configs/default.yaml`, `ood_holdout_seed: 13`
- Train: 795K DAGs, 49 unique workflow types, log10 spread 3.37
- Val: 10K DAGs, 47 unique workflow types
- Test_id: 10K DAGs, 46 unique workflow types
- Test_ood_size: 5K DAGs, 46 unique workflow types (9.7% filtered by max_nodes=100)
- Test_ood_composition: 1.6K DAGs, **2 unique workflow types** (known limitation — not used in verdict)

## Probe configuration (locked)

- Script: `scripts/warmstart_probe.py` at commit `b02a7ba` or later
- Eval split: `val` (10K DAGs)
- `max_samples`: 1000
- Carry state extraction: mean-pool `z_H` over real (non-padded) nodes after final ACT halt
- Classifier: `sklearn.linear_model.LogisticRegression` with `StandardScaler`, `C=1.0`, `max_iter=500`
- **Train/test split: stratified by workflow_type_id, 70/30** (every class present in both partitions)
- Singleton classes (only 1 sample) filtered before stratification; count surfaced in JSON output
- Reported metric: macro-averaged one-vs-rest AUC across classes present in test set
- **Bootstrap: 100 iterations for 95% CI on the verdict run** (use `--bootstrap 100`)
- **AUC computation now raises on failure** (no silent fallback to accuracy)

---

## Pre-registered decision gates

**Primary metric:** `workflow_type_auc` (macro-OvR on val H-states, point estimate)
**Secondary metric:** `mean_edit_distance` (execution faithfulness on val)
**Confidence:** 95% bootstrap CI; verdict refers to the point estimate, but CI is reported

### Path A — Proceed to Phase 1 with HMSC

```
workflow_type_auc ≥ 0.65
AND mean_edit_distance ≤ 0.15
```

**Reading:** Substrate develops workflow-discriminating H-states without explicit workflow supervision. Adding HMSC (workflow + region + primitive auxiliary losses) on top has a working foundation. Recognition Cortex (deployment-time compounding) can be built against this substrate.

**Implication:** Session 7 → Phase 1 with HMSC turned on (`use_hmsc: true`, non-zero lambdas). Architecture commits as-is. Investor narrative: "Cortical Reasoning Architecture substrate validated on synthetic DAGs at workflow-type AUC X."

### Path B — Revisit substrate architecture

```
workflow_type_auc < 0.55
```

**Reading:** Even with 49 distinct workflow types and structurally diverse graphs, H-states do not organize semantically under PC + ACT. The substrate hypothesis is wrong, undersized, or under-trained.

**Implication:** Pause Marifah build. Diagnose whether the issue is:
- Capacity (4/4 layers too shallow for graph task → revisit at 8/8 or larger)
- Training duration (5K steps insufficient → extended Phase 0)
- Architecture (PC + ACT alone doesn't develop workflow-level abstraction → revisit substrate design)

No HMSC work until Path B's diagnostic completes.

### Path C — Extended Phase 0 with revised training

```
0.55 ≤ workflow_type_auc < 0.65
OR (workflow_type_auc ≥ 0.65 AND mean_edit_distance > 0.15)
```

**Reading:** Partial signal. Either AUC is in the gray zone, or AUC is good but the model isn't actually learning the task at val (high edit distance → main loss isn't reaching val, possible train-only memorization).

**Implication:** Extended Phase 0 run (100K+ steps, or with auxiliary self-supervised contrastive objective on workflow_type) before committing to HMSC. Session 7 becomes "Phase 0 extended" rather than "Phase 1 with HMSC."

---

## Threshold rationale

Chance baseline for macro-OvR AUC with ~47 classes (val) is 0.50 (each binary one-vs-rest is at chance for a chance model). The thresholds were chosen as:

- **0.65 Path A:** "Meaningfully above chance, defensibly substrate is working." Not a high bar — with 49 well-separated workflow types and structural input differences, an organized substrate should produce 0.75-0.85+. 0.65 says "substrate organization is real even if modest."
- **0.55 Path B:** "Essentially chance with noise — substrate is doing nothing useful." Below this, calling the substrate validated would be dishonest.
- **0.15 mean_edit_distance ceiling for Path A:** 85% per-token argmax accuracy on val. Given main_loss collapses to ~0 on train within ~80 steps, val edit distance should be well below 0.05 unless val is OOD. 0.15 is a generous ceiling primarily to catch catastrophic generalization failure.

If the resulting AUC point estimate is uncomfortably close to a path boundary (within 0.02 of 0.55 or 0.65), the verdict doc must note this explicitly and use the conservative path (B over C, C over A). No "round up."

---

## Known caveats acknowledged at pre-reg time

1. **`test_ood_size`** split: 9.7% of records (>100 nodes) filtered by `max_nodes=100`. OOD-size eval, if reported, only covers the smaller-graph half. Phase 0 verdict uses `val` only, not OOD splits.

2. **`test_ood_composition`** split: only 2 unique workflow types. Effectively unusable for workflow-type AUC. Phase 0 verdict explicitly excludes this split.

3. **`main_loss` collapse**: by architecture, `main_loss` is structurally trivial (predicting input primitive_assignments from inputs that contain them). It is the gradient signal pushing optimization through the substrate, not a verdict metric. Reaching ~0 on train within ~80 steps is expected behavior and is not evidence of substrate quality.

4. **Sudoku checkpoint warm-start**: handled separately under `warmstart_warm.yaml` after checkpoint remap fix landed (commit `07f8341`). The warm-vs-cold comparison is a supporting study, not the Phase 0 verdict. If warmstart_warm crashes or produces uninterpretable results, Phase 0 verdict still proceeds on cold from-scratch.

5. **The reading-1 vs reading-2 ambiguity**: high workflow_type_auc could arise from either (a) genuine substrate-level semantic organization or (b) downstream-of-main-loss correlation, where z_H encodes which primitives are present and primitives correlate with workflow type. Distinguishing these is Phase 1+ work (ablations, primitive-shuffle controls, OOD probes). The Phase 0 verdict does not claim to distinguish them — it only claims the substrate produces workflow-discriminating H-states.

6. **Bootstrap CI for the verdict run only**: per-iteration probe takes ~30s. 100 bootstraps add ~1 hour to the verdict-quality run. Routine probes (cold smoke, warm smoke) skip the bootstrap (`--bootstrap 0`).

---

## What the verdict doc must contain

When Phase 0 completes, `docs/sessions/session-07-phase0-verdict.md` must record:

1. **Primary metric value(s)** — point estimate AUC with 95% bootstrap CI
2. **Path A / B / C declaration** with explicit threshold comparison
3. **Confusion matrix or per-class AUC breakdown** — which workflow types are easy vs hard
4. **Sanity check**: workflow_type_auc on the cold checkpoint at *step 0* (random init) — should be ~0.50. If it's not, the probe is broken, not the substrate
5. **Acknowledgment of any boundary cases** — e.g. if a threshold was within 0.02 and conservative-path was chosen, note that explicitly
6. **What this verdict does NOT claim**: no claim about HMSC, no claim about Recognition Cortex, no claim about edge deployment, no claim about real-world DAGs (synthetic only), no claim about distinguishing semantic-vs-correlational H-state organization (reading-1 vs reading-2)

---

## Sign-off

Locked by: Anwar Haq, Founder, Aktuator AI
Date: 2026-05-16
Branch: `session-6/phase0-prep` (commit at or after `b02a7ba`)
Status: This pre-registration takes precedence over any later threshold adjustment. Modifications require an explicit "amendment" entry below with date and rationale, *before* seeing new results.

**Amendments:** see Amendment 1 below.

---

## Amendment 1 — 2026-05-16

**Pre-registered before any amended-criteria run is executed.**

**Title:** Replace `workflow_type_auc` as primary metric with two substrate-quality probes (Δ-probe and shuffled-primitive probe).

**Rationale for amendment:**

Post-hoc analysis of the cold-run probe result (`workflow_type_auc = 0.9933`, W&B `w3sn79z5`) revealed that `node_features[b,n,0]` encodes `primitive_id`, and each workflow type has a characteristic primitive distribution. A linear classifier reading only the *input* node features already achieves AUC ≈ 1.0. This means `workflow_type_auc` measured on `z_H` is dominated by input-level leakage, not by substrate-level semantic organization. The metric cannot distinguish:

- A substrate that organized representations (what we want to detect), from
- A substrate that simply passed through the primitive-identity encoding from inputs (no useful work done).

This is not a post-result threshold adjustment — it is a correction to the probe design that makes the measurement meaningful. The original metric was not a valid test of the hypothesis it was supposed to test.

**Replacement metrics:**

### Δ-probe (primary substrate-quality metric)

```
Δ = AUC(z_H) - AUC(node_features)
Script: scripts/delta_probe.py --max_samples 1000 --bootstrap 100
```

- `AUC(node_features)`: linear probe on mean-pooled raw *input* node features (measures input-level leakage baseline)
- `AUC(z_H)`: linear probe on mean-pooled *substrate carry states* (measures what z_H contains)
- Δ is the *information added* by the substrate beyond the inputs

Interpretation:
- Δ ≈ 0: substrate adds nothing beyond inputs — not doing useful work
- Δ > 0.10: substrate is genuinely organizing representations beyond pass-through
- Δ < 0: substrate is destroying input information (degenerate training)

**Path A criterion (Δ-probe): Δ ≥ 0.10**

### Shuffled-primitive probe (structural organization metric)

```
Script: scripts/shuffled_probe.py --max_samples 1000
```

- `auc_unshuffled`: AUC when primitive assignments are intact
- `auc_shuffled`: AUC after permuting primitive_assignments and node_features[:,0] within each DAG (preserves multiset, destroys position-specific identity; graph topology unchanged)

If the substrate reads *structural organization* (which primitive is at which structural position), shuffling destroys that signal → `auc_shuffled` stays high but below `auc_unshuffled`. If the substrate merely reads *primitive identity without structure*, `auc_shuffled ≈ auc_unshuffled ≈ 1.0`.

**Path A criterion (shuffled-probe): `auc_shuffled ≥ 0.65`**

### Combined Path A (both criteria must be met)

```
Δ ≥ 0.10   [delta_probe]
AND
auc_shuffled ≥ 0.65   [shuffled_probe]
```

**Path B (stop, revisit):** Δ < 0.05 AND auc_shuffled < 0.55

**Path C (extended Phase 0):** all other outcomes

### Faithfulness metric unchanged

`mean_edit_distance ≤ 0.15` remains required for Path A, unchanged from original pre-reg.

**Additional change: OOD-composition split restored**

The `test_ood_composition` split had collapsed to 2 unique workflow types (original pre-reg: "known limitation — not used in verdict"). Root causes identified:

1. OOD-acceptance check used only cross-pattern boundary edges (conservative, intended for training filter). With seed=13 holdout pairs, only 2/50 workflows had cross-pattern reserved pairs.
2. Workflow sampling was frequency-weighted: top-5 workflows got 63.8% of attempts.

Fix (component A of session-6/phase0-prep, commits tagged `fix(generator)`):
- OOD-acceptance now uses all directed edges (inclusive check). 49/50 workflows can produce OOD-composition DAGs with seed=13.
- OOD-composition sampling uses round-robin workflow assignment (workflow_type_id = (i % 50) + 1).
- Training filter (cross-pattern only) unchanged.

Updated `test_ood_composition` capability: ≥25 unique workflow types with n=200 (verified by `TestOodCompositionDiversity` test). Not used in Phase 0 primary verdict, but available for supporting probes.

**Scripts added in this amendment:**
- `scripts/delta_probe.py` — Δ-probe CLI
- `scripts/shuffled_probe.py` — shuffled-primitive probe CLI
- `scripts/check_deps.py` — dependency pre-flight for Vast.ai
- `tests/test_substrate_probes.py` — unit tests for both probes (no GPU, synthetic data)
- Updated `scripts/verify_pipeline_e2e.py` — Phase 3 split into Phase 3a (Δ-probe) + Phase 3b (shuffled-probe)

**Sign-off on Amendment 1:** Anwar Haq, Founder, Aktuator AI — 2026-05-16

---

## Amendment 2 — 2026-05-17

**Pre-registered before any amended-criteria run is executed.**

**Title:** (A) Δ-probe baseline changed to dimensionality-matched random projection; (B) ACT-iterative probe variants added as comparison probes.

### Amendment 2A: Dimensionality-matched Δ-baseline

**Rationale:**

The cold checkpoint probe (5K steps) returned Δ = 0.1772, identical to the random-init probe Δ = 0.1767. Since Δ should be ~0 for random init (substrate adds nothing), this should have raised a flag — it didn't, because the Δ was entirely a dimensional artifact.

Root cause: baseline probe used mean-pooled `node_features` of shape (N, 5), while substrate used mean-pooled `z_H` of shape (N, 512). Comparing 5-dim to 512-dim logistic regression separability is not a fair comparison — higher dimensions make any information linearly separable more easily (curse of dimensionality in reverse). The same 5-dim input information projected into 512-dim is more separable than in 5-dim, so AUC(z_H) > AUC(node_features) even when z_H is a random linear transformation of the inputs.

**Fix:** Project `node_features` from 5-dim to d_model-dim via a **fixed random matrix** (seed=0, unit-norm columns) before baseline probing. Both baseline and substrate now operate in d_model-dim space. The random projection is reproducible and committed alongside the probe code.

**Expected behavior after fix:**
- Random init: Δ ≈ 0 (substrate IS a random linear transformation of inputs, so its output ≈ random projection baseline)
- Trained substrate doing semantic work: Δ > 0 (training added structure beyond random projection)

**Change:** In `scripts/delta_probe.py` (and `delta_probe_act.py`), after pooling raw node_features, apply:
```python
torch.manual_seed(0)  # FIXED; never randomize this
proj = torch.randn(raw_dim, d_model)
proj = proj / proj.norm(dim=0, keepdim=True)
node_feats = (torch.from_numpy(node_feats_raw) @ proj).numpy()
```

**Path A threshold unchanged: Δ ≥ 0.10.** With dim-matching, random init is expected at Δ ≈ 0. A trained substrate that genuinely organizes carry states should produce Δ > 0.10.

### Amendment 2B: ACT-iterative probe variants

**Rationale:**

The trainer uses `CoralV3Inner` directly (1-step-gradient training, documented in `trainer.py`). The probes also call `CoralV3Inner`. So both training and measurement happen at single-step. However, `CoralV3ACT` allows 4 iterations of refinement (H→L→halt-check repeated 4 steps) at inference time — the substrate may have capacity to organize representations over 4 refinement steps even if the single-step gradient didn't drive it there.

Adding ACT-iterated probes allows distinguishing:
- **Training bottleneck hypothesis:** substrate CAN organize at 4-step inference (ACT probes show Δ > 0.10 and auc_shuffled > 0.65) but 1-step training didn't push it there. Fix: switch trainer to use ACT.
- **Architecture bottleneck hypothesis:** even 4-step ACT shows null result (Δ ≈ 0 and auc_shuffled ≈ 0.50). The substrate cannot organize from this task/training combination regardless of iteration count. Path B: deeper redesign.

**New scripts:**
- `scripts/delta_probe_act.py` — Δ-probe with ACT-iterated z_H
- `scripts/shuffled_probe_act.py` — shuffled-primitive probe with ACT-iterated z_H
- `scripts/check_act_iteration.py` — sanity check (run before probes to confirm ACT iterates and differs from single-step)
- `tests/test_substrate_probes_act.py` — unit tests (no GPU): iteration count, z_H differs, steps counter

**Path A criteria for ACT-iterated probes** (same thresholds as Amendment 1):
- Δ-probe ACT: delta ≥ 0.10
- Shuffled-probe ACT: auc_shuffled ≥ 0.65

**Scope note:** ACT-iterated probes are **comparison probes**, not replacements. The 1-step probes (delta_probe.py, shuffled_probe.py) remain the primary measurement — they measure exactly what training produced. ACT probes measure inference-time upper bound.

**Note on Vast runbook:** run `check_act_iteration.py` before the ACT probes. If it reports `[WARN] IDENTICAL z_H`, the ACT wrapper is not applying iteration correctly — do not proceed with ACT probes until resolved.

**Sign-off on Amendment 2:** Anwar Haq, Founder, Aktuator AI — 2026-05-17
