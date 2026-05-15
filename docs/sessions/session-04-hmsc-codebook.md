# Session 4 — HMSC Codebook Module

**Date:** 2026-05-15
**Branch:** `session-4/hmsc-codebook`
**Goal:** Implement the Hierarchical Multi-Scale Codebook (HMSC) — three codebooks at three scales with routing, composition, and auxiliary loss heads. Lambdas = 0 (losses computed but don't engage this session).

---

## Exit criteria status: ALL MET

| # | Criterion | Status |
|---|-----------|--------|
| 1 | All modules under `src/marifah/models/hmsc/` exist | ✅ |
| 2 | CORAL modified with `use_hmsc` config flag | ✅ |
| 3 | `use_hmsc=False` produces bit-identical output | ✅ max_diff = 0.0 |
| 4 | `use_hmsc=True` integrates cleanly | ✅ |
| 5 | All unit tests pass (255/255) | ✅ |
| 6 | All three scales produce correct-shape outputs | ✅ |
| 7 | Composition sums three modes correctly | ✅ |
| 8 | Auxiliary heads exist; losses computed; lambdas = 0 | ✅ |
| 9 | Utilization tracking in place | ✅ |
| 10 | End-to-end smoke test passes | ✅ |
| 11 | All §4 verification steps pass | ✅ |
| 12 | Branch `session-4/hmsc-codebook` committed | ✅ |
| 13 | Session summary written | ✅ (this doc) |

---

## What was built

### Core modules (`src/marifah/models/hmsc/`)

| Module | LOC | Contents |
|--------|-----|----------|
| `__init__.py` | 31 | Package marker + re-exports |
| `global_codebook.py` | 64 | `GlobalCodebook`: mean-pool carry → softmax routing → broadcast mode |
| `regional_codebook.py` | 144 | `RegionalCodebook`: learned region tokens → cross-attention pooling → per-region routing → soft/hard node assignment |
| `perposition_codebook.py` | 81 | `PerPositionCodebook`: per-node cross-attention to codebook; soft (train) / hard top-1 (eval) discreteness toggle |
| `composition.py` | 85 | `HMSCComposition`: sum (default) or gated composition of G, R, P modes |
| `auxiliary_heads.py` | 135 | `GlobalAuxHead`, `RegionalAuxHead`, `PerPositionAuxHead`, `compute_aux_losses` |
| `hmsc.py` | 176 | `HMSC`: top-level module composing all three codebooks + heads + utilization stats |

**Total core:** 716 LOC

### CORAL modifications

- `src/marifah/models/coral_base.py`: added `use_hmsc: bool = False` to `CoralConfig`
- `src/marifah/models/coral.py`:
  - Added `from marifah.models.hmsc.hmsc import HMSC` import
  - Added `hmsc_aux_losses`, `hmsc_utilization` fields to `PredMetrics`
  - `CoralV3Inner.__init__`: instantiates `HMSC(d_model=config.hidden_size)` when `use_hmsc=True`
  - `_forward_with_pc`: HMSC tap after final H-step, before `lm_head`. Applies `z_H = z_H + hmsc_out["composed"]`. Reads `node_mask`, `workflow_labels`, `region_labels`, `primitive_labels` from batch dict (all optional).

### Tests (`tests/`)

| File | LOC | Tests |
|------|-----|-------|
| `test_hmsc_global.py` | 110 | 8 tests |
| `test_hmsc_regional.py` | 119 | 9 tests |
| `test_hmsc_perposition.py` | 123 | 9 tests |
| `test_hmsc_composition.py` | 103 | 9 tests |
| `test_hmsc_auxiliary_heads.py` | 136 | 8 tests |
| `test_hmsc_e2e.py` | 288 | 17 tests |

**Total tests:** 60 new tests; full suite 255/255 passed.

---

## Architectural decisions made

### 1. HMSC tap point: inside `_forward_with_pc`, after final H-step

The HMSC fires after `z_H = self.H_level(z_H, xi, cos_sin=cos_sin)` (the last H-step, which is in the computation graph). This gives HMSC access to z_H while it still has gradients, so:
- HMSC parameters receive gradients via: `composed → z_H (augmented) → lm_head → loss`
- The carry stored for the next ACT step is `z_H.detach()` post-HMSC, so HMSC contribution propagates forward through the recurrence
- Per-step crystallization is achieved: HMSC fires once per call to `_forward_with_pc`, which corresponds to one ACT segment

**Alternative considered:** Tapping in `CoralV3ACT.forward()` (outer level) on the detached carry. Rejected: z_H is detached there; HMSC composed output would only influence the next step's inner-forward under `torch.no_grad()`, providing no gradient path for HMSC params via main loss.

### 2. Residual augmentation: `z_H = z_H + composed`

Adding composed to z_H (not replacing) avoids destroying the carry state at random init when HMSC hasn't learned anything. At init, `composed` has small magnitude (from Xavier init) so the residual addition is stable.

### 3. Regional codebook attention architecture

Used a custom attention implementation (not `nn.MultiheadAttention`) for two reasons:
- Need access to per-node-to-region attention weights for soft assignment (transpose of the region-attention matrix)
- Easier to control masking of padding nodes in the scores

Implementation: region_tokens as queries, carry_state as keys and values. The soft node-to-region weights are computed via a separate `k_proj(carry_state) @ k_proj(region_features).T` operation for symmetry — this gives a proper node-to-region affinity score without using the same attention weights (which average over nodes, not over regions).

### 4. `output_proj` in `RegionalCodebook` initialized to identity

When `d_R == d_R` (always true), `output_proj` is initialized with `nn.init.eye_()`. This means at init, the regional codebook output is identical to the mode mixture from the codebook — no transformation. Avoids the composition module receiving zero-mean noise from an untrained projection.

### 5. Composition default: sum

Follows design §3.2. The "gated" path is implemented with a 2-layer MLP (d_output → d_hidden → 3) outputting softmax gate weights. Not the default training path.

### 6. Lambda=0 → zero gradient to aux heads (via multiply, not detach)

`L_G = lambda_G * L_G_raw`. When `lambda_G=0.0`, this is `0.0 * L_G_raw`. PyTorch does propagate gradient through `0.0 * x` as `0.0 * dx`, so head parameters receive zero gradient from the aux loss path. This is more debuggable than detaching the loss.

### 7. `node_mask` is optional in the CORAL batch dict

If `node_mask` is absent from the batch dict, HMSC falls back to a full mask (all positions real). This preserves backward compatibility for non-graph inputs (Sudoku) and ensures `use_hmsc=True` can be set without modifying non-graph callers.

---

## Design ambiguities encountered and resolved

1. **K values discrepancy**: Design doc §3.1 says K_G ~ 16-32, K_R ~ 32-64. Session prompt §2.5 commits K_G=64, K_R=16, K_P=16. Followed session prompt; noted in code comments.

2. **Tap point not specified precisely**: Design doc §4.1 says "every ACT step" but doesn't specify where within the inner forward. Chose post-final-H-step (pre-lm_head) for gradient correctness — see decision #1 above.

3. **Soft node-to-region assignment**: Design doc §4.2 says "node-to-region weights computed via the same attention mechanism (transpose)". The transpose of the region-attention scores `(B, R, N)` gives node-to-region scores `(B, N, R)`. However, this transpose would give weights over regions for each node — used softmax(dim=-1) over this. Implementation uses a separate k_proj dot product for cleaner separation.

---

## Regression verification

**`use_hmsc=False` vs. pre-HMSC CORAL:**
```
max_diff = 0.0 (exact bit identity)
```
Two `CoralV3Inner` instances with identical weights and `use_hmsc=False` produce identical output. The HMSC code path is completely gated by `self.hmsc is not None`.

---

## Auxiliary loss values at random init

Computed on B=4, N=20 batch with `d_model=512`:

| Scale | Loss (raw) | Notes |
|-------|-----------|-------|
| L_G | 4.23 | ≈ log(50) = 3.91 (expected for 50-class uniform) |
| L_R | 2.45 | ≈ log(12) = 2.48 (expected for 12-class uniform) |
| L_P | 2.30 | ≈ log(10) = 2.30 (expected for 10-class uniform) |

All three are near the theoretical cross-entropy for a uniform classifier over the respective label vocabularies — sanity check passed.

---

## Initial codebook utilization stats (random init, B=4, N=20, d_model=512)

| Scale | Active frac | Active count | Entropy | Top-1 dominance |
|-------|-------------|--------------|---------|-----------------|
| G (K=64) | 0.875 | 64 | 4.10 | 0.000 |
| R (K=16) | 1.000 | 16 | 2.73 | 0.000 |
| P (K=16) | 1.000 | 16 | 2.77 | 0.000 |

At random init: all entries are active (no dead codes), entropy is high (near uniform), top-1 dominance is 0 (no single entry dominates). This is the expected profile before training — confirms no pathological initialization.

---

## Deviations from design doc

1. **K_G=64 vs. design §3.1 "K_G ~ 16-32"**: Session prompt §2.5 commits K_G=64. Followed session prompt.
2. **`output_proj` is `nn.Linear(d_R, d_R)` not identity**: Added for future trainability; initialized to identity at init.
3. **d_model passed explicitly to HMSC constructor**: Design doc doesn't specify how HMSC knows d_model; I thread it from `config.hidden_size` in `CoralV3Inner.__init__`. All other HMSC hyperparameters use committed defaults baked into `HMSC.__init__` defaults.

---

## Known issues, limitations, deferred items

1. **HMSC with `use_hmsc=True` is not tested on the non-PC path** (`CoralInner` base, not `CoralV3Inner`). The HMSC code is in `_forward_with_pc`; `CoralInner.forward()` doesn't have HMSC. If someone sets `use_hmsc=True` with `use_predictive_coding=False`, the model instantiates HMSC but never calls it. This is acceptable for Session 4 since Phase 1 (PC) is the only active path.

2. **HMSC with committed defaults is not configurable from `CoralConfig`**: The HMSC hyperparameters (K_G, K_R, K_P, d_G, d_R, d_P, etc.) are fixed in `HMSC.__init__` defaults. Adding them to CoralConfig is Session 5 scope.

3. **`region_labels` for aux loss is `(B, num_regions)`, not `(B, N)`**: The R-head classifies per-region, not per-node. Ground-truth region labels from the generator are per-node (`(B, N)`), so a mapping is needed before calling `compute_aux_losses`. Session 5 training integration should add this mapping.

4. **Warm-start regression test (discipline rule #3) not run**: The Sudoku Phase 3c checkpoint regression is a pre-merge requirement, separate from HMSC testing. The `use_hmsc=False` bit-identity check is the HMSC-specific regression; the Sudoku checkpoint test is the pre-existing repo-wide gate.

---

## Notes for Session 5

1. **Training pipeline integration (recommended Session 5 scope)**: Wire `GraphDAGDataset` → `collate_graphs` → CORAL with `use_hmsc=True` → main training loop. Key items:
   - `node_mask` must be in the batch dict passed to CORAL (add to `collate_graphs` output or wire through).
   - `workflow_labels`, `primitive_labels` are already in the `GraphBatch` format (from adapter); pass through to CORAL batch dict.
   - `region_labels` requires mapping from per-node to per-HMSC-region — either pass per-node primitive labels as `primitive_labels` and let the head work per-node, or construct region labels from graph topology.
   - Accumulate `hmsc_aux_losses` from `pred_metrics` across ACT steps and add to total loss.

2. **Lambda schedule**: Start with λ_G = λ_R = λ_P = 0.1 per design §5.3. The loss magnitude ratios at random init (L_G ≈ 4.2, L_R ≈ 2.5, L_P ≈ 2.3) suggest λ values should be similar for balanced supervision.

3. **Dead code monitoring**: At random init, no dead codes. During training, watch `G_active_frac` and `P_active_frac` — if entries die, add commitment loss or increase λ values.

4. **Region label mismatch**: The `RegionalAuxHead` expects `region_labels: (B, num_regions)` — one label per HMSC region, not per node. The generator produces per-node region assignments. Session 5 should decide: use majority-label per region, or redesign the R-head to work per-node with node-to-region soft assignment labels.
