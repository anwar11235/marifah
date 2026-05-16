# Session 06 — Generator Workflow-Type Distribution Finding

**Status: FIXED** — 1-line config change (`ood_holdout_seed: 0 → 13`). See §Fix below.

---

## Root cause

**Type 2: Filter-driven extinction.**

The OOD-composition holdout filter (which rejects training DAGs whose cross-pattern boundary edges contain any reserved primitive pair) drove 40 of 50 workflow types to extinction. Those 40 types could not generate any valid training DAG in 20 retries, producing only 10 surviving types — exactly the 10 observed in the Vast dataset.

---

## Evidence

**Diagnostic 1 — Per-workflow pass rate (100 attempts per type, seed=0):**

| Category | Workflow count | Example IDs |
|----------|---------------|-------------|
| Pass rate = 100% | 2 | 1, 17 |
| Pass rate 5–50% | 8 | 3, 4, 5, 7, 11, 26, 35, 44 |
| Pass rate = 0% (extinct) | 40 | 2, 6, 8–10, 12–16, 18–25, 27–34, 36–43, 45–50 |

The 10 surviving types (pass rate > 0%) match **exactly** the 10 types observed in the Vast training distribution: `{1, 3, 4, 5, 7, 11, 17, 26, 35, 44}`.

**Diagnostic 2 — Boundary pair coverage:**

Only 33 of 100 primitive pairs ever appear at cross-pattern boundaries. The current reserved set (seed=0) includes **5 pairs that are also boundary pairs**:
- `(4, 9)` — appears in 43 of 50 workflow types
- `(9, 7)` — appears in 29 of 50 workflow types
- `(2, 7)` — appears in 11 of 50 workflow types
- `(4, 4)` — appears in 9 of 50 workflow types
- `(8, 8)` — appears in 2 of 50 workflow types

These 5 pairs, especially `(4,9)`, cover almost all 50 workflow types. Any workflow whose cross-pattern boundaries always include at least one of these 5 pairs cannot pass the training filter regardless of retries.

**Diagnostic 3 — Val/test splits (no training OOD filter):**

Val, test_id, and test_ood_size splits use weight-based random sampling with no per-DAG OOD filter. They sample from all 50 workflow types. This creates a **train–val distribution mismatch**: model trains on 10 types, is evaluated on 36+ types.

**Diagnostic 4 — Budget impact:**

The 40 extinct workflow types represent 331,750 / 782,750 = **42.4% of the training instance budget** that was allocated but generated 0 records. This reduced the observable train set from ~800K to ~448K DAGs.

---

## What was NOT the root cause

- **Not a sampler bug (type 1):** `_build_train_tasks` correctly creates tasks for all 50 workflow types with their full instance counts.
- **Not a spec mismatch (type 3):** `WORKFLOW_INSTANCES_MAP` has all 50 types with non-zero counts.
- **Not a code bug:** The OOD filter logic itself is correct. The problem is the random seed used to select reserved pairs produced a set that overlaps destructively with cross-pattern boundaries.

---

## Fix

**Change `ood_holdout_seed` from 0 to 13 in `configs/default.yaml`.**

With seed=13, the reserved pairs are: `{(0,6),(0,7),(1,5),(2,4),(5,9),(6,0),(7,2),(7,3),(7,5),(7,6),(7,7),(7,8),(8,6),(9,1),(9,5)}`.

Only 2 of these are boundary pairs: `(7,2)` [appears in 2 workflow types] and `(7,8)` [appears in 1 workflow type].

**Post-fix per-workflow pass rate (100 attempts, seed=13):**

| Category | Count |
|----------|-------|
| Pass rate ≥ 50% (surviving) | 49 |
| Pass rate = 0% (extinct) | 1 (wf_id=27 only) |

wf_id=27 is extinct from training but is naturally suitable for test_ood_composition (it always has a reserved boundary pair and thus satisfies `require_reserved_pair=True`). The test_ood_composition split of 5,000 DAGs is fully generatable from wf_id=27 alone.

**Why this seed was chosen:** Systematic search over 500 seeds optimizing for (OOD-capable types) − 10 × (extinct types). Seed=13 gives 49 surviving types with only 1 extinct — the best balance found.

---

## Per-split distribution (local tiny dataset, pre-fix)

| Split | Unique workflow types | Notes |
|-------|----------------------|-------|
| train | 1 (wf_id=1 only) | Tiny config truncates tasks to 963 — all from wf_id=1's 100K task block |
| val | 9 | No OOD filter; weight-based random sampling |
| test_id | 9 | Same |
| test_ood_size | 8 | Same |
| test_ood_composition | 8 | Requires reserved pair — wf_ids 2, 3, 4, 6 etc always have them |

Note: the tiny dataset's train showing only 1 type is a separate issue from the full dataset's 10-type result. In the tiny config (963 train records), `_build_train_tasks` creates 782,750 tasks in order (all wf_id=1 tasks first); with n=963 the truncation never reaches wf_id=2+. At full scale (800K), all 50 types have tasks submitted, but 40 fail the OOD filter.

---

## Should the test suite have caught this?

Yes. The test suite asserts record counts and distribution shape but does NOT assert unique workflow-type count post-generation. The tiny-config test generates too few records to see all 50 types even with a correct generator (very rare types have 50 instances; tiny config scales them to ~1 which rounds away). A proper test would need a moderate-size config with enough instances per type to guarantee they appear.

A distribution test was added in the fix commit: `test_train_split_workflow_type_coverage` — generates 5K train records and asserts ≥ 45 unique workflow types (leaves margin for stochastic variation and the 1 naturally extinct type).

---

## Fix complexity

**Minimal** — 1 line in `configs/default.yaml`, 1 new test. No code changes to generator, sampler, or filter logic. Holdout fraction unchanged at 15%.

---

## Verification (post-fix)

Generated 5K-record moderate local dataset and confirmed:
- 49 unique workflow types in train (vs 10 pre-fix)
- Log10 spread ≈ 3.0 (max/min ≈ 1000)
- All 291 + 1 new tests pass

(Results appended after fix commit.)
