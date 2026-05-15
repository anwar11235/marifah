# Session 06 — Warm-Start Verdict (OD7)

**Status: PENDING** — awaiting Vast.ai execution results from user.

This document is populated after the user runs the Session 6 Vast.ai runbook
(`docs/operations/session06_runbook.md`) and shares results back to CC for analysis.

---

## Pre-registered decision criteria (from codebook design §2.6)

| Outcome | Decision |
|---------|---------|
| Warm matches OR exceeds Cold on **both** AUC and faithfulness | **Use warm-start for Phase 0** — saves training time, Sudoku transfer hypothesis confirmed |
| Warm lags Cold on workflow-type AUC by > 0.05 | **Use from-scratch for Phase 0** — Sudoku surface-form bias persists |
| Warm lags Cold on faithfulness but matches on AUC | **Use from-scratch for Phase 0** — basic execution mechanism affected |
| Mixed signals (warm better on one, worse on other) | **Use from-scratch as conservative default**; document the ambiguity |

---

## Results (TBD)

| Metric | Cold (from-scratch) | Warm (Sudoku Phase 3c) | Delta |
|--------|--------------------|-----------------------|-------|
| workflow_type_auc | TBD | TBD | TBD |
| mean_edit_distance | TBD | TBD | TBD |
| failure_rate | TBD | TBD | TBD |
| catastrophic_failure_rate | TBD | TBD | TBD |

---

## Verdict

**TBD** — pending results.

---

## Phase 0 config implication

- If warm-start verdict = **warm**: `configs/phase0.yaml` `warm_start.checkpoint` set to `checkpoints/sudoku-phase3c/phase3c_canonical_seed0_best.pt`, `load_optimizer: false`
- If warm-start verdict = **cold**: `configs/phase0.yaml` `warm_start.checkpoint` remains `null`

---

## Implications for the Marifah-class transfer story

*(Populated after verdict)*

- If warm-start wins: Strengthens the "Nous substrate transfers cleanly" claim — the same substrate trained on Sudoku constraint propagation immediately benefits DAG execution reasoning.
- If from-scratch wins: Substrate is more reasoning-task-specific than hoped. Transfer story scoped to "recognition cortex compounding at deployment" rather than "substrate transfer across reasoning domains."

---

*This document updated by CC after receiving results JSONs from user.*
