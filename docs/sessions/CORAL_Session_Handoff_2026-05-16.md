# Session Handoff: Marifah / CORAL Build Sessions 1–6

**Date:** 2026-05-16
**Last active session:** Session 6 (Phase 0 Prep) — in progress, on Vast.ai
**Repo:** `marifah-core` (local-only, no remote configured yet)
**Active branch:** `session-6/phase0-prep`
**Vast instance:** A100 SXM4 40GB, accessed at `ssh -p 13595 root@212.13.234.23 -L 8080:localhost:8080`
**Working directory on Vast:** `/workspace/marifah` (note: not `/workspace/marifah-core`)

---

## Strategic context (don't re-derive)

**The pivot (2026-05-13).** ARC validation track formally CLOSED (Phase 0 verdict reached; eval-wedge was resolved without disable-compile). Marifah pivoted to **graph DAG execution for commercial enterprise workflows.** 16-week horizon to fundraise-ready validation.

**Vocabulary (canonical, drawn from `Marifah_Naming_and_Taxonomy.md`):**
- **Marifah** — company name + Recognition Cortex mechanism
- **Nous** — Reasoning Cortex substrate (the CORAL substrate code)
- **CORAL** — first architecture in the family = Cortical Reasoning via Abstraction Layers
- **CRA** — Cortical Reasoning Architecture (category, sits alongside Connectionism / Symbolic / NSAI)
- **CWA** — Cognitive Workflow Automation (customer-facing product category)
- **Taxonomy:** ʿIlm-class (LLMs) / Nous-class (reasoning archs) / Marifah-class (recognition-based compounding)

**Recognition Cortex commitment (Option 3, 2026-05-13).** Recognition Cortex MVP deferred to Weeks 11–13 (demo / deck-rebuild weeks). Not production grade; minimal demo showing H-state cache + learned head doing online compounding on synthetic recurring workflows. HMSC (training-time crystallization) is implemented; Recognition Cortex (deployment-time compounding) is the separate layer that stacks on top per codebook design §1.

**Encoder layer (NL → structured DAG).** Deferred to Layer C scoping (Week 8+). LLM-based, runs once per SOP, amortized. Positioning is "no LLM in the runtime execution path" — not "no LLM anywhere." Not the moat; CORAL is.

---

## Build state — Sessions 1–5 (all merged to main)

**Session 1 — Repo Bootstrap (complete).** Ported CORAL substrate from CORAL-v3: PC mechanism, ACT loop, codebook scaffolding (SpatialMoECodebook stays as scaffolding; gets replaced by HMSC in Session 4), transformer block, layers, training loop. NOT ported: ARC adapter, ConceptARC, v2 metric-shape, R0/R0.5 probes. Sudoku Phase 3c checkpoint at `checkpoints/sudoku-phase3c/phase3c_canonical_seed0_best.pt` (116 MB). CORAL-v3 tagged `v3.0-pre-marifah-pivot`. CLAUDE.md established with discipline rules.

**Session 2 — Synthetic Generator (complete).** 130/130 tests. ~1K DAGs across 5 splits generated in ~2s. Byte-identical determinism, ~950 DAGs/sec single-threaded. 7 bugs fixed during build, two notable: cross-pattern boundary OOD filter rejecting nearly 100% of training DAGs (fixed but residual rejection ~44% still present — see "Known issues" below), and JSON integer-key round-trip corruption in LOOKUP tables (fixed via parquet for typed-key dicts).

**Session 3 — Graph Adapter (complete).** 23 new files, 195/195 tests. SDPA vs flash_varlen equivalence: **max_diff=0.0 bit-identical.** Salvaged from CORAL-v3: additive-bias attention mask infrastructure (arc/padding-attention-mask c7e784d), flash-attn-varlen path (arc/flash-attn-varlen 28af53a), dataloader underfull-batch fix (arc/phase0-config 7367d6e). Laplacian PE top-8, edge-induced directed attention default. Modules at `src/marifah/data/adapter/`.

**Session 4 — HMSC Codebook (complete).** 7 modules at `src/marifah/models/hmsc/`. 60 new tests, 255/255 total. **Critical regression check: `use_hmsc=False` max_diff=0.0 with pre-HMSC CORAL** (bit-identical regression preserved). Auxiliary losses at random init: L_G=4.23, L_R=2.45, L_P=2.30 (≈ log(vocab_size), correct). Utilization at init: G 88%, R 100%, P 100% active (no dead codes). HMSC taps z_H after final H-step via residual addition in `_forward_with_pc` (`coral.py` lines 294–317) — **per-ACT-segment tap topology confirmed matching spec §4.** Verified: HMSC.forward fires once per ACT segment (each `CoralV3ACT.forward()` call), not once per H-step or once per full forward pass.

**Session 5 — Training Pipeline (complete).** 9 modules at `src/marifah/training/`. 4 config files (`configs/phase0.yaml`, `configs/phase1.yaml`, `configs/smoke.yaml`, `configs/smoke_hmsc_off.yaml`). 280/280 tests (25 new). Smoke A (HMSC off): main loss 1.68 → 0.08 over 3 epochs. Smoke B (HMSC on): main 1.69 → 0.11, aux losses 0.85 → 0.71 (decreasing). Regression check: max_diff=0.0. Key decisions: HMSC attached post-init with training lambdas baked in; eval_interval enforced in EPOCHS not steps; 1-step gradient training with fresh carry per batch; `derive_region_labels` uses majority-vote heuristic mapping per-node pattern_ids to per-region slots.

---

## Session 6 — Phase 0 Prep (in progress)

### Goal
Container verification, full synthetic dataset generation, warm-start comparison (codebook OD7 resolution), Phase 0 launch config finalization. **Does not** launch Phase 0 — that's Session 7.

### Current state (as of handoff)

**Completed:**
- Branch `session-6/phase0-prep` created and pushed to origin (wait, no remote — local-only; pushed to local branch)
- Configs `warmstart_cold.yaml` and `warmstart_warm.yaml` created
- `scripts/warmstart_probe.py` created  
- Runbook `docs/operations/session06_runbook.md` written (parallel + sequential paths documented)
- Container image documentation drafted at `docs/operations/container_image.md`
- Container deps verified on Vast: torch 2.6.0+cu124, flash-attn 2.7.4.post1, all dep imports succeed (scipy was missing from `requirements.txt`; installed at runtime — needs commit as follow-up)
- Sudoku checkpoint copied to Vast at `/workspace/marifah/checkpoints/sudoku-phase3c/phase3c_canonical_seed0_best.pt`
- `configs/warmstart_warm.yaml` points to correct checkpoint path
- **Two synthetic-generator bugs discovered and fixed mid-Session 6** — see "Bug fixes during Session 6" below
- Full dataset generated on Vast: 478,188 DAGs total in 423.7s, at `/workspace/data/marifah_full_dataset`

**In progress:**
- Frequency distribution verification (just ran the diagnostic)
- Awaiting paste of distribution output to decide if warm-start launch is approved

**Not yet done (remaining Session 6 work):**
1. Verify frequency distribution from data (computed, not from manifest — manifest doesn't track it)
2. Run `validate-dataset` for well-formedness check
3. Launch cold warm-start run: `python -m marifah.training.cli train --config configs/warmstart_cold.yaml`
4. Launch warm warm-start run (after cold completes): `python -m marifah.training.cli train --config configs/warmstart_warm.yaml`
5. Run probes on both checkpoints via `scripts/warmstart_probe.py`
6. Share probe result JSONs back to local
7. **Claude analyzes:** write `docs/sessions/session-06-warmstart-verdict.md` per pre-registered §2.6 decision criteria
8. Finalize `configs/phase0.yaml` based on verdict
9. Container image doc updated with validation outcome at `docs/operations/container_image.md`
10. Pre-launch checklist completed at `docs/sessions/session-06-prep-complete.md`
11. Session summary written at `docs/sessions/session-06-phase0-prep.md`
12. Commit scipy addition to `requirements.txt` (small fix)
13. **THEN** merge `session-6/phase0-prep` → `main`
14. **THEN** Session 7 = Phase 0 main launch (multi-day; ~100K steps; decision gate Path A/B/C per codebook §7)

### Pre-registered warm-start decision criteria (do not modify after seeing results)

| Outcome | Decision |
|---|---|
| Warm matches OR exceeds Cold on **both** AUC and faithfulness | Use warm-start for Phase 0 |
| Warm lags Cold on workflow-type AUC by > 0.05 | Use from-scratch for Phase 0 |
| Warm lags Cold on faithfulness but matches on AUC | Use from-scratch (basic execution mechanism affected) |
| Mixed signals (warm better on one, worse on other) | Use from-scratch (conservative default) |

### Bug fixes during Session 6 (CC_Fix_Synthetic_Generator_Performance.md handed off and completed)

**Bug 1:** Generator defaulted to `--workers=1` (single-threaded). Fixed: now defaults to `max(1, os.cpu_count() - 1)` (12 local, 63 Vast).

**Bug 2:** Generator buffered all records per split before any disk writes. Fixed: streaming shard writes via `generate_split_streaming` yielding 10K-record batches. Used `imap` (ordered) not `imap_unordered` to preserve per-shard determinism and manifest hash verification.

**Result:** 291/291 tests pass. Pushed locally to `session-6/phase0-prep`. Dataset regeneration succeeded.

### Known issues surfaced in Session 6 (capture in session summary)

1. **Train split size = 448K vs expected 800K (56% of target).** Likely the OOD-composition holdout filter still rejecting ~44% of training DAGs that use any of the 15% reserved primitive pairs. Math: `1 - 0.85^4 ≈ 48%` rejection rate for DAGs with ~4 primitive pairs each. The Session 2 fix addressed 100% rejection but didn't fully optimize the holdout strategy. **For warm-start (5K steps × batch 64 = 320K examples), 448K is sufficient.** Worth fixing before Phase 0 main run if practical.

2. **Throughput on 63 workers ~1130 DAGs/sec vs single-threaded ~950.** Only ~18% improvement from 63 workers — parallelism not engaging effectively. Possible cause: serial overhead in rejection logic, or `imap` ordering forcing serial output collection. Worth investigating but not blocking.

3. **`load_manifest` requires `Path`, not string — API papercut.** Easy CC follow-up.

4. **Manifest doesn't track `frequency_distribution`.** Spec §4.2's log-spaced distribution is critical for A4 compounding probe; not surfaced anywhere in the manifest. Worth adding to generator manifest emission. Workaround: compute from data directly (currently running this diagnostic).

5. **`scipy` was missing from `requirements.txt`.** Used by `src/marifah/data/adapter/positional.py` (Laplacian PE). Runtime-installed on Vast; needs commit to `requirements.txt`.

---

## Discipline rules (active across sessions)

1. Two consecutive evals same direction before treating trajectory as real
2. Two consecutive same-signature failures = diagnose mode (laptop), not another knob turn
3. When fixing a bug in a new component, audit adjacent same-class issues
4. Hold conclusions against outcome metrics, not intermediate signals
5. No merge to main without warm-start regression test
6. **Tests passing ≠ launch-ready** — verify: branch ancestry, configs against trainer semantics, test fixtures match real data shapes
7. Read trainer eval semantics before launching (`eval_interval` is in EPOCHS not steps)
8. Pre-register kill criteria before runs
9. ARC closed at Phase 0 verdict regardless of outcome (no reopening); one validation track at a time post-Phase-0

---

## Operational state on Vast

**Instance:** A100 SXM4 40GB at `ssh -p 13595 root@212.13.234.23`
**Image:** `anwar1919/coral-v3:2026-04-20` with runtime `pip install -r requirements.txt --break-system-packages`
**Repo path:** `/workspace/marifah` (not marifah-core; user clones differently)
**Branch checked out:** `session-6/phase0-prep`
**Dataset:** `/workspace/data/marifah_full_dataset/` — 478K DAGs total, 45 train shards
**Sudoku checkpoint:** `/workspace/marifah/checkpoints/sudoku-phase3c/phase3c_canonical_seed0_best.pt`

**Earlier failures observed:**
- 2-GPU Quebec instance attempted, failed — fell back to single SXM4 40GB
- Alberta instance (prior experience): broken, never pulls container properly — avoid

---

## Immediate next steps when resuming

1. **Run frequency-distribution diagnostic and validate-dataset on Vast.** Commands (already provided in last message):

```bash
python -c "
from pathlib import Path
from collections import Counter
import math
import pyarrow.parquet as pq
from marifah.data.synthetic.storage import load_manifest

m = load_manifest(Path('/workspace/data/marifah_full_dataset'))
print('=== Manifest splits info ===')
print(m.get('splits', 'N/A'))
print()
print('=== Computed workflow distribution (train split) ===')
train_dir = Path('/workspace/data/marifah_full_dataset/train')
counter = Counter()
for shard in sorted(train_dir.glob('shard_*.parquet')):
    t = pq.read_table(shard, columns=['workflow_type_id'])
    counter.update(t['workflow_type_id'].to_pylist())

print(f'Total DAGs: {sum(counter.values())}')
print(f'Unique workflow_type_ids: {len(counter)}')

sorted_counts = sorted(counter.items(), key=lambda x: -x[1])
print()
print('Top 10:')
for wf, n in sorted_counts[:10]:
    print(f'  workflow_type_id={wf}: {n}')
print('Bottom 10:')
for wf, n in sorted_counts[-10:]:
    print(f'  workflow_type_id={wf}: {n}')

counts = sorted([n for _, n in sorted_counts])
print()
print(f'Max count: {counts[-1]}')
print(f'Min count: {counts[0]}')
print(f'Ratio (max/min): {counts[-1] / counts[0]:.1f}')
print(f'Log10 spread: {math.log10(counts[-1] / counts[0]):.2f}')
"

python -m marifah.data.synthetic.cli validate-dataset /workspace/data/marifah_full_dataset
```

Expected: ~50 unique workflow_type_ids, log10 spread ~3 (5 tiers spanning ~3 orders of magnitude), validate-dataset passes.

2. **Launch cold warm-start run (sequential, single GPU):**

```bash
cd /workspace/marifah
python -m marifah.training.cli train --config configs/warmstart_cold.yaml
```

Expected: ~1–2 hours on A100 SXM4 40GB for 5K steps.

3. **Launch warm warm-start run (after cold completes):**

```bash
python -m marifah.training.cli train --config configs/warmstart_warm.yaml
```

4. **Run probes:**

```bash
python scripts/warmstart_probe.py --checkpoint checkpoints/warmstart_cold/final.pt \
    --dataset /workspace/data/marifah_full_dataset \
    --split val \
    --output results/cold_results.json

python scripts/warmstart_probe.py --checkpoint checkpoints/warmstart_warm/final.pt \
    --dataset /workspace/data/marifah_full_dataset \
    --split val \
    --output results/warm_results.json
```

5. **Transfer results back to local** (rsync or scp). Share JSONs with Claude.

6. **Claude writes warm-start verdict** per pre-registered §2.6 decision criteria.

7. **Finalize Phase 0 config and complete remaining Session 6 work.**

---

## Documents and prompts produced this conversation

All in `/mnt/user-data/outputs/` of the Claude conversation (Anwar should download/save):

1. `Marifah_Naming_and_Taxonomy.md` — vocabulary source of truth (~2,400 words)
2. `CC_Session_01_Repo_Bootstrap.md` — handed off, completed
3. `CC_Session_02_Synthetic_Generator.md` — handed off, completed
4. `CC_Session_03_Graph_Adapter.md` — handed off, completed
5. `CC_Session_04_HMSC_Codebook.md` — handed off, completed
6. `CC_Session_05_Training_Pipeline.md` — handed off, completed
7. `CC_Session_06_Phase0_Prep_v2.md` — handed off, in progress
8. `CC_Task_Session6_Parallel_Runbook.md` — handed off, completed (then 2-GPU pivot moot)
9. `CC_Fix_Synthetic_Generator_Performance.md` — handed off, completed (bug fix landed)

---

## Resume protocol (for new conversation)

1. Start new conversation with this handoff doc
2. Paste output of frequency distribution diagnostic + validate-dataset (if not already pasted)
3. Confirm whether to proceed with warm-start launch
4. From there, the workflow continues as documented in §"Immediate next steps"

When the warm-start runs complete and results are shared back, the next major task is the warm-start verdict + Phase 0 config finalization + session summary writing + merge to main + Session 7 prompt drafting.

---

*End of handoff.*
