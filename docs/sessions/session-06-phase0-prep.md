# Session 06 — Phase 0 Pre-Launch Preparation

**Date:** 2026-05-15
**Branch:** `session-6/phase0-prep`
**Goal:** Everything required before Phase 0 main training launch — container image documentation, warm-start comparison setup (resolves OD7), runbook for Vast.ai execution, and trainer `max_steps` support.

---

## What was built

### Code changes

| File | Change | Notes |
|------|--------|-------|
| `src/marifah/training/config.py` | Added `max_steps: Optional[int] = None` to `TrainingPhaseConfig` | Hard step cap for comparison runs; exits at whichever of `max_steps` / `max_epochs` is hit first |
| `src/marifah/training/trainer.py` | Wired `max_steps` into `Trainer.train()` inner loop | Sets `_stop_training = True` when `global_step >= max_steps`; breaks inner+outer loop cleanly |
| `configs/phase0.yaml` | Fixed: moved `drop_last: true` from `training:` to `data:` section | It was silently ignored before (wrong section); now reads correctly from `DataConfig` |

### New files

| File | Contents |
|------|----------|
| `configs/warmstart_cold.yaml` | Warm-start comparison cold run (from-scratch init); max_steps=5000, eval_interval_epochs=1 |
| `configs/warmstart_warm.yaml` | Warm-start comparison warm run (Sudoku Phase 3c init); identical except `warm_start.checkpoint` set |
| `scripts/warmstart_probe.py` | ~200 LOC probe script: workflow-type AUC + execution faithfulness; emits results JSON |
| `docs/operations/container_image.md` | Container image verification docs; dep list; Vast.ai provisioning workflow |
| `docs/operations/session06_runbook.md` | User-facing runbook for all Vast.ai work (verification + dataset generation + comparison runs + probe runs) |
| `docs/sessions/session-06-warmstart-verdict.md` | Placeholder — populated after user executes the Vast.ai runbook and shares results back |

---

## Configuration decisions

- **`max_steps` implementation:** Added as an optional per-run override, not a permanent default. Phase 0 and Phase 1 configs use only `max_epochs`. The warmstart comparison configs set `max_steps=5000` to enforce the OD7 comparison protocol (5K steps each, one variable: initialization).
- **Warm-start checkpoint path:** The actual file is `checkpoints/sudoku-phase3c/phase3c_canonical_seed0_best.pt` (not `best.pt` as written in the prompt). The config references the correct path.
- **`load_optimizer: false` on warm run:** Resets the optimizer so the comparison tests only the effect of initialization on learning dynamics, not Sudoku optimization trajectory.
- **Probe: pooled carry state for AUC:** Mean-pools `z_H` over real (non-padded) nodes, masked by `node_mask`. This is the "pooled readout" referenced in codebook design §7.1.
- **Probe: faithfulness as primitive-id edit distance:** Predicted trace = argmax of model logits at each real node. Ground truth = `primitive_assignments` labels. Standard Levenshtein distance, normalized by trace length.

---

## CC-local verification results (§4, items 1–6)

| # | Check | Result |
|---|-------|--------|
| 1 | Sudoku Phase 3c checkpoint intact and loadable | ✅ OrderedDict, 51 keys, loads via `torch.load` |
| 2 | `warmstart_cold.yaml` parses correctly | ✅ `max_steps=5000`, `warm_start.checkpoint=None` |
| 3 | `warmstart_warm.yaml` parses correctly | ✅ `max_steps=5000`, `warm_start.checkpoint=...phase3c...` |
| 4 | Probe runs end-to-end on small input | ✅ Sudoku ckpt + tiny val (20 samples): AUC=0.667, mean_ED=0.896, no errors |
| 5 | `pytest tests/` passes with trainer changes | ✅ 280/280 |
| 6 | `container_image.md` and `session06_runbook.md` complete | ✅ |

---

## Pending (awaiting user's Vast.ai execution)

Items 7–11 of the §4 verification sequence, the warm-start verdict, and Phase 0 config finalization all require the user to execute the §2.4 runbook on Vast.ai and share results back. These are tracked in `docs/sessions/session-06-warmstart-verdict.md` (currently a placeholder).

**Pre-launch checklist status:**

| Item | Status |
|------|--------|
| Container image verified on Vast.ai | ⏳ Pending user execution |
| Full synthetic dataset generated + validated | ⏳ Pending user execution |
| Workflow-frequency distribution correct | ⏳ Pending user execution |
| Warm-start comparison executed (both runs) | ⏳ Pending user execution |
| Warm-start verdict documented | ⏳ Pending results |
| Phase 0 config finalized per verdict | ⏳ Pending verdict |
| Session 6 runbook validated by execution | ⏳ Pending user execution |
| Sudoku Phase 3c checkpoint on Vast.ai | ⏳ Pending rsync |
| W&B project `marifah-core` configured | ⏳ Pending `wandb login` on instance |

---

## Generator performance fix (discovered mid-Session 6)

Two bugs were found during Vast.ai execution and fixed before the runbook was re-executed:

**Bug 1 — `--workers` defaulted to 1.** On a 64-core Vast.ai instance this left 63 cores idle, producing ~14 min generation time instead of ~30 seconds. Fixed: `--workers` now defaults to `max(1, os.cpu_count() - 1)` for both `generate-tiny` and `generate-full`.

**Bug 2 — Per-split in-memory buffering before any shard write.** The old `_generate()` called `generate_train()` etc., which accumulated all records in RAM before writing a single shard. On a 800K-record full dataset this meant no disk activity for 11+ minutes. Fixed: `_generate()` now calls the new `SplitGenerator.generate_split_streaming()` method, which yields batches of 10 000 records; each batch is written to disk immediately as `shard_NNNN.parquet`. Shards appear on disk within seconds of generation starting.

**New API (all backwards-compatible additions):**
- `storage.write_shard(records, shard_path)` — write a single shard
- `storage.write_manifest_from_counts(output_dir, config, split_counts, split_shard_paths)` — manifest without holding all records in memory
- `generator.DagGenerator.generate_split_streaming(...)` — yields batches; uses `mp.Pool.imap` (ordered) for determinism
- `SplitGenerator.generate_split_streaming(split_name, batch_size)` — delegates to DagGenerator

**Determinism contract:** `imap` (ordered, not `imap_unordered`) is used, so per-shard contents are byte-identical across runs with the same seed and `num_workers`. `verify_manifest` shard-hash checks still pass.

**Verification:** 291/291 tests pass (11 new tests in `tests/test_synthetic_generator_performance.py`). `generate-tiny` + `validate-dataset` smoke test: PASSED in 1.1 s.

---

## Known issues / notes for user execution

- **GPU memory not profiled locally** (no GPU on dev machine). Warm-start comparison configs use `batch_size=64, d_model=512, H_layers=2` — same as Phase 0. If OOM occurs, reduce batch_size to 32 and report the observation so Phase 0 config can be adjusted.
- **Dataset path in configs is `/workspace/data/marifah_full_dataset`** — this is the expected Vast.ai path. Do not change it locally.
- **WANDB_MODE** can be set to `offline` if W&B connectivity is problematic on the instance; sync after training with `wandb sync`.
- **Generator `--workers` flag**: recommend `--workers 8` for dataset generation on server-grade CPUs. Report actual throughput (DAGs/sec) for Session 7 planning.

---

## Session 7 handoff notes

Session 7 is Phase 0 main launch:
1. Execute `configs/phase0.yaml` on Vast.ai A100 SXM4 80GB (same instance type as warm-start comparison)
2. Apply early-checkpoint probe at ~5K steps (codebook §8.2 gate: AUC < 0.3 → stop early)
3. Apply midpoint probe at ~20K steps (codebook §8.3 table)
4. Apply final probe at ~50K+ steps (codebook §7 decision gate: Path A/B/C)
5. Any operational learnings from Session 6 Vast.ai execution (timing, memory, failure modes) are captured in the verdict doc and should inform Session 7's runbook

The warm-start verdict (`warm_start.checkpoint` in `phase0.yaml`) must be set before Session 7 launches — this is the last step of Session 6.
