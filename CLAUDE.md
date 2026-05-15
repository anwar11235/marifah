# CLAUDE.md — marifah-core cross-session memory

This file is the continuity mechanism across Claude Code sessions. Each session appends an entry to the Session Log. Do not rewrite existing entries — only append.

---

## Repo purpose

Commercial validation track for CORAL architecture. Pivoted from CORAL-v3 (Sudoku + ARC research) to graph DAG execution for enterprise workflows. The Sudoku Phase 3c checkpoint (68.74% exact accuracy, W&B run `wpmdrf8n`, seed 0) is the warm-start baseline for all graph work.

## Architecture vocabulary

**Do not deviate from this taxonomy.** Source of truth: `Marifah_Naming_and_Taxonomy.md`.

| Term | Meaning |
|------|---------|
| Marifah | The company; also the Recognition Cortex mechanism |
| CRA | Cortical Reasoning Architecture — the architecture family |
| CORAL | First CRA instance (Cortical Reasoning via Abstraction Layers) |
| Nous / Nous substrate | The Reasoning Cortex — the CORAL inference engine (CoralInner / CoralV3Inner) |
| Marifah mechanism | The Recognition Cortex — SpatialMoECodebook crystallization |
| ʿIlm-class | Systems whose knowledge is retrieval/pattern-matching (LLMs) — what CORAL is positioned against |
| Nous-class | Systems that do deliberate iterative reasoning — CORAL's Nous substrate qualifies |
| Marifah-class | Systems with recognition-based compounding at deployment — CORAL + Recognition Cortex is the first commercial instance |
| Phase 1 | Predictive coding (`use_predictive_coding=True`) |
| Phase 2 | Columnar routing — STUBBED, NotImplementedError |
| Phase 3b | Soft MoE crystallization (`use_crystallization=True`) |

## Module map (current)

```
src/marifah/
  models/
    coral_base.py    CoralConfig, CoralInner, InnerCarry
    coral.py         CoralV3Inner, PredMetrics (was coral_v3.py in CORAL-v3)
    act.py           CoralACT, CoralV3ACT (was coral/training/act.py)
    codebook.py      SpatialMoECodebook, CrystallizationBuffer (was crystallization.py)
    columnar.py      ColumnarReasoningModule (STUBBED — Phase 2)
    prediction.py    PredictionNet, PrecisionNet
    reasoning_module.py ReasoningModule
    layers.py        CastedLinear, CastedEmbedding, Attention, SwiGLU, RotaryEmbedding
    transformer_block.py TransformerBlock, TransformerBlockConfig
    sparse_embedding.py  CastedSparseEmbedding, CastedSparseEmbeddingSignSGD_Distributed
  training/
    losses.py        ACTLossHead, CoralV3LossHead, stablemax_cross_entropy
    scheduler.py     cosine_schedule_with_warmup_lr_lambda
    adam_atan2.py    Pure-PyTorch AdamATan2 fallback
    train.py         Full training loop (Hydra entry point)
  data/
    base_dataset.py  PuzzleDataset, create_dataloader (HRM-format)
checkpoints/
  sudoku-phase3c/
    phase3c_canonical_seed0_best.pt   Best Sudoku checkpoint (68.74%)
    config.yaml                       Run config from W&B wpmdrf8n
```

## Discipline rules (from prior strategic work)

1. **Two consecutive evals same direction** before treating a trajectory as real (not noise).
2. **Two consecutive same-signature failures = diagnose mode**, not another knob turn.
3. **No merge to main without warm-start regression test** — the Sudoku Phase 3c checkpoint must load cleanly and the smoke test must pass.
4. **Pre-register kill criteria before runs** — define the threshold before seeing the number.
5. **Tests passing ≠ launch-ready** — verify branch ancestry, configs against trainer semantics, test fixtures match real data shapes.

## Key architectural constraints (never change without deliberate session decision)

- Do NOT modify the predictive coding mechanism or the symmetric log-normal precision regularizer (`lambda_pi * 0.5 * (log(π))²`). Extensively validated in Phase 3c.
- Do NOT modify the ACT halt mechanism or Q-halt loss. Port as-is.
- Do NOT modify the SpatialMoECodebook before HMSC session. Port as scaffold; replacement is deliberate future work.
- The Sudoku checkpoint is bit-identical to the CORAL-v3 warm-start. Do not transform it.

## Key decisions (permanent record)

- `act.py` moved from `coral/training/` to `src/marifah/models/` (session 1 decision, matches target structure in bootstrap prompt).
- `coral_v3.py` renamed to `coral.py` (the `_v3` suffix was version-tagging not needed in new repo).
- `crystallization.py` renamed to `codebook.py` (better reflects scope: just the Marifah mechanism scaffold).
- `puzzle_dataset.py` renamed to `base_dataset.py` (emphasizes it's the generic base, not Sudoku-specific).
- W&B workspace: still pointing at `aktuator-ai` — migration is a separate decision.
- Docker image tag: `anwar1919/marifah-core:2026-05-15` (not pushed yet; push on first Vast.ai use).

---

## Session Log

### Session 1 — 2026-05-15

**Goal:** Bootstrap `marifah-core` repository from CORAL-v3 master.

**Exit criteria status:** All criteria met except Docker build (not verifiable on Windows without Docker daemon). See deviations below.

**Pre-flight actions:**
- Dropped 2 stashes from CORAL-v3 (both old housekeeping, user confirmed)
- Tagged CORAL-v3 master as `v3.0-pre-marifah-pivot` and pushed to origin
- Source commit: `85f3a73` (CORAL-v3 master HEAD)

**Files ported (CORAL-v3 source → marifah destination):**

| Source | Destination | Notes |
|--------|-------------|-------|
| `coral/models/coral_v3.py` | `src/marifah/models/coral.py` | Renamed; _v3 suffix dropped |
| `coral/models/coral_base.py` | `src/marifah/models/coral_base.py` | — |
| `coral/training/act.py` | `src/marifah/models/act.py` | Moved from training/ to models/ |
| `coral/models/crystallization.py` | `src/marifah/models/codebook.py` | Renamed |
| `coral/models/prediction.py` | `src/marifah/models/prediction.py` | — |
| `coral/models/transformer_block.py` | `src/marifah/models/transformer_block.py` | — |
| `coral/models/layers.py` | `src/marifah/models/layers.py` | — |
| `coral/models/reasoning_module.py` | `src/marifah/models/reasoning_module.py` | — |
| `coral/models/sparse_embedding.py` | `src/marifah/models/sparse_embedding.py` | — |
| `coral/models/columnar.py` | `src/marifah/models/columnar.py` | Kept for compat; Phase 2 stubbed |
| `coral/models/common.py` | `src/marifah/utils/common.py` | Moved to utils/ |
| `coral/training/losses.py` | `src/marifah/training/losses.py` | — |
| `coral/training/scheduler.py` | `src/marifah/training/scheduler.py` | — |
| `coral/training/adam_atan2.py` | `src/marifah/training/adam_atan2.py` | — |
| `scripts/train.py` | `src/marifah/training/train.py` | — |
| `coral/data/puzzle_dataset.py` | `src/marifah/data/base_dataset.py` | Renamed; Sudoku subclass not ported |
| `requirements.txt` | `requirements.txt` | Updated package name references |
| `.gitignore` | `.gitignore` | Rescoped |
| `Dockerfile` | `Dockerfile` | Updated image tag to marifah-core:2026-05-15 |
| `pyproject.toml` | `pyproject.toml` | Package name: coral→marifah; src layout |

**Files NOT ported (ARC-specific, per §2.4):**
- `coral/data/build_arc_dataset.py`, `build_sudoku_dataset.py` (Sudoku data builder stays in CORAL-v3)
- `coral/models/columnar.py` — ported but stubbed (not ARC-specific, kept for future)
- All ARC-specific tests, configs, scripts
- `.gitmodules` and ARC/ConceptARC submodules
- v2 metric-shape mechanism (stays in CORAL-v3 for Run B verdict)
- R0/R0.5 probe code

**Verification results:**
1. ✅ `git status` clean on `session-1/repo-bootstrap`
2. ✅ `git log` shows 2 logical commits (source port + this summary)
3. ✅ `pip install -e .` succeeded
4. ✅ `pytest tests/test_smoke.py` — 9/9 passed (3.48s, CPU)
5. ⚠️ Docker build — not verified (no Docker daemon on Windows dev machine; build deferred to Vast.ai)
6. ✅ README.md written and reads coherently
7. ✅ CLAUDE.md populated with session log (this entry)
8. ✅ `src/marifah/models/coral.py` imports cleanly; class signature matches CORAL-v3
9. ✅ Sudoku Phase 3c checkpoint loads (51 keys, correct structure)
10. ✅ Branch `session-1/repo-bootstrap` has all work committed (not merged to main)
11. ✅ Tag `v3.0-pre-marifah-pivot` exists on CORAL-v3 and pushed

**Deviations from prompt:**
- Docker build not verified locally (Windows; no Docker daemon). Build verified structurally — Dockerfile is correct, flash-attn pin correct, ARC submodule clones removed. Will verify on first Vast.ai instance.
- `Marifah_Naming_and_Taxonomy.md` not found locally; README references it by name. User should make it available before Session 2.
- Additional model files ported beyond prompt's explicit list: `coral_base.py`, `reasoning_module.py`, `prediction.py`, `sparse_embedding.py`, `columnar.py` — all required dependencies of `coral.py` and `act.py`.
- Company naming: the prompt says "Marifah is the company"; the game plan doc says "Aktuator". Followed the session prompt. Clarify before committing to external messaging.

**Open items / notes for subsequent sessions:**
- HMSC codebook module (Session 3+) — `SpatialMoECodebook` is scaffolding; the real Marifah mechanism is designed in `CORAL_DAG_Codebook_Design.md`.
- Graph adapter (Session 3 candidate) — no graph-specific code exists yet. Synthetic data is ready.
- W&B workspace migration (separate decision) — currently points at `aktuator-ai`.
- Eval-wedge diagnostic (separate CORAL-v3 work, not ported) — may need resolution before warm-start testing.
- The codebook scaffolding (`SpatialMoECodebook`) has a coupling to Sudoku-specific `seq_len=81` in the existing CORAL-v3 configs. The graph adapter will need to reconfigure `seq_len` to match DAG node count. Not a code change, just a config consideration.
- Docker image push: do on first Vast.ai provisioning.

---

### Session 2 — 2026-05-15

**Goal:** Implement the full synthetic DAG benchmark generator per `CORAL_Synthetic_DAG_Benchmark_Spec.md`.

**Exit criteria status:** All criteria met.

**Modules implemented** (`src/marifah/data/synthetic/`):

| Module | Contents |
|--------|----------|
| `primitives.py` | 10 `PrimitiveType` enums, `apply_primitive()`, `sample_attrs()` |
| `executor.py` | `execute_dag()` — topological-order reference executor with branch routing |
| `patterns.py` | 12 pattern classes (`LinearChain` … `ConstrainedTerminate`), `PATTERN_BY_ID` registry |
| `workflows.py` | 50 `WorkflowSpec` definitions (seed=42), frequency tiers, `validate_coverage()`, `build_reserved_primitive_pairs()` |
| `labels.py` | `DAGRecord`, `NodeRecord`, `EdgeRecord`, `RegionAssignment`, `audit_labels()` |
| `vertical_config.py` | `GeneratorConfig`, `SplitSizes`, `load_config()`, `tiny_config()` |
| `generator.py` | `generate_one()` + `DagGenerator` with `multiprocessing.Pool` support |
| `splits.py` | `SplitGenerator` with disjoint seed ranges per split |
| `storage.py` | Parquet write/read, `write_manifest()` + `verify_manifest()` with SHA256 |
| `validate.py` | `validate_record()`, `audit_distribution()`, `spot_check_traces()` |
| `cli.py` | `generate-tiny`, `generate-full`, `validate-dataset` subcommands |
| `cyclic.py` | Stub raising `CyclicNotImplementedError` |

**Config:** `configs/default.yaml` — full split sizes, OOD parameters, seed.

**Tests:** 130 unit tests across 5 test files — all passing.

**Verification results:**
1. ✅ `pytest tests/` — 130/130 passed
2. ✅ `generate-tiny` — 1023 DAGs across all 5 splits in ~2s
3. ✅ `validate-dataset` — PASSED (all splits OK, manifest verified)
4. ✅ Determinism — byte-identical SHA256 shards across two independent runs
5. ✅ Throughput — ~950 DAGs/sec single-threaded
6. ✅ Spot-check traces — executor re-execution matches stored halt_step
7. ✅ Coverage — all 12 patterns in ≥5 workflows; `validate_coverage()` passes

**Bugs found and fixed:**

1. **Training split producing 0 records** — OOD holdout filter checked ALL edges in assembled DAG (40–150+). For complex workflows, P(no reserved pair) ≈ 0.001% with 20 retries. Fix: changed to check only cross-pattern boundary edges (`_cross_pattern_primitive_pairs`), which is the correct spec interpretation. This reduced the check from ~90 edges to ~1-4 inter-pattern edges per DAG.

2. **Executor forwarding branch_id instead of input state** — Conditional/route nodes were writing `result.output_state` (= branch_id integer) into `node_outputs`, causing downstream nodes to receive 0 or 1 instead of the actual data value. Fix: `node_outputs[node_id] = input_for_prim` for branching nodes.

3. **No TERMINATE node in assembled workflows** — Patterns like `linear_chain`, `conditional_fork`, `fork_and_join` don't include TERMINATE. Fix: `_assemble_workflow` now appends a synthetic TERMINATE node when final pattern exits aren't TERMINATE nodes.

4. **Executor falsely reporting `halted=True`** — Old code set `halt_step = trace[-1].step` as fallback. Fix: removed fallback; `halted` is strictly `halt_step >= 0` (only when TERMINATE fires).

5. **JSON integer key round-trip in LOOKUP** — `sample_lookup_attrs` stores `{0: v, 1: v, ...}`. JSON serializes dict keys as strings; `_apply_lookup` then did `table.get(int_key, 0)` → always 0 (key type mismatch). This corrupted all LOOKUP outputs after deserialization, breaking spot-check executor consistency. Fix: `_apply_lookup` now normalizes `{int(k): v for k, v in table.items()}`.

6. **Windows UnicodeEncodeError** — CLI used `→` (U+2192); Windows cp1252 terminal can't encode it. Fix: replaced all `→` with `->` in `cli.py`.

7. **Parquet read bug in CLI** — `t.to_pydict()` extends list with column name strings, not row dicts. Fix: changed to `t.to_pylist()`.

**Open items for Session 3:**
- Graph adapter: wire `DagGenerator` output into a PyTorch Dataset + DataLoader for CORAL training. `seq_len` = max DAG node count (variable; pad to max).
- Full dataset generation on Vast.ai (800K train takes ~14 min at 950 DAGs/sec single-threaded; use `--workers` to parallelize).
- HMSC codebook session (Session 3+).
