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
    hmsc/
      __init__.py          package + re-exports
      global_codebook.py   GlobalCodebook (G-scale: workflow signatures)
      regional_codebook.py RegionalCodebook (R-scale: sub-DAG patterns)
      perposition_codebook.py PerPositionCodebook (P-scale: primitives)
      composition.py       HMSCComposition (sum default, gated alternative)
      auxiliary_heads.py   GlobalAuxHead, RegionalAuxHead, PerPositionAuxHead, compute_aux_losses
      hmsc.py              HMSC top-level module
  training/
    losses.py        ACTLossHead, CoralV3LossHead, stablemax_cross_entropy (Session 1)
    scheduler.py     cosine_schedule_with_warmup_lr_lambda
    adam_atan2.py    Pure-PyTorch AdamATan2 fallback
    train.py         Sudoku training loop (Hydra entry point, Session 1)
    config.py        TrainingConfig Pydantic hierarchy (Session 5)
    graph_losses.py  compute_total_loss for graph DAG (Session 5)
    graph_utils.py   derive_region_labels, prepare_batch_for_model (Session 5)
    data_pipeline.py build_data_loaders, collate_graphs_padded (Session 5)
    checkpointing.py save_checkpoint, load_checkpoint (Session 5)
    logging.py       TrainingLogger — W&B + JSONL (Session 5)
    eval_loop.py     evaluate() for graph val split (Session 5)
    trainer.py       build_model, Trainer class; max_steps support (Sessions 5–6)
    cli.py           train / eval / smoke CLI (Session 5)
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

---

### Session 3 — 2026-05-15

**Goal:** Implement graph adapter (data pipeline: parquet -> GraphBatch -> CORAL forward).

**Exit criteria status:** All criteria met.

**Modules implemented** (`src/marifah/data/adapter/`):

| Module | Contents |
|--------|----------|
| `__init__.py` | Package marker |
| `batch_format.py` | `GraphBatch` dataclass (node_features, attention_mask, node_mask, pos_encoding, label tensors); `primitive_ids` and `attr_vec` properties |
| `tokenizer.py` | `encode_node_attrs()` for all 10 primitives; `NodeTokenizer` nn.Module; `NODE_FEAT_DIM=5`, `ATTR_DIM=4` |
| `positional.py` | `compute_laplacian_pe()` (numpy); `laplacian_pe_tensor()` (torch); dense eigh <= 32 nodes, scipy eigsh otherwise; `K_PE_DEFAULT=8` |
| `attention_mask.py` | `build_attention_mask()` additive bias (0.0 attend, -inf block); `pad_attention_masks()`; `AttentionDirection` literal |
| `dataset.py` | `GraphDAGDataset` — loads shard_*.parquet, filters by max_nodes (with logging), precomputes PE + masks at init |
| `collate.py` | `collate_graphs()` — pads to batch max-nodes, raises ValueError on empty batch (underfull-batch fix) |
| `cli.py` | `precompute-pe` (adds PE column to shards); `inspect-batch` (prints tensor shapes and sample) |

**New model file** (`src/marifah/models/`):

| Module | Contents |
|--------|----------|
| `attention.py` | `sdpa_with_bias()`, `flash_varlen()` with CPU fallback, `GraphAttentionLayer` |

**Tests** (all in `tests/`):

| File | Count |
|------|-------|
| `test_adapter_tokenizer.py` | 7 tests |
| `test_adapter_positional.py` | 7 tests |
| `test_adapter_attention_mask.py` | 7 tests |
| `test_adapter_attention_layer.py` | 8 tests |
| `test_adapter_dataset.py` | 8 tests |
| `test_adapter_collate.py` | 11 tests |
| `test_adapter_e2e.py` | 5 tests |

Total: 195/195 tests passing (includes all Session 1 + 2 tests).

**Verification results:**
1. ✅ 195/195 tests pass (`pytest tests/ -q`)
2. ✅ `inspect-batch` — shapes confirmed: `(B=4, N_max=21, 5)` node_features, `(4, 21, 21)` attention_mask
3. ✅ SDPA vs flash_varlen max abs diff: 0.0 (< 1e-2 fp16 tolerance; §4 item 3)
4. ✅ 3-node chain attention mask verified: node 1 attends to node 0 only; node 2 attends to node 1 only (§4 item 5)
5. ✅ Mixed 5-node + 15-node batch pads correctly; padding positions are -inf / 0 / -1 as required (§4 item 7)
6. ✅ Empty-batch `collate_graphs([])` raises `ValueError("empty")` (underfull-batch fix, salvage §3)
7. ✅ `precompute-pe` CLI adds `pos_encoding` JSON column to shard files
8. ✅ End-to-end: `GraphDAGDataset -> collate_graphs -> TinyGraphModel -> loss -> backward` — gradients are non-None and finite
9. ✅ `CoralInner` integration smoke test: output shape `(B, N, 10)`, all finite

**Bug fixed during implementation:**
- `GraphDAGDataset` node-count filter: `len(rec["nodes"])` measured JSON string length (always >> 512), filtering all records. Fixed to `int(rec.get("num_nodes", 0))` with JSON-parse fallback.

**Key architectural decisions:**
- Additive-bias mask convention: 0.0 = attend, -inf = block. Matches CORAL-v3 arc/padding-attention-mask salvage (commit c7e784d).
- Directed mask: node `dst` attends to `src` iff `src -> dst` is a DAG edge. Self-loops always 0.0.
- Laplacian PE is computed on undirected version of graph (symmetric L = D - A).
- `collate_graphs` pads to max-nodes-in-batch (not dataset max); `DataLoader` controls underfull batches via `drop_last`.
- `NodeTokenizer` sums primitive embedding + attr projection (not concat) to keep d_model fixed.

**Open items for Session 4:**
- HMSC codebook session: replace `SpatialMoECodebook` scaffold with real Marifah mechanism.
- Full dataset generation on Vast.ai (800K train DAGs).
- Wire graph adapter into main training loop (`train.py`) with graph-specific `seq_len = max_nodes`.
- W&B workspace migration from `aktuator-ai`.

---

### Session 4 — 2026-05-15

**Goal:** Implement the Hierarchical Multi-Scale Codebook (HMSC) — three codebooks (G, R, P), composition, and auxiliary loss heads. Lambdas = 0 (losses computed but don't engage this session).

**Exit criteria status:** All criteria met.

**Modules implemented** (`src/marifah/models/hmsc/`):

| Module | LOC | Contents |
|--------|-----|----------|
| `__init__.py` | 31 | Package marker + re-exports |
| `global_codebook.py` | 64 | `GlobalCodebook`: mean-pool → softmax routing → broadcast mode |
| `regional_codebook.py` | 144 | `RegionalCodebook`: region-token attention pooling → per-region routing → soft/hard assignment |
| `perposition_codebook.py` | 81 | `PerPositionCodebook`: per-node cross-attention; soft (train) / hard top-1 (eval) |
| `composition.py` | 85 | `HMSCComposition`: sum (default) or gated |
| `auxiliary_heads.py` | 135 | `GlobalAuxHead`, `RegionalAuxHead`, `PerPositionAuxHead`, `compute_aux_losses` |
| `hmsc.py` | 176 | `HMSC` top-level module with utilization tracking |

**CORAL modifications:**
- `coral_base.py`: added `use_hmsc: bool = False` to `CoralConfig`
- `coral.py`: added HMSC import; `hmsc_aux_losses`/`hmsc_utilization` to `PredMetrics`; HMSC instantiation in `__init__`; HMSC tap in `_forward_with_pc` (post-final-H-step, pre-lm_head, residual addition)

**Tests:** 60 new tests; 255/255 total passing.

**Verification results:**
1. ✅ 255/255 tests pass
2. ✅ composed shape `(B, N, 512)` confirmed
3. ✅ Aux losses finite at random init (L_G=4.23, L_R=2.45, L_P=2.30 — near log(vocab_size) as expected)
4. ✅ Determinism: max_diff = 0.0 across two identical forward passes
5. ✅ Regression check: `use_hmsc=False` max_diff = 0.0 (bit-identical to pre-HMSC)
6. ✅ Gradient flow with lambda>0: all codebook, routing, composition, aux head params have finite gradients
7. ✅ Head gradient = 0.00e+00 with lambda=0
8. ✅ Utilization at random init: G active=87.5%, R active=100%, P active=100%; non-zero entropy (no dead codes)
9. ✅ End-to-end smoke test: CoralV3Inner + HMSC + backward, all finite

**Key architectural decisions:**
- HMSC tap: post-final-H-step in `_forward_with_pc`, where z_H is still in computation graph → gradients flow via main loss path
- Residual addition `z_H = z_H + composed` (not replacement) for stability at random init
- `node_mask` is optional in CORAL batch dict; falls back to all-ones for non-graph inputs
- Regional codebook uses custom attention (not `nn.MultiheadAttention`) for symmetric node-to-region soft assignment

**Open items for Session 5:**
- Training pipeline integration: wire adapter + HMSC + CORAL into main training loop with W&B logging and configurable lambdas
- `node_mask` must be threaded through from graph batch to CORAL batch dict
- `region_labels` mapping: generator produces per-node labels; R-head expects per-region — needs mapping logic
- Full dataset generation on Vast.ai (800K train DAGs)
- W&B workspace migration from `aktuator-ai`

---

### Session 5 — 2026-05-15

**Goal:** Wire the adapter + HMSC + CORAL into a complete training pipeline with checkpointing, W&B logging, configurable lambdas, and Phase 0 / Phase 1 launch configurations. Verify end-to-end via tiny-dataset training smoke tests.

**Exit criteria status:** All criteria met.

**Modules implemented** (`src/marifah/training/`):

| Module | Contents |
|--------|----------|
| `config.py` | `TrainingConfig` Pydantic hierarchy (Experiment/Model/HMSC/Training/Data/Logging/WarmStart subconfigs), `load_config()` |
| `graph_losses.py` | `compute_total_loss()` — main CE + halt BCE + HMSC aux losses (pre-weighted via HMSC lambdas set at init) |
| `graph_utils.py` | `derive_region_labels()` (per-node → per-region majority-vote), `prepare_batch_for_model()` (GraphBatch → CoralV3Inner dict) |
| `data_pipeline.py` | `collate_graphs_padded()` (fixed-length padding), `build_data_loaders()` for all 5 splits |
| `checkpointing.py` | `save_checkpoint()` (atomic write), `load_checkpoint()`, `load_warmstart()` |
| `logging.py` | `TrainingLogger` — W&B + always-on JSONL fallback, step/eval/codebook_stats emitters |
| `eval_loop.py` | `evaluate()` — full val pass in eval mode, aggregated metrics |
| `trainer.py` | `build_model()`, `Trainer` class (epoch-based, 1-step gradient, fresh carry per batch) |
| `cli.py` | `train` / `eval` / `smoke` subcommands; smoke auto-generates tiny dataset |

**Config files implemented** (`configs/`):

| File | Contents |
|------|----------|
| `phase0.yaml` | PC-only, use_hmsc=false, d_model=512, all lambdas=0 |
| `phase1.yaml` | HMSC engaged, λ_G=λ_R=λ_P=0.1, K_G=64/K_R=16/K_P=16 |
| `smoke.yaml` | d_model=64, 3 epochs, HMSC on, wandb disabled |
| `smoke_hmsc_off.yaml` | Same as smoke, HMSC off |

**Verification results:**
1. ✅ `pytest tests/` — 280/280 passed (includes 25 new training tests)
2. ✅ Smoke A (HMSC off) — 3 epochs, loss 1.68→0.08, checkpoint saved + reloaded
3. ✅ Smoke B (HMSC on) — 3 epochs, main 1.69→0.11, aux 0.85→0.71 (decreasing), HMSC util stats in log
4. ✅ Checkpoints load into fresh model instances
5. ✅ Regression check: `use_hmsc=False` max_diff = 0.0
6. ✅ Loss values real (main non-zero, decreasing over epochs)
7. ✅ HMSC codebook utilisation stats in Smoke B JSONL log
8. ✅ Aux=0 in Smoke A, aux>0 in Smoke B
9. ✅ `wandb_mode: disabled` works without W&B connectivity
10. ✅ JSONL log written and parseable for both smokes
11. ✅ CLI `train`, `eval`, `smoke` all execute end-to-end
12. ✅ Branch `session-5/training-pipeline` committed (after this entry)

**Key architectural decisions:**
- **1-step gradient training**: Fresh carry per batch, single CORAL inner segment. Full ACT multi-step is test-time; training uses single segment for simplicity.
- **HMSC attached post-init**: `CoralV3Inner` built with `use_hmsc=False`, then `model.hmsc = HMSC(...)` added with correct params and training lambdas. Avoids CoralConfig needing HMSC-specific fields.
- **eval_interval in EPOCHS**: Enforced by design — `eval_interval_epochs` in config, loop counts epochs.
- **region_labels heuristic**: `pattern_id % num_regions` majority-vote mapping. Pipeline-plumbing approximation; may need refinement for production.
- **HMSC lambdas set at model-build time** from training config, not as separate config section. Keeps aux loss weighting co-located with the model.

**Smoke test summary:**
| Metric | Smoke A (HMSC off) | Smoke B (HMSC on) |
|--------|--------------------|-------------------|
| Epoch 0 main loss | 1.681 | 1.686 |
| Epoch 2 main loss | 0.077 | 0.110 |
| Epoch 2 val acc_node | 0.824 | 0.824 |
| Aux loss epoch 0 | 0.000 | 0.853 |
| Aux loss epoch 2 | 0.000 | 0.710 |
| Regression max_diff | 0.0 | — |

**Open items for Session 6:**
- Phase 0 training launch on Vast.ai A100 SXM4 80GB
- Full dataset generation (800K train DAGs; ~14 min single-threaded)
- Update `configs/phase0.yaml` `data.dataset_root` to Vast.ai path
- Verify GPU memory at batch_size=64, d_model=512, H_layers=2
- Workflow-type AUC probe after Phase 0 checkpoint (§7 decision gate)

---

### Session 6 — 2026-05-15

**Goal:** Pre-launch preparation — trainer `max_steps` support, warm-start comparison configs + probe (resolves OD7), container image docs, and Vast.ai runbook. Does not launch Phase 0 (that's Session 7).

**Exit criteria status:** CC-local work complete. Items 7–11 (Vast.ai execution + analysis) pending user runbook execution.

**Code changes:**

| Change | Location | Notes |
|--------|----------|-------|
| `max_steps: Optional[int] = None` | `config.py` `TrainingPhaseConfig` | Hard step cap; used by warmstart comparison runs (5K steps each) |
| `max_steps` wired in training loop | `trainer.py` `Trainer.train()` | Breaks inner + outer loop cleanly when hit; `final.pt` always saved |
| `drop_last` moved to correct section | `phase0.yaml` | Was under `training:` (ignored); moved to `data:` where `DataConfig` reads it |

**New files:**

| File | Contents |
|------|----------|
| `configs/warmstart_cold.yaml` | OD7 cold run — fresh init, max_steps=5000, eval_interval_epochs=1 |
| `configs/warmstart_warm.yaml` | OD7 warm run — Sudoku Phase 3c init, same otherwise |
| `scripts/warmstart_probe.py` | Probe script: workflow-type AUC (linear probe on pooled z_H) + execution faithfulness (edit distance) |
| `docs/operations/container_image.md` | Container image deps, Vast.ai provisioning steps, validation status (pending Vast execution) |
| `docs/operations/session06_runbook.md` | User-facing Vast.ai runbook: verification → dataset generation → cold run → warm run → probes → share results |
| `docs/sessions/session-06-phase0-prep.md` | Session summary (this file) |
| `docs/sessions/session-06-warmstart-verdict.md` | Placeholder — verdict table + decision populated after user shares results |

**CC-local verification results:**
1. ✅ Sudoku Phase 3c checkpoint intact and loadable (51 keys, `OrderedDict`)
2. ✅ `warmstart_cold.yaml` parses: `max_steps=5000`, `warm_start.checkpoint=None`
3. ✅ `warmstart_warm.yaml` parses: `max_steps=5000`, `warm_start.checkpoint=...phase3c...`
4. ✅ Probe runs end-to-end on tiny dataset (Sudoku ckpt as dummy): no errors, results JSON written
5. ✅ `pytest tests/` — 280/280 pass with trainer changes
6. ✅ Container image doc and runbook complete

**Key architectural decisions:**
- **Probe operationalizes codebook §7.1**: Pools `z_H` over `node_mask` real positions → mean pooled carry state → linear probe → macro OvR AUC. This is the "pooled readout" referenced in the gating logic.
- **Faithfulness probe as primitive-id edit distance**: Argmax logits at real nodes vs. `primitive_assignments` ground truth. Normalized Levenshtein per DAG, averaged. Covers codebook design §6.1.
- **Warm run uses `load_optimizer: false`**: Tests only initialization effect, not Sudoku optimization trajectory.
- **Warm-start checkpoint path**: actual file is `phase3c_canonical_seed0_best.pt` (not `best.pt`).

**Pending (Session 6 not complete until user executes Vast.ai runbook):**
- Container image verification (pass/fail + any errors)
- Full dataset generation (time, distribution stats)
- Warmstart cold + warm runs (5K steps each, ~1–2 hours each)
- Probe runs on both final checkpoints
- Results JSONs shared back to CC
- Warm-start verdict written + Phase 0 config finalized

**Bug fix (mid-Session 6, after first Vast.ai launch): flash_attn float32 crash**

Two bugs in `trainer.py` hardcoded float32, causing immediate crash on GPU launch (W&B `79lxh45j`):
1. `build_model()` line 61: `forward_dtype="float32"` — ignored `config.model.forward_dtype`
2. `step()` lines 254-255: carry tensor `dtype=torch.float32` — hardcoded, bypassed model dtype

Both lines now use `config.model.forward_dtype`. Three production configs updated: `forward_dtype: "bfloat16"` under `model:` in `warmstart_cold.yaml`, `warmstart_warm.yaml`, `phase0.yaml`. Smoke configs remain `float32` (CPU — flash_attn not triggered).

Checkpoint dtype audit: `load_warmstart` → `strict=False` `load_state_dict` → `copy_()` safely handles dtype transitions. No garbage activation risk.

Note: `mixed_precision` config key is never consumed (no `torch.autocast` in trainer). bf16 is achieved entirely through the Casted layers pattern (`forward_dtype`). `mixed_precision` is dead config — flagged, not fixed.

**Bug fix (2026-05-16): architecture mismatch — warmstart/phase0 configs had H_layers=2, L_layers=2**

Sudoku Phase 3c checkpoint (warm-start source) has H_layers=4, L_layers=4, d_model=512 (~30.4M params).
Warmstart and phase0 configs had H_layers=2, L_layers=2 (~15.2M params) — set conservatively in Session 5 with no GPU available.
This caused `load_state_dict(strict=False)` to silently drop checkpoint layers 2-3, making warm-vs-cold meaningless.

Changes:
- `configs/warmstart_cold.yaml`: H_layers 2→4, L_layers 2→4
- `configs/warmstart_warm.yaml`: H_layers 2→4, L_layers 2→4
- `configs/phase0.yaml`: H_layers 2→4, L_layers 2→4

New test file `tests/test_configs.py` (2 tests, 294 total pass):
- `test_warmstart_configs_match_sudoku_checkpoint_architecture`: asserts H/L/d_model match Sudoku Phase 3c constants
- `test_warmstart_cold_and_warm_have_identical_model_architecture`: asserts cold/warm model sections are bit-identical

Audit findings (not patched — flagged for review):

1. **load_state_dict silent truncation** (`checkpointing.py:95,98`): `load_state_dict(..., strict=False)` discards the `_IncompatibleKeys` return value. Missing/unexpected keys produce no log warning. This is the exact mechanism that allowed the 2-vs-4-layer mismatch to go unnoticed. Proposed fix (separate PR): capture return and emit `logger.warning` for any non-empty `missing_keys` or `unexpected_keys`.

2. **vocab_size=10** (`warmstart_cold/warm.yaml`, `phase0.yaml`): 10 primitive types (IntEnum 0–9), so graph task uses indices 0–9. `vocab_size=10` is correct for the graph model. The Sudoku checkpoint's lm_head has shape `(11, 512)` (11-class Sudoku vocab) vs. graph model `(10, 512)`. Under `strict=False`, same-key shape mismatches still raise `RuntimeError`; needs checkpoint key-name inspection on Vast to confirm whether the lm_head key is present in the Sudoku checkpoint. Cannot verify locally without the full checkpoint dump.

3. **max_nodes=100** (`warmstart_cold/warm.yaml`, `phase0.yaml`): Largest single pattern has `max_size=18`; complex workflows have up to 8 patterns → theoretical max ~145 nodes. `GraphDAGDataset` silently filters records with `num_nodes > max_nodes`, so complex DAGs >100 nodes are dropped from training. `test_ood_size` split DAGs scale 2–5× training max in size — if empirical training cap is ~100 nodes, OOD-size DAGs (200–500 nodes) will be entirely filtered, making that eval split empty. Verify empirical size distribution before phase 0 launch; consider raising max_nodes or confirming OOD size split is not used in Phase 0 eval.

**Three-fix session (2026-05-16): data pipeline, checkpoint remap, console progress**

*Context:* warmstart_cold launched at 18:48 (W&B `0ac8373c`) and is training. These fixes done in parallel locally.

**Issue 1 — DataLoader startup (~5 min observed on Vast):**
Root cause: `GraphDAGDataset.__init__` precomputed Laplacian PE + attention masks for ALL records eagerly in a single-threaded loop. Fix: `_make_loader` now passes `precompute=False`; computation happens lazily in DataLoader worker processes. Timing: 45× faster init (0.013s vs 0.584s for 963 records); extrapolated ~8 min → <30s on 800K rows.

Side-fix: Laplacian eigenvectors lacked sign normalization. `numpy.linalg.eigh` sign is non-deterministic across independent calls; without normalization, the same graph gets different-signed PE from different workers on different epochs. Added sign normalization in `positional.py` (first non-zero element per eigenvector column forced positive).

Timing logs added to `data_pipeline.py` per-split and total.

**Issue 2 — Checkpoint remap (warm-start was silently producing warm ≡ cold):**
Three root causes identified and fixed via `remap_sudoku_checkpoint()` in `checkpointing.py`:
1. `model.inner.` prefix (Sudoku structure) → stripped
2. `_orig_mod.` infix (torch.compile artifact) → stripped  
3. vocab_size shape mismatch embed/lm_head (11 vs 10) → dropped with WARNING, train from scratch

`load_checkpoint` now:
- Accepts `remap='sudoku_to_marifah'` kwarg
- Logs `missing_keys` / `unexpected_keys` at WARNING (Session 6 Audit Finding 1 resolved)
- Raises `RuntimeError` if substrate param (H_level.*.weight) unchanged after load (prevents silent-drop-to-random-init)

`WarmStartConfig` schema: added `remap: Optional[Literal['sudoku_to_marifah']]`.
`warmstart_warm.yaml`: `remap: sudoku_to_marifah` added to `warm_start:` block.

The real Sudoku Phase 3c checkpoint loaded successfully via remap (test 5 PASSED — not skipped).

**Issue 3 — Console progress visibility:**
- tqdm progress bar in TTY mode (step, lm_loss, lr, it/s) — `mininterval=1.0`, `disable=not isatty()`
- Heartbeat log in non-TTY mode: `step=N | lm_loss=X.XX` every `heartbeat_interval_steps=10` steps (configurable, 0=disable)
- `heartbeat_interval_steps: int = 10` added to `LoggingConfig`
- No tensor sync overhead — reads pre-converted CPU floats from `step_losses` dict

**Test count: 305/305 (was 294 before this work)**
New files: `tests/test_checkpointing.py` (10 tests), 1 new test in `tests/test_adapter_dataset.py`

**Carry-dtype audit (2026-05-16): propagate forward_dtype to all carry constructions**

*Context:* Round 1 (c92cfcb) fixed `trainer.step` and `build_model`. This round fixes the remaining sites and adds the `InnerCarry.zeros` factory as single source of truth.

**Bug class:** Carry tensors hardcoded `dtype=torch.float32` instead of reading `config.model.forward_dtype`. On GPU with `forward_dtype=bfloat16`, fp32 carry flows into flash_attn_func → `RuntimeError: FlashAttention only supports fp16 and bf16`.

**Sites fixed:**

| Site | File | Change |
|------|------|--------|
| `eval_loop.evaluate` | `src/marifah/training/eval_loop.py` | Replaced `InnerCarry(z_H=torch.zeros(..., dtype=torch.float32, ...))` with `InnerCarry.zeros(B, config.model, device)` |
| `warmstart_probe._extract_carry_states` | `scripts/warmstart_probe.py` | Added `_make_carry()` helper; replaced hardcoded fp32 |
| `warmstart_probe.compute_execution_faithfulness` | `scripts/warmstart_probe.py` | Same `_make_carry()` helper |
| `trainer.step` | `src/marifah/training/trainer.py` | Simplified to `InnerCarry.zeros(B, self.config.model, self.device)` |

**`InnerCarry.zeros` factory** added to `coral_base.py`:
- Reads `model_config.forward_dtype` (duck-typed `Any` to avoid circular imports)
- Accepts `dtype_override` for CPU bf16 fallback in probe script
- Single source of truth: all four call sites now funnel through this method

**`_make_carry()` helper in `warmstart_probe.py`:**
- CPU + forward_dtype=bfloat16 → warns and falls back to float32 (flash-attn requires CUDA)
- GPU + forward_dtype=bfloat16 → passes through correctly

**New test file:** `tests/test_dtype_propagation.py` (3 passed, 1 skipped-no-CUDA):
- `test_carry_dtype_propagates_through_all_call_sites` — monkey-patches `InnerCarry.__init__` to record z_H dtypes; asserts all 4+ carry constructions use `config.model.forward_dtype`. float32 runs on CPU; bfloat16 skipped without CUDA.
- `test_inner_carry_zeros_factory_uses_config_dtype` — factory unit test, float32 + bfloat16
- `test_inner_carry_zeros_dtype_override` — dtype_override wins over model_config

**New script:** `scripts/verify_pipeline_e2e.py`:
- 4-phase: (1) 4-step training, (2) eval pass, (3) probe pass (n=16), (4) PASS/FAIL summary
- Target runtime: <5 min on A100 SXM4 80GB
- Usage: `python scripts/verify_pipeline_e2e.py --dataset /workspace/data/... --config configs/warmstart_cold.yaml`
- Do NOT run locally — GPU-only verification

**Test count: 308/308 (was 305 before this work)**

**Open items for Session 7:**
- Phase 0 main launch (same instance, same runbook pattern as warm-start comparison)
- Early checkpoint probe at ~5K steps (codebook §8.2: AUC < 0.3 → stop)
- Midpoint probe at ~20K steps
- Final probe at ~50K+ steps (§7 decision gate: Path A/B/C)
- Run `verify_pipeline_e2e.py` on Vast before Phase 0 launch
