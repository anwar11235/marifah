# Session 2 — Synthetic DAG Benchmark Generator

**Date:** 2026-05-15  
**Branch:** `session-2/synthetic-generator`  
**Prompt:** `docs/prompts/CC_Session_02_Synthetic_Generator.md`  
**Spec:** `CORAL_Synthetic_DAG_Benchmark_Spec.md`

---

## Goal

Implement the full synthetic DAG benchmark generator — the data engine that produces the Claim A4 training dataset and OOD evaluation splits.

---

## Modules implemented

All 12 modules under `src/marifah/data/synthetic/`:

| Module | LOC | Contents |
|--------|-----|----------|
| `primitives.py` | ~250 | 10 `PrimitiveType` enums, `apply_primitive()`, `sample_attrs()` |
| `executor.py` | ~150 | `execute_dag()` — topological-order reference executor with branch routing |
| `patterns.py` | ~350 | 12 pattern classes, `PATTERN_BY_ID` registry |
| `workflows.py` | ~180 | 50 `WorkflowSpec` definitions, frequency tiers, coverage validation, reserved pairs |
| `labels.py` | ~100 | `DAGRecord` dataclass, `audit_labels()` |
| `vertical_config.py` | ~120 | `GeneratorConfig`, YAML loader, `tiny_config()` |
| `generator.py` | ~250 | `generate_one()`, `DagGenerator` with multiprocessing |
| `splits.py` | ~80 | `SplitGenerator` with disjoint seed ranges |
| `storage.py` | ~120 | Parquet write/read, SHA256 manifest |
| `validate.py` | ~150 | `validate_record()`, `audit_distribution()`, `spot_check_traces()` |
| `cli.py` | ~220 | `generate-tiny`, `generate-full`, `validate-dataset` argparse commands |
| `cyclic.py` | ~20 | Stub raising `CyclicNotImplementedError` |

Additional: `configs/default.yaml`, 5 test files (130 unit tests total).

---

## Architecture decisions

### Three-scale structure

- **Workflow** (global): 50 workflow signatures across 5 frequency tiers. Training follows per-workflow instance counts (100K, 20K, 5K, 500, 50).
- **Pattern** (regional): 12 sub-DAG patterns wired sequentially. Each pattern is independently instantiated with an RNG-derived structure.
- **Primitive** (per-node): 10 generic primitives. Each node has a primitive type + sampled attributes (e.g., CONDITIONAL gets a condition string; LOOKUP gets an 8-entry int table).

### OOD design

- **OOD-size**: Repeat the workflow's pattern sequence 2–5× for larger-than-training DAGs.
- **OOD-composition**: 15% of (src_prim, dst_prim) cross-pattern adjacent pairs are reserved. Training DAGs are rejected if they contain any reserved cross-pattern pair. `test_ood_composition` DAGs must contain at least one.

### Determinism

`(config_hash, seed)` → byte-identical parquet shards. The config hash is a SHA256 of all YAML fields; per-DAG seeds are sequential integers. The manifest stores per-shard SHA256 hashes for downstream verification.

### Multiprocessing

`multiprocessing.Pool.imap_unordered` with a top-level `_worker()` function (required for pickle on Windows).

---

## Bugs found and fixed

### 1. Training split — 0 records produced

**Root cause:** `generate_one()` checked ALL edges in the assembled DAG against the reserved primitive pairs. Complex/medium workflows produce DAGs with 40–150+ edges. With 15 reserved pairs out of 100 possible, P(no reserved pair in 90 edges) = 0.85^90 ≈ 0.001%. Almost every training DAG was rejected within 20 retries.

**Fix:** Changed from checking all edges to checking only cross-pattern boundary edges (`_cross_pattern_primitive_pairs`). The reserved pair semantics are about cross-pattern composition — which primitives appear adjacent BETWEEN pattern instances, not within a single pattern. This reduces the check from ~90 edges to ~1–4 inter-pattern edges per workflow.

### 2. Executor forwarding branch_id instead of input state

**Root cause:** After a CONDITIONAL or ROUTE node fires, the executor was writing `result.output_state` (= branch_id, either 0 or 1) into `node_outputs[node_id]`. Downstream nodes then received 0 or 1 as their input instead of the actual data value.

**Fix:** For branching nodes, set `node_outputs[node_id] = input_for_prim` (the input is forwarded, not the branch_id).

### 3. No TERMINATE node in assembled workflows

**Root cause:** Many patterns (`linear_chain`, `conditional_fork`, `fork_and_join`, `hierarchical_aggregate`, `multi_way_route`, `branch_merge_resolve`) don't include a TERMINATE node — they're designed to be chained. When assembled alone or as the final pattern, the resulting DAG had no TERMINATE, causing execution to never halt.

**Fix:** `_assemble_workflow()` appends a synthetic TERMINATE node connected to all final pattern exits when none of those exits is already a TERMINATE node.

### 4. Executor falsely reporting `halted=True`

**Root cause:** The executor had a fallback: if TERMINATE never fired, it set `halt_step = trace[-1].step`. This made `halted` always True, masking generation failures.

**Fix:** Removed the fallback. `halted = (halt_step >= 0)` is now the exclusive indicator that TERMINATE fired.

### 5. JSON integer key round-trip in LOOKUP

**Root cause:** `sample_lookup_attrs` stores `{0: v, 1: v, ...}` (integer keys). JSON serializes dict keys as strings. After `json.loads`, the table is `{"0": v, "1": v, ...}`. `_apply_lookup` then did `table.get(int_key, 0)` — integer key vs. string key mismatch → returned 0 for every lookup. This corrupted all LOOKUP outputs after parquet deserialization, changing downstream branching and making spot-check halt_step mismatches.

**Fix:** `_apply_lookup` now normalizes: `table = {int(k): v for k, v in raw_table.items()}`. Handles both int keys (during generation) and string keys (after JSON deserialization).

### 6. Windows UnicodeEncodeError in CLI

**Root cause:** CLI used `→` (U+2192). Windows cp1252 terminal can't encode it.

**Fix:** Replaced all `→` with `->`.

### 7. Parquet read bug in CLI validate-dataset

**Root cause:** `t.to_pydict()` returns `{col_name: [values]}`. `rows.extend(to_pydict())` extends the list with column name strings, not row dicts.

**Fix:** Changed to `t.to_pylist()` which returns `[{col: val, ...}]` per row.

---

## Verification results

| Check | Result |
|-------|--------|
| `pytest tests/` | 130/130 passed |
| `generate-tiny` | 1023 DAGs in ~2s across all 5 splits |
| `validate-dataset` | PASSED — all splits OK, manifest verified |
| Determinism | Byte-identical SHA256 shards across 2 independent runs |
| Throughput | ~950 DAGs/sec single-threaded (CPU, Windows) |
| Spot-check traces | Executor re-execution matches stored halt_step in all splits |
| Coverage | All 12 patterns in ≥14 workflows each; `validate_coverage()` passes |

---

## Performance projections

At 950 DAGs/sec single-threaded:
- Train split (800K): ~14 min with `--workers 1`; ~2 min with `--workers 8`
- Full dataset (~830K): ~15 min with `--workers 1`

Use `--workers N` matching available CPU cores on Vast.ai.

---

## Open items for Session 3

1. **Graph adapter** — Wire `DagGenerator` output into a PyTorch `Dataset` + `DataLoader` for CORAL training. `seq_len` is variable (max DAG node count); needs padding strategy.
2. **Full dataset generation** — Run on Vast.ai with multi-worker parallelism.
3. **HMSC codebook session** — `SpatialMoECodebook` is currently scaffolding.
4. **Model eval on DAG benchmark** — Define accuracy metrics for DAG execution prediction.
