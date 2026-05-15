# Session 3 — Graph Adapter

**Date:** 2026-05-15
**Branch:** `session-3/graph-adapter`
**Merged from:** `session-2/synthetic-generator` (via main)

---

## Goal

Bridge the synthetic DAG generator (parquet shards) to the CORAL forward pass via a typed PyTorch data pipeline.

## What was built

### `src/marifah/data/adapter/`

| Module | Purpose |
|--------|---------|
| `batch_format.py` | `GraphBatch` dataclass — the canonical in-memory batch format for all graph training |
| `tokenizer.py` | `encode_node_attrs()` attribute encoder; `NodeTokenizer` learnable embedding module |
| `positional.py` | Laplacian PE via dense `eigh` (≤32 nodes) or `scipy.sparse.linalg.eigsh` (>32) |
| `attention_mask.py` | Additive-bias mask builder (0.0 = attend, −∞ = block); directed and bidirectional modes |
| `dataset.py` | `GraphDAGDataset` — parquet loader with node-count filtering and init-time precompute |
| `collate.py` | `collate_graphs()` — pads to per-batch max, raises `ValueError` on empty input |
| `cli.py` | `precompute-pe` and `inspect-batch` subcommands |

### `src/marifah/models/attention.py`

- `sdpa_with_bias(q, k, v, attention_mask)` — PyTorch SDPA with (B, N, N) additive bias
- `flash_varlen(q_packed, k_packed, v_packed, cu_seqlens, max_seqlen)` — flash-attn-varlen with CPU fallback
- `GraphAttentionLayer` — full attention layer with QKV projection, backend dispatch, output projection

## CORAL-v3 salvage operations

| Commit | Salvaged |
|--------|---------|
| `c7e784d` (arc/padding-attention-mask) | Additive-bias mask convention and SDPA insertion point |
| `28af53a` (arc/flash-attn-varlen) | `flash_varlen` packing/unpacking logic; CPU fallback structure |
| `7367d6e` (arc/phase0-config) | Underfull-batch fix: `collate_graphs` raises instead of silently dropping |

## Verification

All §4 exit criteria from `CC_Session_03_Graph_Adapter.md` met:

| Check | Result |
|-------|--------|
| 195/195 tests pass | ✅ |
| inspect-batch shapes correct | ✅ |
| SDPA vs flash_varlen max diff = 0.0 (< 1e-2) | ✅ |
| 3-node chain directed mask structure | ✅ |
| Mixed 5+15 node batch padding | ✅ |
| Empty batch raises ValueError | ✅ |
| precompute-pe CLI adds PE column | ✅ |
| Forward/loss/backward gradient flow | ✅ |
| CoralInner integration smoke test | ✅ |

## Key decisions

**Additive-bias mask convention:** 0.0 = attend, −∞ = block. This is PyTorch SDPA's native additive bias format. Matches flash-attn's `softmax_scale`-aware behavior on CUDA.

**Directed mask semantics:** For edge `src → dst`, mask is set so `dst` attends to `src` (information flows forward along the edge). Self-loops always 0.0.

**Laplacian PE:** Undirected graph (symmetrized). Skip zero eigenvalue; take next K. Returns zeros if insufficient non-zero eigenvalues (isolated nodes / trivial graphs).

**NodeTokenizer sums (not concat):** `prim_emb + attr_proj(attr_vec)` keeps output dimension = d_model independent of attribute encoding choices.

**Padding:** All padding to per-batch max-nodes, not dataset max. `DataLoader`'s `drop_last` controls underfull batches; `collate_graphs` never decides for the caller.

## Bug fixed

`GraphDAGDataset` was filtering all records because `len(rec["nodes"])` measured the JSON string length (hundreds of characters) rather than the node count. Fixed to `int(rec.get("num_nodes", 0))` with a JSON-parse fallback for records without the `num_nodes` column.

## Open items for Session 4

- HMSC codebook: replace `SpatialMoECodebook` scaffold with real Marifah mechanism (see `CORAL_DAG_Codebook_Design.md`)
- Full dataset: 800K train DAGs on Vast.ai (`--workers` flag available)
- Training loop: wire graph adapter into `train.py`; `seq_len = batch.max_nodes` (variable per batch)
- W&B: migrate workspace from `aktuator-ai` to Marifah workspace
