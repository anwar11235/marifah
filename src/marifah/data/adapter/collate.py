"""Batch collation for variable-node-count DAG records.

collate_graphs(batch) pads each per-DAG dict to the maximum node count in
the current batch, stacks all tensors, and returns a GraphBatch.

Underfull-batch handling (Salvage 3)
-------------------------------------
Salvage source: CORAL-v3 arc/phase0-config commit 7367d6e,
                coral/data/puzzle_dataset.py _iter_train_coherent().

The bug in the original ARC code: the iteration dropped ALL underfull batches
because it couldn't distinguish mid-epoch underfull (real data, should be
padded and yielded) from end-of-epoch partial (legitimate to drop).

Condition before fix:
    if total < gbs:
        break   # fired on every batch if demo counts didn't divide gbs

Condition after fix:
    if total < gbs and p_idx >= len(puzzle_queue):
        break   # only drop the genuinely last partial batch

Adapted pattern here: the collate function never drops underfull batches.
The DataLoader decides drop_last policy.  What the fix *prevents* is the
collate function itself silently failing on batches smaller than some assumed
minimum.  Implementation: collate_graphs accepts any batch size ≥ 1.
"""

from __future__ import annotations

from typing import Any, Dict, List

import torch

from marifah.data.adapter.attention_mask import pad_attention_masks
from marifah.data.adapter.batch_format import GraphBatch
from marifah.data.adapter.positional import K_PE_DEFAULT
from marifah.data.adapter.tokenizer import NODE_FEAT_DIM

_PAD_LABEL = -1  # padding sentinel for integer label tensors


def collate_graphs(batch: List[Dict[str, Any]]) -> GraphBatch:
    """Collate a list of per-DAG dicts into a GraphBatch.

    Pads to the maximum node count in the current batch (not the dataset maximum).
    Never drops underfull batches — that is the DataLoader's responsibility via
    drop_last (adapting the underfull-batch fix from arc/phase0-config 7367d6e).

    Parameters
    ----------
    batch:
        List of dicts from GraphDAGDataset.__getitem__.  Each dict must contain:
        node_feat, attention_mask, pos_encoding, edges, primitive_ids,
        region_assignments, primitive_assignments, halt_step,
        workflow_type_id, execution_trace.

    Returns
    -------
    GraphBatch with all tensors padded to max_nodes_in_batch.
    """
    if not batch:
        raise ValueError("collate_graphs received an empty batch")

    B = len(batch)
    node_counts = [item["num_nodes"] for item in batch]
    N_max = max(node_counts)
    k_pe = batch[0]["pos_encoding"].shape[-1] if "pos_encoding" in batch[0] else K_PE_DEFAULT

    # --- node_features: (B, N_max, NODE_FEAT_DIM) ---
    node_features = torch.zeros(B, N_max, NODE_FEAT_DIM, dtype=torch.float32)
    for i, item in enumerate(batch):
        nf = item["node_feat"]
        n = nf.shape[0] if isinstance(nf, torch.Tensor) else len(nf)
        nf_t = torch.as_tensor(nf, dtype=torch.float32)
        node_features[i, :n] = nf_t

    # --- attention_mask: (B, N_max, N_max) ---
    masks = [item["attention_mask"] for item in batch]
    attention_mask = pad_attention_masks(masks, N_max)

    # --- node_mask: (B, N_max) bool ---
    node_mask = torch.zeros(B, N_max, dtype=torch.bool)
    for i, nc in enumerate(node_counts):
        node_mask[i, :nc] = True

    # --- pos_encoding: (B, N_max, k_pe) ---
    pos_encoding = torch.zeros(B, N_max, k_pe, dtype=torch.float32)
    for i, item in enumerate(batch):
        pe = item["pos_encoding"]
        n = pe.shape[0]
        pos_encoding[i, :n] = pe if isinstance(pe, torch.Tensor) else torch.as_tensor(pe)

    # --- workflow_type_id: (B,) ---
    workflow_type_id = torch.tensor(
        [item["workflow_type_id"] for item in batch], dtype=torch.int64
    )

    # --- region_assignments: (B, N_max) — pattern_id per node, -1 for padding ---
    region_assignments = torch.full((B, N_max), _PAD_LABEL, dtype=torch.int64)
    for i, item in enumerate(batch):
        ra = item["region_assignments"]
        n = len(ra)
        region_assignments[i, :n] = torch.tensor(ra, dtype=torch.int64)

    # --- primitive_assignments: (B, N_max) ---
    primitive_assignments = torch.full((B, N_max), _PAD_LABEL, dtype=torch.int64)
    for i, item in enumerate(batch):
        pa = item["primitive_assignments"]
        n = len(pa)
        primitive_assignments[i, :n] = torch.tensor(pa, dtype=torch.int64)

    # --- halt_step: (B,) ---
    halt_step = torch.tensor(
        [item["halt_step"] for item in batch], dtype=torch.int64
    )

    # --- execution_trace: list of per-DAG trace lists (not padded) ---
    execution_trace = [item["execution_trace"] for item in batch]

    return GraphBatch(
        node_features=node_features,
        attention_mask=attention_mask,
        node_mask=node_mask,
        pos_encoding=pos_encoding,
        workflow_type_id=workflow_type_id,
        region_assignments=region_assignments,
        primitive_assignments=primitive_assignments,
        halt_step=halt_step,
        execution_trace=execution_trace,
        cycle_annotations=None,
    )
