"""Shared utilities for graph-DAG training: batch preparation, region label derivation.

Factored out to avoid circular imports between trainer.py and eval_loop.py.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict

import torch

if TYPE_CHECKING:
    from marifah.data.adapter.batch_format import GraphBatch
    from marifah.training.config import TrainingConfig


def derive_region_labels(
    region_assignments: torch.Tensor,
    num_regions: int,
    num_pattern_types: int,
) -> torch.Tensor:
    """Map per-node pattern assignments to per-learnable-region labels.

    The HMSC R auxiliary head expects (B, num_regions) labels in [0, num_pattern_types).
    Our data has per-node pattern_ids.  This function assigns each learnable region slot r
    the majority pattern_id among nodes whose pattern_id % num_regions == r.

    Args:
        region_assignments: (B, N) int64, pattern_id per node (-1 for padding)
        num_regions:        number of learnable region tokens in the HMSC
        num_pattern_types:  number of distinct pattern classes

    Returns:
        (B, num_regions) int64, labels in [0, num_pattern_types)
    """
    B = region_assignments.shape[0]
    labels = torch.zeros(B, num_regions, dtype=torch.long, device=region_assignments.device)

    for b in range(B):
        valid = region_assignments[b][region_assignments[b] >= 0]
        if valid.numel() == 0:
            for r in range(num_regions):
                labels[b, r] = r % num_pattern_types
            continue
        for r in range(num_regions):
            bucket = valid[valid % num_regions == r]
            if bucket.numel() > 0:
                mode_val = int(bucket.mode().values.item())
                labels[b, r] = mode_val % num_pattern_types
            else:
                labels[b, r] = r % num_pattern_types

    return labels


def prepare_batch_for_model(
    graph_batch: "GraphBatch",
    config: "TrainingConfig",
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """Convert a GraphBatch to the dict format expected by CoralV3Inner.

    Pads all tensors to config.model.max_nodes so CORAL's fixed seq_len is respected.
    Derives region_labels from region_assignments using majority-vote heuristic.
    Converts workflow_type_id from 1-indexed (dataset) to 0-indexed (CE loss).

    Returns:
        dict with keys: inputs, node_mask, workflow_labels, region_labels,
                        primitive_labels
    """
    B = graph_batch.batch_size
    max_nodes = config.model.max_nodes
    vocab_size = config.model.vocab_size

    hmsc_cfg = config.model.hmsc if config.model.use_hmsc else None
    num_regions = hmsc_cfg.num_regions if hmsc_cfg else 4
    num_pattern_types = hmsc_cfg.num_pattern_types if hmsc_cfg else 12
    num_workflow_types = hmsc_cfg.num_workflow_types if hmsc_cfg else 50

    prim_ids = graph_batch.primitive_ids.to(device)   # (B, N)
    N = prim_ids.shape[1]

    def _pad_or_crop(t: torch.Tensor, pad_val: int, dtype: torch.dtype) -> torch.Tensor:
        if N < max_nodes:
            p = torch.full((B, max_nodes - N), pad_val, dtype=dtype, device=device)
            return torch.cat([t, p], dim=1)
        return t[:, :max_nodes]

    # inputs: primitive IDs clamped to vocab range
    prim_ids_pad = _pad_or_crop(prim_ids, 0, torch.long).clamp(0, vocab_size - 1)

    # node_mask: float32 (B, max_nodes)
    nm = graph_batch.node_mask.to(device).float()
    if N < max_nodes:
        nm = torch.cat([nm, torch.zeros(B, max_nodes - N, device=device)], dim=1)
    else:
        nm = nm[:, :max_nodes]

    # workflow_labels: 1-indexed → 0-indexed
    wf = (graph_batch.workflow_type_id.to(device) - 1).clamp(0, num_workflow_types - 1)

    # region_labels from per-node region_assignments
    reg_assign = graph_batch.region_assignments.to(device)
    region_labels = derive_region_labels(reg_assign, num_regions, num_pattern_types).to(device)

    # primitive_labels padded with -1 sentinel
    prim_labels = _pad_or_crop(graph_batch.primitive_assignments.to(device), -1, torch.long)

    return {
        "inputs": prim_ids_pad,
        "node_mask": nm,
        "workflow_labels": wf,
        "region_labels": region_labels,
        "primitive_labels": prim_labels,
    }
