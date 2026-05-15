"""GraphBatch dataclass and tensor layout specification.

Tensor layout
-------------
node_features       (B, N_max, node_feat_dim)
    Raw per-node features before the NodeTokenizer embedding step.
    Consists of primitive_ids packed alongside an attribute float vector.
    Specifically:
        node_features[b, n, 0]     = primitive_id (int, cast to float for padding)
        node_features[b, n, 1:5]   = attribute encoding vector (ATTR_DIM = 4 floats)
    Padded positions have node_features set to zero.

attention_mask      (B, N_max, N_max)
    Additive bias mask (float32): 0.0 where attention is allowed, -inf elsewhere.
    Edge-induced and directed: node i can attend to node j iff there exists a
    directed edge j→i in the DAG, OR i == j (self-loop always allowed).
    Padding rows/columns are -inf throughout.
    Conventions match the SDPA additive_mask semantics of torch.scaled_dot_product_attention.
    Salvaged from CORAL-v3 arc/padding-attention-mask (c7e784d).

node_mask           (B, N_max)
    Boolean: True for real nodes, False for padding.

pos_encoding        (B, N_max, K_pe)
    Laplacian eigenvector positional encoding; zeros for padding nodes.
    K_pe = 8 by default.

workflow_type_id    (B,) int64
    Integer workflow type label 1-50.

region_assignments  (B, N_max) int64
    Per-node pattern ID (0-11) for regional codebook auxiliary loss.
    Padding positions have value -1.

primitive_assignments   (B, N_max) int64
    Per-node primitive ID (0-9) for per-position codebook auxiliary loss.
    Padding positions have value -1.

halt_step           (B,) int64
    Ground-truth halt step index.

execution_trace     list[list[dict]]
    Per-DAG execution traces; variable length; stored as nested Python list
    (no padding — consumed by faithfulness probe, not by CORAL forward pass).

cycle_annotations   None
    Reserved for cyclic-mode future work (§5 out of scope).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch


@dataclass
class GraphBatch:
    """One collated batch of DAG records, ready for the adapter→CORAL pipeline."""

    node_features: torch.Tensor           # (B, N_max, node_feat_dim=5)
    attention_mask: torch.Tensor          # (B, N_max, N_max) float32 additive bias
    node_mask: torch.Tensor               # (B, N_max) bool
    pos_encoding: torch.Tensor            # (B, N_max, K_pe)
    workflow_type_id: torch.Tensor        # (B,) int64
    region_assignments: torch.Tensor      # (B, N_max) int64, -1 for padding
    primitive_assignments: torch.Tensor   # (B, N_max) int64, -1 for padding
    halt_step: torch.Tensor               # (B,) int64
    execution_trace: List[List[Dict[str, Any]]]
    cycle_annotations: None = None

    @property
    def batch_size(self) -> int:
        return int(self.node_features.shape[0])

    @property
    def max_nodes(self) -> int:
        return int(self.node_features.shape[1])

    @property
    def primitive_ids(self) -> torch.Tensor:
        """(B, N_max) int64 — first channel of node_features."""
        return self.node_features[..., 0].long()

    @property
    def attr_vec(self) -> torch.Tensor:
        """(B, N_max, ATTR_DIM) float — remaining channels of node_features."""
        return self.node_features[..., 1:]

    def to(self, device: torch.device) -> "GraphBatch":
        return GraphBatch(
            node_features=self.node_features.to(device),
            attention_mask=self.attention_mask.to(device),
            node_mask=self.node_mask.to(device),
            pos_encoding=self.pos_encoding.to(device),
            workflow_type_id=self.workflow_type_id.to(device),
            region_assignments=self.region_assignments.to(device),
            primitive_assignments=self.primitive_assignments.to(device),
            halt_step=self.halt_step.to(device),
            execution_trace=self.execution_trace,
            cycle_annotations=None,
        )
