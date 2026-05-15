"""Edge-induced attention mask construction for graph adapter.

Salvage report (CORAL-v3 arc/padding-attention-mask, commit c7e784d)
----------------------------------------------------------------------
Source: coral/models/layers.py, Attention.forward()
What was ported:
  - Additive bias mask convention: 0.0 = attend, -inf = block
  - Mask construction via torch.full + masked_fill
  - Compatibility with torch.scaled_dot_product_attention's attn_mask argument
What was adapted:
  - Source was a 1D padding mask [B, S] for sequence padding (True = valid token).
  - Here the mask is a 2D per-graph edge-induced mask (N, N) where structure
    comes from DAG adjacency rather than padding positions.
  - Self-loops are always added (every node attends to itself).
  - Direction flag controls whether attention is directed (DAG-semantic) or
    bidirectional (undirected graph).
What was not ported:
  - ARC-specific grid-shape assumptions (not applicable to arbitrary DAGs).
  - The pack/unpack logic (that lives in models/attention.py for the varlen path).
"""

from __future__ import annotations

from typing import List, Literal, Tuple

import torch

AttentionDirection = Literal["directed", "bidirectional"]


def build_attention_mask(
    edges: List[Tuple[int, int]],
    num_nodes: int,
    direction: AttentionDirection = "directed",
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """Build (num_nodes, num_nodes) additive-bias attention mask from an edge list.

    Parameters
    ----------
    edges:
        List of (src, dst) directed edge pairs.  src→dst means state flows
        from src to dst in the DAG.
    num_nodes:
        Total node count.
    direction:
        "directed"  — node dst can attend to node src iff src→dst is an edge.
        "bidirectional" — symmetric: both directions allow attention.
    device:
        Target device.

    Returns
    -------
    mask : (num_nodes, num_nodes) float32 tensor
        mask[i, j] = 0.0   if node i can attend to node j
                   = -inf   otherwise
    """
    mask = torch.full(
        (num_nodes, num_nodes), float("-inf"), dtype=torch.float32, device=device
    )

    # Self-loops: every node can attend to itself
    mask.fill_diagonal_(0.0)

    for src, dst in edges:
        if not (0 <= src < num_nodes and 0 <= dst < num_nodes):
            continue
        if direction == "directed":
            # dst attends to src (src is a predecessor of dst)
            mask[dst, src] = 0.0
        else:
            mask[dst, src] = 0.0
            mask[src, dst] = 0.0

    return mask


def pad_attention_masks(
    masks: List[torch.Tensor],
    max_nodes: int,
) -> torch.Tensor:
    """Stack a list of per-graph (N_i, N_i) masks into a (B, max_nodes, max_nodes) batch.

    Padding rows and columns (beyond each graph's actual node count) are
    initialized to -inf so padded positions are fully blocked.
    """
    B = len(masks)
    batched = torch.full(
        (B, max_nodes, max_nodes), float("-inf"), dtype=torch.float32
    )
    for i, mask in enumerate(masks):
        n = mask.shape[0]
        batched[i, :n, :n] = mask
    return batched
