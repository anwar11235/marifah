"""Graph DAG training loss aggregation.

Centralises all loss sources:
  - main:  cross-entropy on per-node primitive-type predictions (mask-aware)
  - halt:  binary cross-entropy on the Q-halt head (1-step training target)
  - aux_*: HMSC hierarchical auxiliary losses (G / R / P scale), weighted by lambdas

Keeps the Session-1 losses.py (ACTLossHead, CoralV3LossHead) untouched.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Optional

import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    from marifah.data.adapter.batch_format import GraphBatch
    from marifah.models.coral import PredMetrics
    from marifah.training.config import TrainingConfig


def compute_total_loss(
    logits: torch.Tensor,
    q_halt_logits: Optional[torch.Tensor],
    pred_metrics: Optional["PredMetrics"],
    graph_batch: "GraphBatch",
    config: "TrainingConfig",
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """Aggregate all loss components into a single backward-able total.

    Args:
        logits:        (B, N, vocab_size) model output before softmax
        q_halt_logits: (B,) Q-halt head output; None if model ran without ACT
        pred_metrics:  PredMetrics from CoralV3Inner, carries hmsc_aux_losses
        graph_batch:   collated batch from the adapter
        config:        TrainingConfig
        device:        target device

    Returns:
        dict with keys: total (backward), main, halt, aux_G, aux_R, aux_P, aux_total
        All values except 'total' are detached scalars for logging.
    """
    zero = torch.tensor(0.0, device=device)
    node_mask = graph_batch.node_mask.to(device)          # (B, N) bool
    primitive_labels = graph_batch.primitive_assignments.to(device)  # (B, N) int64

    # --- Main loss: per-node primitive prediction (mask-aware) ---
    N_model = logits.shape[1]
    N_batch = node_mask.shape[1]

    if N_batch < N_model:
        pad = torch.zeros(node_mask.shape[0], N_model - N_batch, dtype=torch.bool, device=device)
        node_mask_padded = torch.cat([node_mask, pad], dim=1)
        pad_labels = torch.full((primitive_labels.shape[0], N_model - N_batch), -1,
                                dtype=torch.long, device=device)
        primitive_labels_padded = torch.cat([primitive_labels, pad_labels], dim=1)
    else:
        node_mask_padded = node_mask[:, :N_model]
        primitive_labels_padded = primitive_labels[:, :N_model]

    valid_mask = node_mask_padded & (primitive_labels_padded >= 0)
    flat_logits = logits.float()[valid_mask]          # (n_valid, vocab_size)
    flat_labels = primitive_labels_padded[valid_mask] # (n_valid,)

    if flat_labels.numel() > 0:
        main_loss = F.cross_entropy(flat_logits, flat_labels)
    else:
        main_loss = zero

    # --- Halt loss: binary CE with all-halt target (1-step gradient) ---
    halt_loss = zero
    if q_halt_logits is not None and config.training.halt_loss_weight > 0:
        halt_targets = torch.ones_like(q_halt_logits.float())
        halt_loss = F.binary_cross_entropy_with_logits(q_halt_logits.float(), halt_targets)

    # --- HMSC auxiliary losses (already weighted by lambdas set at HMSC init) ---
    aux_G = zero
    aux_R = zero
    aux_P = zero
    if pred_metrics is not None and pred_metrics.hmsc_aux_losses is not None:
        h = pred_metrics.hmsc_aux_losses
        aux_G = h.get("L_G", zero).to(device)
        aux_R = h.get("L_R", zero).to(device)
        aux_P = h.get("L_P", zero).to(device)

    aux_total = aux_G + aux_R + aux_P

    total = (
        config.training.main_loss_weight * main_loss
        + config.training.halt_loss_weight * halt_loss
        + aux_total
    )

    return {
        "total": total,
        "main": main_loss.detach(),
        "halt": halt_loss.detach(),
        "aux_G": aux_G.detach(),
        "aux_R": aux_R.detach(),
        "aux_P": aux_P.detach(),
        "aux_total": aux_total.detach(),
    }
