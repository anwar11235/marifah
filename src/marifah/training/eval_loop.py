"""Validation evaluation loop for graph DAG training.

Runs one full pass over a DataLoader with the model in eval mode.
Returns a dict of aggregated metrics (mean over all batches).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Optional

import torch
from torch import nn
from torch.utils.data import DataLoader

from marifah.training.graph_utils import prepare_batch_for_model
from marifah.training.graph_losses import compute_total_loss

if TYPE_CHECKING:
    from marifah.training.config import TrainingConfig


def evaluate(
    model: nn.Module,
    data_loader: DataLoader,
    config: "TrainingConfig",
    device: torch.device,
) -> Dict[str, float]:
    """Evaluate model on a DataLoader in eval mode (no grad).

    Args:
        model:       CoralV3Inner whose forward returns
                     (carry, logits, q_logits_tuple, pred_metrics)
        data_loader: DataLoader yielding GraphBatch items
        config:      TrainingConfig
        device:      Computation device

    Returns:
        Dict with mean scalar metrics:
            loss_main, loss_halt, loss_aux_G, loss_aux_R, loss_aux_P,
            accuracy_node, codebook_<scale>_<stat> (when HMSC enabled)
    """
    from marifah.models.coral_base import InnerCarry

    model.eval()
    totals: Dict[str, float] = {}
    n_batches = 0

    def _add(key: str, val: float) -> None:
        totals[key] = totals.get(key, 0.0) + val

    with torch.no_grad():
        for batch in data_loader:
            graph_batch = batch.to(device)
            B = graph_batch.batch_size

            carry = InnerCarry.zeros(B, config.model, device)

            coral_batch = prepare_batch_for_model(graph_batch, config, device)
            result = model(carry, coral_batch, is_last_segment=True)

            logits = result[1]
            q_tuple = result[2]
            pred_metrics = result[3] if len(result) > 3 else None
            q_halt_logits = q_tuple[0] if isinstance(q_tuple, tuple) else None

            loss_dict = compute_total_loss(
                logits=logits,
                q_halt_logits=q_halt_logits,
                pred_metrics=pred_metrics,
                graph_batch=graph_batch,
                config=config,
                device=device,
            )
            _add("loss_main", float(loss_dict["main"].item()))
            _add("loss_halt", float(loss_dict["halt"].item()))
            _add("loss_aux_G", float(loss_dict["aux_G"].item()))
            _add("loss_aux_R", float(loss_dict["aux_R"].item()))
            _add("loss_aux_P", float(loss_dict["aux_P"].item()))

            # Node accuracy
            node_mask = graph_batch.node_mask.to(device)
            prim_labels = graph_batch.primitive_assignments.to(device)
            N_b = node_mask.shape[1]
            N_m = logits.shape[1]
            if N_b < N_m:
                pad_m = torch.zeros(B, N_m - N_b, dtype=torch.bool, device=device)
                node_mask_p = torch.cat([node_mask, pad_m], dim=1)
                pad_l = torch.full((B, N_m - N_b), -1, dtype=torch.long, device=device)
                prim_labels_p = torch.cat([prim_labels, pad_l], dim=1)
            else:
                node_mask_p = node_mask[:, :N_m]
                prim_labels_p = prim_labels[:, :N_m]

            valid = node_mask_p & (prim_labels_p >= 0)
            if valid.any():
                preds = logits.argmax(dim=-1)
                _add("_correct", float((preds == prim_labels_p)[valid].sum().item()))
                _add("_valid", float(valid.sum().item()))

            # HMSC codebook utilisation
            if pred_metrics is not None and pred_metrics.hmsc_utilization:
                for k, v in pred_metrics.hmsc_utilization.items():
                    val = float(v.item()) if hasattr(v, "item") else float(v)
                    _add(f"codebook_{k}", val)

            n_batches += 1

    n = max(n_batches, 1)
    metrics: Dict[str, float] = {k: v / n for k, v in totals.items()
                                  if not k.startswith("_")}

    # Accuracy: ratio over all real nodes seen
    if totals.get("_valid", 0.0) > 0:
        metrics["accuracy_node"] = totals["_correct"] / totals["_valid"]
    else:
        metrics["accuracy_node"] = 0.0

    return metrics
