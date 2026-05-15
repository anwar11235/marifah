"""Auxiliary loss heads — one per HMSC scale.

All three heads are built and wired. Lambdas = 0.0 by default (Session 4);
losses are computed for logging but don't contribute to the gradient.
Session 5/6 engages them with non-zero lambdas.

Critical: when lambda=0, the loss tensor carries 0.0 * cross_entropy. This
is computed (for monitoring) but contributes zero to the backward graph.
"""

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class GlobalAuxHead(nn.Module):
    """Map G routing logits → workflow type classification."""

    def __init__(self, K_G: int, num_workflow_types: int) -> None:
        super().__init__()
        self.classifier = nn.Linear(K_G, num_workflow_types, bias=True)
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, routing_logits: torch.Tensor) -> torch.Tensor:
        """routing_logits: (B, K_G) → workflow_logits: (B, num_workflow_types)."""
        return self.classifier(routing_logits)


class RegionalAuxHead(nn.Module):
    """Map R routing logits → per-region pattern classification."""

    def __init__(self, K_R: int, num_regions: int, num_pattern_types: int) -> None:
        super().__init__()
        self.num_regions = num_regions
        self.classifier = nn.Linear(K_R, num_pattern_types, bias=True)
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, routing_logits: torch.Tensor) -> torch.Tensor:
        """routing_logits: (B, num_regions, K_R) → (B, num_regions, num_pattern_types)."""
        B, R, K = routing_logits.shape
        flat = routing_logits.reshape(B * R, K)
        out = self.classifier(flat)
        return out.reshape(B, R, -1)


class PerPositionAuxHead(nn.Module):
    """Map P attention weights → per-node primitive classification."""

    def __init__(self, K_P: int, num_primitives: int) -> None:
        super().__init__()
        self.classifier = nn.Linear(K_P, num_primitives, bias=True)
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, attention_weights: torch.Tensor) -> torch.Tensor:
        """attention_weights: (B, N, K_P) → (B, N, num_primitives)."""
        B, N, K = attention_weights.shape
        flat = attention_weights.reshape(B * N, K)
        out = self.classifier(flat)
        return out.reshape(B, N, -1)


def compute_aux_losses(
    g_logits: torch.Tensor,            # (B, K_G) routing logits from G-codebook
    r_logits: torch.Tensor,            # (B, num_regions, K_R) routing logits from R-codebook
    p_attn: torch.Tensor,              # (B, N, K_P) attention weights from P-codebook
    g_head: GlobalAuxHead,
    r_head: RegionalAuxHead,
    p_head: PerPositionAuxHead,
    workflow_labels: Optional[torch.Tensor],   # (B,) int64
    region_labels: Optional[torch.Tensor],     # (B, num_regions) int64
    primitive_labels: Optional[torch.Tensor],  # (B, N) int64
    node_mask: torch.Tensor,                   # (B, N) bool — mask padding
    lambda_G: float = 0.0,
    lambda_R: float = 0.0,
    lambda_P: float = 0.0,
) -> Dict[str, torch.Tensor]:
    """Compute auxiliary cross-entropy losses at all three scales.

    Losses are always computed (for logging) but lambda=0 → zero gradient
    contribution. Returns dict with L_G, L_R, L_P, L_aux_total.
    """
    device = g_logits.device
    zero = torch.zeros(1, device=device, dtype=torch.float32)

    # --- Global loss ---
    if workflow_labels is not None:
        g_pred = g_head(g_logits)                    # (B, num_workflow_types)
        L_G_raw = F.cross_entropy(g_pred, workflow_labels.long())
    else:
        L_G_raw = zero.squeeze()
    L_G = lambda_G * L_G_raw

    # --- Regional loss ---
    if region_labels is not None:
        B, num_regions, _ = r_logits.shape
        r_pred = r_head(r_logits)                    # (B, num_regions, num_pattern_types)
        # Flatten regions: each region is an independent classification
        r_pred_flat = r_pred.reshape(B * num_regions, -1)
        r_labels_flat = region_labels.reshape(B * num_regions).long()
        L_R_raw = F.cross_entropy(r_pred_flat, r_labels_flat)
    else:
        L_R_raw = zero.squeeze()
    L_R = lambda_R * L_R_raw

    # --- Per-position loss ---
    if primitive_labels is not None:
        p_pred = p_head(p_attn)                      # (B, N, num_primitives)
        # Only compute loss on real (non-padding) nodes
        mask = node_mask.bool()
        p_pred_real = p_pred[mask]                   # (real_nodes, num_primitives)
        p_labels_real = primitive_labels[mask].long()
        if p_pred_real.shape[0] > 0:
            L_P_raw = F.cross_entropy(p_pred_real, p_labels_real)
        else:
            L_P_raw = zero.squeeze()
    else:
        L_P_raw = zero.squeeze()
    L_P = lambda_P * L_P_raw

    L_aux_total = L_G + L_R + L_P

    return {
        "L_G": L_G,
        "L_R": L_R,
        "L_P": L_P,
        "L_G_raw": L_G_raw.detach(),
        "L_R_raw": L_R_raw.detach(),
        "L_P_raw": L_P_raw.detach(),
        "L_aux_total": L_aux_total,
    }
