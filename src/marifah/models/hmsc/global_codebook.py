"""Global codebook (G-scale): workflow-signature crystallization.

Reads mean-pooled carry over real nodes; emits a broadcast mode vector
identical for all nodes within a DAG. Captures workflow-level signatures.
"""

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class GlobalCodebook(nn.Module):
    """G-codebook: pooled-carry routing, broadcast mode output.

    K_G = 64, d_G = 512 (matches d_model) per Session 4 committed defaults.
    """

    def __init__(
        self,
        K_G: int = 64,
        d_model: int = 512,
        d_G: int = 512,
    ) -> None:
        super().__init__()
        self.K_G = K_G
        self.d_model = d_model
        self.d_G = d_G

        self.codebook = nn.Parameter(torch.empty(K_G, d_G))
        self.routing_proj = nn.Linear(d_model, K_G, bias=True)

        self._init_weights()

    def _init_weights(self) -> None:
        # Orthogonal-ish init: spread codebook entries across the sphere
        nn.init.normal_(self.codebook, mean=0.0, std=1.0 / math.sqrt(self.d_G))
        nn.init.xavier_uniform_(self.routing_proj.weight)
        nn.init.zeros_(self.routing_proj.bias)

    def forward(
        self,
        carry_state: torch.Tensor,   # (B, N, d_model)
        node_mask: torch.Tensor,     # (B, N) — 1 real, 0 padding
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            mode_output: (B, N, d_G) — broadcast mode, same for all nodes in a DAG
            routing_logits: (B, K_G) — pre-softmax routing scores for auxiliary head
        """
        B, N, _ = carry_state.shape
        mask = node_mask.float().unsqueeze(-1)           # (B, N, 1)

        # Mask-aware mean pool over real nodes
        pooled = (carry_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)  # (B, d_model)

        logits = self.routing_proj(pooled)               # (B, K_G)
        weights = F.softmax(logits, dim=-1)              # (B, K_G)
        mode_mixture = weights @ self.codebook           # (B, d_G)
        mode_output = mode_mixture.unsqueeze(1).expand(-1, N, -1)   # (B, N, d_G)

        return mode_output, logits
