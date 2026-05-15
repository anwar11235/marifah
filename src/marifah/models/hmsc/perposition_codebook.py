"""Per-position codebook (P-scale): reasoning primitive crystallization.

Cross-attention from each node to K_P codebook entries. Soft during training
(gradients flow); hard top-1 at eval for interpretability probes (OD3).

K_P = 16, d_P = 128 per Session 4 committed defaults.
"""

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class PerPositionCodebook(nn.Module):
    """P-codebook: cross-attention to codebook entries per node.

    discreteness="soft": differentiable softmax weights (training default).
    discreteness="hard": top-1 one-hot (eval/probing only; no gradient through selection).
    """

    def __init__(
        self,
        K_P: int = 16,
        d_model: int = 512,
        d_P: int = 128,
        attention_temperature: float = 1.0,
        discreteness: str = "soft",
    ) -> None:
        super().__init__()
        assert discreteness in ("soft", "hard"), f"discreteness must be 'soft' or 'hard', got {discreteness!r}"
        self.K_P = K_P
        self.d_model = d_model
        self.d_P = d_P
        self.attention_temperature = attention_temperature
        self.discreteness = discreteness

        self.codebook = nn.Parameter(torch.empty(K_P, d_P))
        self.query_proj = nn.Linear(d_model, d_P, bias=False)
        self.key_proj = nn.Linear(d_P, d_P, bias=False)
        self.output_proj = nn.Linear(d_P, d_P, bias=False)

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.codebook, mean=0.0, std=1.0 / math.sqrt(self.d_P))
        nn.init.xavier_uniform_(self.query_proj.weight)
        nn.init.eye_(self.key_proj.weight)
        nn.init.xavier_uniform_(self.output_proj.weight)

    def forward(
        self,
        carry_state: torch.Tensor,   # (B, N, d_model)
        node_mask: torch.Tensor,     # (B, N) — 1 real, 0 padding
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            mode_output: (B, N, d_P) — per-node mode mixture
            attention_weights: (B, N, K_P) — softmax weights (for auxiliary head)
        """
        scale = 1.0 / (math.sqrt(self.d_P) * self.attention_temperature)

        queries = self.query_proj(carry_state)          # (B, N, d_P)
        keys = self.key_proj(self.codebook)             # (K_P, d_P)

        # Attention scores: (B, N, K_P)
        scores = torch.einsum("bnd,kd->bnk", queries, keys) * scale

        if self.discreteness == "soft":
            weights = F.softmax(scores, dim=-1)         # (B, N, K_P)
        else:
            # Hard top-1: one-hot, no gradient through selection
            top1 = scores.argmax(dim=-1)                # (B, N)
            weights = F.one_hot(top1, num_classes=self.K_P).float()

        mode_mixture = weights @ self.codebook          # (B, N, d_P)
        mode_output = self.output_proj(mode_mixture)    # (B, N, d_P)

        return mode_output, weights
