"""Regional codebook (R-scale): sub-DAG pattern crystallization.

Regions are discovered via learned attention pooling — the model decides what
counts as a region, not graph topology. Ground-truth region labels are used only
for auxiliary supervision and probing (per codebook design §5.4).

K_R = 16, d_R = 256, num_regions = 8 per Session 4 committed defaults.
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class RegionalCodebook(nn.Module):
    """R-codebook: region attention pooling + per-region routing.

    Soft assignment (default forward path): each node's mode_output is a
    weighted sum over region_modes, where weights come from the transpose of
    the region-attention mechanism.

    Hard assignment (for probing / aux loss with ground-truth labels): each
    node maps directly to its labeled region's mode.
    """

    def __init__(
        self,
        K_R: int = 16,
        d_model: int = 512,
        d_R: int = 256,
        num_regions: int = 8,
    ) -> None:
        super().__init__()
        self.K_R = K_R
        self.d_model = d_model
        self.d_R = d_R
        self.num_regions = num_regions

        self.codebook = nn.Parameter(torch.empty(K_R, d_R))
        # Learnable query tokens for region attention pooling
        self.region_tokens = nn.Parameter(torch.empty(num_regions, d_model))

        # Projections for multi-head attention (num_heads=1 for simplicity)
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        self.routing_proj = nn.Linear(d_model, K_R, bias=True)
        self.output_proj = nn.Linear(d_R, d_R, bias=False)

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.codebook, mean=0.0, std=1.0 / math.sqrt(self.d_R))
        nn.init.normal_(self.region_tokens, mean=0.0, std=1.0 / math.sqrt(self.d_model))
        for proj in (self.q_proj, self.k_proj, self.v_proj, self.out_proj):
            nn.init.xavier_uniform_(proj.weight)
        nn.init.xavier_uniform_(self.routing_proj.weight)
        nn.init.zeros_(self.routing_proj.bias)
        nn.init.eye_(self.output_proj.weight) if self.d_R == self.d_R else \
            nn.init.xavier_uniform_(self.output_proj.weight)

    def _region_attention(
        self,
        carry_state: torch.Tensor,   # (B, N, d_model)
        node_mask: torch.Tensor,     # (B, N)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Attend region tokens (queries) to carry_state (keys/values).

        Returns:
            region_features: (B, num_regions, d_model)
            node_to_region_weights: (B, N, num_regions) softmax weights for soft assignment
        """
        B, N, D = carry_state.shape
        R = self.num_regions
        scale = 1.0 / math.sqrt(D)

        # Region tokens as queries: (B, R, D)
        queries = self.q_proj(self.region_tokens.unsqueeze(0).expand(B, -1, -1))
        keys = self.k_proj(carry_state)    # (B, N, D)
        values = self.v_proj(carry_state)  # (B, N, D)

        # Attention scores: (B, R, N)
        scores = torch.bmm(queries, keys.transpose(1, 2)) * scale

        # Mask padding nodes: -inf so they don't contribute to region pooling
        if node_mask is not None:
            pad_mask = (~node_mask.bool()).unsqueeze(1)  # (B, 1, N)
            scores = scores.masked_fill(pad_mask, float("-inf"))

        # Region-to-node attention weights: (B, R, N)
        region_attn = F.softmax(scores, dim=-1)
        # Guard against all-inf rows (fully-padded — shouldn't happen in practice)
        region_attn = torch.nan_to_num(region_attn, nan=0.0)

        # Region features: (B, R, D)
        region_features = self.out_proj(torch.bmm(region_attn, values))

        # Transpose for soft node-to-region assignment: (B, N, R)
        # Use carry_state @ region_features^T (re-project for symmetry)
        node_region_scores = torch.bmm(
            self.k_proj(carry_state),
            region_features.transpose(1, 2),
        ) * scale   # (B, N, R)
        node_to_region_weights = F.softmax(node_region_scores, dim=-1)  # (B, N, R)

        return region_features, node_to_region_weights

    def forward(
        self,
        carry_state: torch.Tensor,                      # (B, N, d_model)
        node_mask: torch.Tensor,                        # (B, N)
        region_assignments: Optional[torch.Tensor] = None,  # (B, N) ground-truth region IDs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            mode_output: (B, N, d_R) — per-node mode via soft (or hard) region assignment
            routing_logits: (B, num_regions, K_R) — for auxiliary head
        """
        B, N, _ = carry_state.shape

        region_features, node_to_region_weights = self._region_attention(carry_state, node_mask)
        # region_features: (B, R, d_model), node_to_region_weights: (B, N, R)

        logits = self.routing_proj(region_features)     # (B, R, K_R)
        weights = F.softmax(logits, dim=-1)             # (B, R, K_R)
        region_modes = weights @ self.codebook          # (B, R, d_R)
        region_modes = self.output_proj(region_modes)   # (B, R, d_R)

        if region_assignments is not None:
            # Hard assignment path (probing / eval with ground-truth labels)
            # region_assignments: (B, N) with values in [0, num_regions)
            idx = region_assignments.clamp(0, self.num_regions - 1)  # (B, N)
            idx_exp = idx.unsqueeze(-1).expand(-1, -1, self.d_R)     # (B, N, d_R)
            mode_output = region_modes.gather(1, idx_exp)             # (B, N, d_R)
        else:
            # Soft assignment (default training path): weighted sum over regions
            mode_output = torch.bmm(node_to_region_weights, region_modes)  # (B, N, d_R)

        return mode_output, logits
