"""Sparse columnar routing — replaces monolithic TransformerBlocks with S columns + index-select router.

Strategy C (index-select routing). Phase 2 columnar routing is currently STUBBED
(NotImplementedError in CoralV3Inner dispatch). This module is kept for forward
compatibility with checkpoints that have routing weights and for future revival.
"""

from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from marifah.models.layers import CastedLinear, CosSin
from marifah.models.transformer_block import TransformerBlock, TransformerBlockConfig


class ColumnarTransformerBlock(nn.Module):
    """S smaller columns + index-select router replacing a single TransformerBlock."""

    def __init__(self, config: TransformerBlockConfig, S: int = 8, k: int = 2) -> None:
        super().__init__()
        if k > S:
            raise ValueError(f"k={k} must be <= S={S}")
        self.S = S
        self.k = k

        col_expansion = max(1.0, config.expansion * 2 / S)
        col_config = TransformerBlockConfig(
            hidden_size=config.hidden_size,
            num_heads=config.num_heads,
            expansion=col_expansion,
            rms_norm_eps=config.rms_norm_eps,
        )
        self.columns = nn.ModuleList([TransformerBlock(col_config) for _ in range(S)])
        self.router = CastedLinear(config.hidden_size, S, bias=False)
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(
        self,
        cos_sin: Optional[CosSin],
        hidden_states: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, seq, D = hidden_states.shape

        temp = self.temperature.clamp(0.1, 10.0)
        routing_logits = self.router(hidden_states.mean(dim=1)) / temp
        topk_vals, topk_idx = routing_logits.topk(self.k, dim=-1)
        weights = F.softmax(topk_vals, dim=-1)

        flat_idx = topk_idx.reshape(-1)
        flat_weights = weights.reshape(-1)
        sample_idx = (
            torch.arange(B, device=hidden_states.device)
            .unsqueeze(1)
            .expand(B, self.k)
            .reshape(-1)
        )

        result = torch.zeros_like(hidden_states)

        active_cols = flat_idx.unique()
        for s_tensor in active_cols:
            s = s_tensor.item()
            col_mask = flat_idx == s
            entries = col_mask.nonzero(as_tuple=True)[0]
            src_samples = sample_idx[entries]
            src_weights = flat_weights[entries]
            sub_batch = hidden_states[src_samples]
            col_out = self.columns[s](cos_sin=cos_sin, hidden_states=sub_batch)
            result.index_add_(
                0,
                src_samples,
                (src_weights.unsqueeze(-1).unsqueeze(-1) * col_out).to(result.dtype),
            )

        return result, routing_logits


class ColumnarReasoningModule(nn.Module):
    """Drop-in replacement for ReasoningModule using ColumnarTransformerBlocks."""

    def __init__(
        self,
        config: TransformerBlockConfig,
        num_layers: int,
        S: int = 8,
        k: int = 2,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [ColumnarTransformerBlock(config, S=S, k=k) for _ in range(num_layers)]
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_injection: torch.Tensor,
        cos_sin: Optional[CosSin] = None,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        hidden_states = hidden_states + input_injection
        all_routing_logits: List[torch.Tensor] = []
        for layer in self.layers:
            hidden_states, routing_logits = layer(cos_sin=cos_sin, hidden_states=hidden_states)
            all_routing_logits.append(routing_logits)
        return hidden_states, all_routing_logits
