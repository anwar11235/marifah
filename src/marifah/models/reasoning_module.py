"""ReasoningModule — a stack of TransformerBlocks with additive input injection."""

from typing import List, Optional

import torch
from torch import nn

from marifah.models.layers import CosSin
from marifah.models.transformer_block import TransformerBlock


class ReasoningModule(nn.Module):
    """A stack of TransformerBlocks with an additive input injection at the entry.

    Forward pass:
        1. hidden_states = hidden_states + input_injection
        2. hidden_states = block_N(...(block_1(hidden_states))...)
    """

    def __init__(self, layers: List[TransformerBlock]) -> None:
        super().__init__()
        self.layers = nn.ModuleList(layers)

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_injection: torch.Tensor,
        cos_sin: Optional[CosSin] = None,
    ) -> torch.Tensor:
        hidden_states = hidden_states + input_injection
        for layer in self.layers:
            hidden_states = layer(cos_sin=cos_sin, hidden_states=hidden_states)
        return hidden_states
