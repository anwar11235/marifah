"""Predictive coding components for CORAL Phase 1.

PredictionNet: Maps H-state to a prediction of L-state (mu_L).
PrecisionNet:  Produces per-dimension precision (inverse variance) from L-state.
"""

import torch
import torch.nn.functional as F
from torch import nn

from marifah.models.layers import CastedLinear


class PredictionNet(nn.Module):
    """Maps H-module hidden state to a prediction of L-module hidden state.

    Two-layer MLP: h_dim → l_dim*2 → l_dim.
    """

    def __init__(self, h_dim: int, l_dim: int) -> None:
        super().__init__()
        self.fc1 = CastedLinear(h_dim, l_dim * 2, bias=False)
        self.fc2 = CastedLinear(l_dim * 2, l_dim, bias=False)

    def forward(self, z_H: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(z_H)))


class PrecisionNet(nn.Module):
    """Produces per-dimension precision (inverse variance) from L-module state.

    Two-layer MLP with softplus + eps_min output.
    The eps_min = 0.01 floor ensures precision never reaches zero.
    """

    EPS_MIN: float = 0.01

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.fc1 = CastedLinear(dim, dim, bias=False)
        self.fc2 = CastedLinear(dim, dim, bias=False)

    def forward(self, z_L: torch.Tensor) -> torch.Tensor:
        return F.softplus(self.fc2(F.gelu(self.fc1(z_L)))) + self.EPS_MIN
