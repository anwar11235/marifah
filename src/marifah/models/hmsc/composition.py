"""HMSC composition module: combine three scale mode vectors per node.

Default: sum of projected modes (method="sum").
Alternative: learned gating via small MLP (method="gated") — implemented but
not the default training path per design §3.2.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class HMSCComposition(nn.Module):
    """Compose G, R, P mode vectors into a single per-node output.

    sum (default): composed = proj_G(G) + proj_R(R) + proj_P(P)
    gated: alpha weights from carry state MLP; softmax-normalized weighted sum.

    NOTE: "gated" is implemented for completeness but is NOT the default
    training path. Use method="sum" for all training in Session 4.
    """

    def __init__(
        self,
        d_G: int = 512,
        d_R: int = 256,
        d_P: int = 128,
        d_output: int = 512,
        method: str = "sum",
    ) -> None:
        super().__init__()
        assert method in ("sum", "gated"), f"method must be 'sum' or 'gated', got {method!r}"
        self.method = method
        self.d_output = d_output

        self.proj_G = nn.Linear(d_G, d_output, bias=False)
        self.proj_R = nn.Linear(d_R, d_output, bias=False)
        self.proj_P = nn.Linear(d_P, d_output, bias=False)

        if method == "gated":
            # Small MLP: carry (d_output) → 3 gate logits
            d_hidden = max(d_output // 4, 8)
            self.gate_net = nn.Sequential(
                nn.Linear(d_output, d_hidden, bias=True),
                nn.ReLU(),
                nn.Linear(d_hidden, 3, bias=True),
            )

        self._init_weights()

    def _init_weights(self) -> None:
        for proj in (self.proj_G, self.proj_R, self.proj_P):
            nn.init.xavier_uniform_(proj.weight)
        if self.method == "gated":
            for m in self.gate_net.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        G_mode: torch.Tensor,                        # (B, N, d_G)
        R_mode: torch.Tensor,                        # (B, N, d_R)
        P_mode: torch.Tensor,                        # (B, N, d_P)
        carry_state: Optional[torch.Tensor] = None,  # (B, N, d_output) — gated path only
    ) -> torch.Tensor:
        """Returns composed: (B, N, d_output)."""
        G_proj = self.proj_G(G_mode)   # (B, N, d_output)
        R_proj = self.proj_R(R_mode)   # (B, N, d_output)
        P_proj = self.proj_P(P_mode)   # (B, N, d_output)

        if self.method == "sum":
            return G_proj + R_proj + P_proj

        # Gated path
        assert carry_state is not None, "carry_state required for method='gated'"
        gate_input = carry_state.float()
        gate_logits = self.gate_net(gate_input)         # (B, N, 3)
        alphas = F.softmax(gate_logits, dim=-1)         # (B, N, 3)
        aG = alphas[..., 0:1]
        aR = alphas[..., 1:2]
        aP = alphas[..., 2:3]
        return aG * G_proj + aR * R_proj + aP * P_proj
