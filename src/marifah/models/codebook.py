"""Phase 3b Soft MoE Crystallization components.

SpatialMoECodebook:
  K_modes full-spatial [seq_len, l_dim] expert templates + 1 passthrough expert.
  Softmax router; soft blend z_L_out = w_pt*z_L_rec + (1-w_pt)*z_bypass.

CrystallizationBuffer:
  Ring buffer of (recognition_key, pooled_z_L, spatial_z_L) triples on CPU.
  Used during bootstrap to collect spatial z_L for k-means consolidation.
"""

from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn

from marifah.models.layers import CastedLinear
from marifah.models.coral_base import CoralConfig


class SpatialMoECodebook(nn.Module):
    """Spatially-structured Soft MoE codebook for Phase 3b.

    K_modes full-spatial [seq_len, l_dim] expert templates + one passthrough expert.
    Router MLP computes softmax weights over all K+1 experts.
    Unweighted L_recon makes w_pt=1.0 an unstable equilibrium (anti-passthrough-dominance).
    """

    def __init__(self, config: CoralConfig, seq_len: int) -> None:
        super().__init__()
        l_dim = config.hidden_size
        proj_dim = config.crystal_proj_dim
        key_dim = proj_dim * 2
        K_modes = config.moe_num_modes

        self.K_modes = K_modes
        self.seq_len = seq_len
        self.l_dim = l_dim
        self.key_dim = key_dim

        self.proj_h = CastedLinear(l_dim, proj_dim, bias=False)
        self.proj_l = CastedLinear(l_dim, proj_dim, bias=False)

        self.codebook_values = nn.Parameter(
            torch.randn(K_modes, seq_len, l_dim) * 0.02
        )
        self.codebook_keys = nn.Parameter(
            torch.randn(K_modes, key_dim) * 0.02
        )

        self.router_mlp = nn.Sequential(
            nn.Linear(key_dim, 64, bias=True),
            nn.GELU(),
            nn.Linear(64, K_modes + 1, bias=True),
        )

        self._bootstrap_mask_active: bool = False

    def forward(
        self, z_H: torch.Tensor, z_L: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h_pool = self.proj_h(z_H).mean(dim=1)
        l_pool = self.proj_l(z_L).mean(dim=1)
        key = torch.cat([h_pool, l_pool], dim=-1)

        if self._bootstrap_mask_active:
            B = z_H.shape[0]
            K = self.K_modes + 1
            w = torch.zeros(B, K, device=z_H.device, dtype=z_H.dtype)
            w[:, -1] = 1.0
            z_bypass = torch.zeros(
                B, self.seq_len, self.l_dim, device=z_H.device, dtype=z_H.dtype
            )
            return w, z_bypass, key

        logits = self.router_mlp(key.float())
        w = torch.softmax(logits, dim=-1).to(z_H.dtype)
        w_cb = w[:, : self.K_modes]

        z_bypass = torch.einsum(
            "bk,ksd->bsd", w_cb.float(), self.codebook_values
        ).to(z_H.dtype)

        return w, z_bypass, key

    def moe_losses(
        self,
        z_L_final: torch.Tensor,
        w: torch.Tensor,
        z_bypass: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        L_recon = (z_L_final.detach().float() - z_bypass.float()).pow(2).mean()

        w_mean = w.float().mean(dim=0)
        K_plus_one = float(w_mean.shape[-1])
        target = torch.full_like(w_mean, 1.0 / K_plus_one)
        eps = 1e-10
        L_lb = (w_mean * (torch.log(w_mean + eps) - torch.log(target))).sum()

        return L_recon, L_lb

    def bootstrap_mask_router(self, active: bool) -> None:
        self._bootstrap_mask_active = active


class CrystallizationBuffer:
    """Ring buffer collecting (recognition_key, converged_z_L) pairs during training.

    Stored on CPU to avoid long-term GPU memory accumulation.
    Consolidation runs spatial k-means on buffered spatial z_L (Lloyd's algorithm).
    """

    def __init__(self, capacity: int = 10000) -> None:
        self.capacity = capacity
        self.keys: Optional[torch.Tensor] = None
        self.values: Optional[torch.Tensor] = None
        self.spatial_buffer: Optional[torch.Tensor] = None
        self.pointer: int = 0
        self.size: int = 0

    def _lazy_init(self, key_dim: int, value_dim: int) -> None:
        if self.keys is None:
            self.keys = torch.zeros(self.capacity, key_dim, dtype=torch.float32)
            self.values = torch.zeros(self.capacity, value_dim, dtype=torch.float32)

    def _lazy_init_spatial(self, seq_len: int, l_dim: int) -> None:
        if self.spatial_buffer is None:
            self.spatial_buffer = torch.zeros(
                self.capacity, seq_len, l_dim, dtype=torch.float32
            )

    @torch._dynamo.disable
    def add(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
        z_L_spatial: Optional[torch.Tensor] = None,
    ) -> None:
        keys_cpu = keys.detach().to(dtype=torch.float32, device="cpu", non_blocking=True)
        values_cpu = values.detach().to(dtype=torch.float32, device="cpu", non_blocking=True)

        B = keys_cpu.shape[0]
        self._lazy_init(keys_cpu.shape[1], values_cpu.shape[1])

        if B > self.capacity:
            keys_cpu = keys_cpu[-self.capacity :]
            values_cpu = values_cpu[-self.capacity :]
            if z_L_spatial is not None:
                z_L_spatial = z_L_spatial[-self.capacity :]
            B = self.capacity

        indices = (torch.arange(B, dtype=torch.long) + self.pointer) % self.capacity

        self.keys[indices] = keys_cpu
        self.values[indices] = values_cpu

        if z_L_spatial is not None:
            z_L_cpu = z_L_spatial.detach().to(
                dtype=torch.float32, device="cpu", non_blocking=True
            )
            self._lazy_init_spatial(z_L_cpu.shape[1], z_L_cpu.shape[2])
            self.spatial_buffer[indices] = z_L_cpu

        self.pointer = int((self.pointer + B) % self.capacity)
        self.size = min(self.size + B, self.capacity)

    def __len__(self) -> int:
        return self.size

    def clear(self) -> None:
        self.pointer = 0
        self.size = 0
        if self.spatial_buffer is not None:
            self.spatial_buffer.zero_()

    def consolidate_spatial(
        self, k_modes: int, num_iterations: int = 100
    ) -> Optional[Tuple[torch.Tensor, int]]:
        """Run Euclidean k-means on buffered spatial z_L to initialise codebook_values."""
        if self.spatial_buffer is None or self.size < k_modes:
            return None

        N = self.size
        all_spatial = self.spatial_buffer[:N]
        S, D = all_spatial.shape[1], all_spatial.shape[2]
        data_flat = all_spatial.view(N, -1).clone()

        perm = torch.randperm(N)[:k_modes]
        centroids_flat = data_flat[perm].clone()

        assignments = torch.zeros(N, dtype=torch.long)
        SD = data_flat.shape[1]

        with torch.no_grad():
            for _ in range(num_iterations):
                dists = torch.cdist(data_flat, centroids_flat)
                assignments = dists.argmin(dim=1)

                new_centroids = torch.zeros(k_modes, SD, dtype=torch.float32)
                counts = torch.zeros(k_modes, dtype=torch.float32)

                new_centroids.scatter_add_(
                    0,
                    assignments.unsqueeze(1).expand(-1, SD),
                    data_flat,
                )
                counts.scatter_add_(0, assignments, torch.ones(N, dtype=torch.float32))

                filled = counts > 0
                new_centroids[filled] /= counts[filled].unsqueeze(1)
                new_centroids[~filled] = centroids_flat[~filled]
                centroids_flat = new_centroids

        utilization = int((assignments.bincount(minlength=k_modes) > 0).sum().item())
        centroids = centroids_flat.view(k_modes, S, D)
        return centroids, utilization
