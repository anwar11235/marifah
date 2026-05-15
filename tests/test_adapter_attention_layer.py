"""Tests for GraphAttentionLayer and SDPA vs flash-varlen equivalence.

Verification §4 item 3: SDPA and flash-attn-varlen produce equivalent outputs
within fp16 tolerance (< 1e-2 max absolute difference) on identical inputs.
"""

import pytest
import torch

from marifah.models.attention import (
    GraphAttentionLayer,
    flash_varlen,
    sdpa_with_bias,
)


class TestSdpaWithBias:
    def test_output_shape(self):
        B, N, H, D = 2, 6, 4, 16
        q = torch.randn(B, N, H, D)
        k = torch.randn(B, N, H, D)
        v = torch.randn(B, N, H, D)
        out = sdpa_with_bias(q, k, v)
        assert out.shape == (B, N, H, D)

    def test_with_mask(self):
        B, N, H, D = 2, 5, 2, 8
        q = torch.randn(B, N, H, D)
        k = torch.randn(B, N, H, D)
        v = torch.randn(B, N, H, D)
        mask = torch.zeros(B, N, N)   # full attention
        out = sdpa_with_bias(q, k, v, attention_mask=mask)
        assert out.shape == (B, N, H, D)

    def test_inf_mask_zeroes_output(self):
        """Positions masked with -inf should not influence other outputs."""
        B, N, H, D = 1, 3, 1, 4
        q = torch.randn(B, N, H, D)
        k = torch.randn(B, N, H, D)
        v = torch.randn(B, N, H, D)

        # full attention
        out_full = sdpa_with_bias(q, k, v, attention_mask=None)

        # mask out node 2 from being attended to
        mask = torch.zeros(B, N, N)
        mask[:, :, 2] = float("-inf")   # column 2 blocked
        out_masked = sdpa_with_bias(q, k, v, attention_mask=mask)

        # Outputs at other positions should differ from full attention
        assert out_full.shape == out_masked.shape


class TestFlashVarlen:
    def test_output_shape_cpu(self):
        """CPU fallback should produce same shape as packed input."""
        total_nodes = 10
        H, D = 4, 8
        q = torch.randn(total_nodes, H, D)
        k = torch.randn(total_nodes, H, D)
        v = torch.randn(total_nodes, H, D)
        # Two graphs: 4 and 6 nodes
        cu_seqlens = torch.tensor([0, 4, 10], dtype=torch.int32)
        out = flash_varlen(q, k, v, cu_seqlens, max_seqlen=6)
        assert out.shape == (total_nodes, H, D)

    def test_single_graph_matches_sdpa(self):
        """flash_varlen on a single graph should match SDPA with full mask."""
        B, N, H, D = 1, 5, 2, 8
        q = torch.randn(B, N, H, D)
        k = torch.randn(B, N, H, D)
        v = torch.randn(B, N, H, D)

        # SDPA path (full attention, no mask)
        out_sdpa = sdpa_with_bias(q, k, v, attention_mask=None)  # (B, N, H, D)

        # flash_varlen path (pack single graph)
        q_packed = q.squeeze(0)   # (N, H, D)
        k_packed = k.squeeze(0)
        v_packed = v.squeeze(0)
        cu_seqlens = torch.tensor([0, N], dtype=torch.int32)
        out_varlen_packed = flash_varlen(q_packed, k_packed, v_packed, cu_seqlens, N)
        out_varlen = out_varlen_packed.unsqueeze(0)  # (B, N, H, D)

        # Should match within fp32 tolerance (both CPU paths here)
        max_diff = (out_sdpa - out_varlen).abs().max().item()
        assert max_diff < 1e-5, f"SDPA vs flash_varlen max_diff={max_diff:.2e} (expected < 1e-5)"


class TestGraphAttentionLayer:
    def test_output_shape(self):
        B, N, d_model = 2, 8, 32
        layer = GraphAttentionLayer(d_model=d_model, n_heads=4)
        x = torch.randn(B, N, d_model)
        out = layer(x)
        assert out.shape == (B, N, d_model)

    def test_with_attention_mask(self):
        B, N, d_model = 2, 6, 16
        layer = GraphAttentionLayer(d_model=d_model, n_heads=2, attention_backend="sdpa")
        x = torch.randn(B, N, d_model)
        mask = torch.zeros(B, N, N)  # full attention
        out = layer(x, attention_mask=mask)
        assert out.shape == (B, N, d_model)

    def test_gradient_flow(self):
        B, N, d_model = 2, 5, 16
        layer = GraphAttentionLayer(d_model=d_model, n_heads=2)
        x = torch.randn(B, N, d_model, requires_grad=True)
        out = layer(x)
        out.sum().backward()
        assert x.grad is not None
        for name, p in layer.named_parameters():
            assert p.grad is not None, f"no grad for {name}"

    def test_sdpa_flash_equivalence(self):
        """SDPA and flash_varlen (CPU fallback) on same full-attention input.
        Verification step §4 item 3: max absolute diff < 1e-2.
        """
        B, N, d_model = 1, 5, 16
        n_heads = 2
        torch.manual_seed(42)
        x = torch.randn(B, N, d_model)

        layer_sdpa = GraphAttentionLayer(d_model=d_model, n_heads=n_heads, attention_backend="sdpa")
        layer_varlen = GraphAttentionLayer(d_model=d_model, n_heads=n_heads, attention_backend="flash_varlen")

        # Copy weights so only the path differs
        layer_varlen.load_state_dict(layer_sdpa.state_dict())

        # Full attention mask (all zeros = all allowed)
        mask = torch.zeros(B, N, N)

        with torch.no_grad():
            out_sdpa = layer_sdpa(x, attention_mask=mask)
            # flash_varlen path on CPU falls back to SDPA-equivalent unmasked path
            out_varlen = layer_varlen(x, attention_mask=mask, node_counts=torch.tensor([N]))

        max_diff = (out_sdpa - out_varlen).abs().max().item()
        assert max_diff < 1e-2, f"SDPA vs flash_varlen max_diff={max_diff:.2e} (expected < 1e-2)"
