"""Unit tests for GlobalCodebook (G-scale)."""

import torch
import pytest

from marifah.models.hmsc.global_codebook import GlobalCodebook


def make_cb(**kw):
    defaults = dict(K_G=8, d_model=16, d_G=16)
    defaults.update(kw)
    return GlobalCodebook(**defaults)


def make_input(B=3, N=7, d=16):
    carry = torch.randn(B, N, d)
    mask = torch.ones(B, N)
    mask[0, 5:] = 0   # first DAG has only 5 real nodes
    return carry, mask


class TestGlobalCodebookShape:
    def test_output_shapes(self):
        cb = make_cb()
        carry, mask = make_input(B=3, N=7, d=16)
        mode_out, logits = cb(carry, mask)
        assert mode_out.shape == (3, 7, 16), f"mode_output shape wrong: {mode_out.shape}"
        assert logits.shape == (3, 8), f"logits shape wrong: {logits.shape}"

    def test_broadcast_identical_across_nodes(self):
        """All nodes in a DAG get the same mode vector."""
        cb = make_cb()
        carry, mask = make_input(B=2, N=5, d=16)
        mode_out, _ = cb(carry, mask)
        # All N positions should be identical within each batch item
        for b in range(2):
            diffs = (mode_out[b] - mode_out[b, 0:1]).abs().max()
            assert diffs < 1e-6, f"batch {b}: mode vectors differ across nodes ({diffs})"

    def test_different_d_G(self):
        cb = GlobalCodebook(K_G=4, d_model=8, d_G=12)
        carry = torch.randn(2, 5, 8)
        mask = torch.ones(2, 5)
        mode_out, logits = cb(carry, mask)
        assert mode_out.shape == (2, 5, 12)
        assert logits.shape == (2, 4)


class TestGlobalCodebookMask:
    def test_padding_doesnt_pollute_pool(self):
        """Two inputs identical except for padded positions should produce same output."""
        cb = make_cb(K_G=4, d_model=8, d_G=8)
        carry = torch.randn(1, 6, 8)
        mask_short = torch.zeros(1, 6)
        mask_short[0, :3] = 1   # only first 3 nodes real

        # Perturb padded positions — should not affect output
        carry_perturbed = carry.clone()
        carry_perturbed[0, 3:] = 999.0

        with torch.no_grad():
            out1, log1 = cb(carry, mask_short)
            out2, log2 = cb(carry_perturbed, mask_short)

        assert torch.allclose(log1, log2, atol=1e-5), "Perturbing padding changed logits"
        assert torch.allclose(out1, out2, atol=1e-5), "Perturbing padding changed output"

    def test_all_nodes_real(self):
        cb = make_cb()
        carry, _ = make_input()
        mask_all = torch.ones(3, 7)
        mode_out, logits = cb(carry, mask_all)
        assert torch.isfinite(mode_out).all()
        assert torch.isfinite(logits).all()


class TestGlobalCodebookGradient:
    def test_gradients_flow_to_codebook(self):
        cb = make_cb()
        carry, mask = make_input()
        mode_out, logits = cb(carry, mask)
        loss = mode_out.sum() + logits.sum()
        loss.backward()
        assert cb.codebook.grad is not None, "No grad on codebook"
        assert cb.routing_proj.weight.grad is not None, "No grad on routing_proj.weight"
        assert torch.isfinite(cb.codebook.grad).all()

    def test_determinism(self):
        cb = make_cb()
        carry, mask = make_input(B=2, N=4, d=16)
        with torch.no_grad():
            out1, log1 = cb(carry, mask)
            out2, log2 = cb(carry, mask)
        assert torch.allclose(out1, out2), "Non-deterministic output"
        assert torch.allclose(log1, log2), "Non-deterministic logits"


class TestGlobalCodebookInit:
    def test_codebook_not_degenerate(self):
        """Codebook entries should not all be near-zero or near-identical."""
        cb = GlobalCodebook(K_G=64, d_model=512, d_G=512)
        norms = cb.codebook.data.norm(dim=-1)
        assert norms.min() > 1e-4, f"Some codebook entries near-zero: min norm = {norms.min()}"
        # Check entries are diverse: pairwise cosine sim should not all be ~1
        c = cb.codebook.data / norms.unsqueeze(-1).clamp(min=1e-8)
        sim = c @ c.T   # (K_G, K_G)
        # Off-diagonal max cosine similarity
        mask = ~torch.eye(64, dtype=torch.bool)
        max_off_diag = sim[mask].abs().max()
        assert max_off_diag < 0.99, f"Codebook entries are near-identical (max cosine sim = {max_off_diag})"
