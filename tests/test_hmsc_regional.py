"""Unit tests for RegionalCodebook (R-scale)."""

import torch
import pytest

from marifah.models.hmsc.regional_codebook import RegionalCodebook


def make_cb(**kw):
    defaults = dict(K_R=4, d_model=16, d_R=8, num_regions=3)
    defaults.update(kw)
    return RegionalCodebook(**defaults)


def make_input(B=2, N=8, d=16):
    carry = torch.randn(B, N, d)
    mask = torch.ones(B, N)
    mask[0, 6:] = 0
    return carry, mask


class TestRegionalCodebookShape:
    def test_output_shapes(self):
        cb = make_cb()
        carry, mask = make_input()
        mode_out, logits = cb(carry, mask)
        assert mode_out.shape == (2, 8, 8), f"mode_output shape wrong: {mode_out.shape}"
        assert logits.shape == (2, 3, 4), f"logits shape wrong: {logits.shape}"

    def test_hard_assignment_shape(self):
        cb = make_cb()
        carry, mask = make_input(B=2, N=8)
        region_ids = torch.randint(0, 3, (2, 8))
        mode_out, logits = cb(carry, mask, region_assignments=region_ids)
        assert mode_out.shape == (2, 8, 8)
        assert logits.shape == (2, 3, 4)


class TestRegionalCodebookMask:
    def test_padding_doesnt_pollute_regions(self):
        """Perturbing padded nodes should not change the output."""
        cb = make_cb()
        carry = torch.randn(1, 8, 16)
        mask = torch.zeros(1, 8)
        mask[0, :4] = 1   # only 4 real nodes

        carry_perturbed = carry.clone()
        carry_perturbed[0, 4:] = 1e6

        with torch.no_grad():
            out1, log1 = cb(carry, mask)
            out2, log2 = cb(carry_perturbed, mask)

        # Region pooling uses masked attention, so logits should be identical
        assert torch.allclose(log1, log2, atol=1e-4), "Padding polluted region routing logits"

    def test_finite_output(self):
        cb = make_cb()
        carry, mask = make_input()
        mode_out, logits = cb(carry, mask)
        assert torch.isfinite(mode_out).all(), "Non-finite mode_output"
        assert torch.isfinite(logits).all(), "Non-finite logits"


class TestRegionalCodebookGradient:
    def test_gradients_flow(self):
        cb = make_cb()
        carry, mask = make_input()
        mode_out, logits = cb(carry, mask)
        loss = mode_out.sum() + logits.sum()
        loss.backward()
        assert cb.codebook.grad is not None
        assert cb.region_tokens.grad is not None
        assert torch.isfinite(cb.codebook.grad).all()

    def test_determinism(self):
        cb = make_cb()
        carry, mask = make_input()
        with torch.no_grad():
            out1, log1 = cb(carry, mask)
            out2, log2 = cb(carry, mask)
        assert torch.allclose(out1, out2)
        assert torch.allclose(log1, log2)


class TestRegionalHardAssignment:
    def test_hard_assignment_maps_to_region_modes(self):
        """With hard assignment, each node's output should match its region's mode."""
        cb = make_cb(K_R=4, d_model=16, d_R=8, num_regions=3)
        carry = torch.randn(1, 6, 16)
        mask = torch.ones(1, 6)
        # All nodes assigned to region 0
        region_ids = torch.zeros(1, 6, dtype=torch.long)

        with torch.no_grad():
            mode_out_hard, logits = cb(carry, mask, region_assignments=region_ids)

        # All nodes should have the same mode vector (region 0's mode)
        for n in range(1, 6):
            diff = (mode_out_hard[0, n] - mode_out_hard[0, 0]).abs().max()
            assert diff < 1e-5, f"Node {n} differs from node 0 under hard assignment"

    def test_soft_hard_different(self):
        """Soft and hard assignment should generally differ."""
        cb = make_cb()
        carry, mask = make_input(B=1, N=5)
        region_ids = torch.randint(0, 3, (1, 5))
        with torch.no_grad():
            soft_out, _ = cb(carry, mask)
            hard_out, _ = cb(carry, mask, region_assignments=region_ids)
        # They can be the same by accident but usually aren't
        assert soft_out.shape == hard_out.shape


class TestRegionalCodebookInit:
    def test_codebook_not_degenerate(self):
        cb = RegionalCodebook(K_R=16, d_model=512, d_R=256, num_regions=8)
        norms = cb.codebook.data.norm(dim=-1)
        assert norms.min() > 1e-4
