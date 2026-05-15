"""Unit tests for auxiliary loss heads and compute_aux_losses."""

import torch
import pytest

from marifah.models.hmsc.auxiliary_heads import (
    GlobalAuxHead,
    RegionalAuxHead,
    PerPositionAuxHead,
    compute_aux_losses,
)


# Small head instances for testing
K_G, K_R, K_P = 8, 4, 6
N_WF, N_PAT, N_PRIM = 5, 3, 10
NUM_REGIONS = 3


def make_heads():
    g = GlobalAuxHead(K_G=K_G, num_workflow_types=N_WF)
    r = RegionalAuxHead(K_R=K_R, num_regions=NUM_REGIONS, num_pattern_types=N_PAT)
    p = PerPositionAuxHead(K_P=K_P, num_primitives=N_PRIM)
    return g, r, p


def make_batch_tensors(B=2, N=6):
    g_logits = torch.randn(B, K_G)
    r_logits = torch.randn(B, NUM_REGIONS, K_R)
    p_attn = torch.softmax(torch.randn(B, N, K_P), dim=-1)
    node_mask = torch.ones(B, N, dtype=torch.bool)
    node_mask[0, 4:] = False
    wf_labels = torch.randint(0, N_WF, (B,))
    reg_labels = torch.randint(0, NUM_REGIONS, (B, NUM_REGIONS))
    prim_labels = torch.randint(0, N_PRIM, (B, N))
    return g_logits, r_logits, p_attn, node_mask, wf_labels, reg_labels, prim_labels


class TestAuxHeadShapes:
    def test_global_head_shape(self):
        g, _, _ = make_heads()
        logits = torch.randn(3, K_G)
        out = g(logits)
        assert out.shape == (3, N_WF)

    def test_regional_head_shape(self):
        _, r, _ = make_heads()
        logits = torch.randn(3, NUM_REGIONS, K_R)
        out = r(logits)
        assert out.shape == (3, NUM_REGIONS, N_PAT)

    def test_perposition_head_shape(self):
        _, _, p = make_heads()
        attn = torch.randn(3, 7, K_P)
        out = p(attn)
        assert out.shape == (3, 7, N_PRIM)


class TestAuxLossValues:
    def test_losses_are_real_numbers(self):
        g, r, p = make_heads()
        g_log, r_log, p_attn, mask, wf_lab, reg_lab, prim_lab = make_batch_tensors()
        result = compute_aux_losses(
            g_logits=g_log, r_logits=r_log, p_attn=p_attn,
            g_head=g, r_head=r, p_head=p,
            workflow_labels=wf_lab, region_labels=reg_lab, primitive_labels=prim_lab,
            node_mask=mask, lambda_G=0.1, lambda_R=0.1, lambda_P=0.1,
        )
        for key in ("L_G", "L_R", "L_P", "L_aux_total"):
            assert torch.isfinite(result[key]), f"{key} is not finite: {result[key]}"

    def test_losses_zero_gradient_when_lambda_zero(self):
        """With lambda=0, aux head parameters should get zero gradient."""
        g, r, p = make_heads()
        g_log, r_log, p_attn, mask, wf_lab, reg_lab, prim_lab = make_batch_tensors()

        result = compute_aux_losses(
            g_logits=g_log, r_logits=r_log, p_attn=p_attn,
            g_head=g, r_head=r, p_head=p,
            workflow_labels=wf_lab, region_labels=reg_lab, primitive_labels=prim_lab,
            node_mask=mask, lambda_G=0.0, lambda_R=0.0, lambda_P=0.0,
        )
        result["L_aux_total"].backward()

        # With all lambdas = 0, head parameters receive zero gradient
        for name, param in list(g.named_parameters()) + list(r.named_parameters()) + list(p.named_parameters()):
            if param.grad is not None:
                assert param.grad.abs().max() < 1e-8, \
                    f"Head param {name} has non-zero grad with lambda=0: {param.grad.abs().max()}"

    def test_losses_nonzero_gradient_when_lambda_nonzero(self):
        """With lambda>0, gradient should flow to head parameters."""
        g, r, p = make_heads()
        g_log, r_log, p_attn, mask, wf_lab, reg_lab, prim_lab = make_batch_tensors()

        result = compute_aux_losses(
            g_logits=g_log.requires_grad_(True),
            r_logits=r_log.requires_grad_(True),
            p_attn=p_attn.requires_grad_(True),
            g_head=g, r_head=r, p_head=p,
            workflow_labels=wf_lab, region_labels=reg_lab, primitive_labels=prim_lab,
            node_mask=mask, lambda_G=0.1, lambda_R=0.1, lambda_P=0.1,
        )
        result["L_aux_total"].backward()

        grads_found = any(
            p.grad is not None and p.grad.abs().max() > 0
            for p in list(g.parameters()) + list(r.parameters()) + list(p.parameters())
        )
        assert grads_found, "No non-zero gradients on head params with lambda > 0"

    def test_no_labels_returns_zero_raw_loss(self):
        g, r, p = make_heads()
        g_log, r_log, p_attn, mask, _, _, _ = make_batch_tensors()
        result = compute_aux_losses(
            g_logits=g_log, r_logits=r_log, p_attn=p_attn,
            g_head=g, r_head=r, p_head=p,
            workflow_labels=None, region_labels=None, primitive_labels=None,
            node_mask=mask, lambda_G=0.5, lambda_R=0.5, lambda_P=0.5,
        )
        # Raw losses should be zero when no labels
        assert result["L_G_raw"].item() == 0.0
        assert result["L_R_raw"].item() == 0.0
        assert result["L_P_raw"].item() == 0.0

    def test_result_keys_present(self):
        g, r, p = make_heads()
        g_log, r_log, p_attn, mask, wf_lab, reg_lab, prim_lab = make_batch_tensors()
        result = compute_aux_losses(
            g_logits=g_log, r_logits=r_log, p_attn=p_attn,
            g_head=g, r_head=r, p_head=p,
            workflow_labels=wf_lab, region_labels=reg_lab, primitive_labels=prim_lab,
            node_mask=mask,
        )
        for key in ("L_G", "L_R", "L_P", "L_aux_total", "L_G_raw", "L_R_raw", "L_P_raw"):
            assert key in result, f"Missing key: {key}"
