"""Unit tests for graph_losses.compute_total_loss."""

import torch
import pytest
from unittest.mock import MagicMock

from marifah.training.graph_losses import compute_total_loss
from marifah.training.config import TrainingConfig


def _make_config(lambda_G=0.0, lambda_R=0.0, lambda_P=0.0, use_hmsc=False):
    raw = {
        "model": {"use_hmsc": use_hmsc, "max_nodes": 10, "vocab_size": 5},
        "training": {
            "lambda_G": lambda_G, "lambda_R": lambda_R, "lambda_P": lambda_P,
            "main_loss_weight": 1.0, "halt_loss_weight": 0.1,
        },
    }
    return TrainingConfig(**raw)


def _make_graph_batch(B=2, N=6, vocab_size=5):
    """Minimal GraphBatch-like object."""
    from marifah.data.adapter.batch_format import GraphBatch
    node_mask = torch.ones(B, N, dtype=torch.bool)
    prim_assign = torch.randint(0, vocab_size, (B, N))
    return GraphBatch(
        node_features=torch.zeros(B, N, 5),
        attention_mask=torch.zeros(B, N, N),
        node_mask=node_mask,
        pos_encoding=torch.zeros(B, N, 4),
        workflow_type_id=torch.ones(B, dtype=torch.int64),
        region_assignments=torch.zeros(B, N, dtype=torch.int64),
        primitive_assignments=prim_assign,
        halt_step=torch.zeros(B, dtype=torch.int64),
        execution_trace=[[] for _ in range(B)],
    )


class TestComputeTotalLoss:
    def test_returns_expected_keys(self):
        cfg = _make_config()
        gb = _make_graph_batch()
        B, N = 2, 6
        logits = torch.randn(B, 10, 5)  # max_nodes=10 > N=6
        q_halt = torch.zeros(B)
        result = compute_total_loss(logits, q_halt, None, gb, cfg, torch.device("cpu"))
        for key in ("total", "main", "halt", "aux_G", "aux_R", "aux_P", "aux_total"):
            assert key in result, f"Missing key: {key}"

    def test_all_finite(self):
        cfg = _make_config()
        gb = _make_graph_batch()
        logits = torch.randn(2, 10, 5)
        q_halt = torch.zeros(2)
        result = compute_total_loss(logits, q_halt, None, gb, cfg, torch.device("cpu"))
        for k, v in result.items():
            assert torch.isfinite(v), f"{k} is not finite: {v}"

    def test_aux_zero_when_no_hmsc(self):
        cfg = _make_config(lambda_G=0.1, lambda_R=0.1, lambda_P=0.1, use_hmsc=False)
        gb = _make_graph_batch()
        logits = torch.randn(2, 10, 5)
        result = compute_total_loss(logits, None, None, gb, cfg, torch.device("cpu"))
        assert result["aux_G"].item() == 0.0
        assert result["aux_R"].item() == 0.0
        assert result["aux_P"].item() == 0.0

    def test_aux_nonzero_when_hmsc_active(self):
        cfg = _make_config(lambda_G=0.1, lambda_R=0.1, lambda_P=0.1, use_hmsc=True)
        gb = _make_graph_batch()
        logits = torch.randn(2, 10, 5)

        pred_metrics = MagicMock()
        pred_metrics.hmsc_aux_losses = {
            "L_G": torch.tensor(0.5),
            "L_R": torch.tensor(0.3),
            "L_P": torch.tensor(0.2),
        }
        result = compute_total_loss(logits, None, pred_metrics, gb, cfg, torch.device("cpu"))
        assert result["aux_G"].item() == pytest.approx(0.5, abs=1e-5)
        assert result["aux_total"].item() > 0.0

    def test_total_is_backward_able(self):
        cfg = _make_config()
        gb = _make_graph_batch()
        logits = torch.randn(2, 10, 5, requires_grad=True)
        result = compute_total_loss(logits, None, None, gb, cfg, torch.device("cpu"))
        result["total"].backward()
        assert logits.grad is not None

    def test_masking_excludes_padding(self):
        cfg = _make_config()
        B, N = 1, 4
        from marifah.data.adapter.batch_format import GraphBatch
        node_mask = torch.tensor([[True, True, False, False]])
        prim_assign = torch.tensor([[0, 1, -1, -1]])
        gb = GraphBatch(
            node_features=torch.zeros(B, N, 5),
            attention_mask=torch.zeros(B, N, N),
            node_mask=node_mask,
            pos_encoding=torch.zeros(B, N, 4),
            workflow_type_id=torch.ones(B, dtype=torch.int64),
            region_assignments=torch.zeros(B, N, dtype=torch.int64),
            primitive_assignments=prim_assign,
            halt_step=torch.zeros(B, dtype=torch.int64),
            execution_trace=[[]],
        )
        logits = torch.zeros(B, 10, 5)
        logits[0, 0, 0] = 10.0
        logits[0, 1, 1] = 10.0
        result = compute_total_loss(logits, None, None, gb, cfg, torch.device("cpu"))
        assert result["main"].item() < 0.1, "Loss should be ~0 for correct predictions"
