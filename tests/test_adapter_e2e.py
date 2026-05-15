"""End-to-end smoke test: dataset -> collate -> CORAL forward -> loss -> backward.

Verification §2.6 and §4 item 6.  Does NOT validate convergence — only that
the full pipeline plumbing works without errors and gradients flow correctly.

The CORAL model receives primitive IDs as input tokens (primitive vocabulary
size = 10), with seq_len set to the batch's max_nodes.  This is the minimal
integration that exercises the adapter -> model interface.
"""

import pytest
import torch
import torch.nn.functional as F
from pathlib import Path

from marifah.data.adapter.dataset import GraphDAGDataset
from marifah.data.adapter.collate import collate_graphs
from marifah.data.adapter.tokenizer import NodeTokenizer
from marifah.models.attention import GraphAttentionLayer


# ---------------------------------------------------------------------------
# Minimal graph model for the smoke test
# (wraps NodeTokenizer + GraphAttentionLayer, avoids modifying CORAL)
# ---------------------------------------------------------------------------

class TinyGraphModel(torch.nn.Module):
    """Minimal learnable model: tokenize nodes, apply attention, predict primitive."""

    def __init__(self, d_model: int = 32, n_heads: int = 2, num_primitives: int = 10):
        super().__init__()
        self.tokenizer = NodeTokenizer(d_model=d_model)
        self.attn = GraphAttentionLayer(d_model=d_model, n_heads=n_heads, attention_backend="sdpa")
        self.head = torch.nn.Linear(d_model, num_primitives)

    def forward(self, batch):
        # Embed nodes: (B, N, d_model)
        x = self.tokenizer(batch.primitive_ids, batch.attr_vec)
        # Apply attention: (B, N, d_model)
        x = self.attn(x, attention_mask=batch.attention_mask)
        # Predict per-node primitive type: (B, N, 10)
        return self.head(x)


def _make_tiny_val_dataset(tmp_path: Path) -> Path:
    """Generate a tiny validation split."""
    from marifah.data.synthetic.generator import DagGenerator
    from marifah.data.synthetic.storage import write_split
    from marifah.data.synthetic.vertical_config import GeneratorConfig, SplitSizes
    from marifah.data.synthetic.vertical_config import _hash_config

    cfg = GeneratorConfig(seed=77)
    cfg.split_sizes = SplitSizes(train=5, val=10, test_id=5, test_ood_size=3, test_ood_composition=3)
    cfg.config_hash = _hash_config(cfg)

    gen = DagGenerator(cfg)
    records = gen.generate_split("val", 10, seed_offset=60_000)
    write_split(records, tmp_path, "val")
    return tmp_path / "val"


@pytest.fixture(scope="module")
def e2e_val_dir(tmp_path_factory):
    return _make_tiny_val_dataset(tmp_path_factory.mktemp("e2e_ds"))


class TestEndToEnd:
    def test_forward_pass_no_error(self, e2e_val_dir):
        from torch.utils.data import DataLoader
        ds = GraphDAGDataset(e2e_val_dir, k_pe=4)
        assert len(ds) > 0

        loader = DataLoader(ds, batch_size=4, collate_fn=collate_graphs, shuffle=False)
        batch = next(iter(loader))

        model = TinyGraphModel(d_model=32)
        with torch.no_grad():
            logits = model(batch)

        assert logits.shape == (batch.batch_size, batch.max_nodes, 10)

    def test_loss_is_finite(self, e2e_val_dir):
        from torch.utils.data import DataLoader
        ds = GraphDAGDataset(e2e_val_dir, k_pe=4)
        loader = DataLoader(ds, batch_size=4, collate_fn=collate_graphs, shuffle=False)
        batch = next(iter(loader))

        model = TinyGraphModel(d_model=32)
        logits = model(batch)   # (B, N_max, 10)

        # Compute cross-entropy only on real (non-padding) nodes
        node_mask = batch.node_mask   # (B, N_max) bool
        targets = batch.primitive_assignments   # (B, N_max) int64, -1 for padding

        flat_logits = logits[node_mask]         # (total_real_nodes, 10)
        flat_targets = targets[node_mask]       # (total_real_nodes,)

        loss = F.cross_entropy(flat_logits, flat_targets)
        assert torch.isfinite(loss), f"Loss is not finite: {loss.item()}"
        assert loss.item() > 0.0, "Loss is zero (suspicious)"

    def test_backward_pass_no_error(self, e2e_val_dir):
        from torch.utils.data import DataLoader
        ds = GraphDAGDataset(e2e_val_dir, k_pe=4)
        loader = DataLoader(ds, batch_size=4, collate_fn=collate_graphs, shuffle=False)
        batch = next(iter(loader))

        model = TinyGraphModel(d_model=32)
        logits = model(batch)

        node_mask = batch.node_mask
        targets = batch.primitive_assignments
        flat_logits = logits[node_mask]
        flat_targets = targets[node_mask]
        loss = F.cross_entropy(flat_logits, flat_targets)
        loss.backward()

        # Verify gradients are non-None and finite
        for name, p in model.named_parameters():
            assert p.grad is not None, f"No gradient for {name}"
            assert torch.isfinite(p.grad).all(), f"Non-finite gradient for {name}"

    def test_coral_inner_forward(self, e2e_val_dir):
        """Smoke test CoralInner with primitive IDs as token inputs."""
        from torch.utils.data import DataLoader
        from marifah.models.coral_base import CoralInner, CoralConfig

        ds = GraphDAGDataset(e2e_val_dir, k_pe=4)
        loader = DataLoader(ds, batch_size=2, collate_fn=collate_graphs, shuffle=False)
        batch = next(iter(loader))

        B = batch.batch_size
        N = batch.max_nodes

        # Configure CORAL for graph token vocab (10 primitives)
        config = CoralConfig(
            batch_size=B,
            seq_len=N,
            vocab_size=10,
            H_cycles=1,
            L_cycles=1,
            H_layers=1,
            L_layers=1,
            hidden_size=32,
            num_heads=2,
            forward_dtype="float32",
        )
        model = CoralInner(config)
        carry = model.empty_carry(B)
        carry.z_H = carry.z_H.detach().clone()
        carry.z_L = carry.z_L.detach().clone()

        # Use primitive_assignments as token IDs (clamped to [0, 9])
        inputs = batch.primitive_ids.clamp(0, 9)   # (B, N_max)

        coral_batch = {"inputs": inputs}
        with torch.no_grad():
            new_carry, output, q_logits = model.forward(carry, coral_batch)

        assert output.shape == (B, N, 10), f"Unexpected output shape: {output.shape}"
        assert torch.isfinite(output).all(), "CORAL output contains non-finite values"

    def test_determinism(self, e2e_val_dir):
        """Same dataset, same seed -> identical batch across two loader iterations."""
        from torch.utils.data import DataLoader
        ds = GraphDAGDataset(e2e_val_dir, k_pe=4)
        loader = DataLoader(ds, batch_size=4, collate_fn=collate_graphs, shuffle=False)

        batch1 = next(iter(loader))
        batch2 = next(iter(loader))

        assert torch.allclose(batch1.node_features, batch2.node_features), \
            "Identical-seed batches should produce identical node_features"
        assert torch.allclose(batch1.attention_mask, batch2.attention_mask), \
            "Identical-seed batches should produce identical attention_mask"
