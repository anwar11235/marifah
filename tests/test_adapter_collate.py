"""Tests for collate_graphs: variable-node-count padding and masking.

Verification §4 item 7: batch mixing 5-node and 15-node DAGs collates correctly.
"""

import numpy as np
import pytest
import torch

from marifah.data.adapter.batch_format import GraphBatch
from marifah.data.adapter.collate import collate_graphs
from marifah.data.adapter.attention_mask import build_attention_mask
from marifah.data.adapter.positional import compute_laplacian_pe
from marifah.data.adapter.tokenizer import NODE_FEAT_DIM, ATTR_DIM


def _make_fake_item(num_nodes: int, workflow_type_id: int = 1) -> dict:
    """Create a fake per-DAG dict matching GraphDAGDataset output format."""
    edges = [(i, i + 1) for i in range(num_nodes - 1)] if num_nodes > 1 else []
    node_feat = np.random.rand(num_nodes, NODE_FEAT_DIM).astype(np.float32)
    pe = compute_laplacian_pe(edges, num_nodes, k=4)
    mask = build_attention_mask(edges, num_nodes)
    return {
        "dag_id": f"fake_{num_nodes}",
        "workflow_type_id": workflow_type_id,
        "split": "val",
        "seed": 0,
        "num_nodes": num_nodes,
        "node_feat": node_feat,
        "edges": edges,
        "primitive_ids": list(range(num_nodes)),
        "region_assignments": [0] * num_nodes,
        "primitive_assignments": [i % 10 for i in range(num_nodes)],
        "halt_step": num_nodes - 1,
        "execution_trace": [],
        "pos_encoding": pe,
        "attention_mask": mask,
    }


class TestCollateGraphs:
    def test_basic_batch_shapes(self):
        batch = [_make_fake_item(5), _make_fake_item(10)]
        gb = collate_graphs(batch)
        assert isinstance(gb, GraphBatch)
        assert gb.batch_size == 2
        assert gb.max_nodes == 10  # max in batch

    def test_mixed_node_counts(self):
        """Verification §4 item 7: 5-node and 15-node DAGs in same batch."""
        batch = [_make_fake_item(5), _make_fake_item(15)]
        gb = collate_graphs(batch)
        assert gb.max_nodes == 15
        assert gb.node_features.shape == (2, 15, NODE_FEAT_DIM)
        assert gb.attention_mask.shape == (2, 15, 15)
        assert gb.node_mask.shape == (2, 15)
        assert gb.pos_encoding.shape == (2, 15, 4)

    def test_node_mask_correct(self):
        batch = [_make_fake_item(5), _make_fake_item(10)]
        gb = collate_graphs(batch)
        # item 0: first 5 are True
        assert gb.node_mask[0, :5].all()
        assert not gb.node_mask[0, 5:].any()
        # item 1: all 10 are True
        assert gb.node_mask[1, :10].all()

    def test_padding_in_node_features(self):
        """Padding positions (beyond actual node count) should be zero."""
        batch = [_make_fake_item(3), _make_fake_item(8)]
        gb = collate_graphs(batch)
        # item 0 has 3 nodes; positions 3..7 should be zero
        assert torch.allclose(gb.node_features[0, 3:], torch.zeros(5, NODE_FEAT_DIM))

    def test_attention_mask_padding_is_neg_inf(self):
        batch = [_make_fake_item(4), _make_fake_item(9)]
        gb = collate_graphs(batch)
        # Rows and columns beyond node 4 in item 0 should be -inf
        pad_region = gb.attention_mask[0, 4:, :]
        assert (pad_region == float("-inf")).all()

    def test_label_tensors(self):
        batch = [_make_fake_item(5, workflow_type_id=3), _make_fake_item(8, workflow_type_id=7)]
        gb = collate_graphs(batch)
        assert gb.workflow_type_id[0].item() == 3
        assert gb.workflow_type_id[1].item() == 7
        assert gb.halt_step[0].item() == 4
        assert gb.halt_step[1].item() == 7

    def test_primitive_assignments_padding(self):
        batch = [_make_fake_item(5), _make_fake_item(10)]
        gb = collate_graphs(batch)
        # Padding positions should be -1
        assert (gb.primitive_assignments[0, 5:] == -1).all()

    def test_execution_trace_not_padded(self):
        batch = [_make_fake_item(3), _make_fake_item(5)]
        gb = collate_graphs(batch)
        assert len(gb.execution_trace) == 2

    def test_single_item_batch(self):
        """Underfull batch of 1 should not be dropped by collate."""
        batch = [_make_fake_item(7)]
        gb = collate_graphs(batch)
        assert gb.batch_size == 1

    def test_empty_batch_raises(self):
        with pytest.raises(ValueError, match="empty"):
            collate_graphs([])

    def test_primitive_ids_property(self):
        batch = [_make_fake_item(5)]
        gb = collate_graphs(batch)
        assert gb.primitive_ids.shape == gb.node_features.shape[:2]
        assert gb.primitive_ids.dtype == torch.int64

    def test_attr_vec_property(self):
        batch = [_make_fake_item(5)]
        gb = collate_graphs(batch)
        assert gb.attr_vec.shape[-1] == ATTR_DIM
