"""Tests for edge-induced attention mask construction.

Verification §4 item 5: 3-node chain (1->2->3) mask shape and allowed positions.
"""

import math
import pytest
import torch

from marifah.data.adapter.attention_mask import (
    build_attention_mask,
    pad_attention_masks,
)


class TestBuildAttentionMask:
    def test_shape(self):
        edges = [(0, 1), (1, 2)]
        mask = build_attention_mask(edges, num_nodes=3)
        assert mask.shape == (3, 3)

    def test_self_loops_always_allowed(self):
        edges = []
        mask = build_attention_mask(edges, num_nodes=4)
        for i in range(4):
            assert mask[i, i].item() == 0.0, f"mask[{i},{i}] should be 0"

    def test_chain_directed_3nodes(self):
        """3-node chain (0->1->2).
        Directed: node 0 attends to itself;
                  node 1 attends to 0 (predecessor) and itself;
                  node 2 attends to 1 (predecessor) and itself.
        """
        edges = [(0, 1), (1, 2)]
        mask = build_attention_mask(edges, num_nodes=3, direction="directed")

        # Self-loops
        assert mask[0, 0].item() == 0.0
        assert mask[1, 1].item() == 0.0
        assert mask[2, 2].item() == 0.0

        # node 1 attends to node 0 (0->1)
        assert mask[1, 0].item() == 0.0

        # node 2 attends to node 1 (1->2)
        assert mask[2, 1].item() == 0.0

        # node 0 does NOT attend to 1 or 2 (directed)
        assert mask[0, 1].item() == float("-inf")
        assert mask[0, 2].item() == float("-inf")

        # node 1 does NOT attend to 2 (1->2 means 2 attends to 1, not vice versa)
        assert mask[1, 2].item() == float("-inf")

    def test_bidirectional(self):
        edges = [(0, 1)]
        mask = build_attention_mask(edges, num_nodes=2, direction="bidirectional")
        assert mask[0, 1].item() == 0.0
        assert mask[1, 0].item() == 0.0

    def test_out_of_range_edges_ignored(self):
        edges = [(0, 1), (5, 6)]  # (5, 6) out of range for 3 nodes
        mask = build_attention_mask(edges, num_nodes=3)
        assert mask.shape == (3, 3)

    def test_empty_edges(self):
        mask = build_attention_mask([], num_nodes=3)
        # Only diagonal is 0; everything else is -inf
        for i in range(3):
            for j in range(3):
                expected = 0.0 if i == j else float("-inf")
                assert mask[i, j].item() == expected

    def test_device(self):
        edges = [(0, 1)]
        mask = build_attention_mask(edges, num_nodes=2, device=torch.device("cpu"))
        assert mask.device.type == "cpu"


class TestPadAttentionMasks:
    def test_shapes(self):
        masks = [
            build_attention_mask([(0, 1)], 2),
            build_attention_mask([(0, 1), (1, 2), (2, 3)], 4),
        ]
        batched = pad_attention_masks(masks, max_nodes=5)
        assert batched.shape == (2, 5, 5)

    def test_real_region_preserved(self):
        mask = build_attention_mask([(0, 1)], 2)
        batched = pad_attention_masks([mask], max_nodes=4)
        assert batched[0, :2, :2].allclose(mask)

    def test_padding_region_is_neg_inf(self):
        mask = build_attention_mask([], 2)
        batched = pad_attention_masks([mask], max_nodes=4)
        # Positions [2:, :] and [:, 2:] should be -inf
        assert (batched[0, 2:, :] == float("-inf")).all()
        assert (batched[0, :, 2:] == float("-inf")).all()
