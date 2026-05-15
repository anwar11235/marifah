"""Tests for NodeTokenizer and encode_node_attrs."""

import pytest
import torch

from marifah.data.adapter.tokenizer import (
    NodeTokenizer,
    encode_node_attrs,
    ATTR_DIM,
    NODE_FEAT_DIM,
    NUM_PRIMITIVES,
)
from marifah.data.synthetic.primitives import PrimitiveType


class TestEncodeNodeAttrs:
    def test_output_length(self):
        for prim in PrimitiveType:
            attrs = {}
            vec = encode_node_attrs(prim.value, attrs)
            assert len(vec) == ATTR_DIM, f"primitive {prim}: expected {ATTR_DIM} dims"

    def test_conditional_encodes_condition(self):
        vec_pos = encode_node_attrs(PrimitiveType.CONDITIONAL.value, {"condition": "positive"})
        vec_zero = encode_node_attrs(PrimitiveType.CONDITIONAL.value, {"condition": "zero"})
        assert vec_pos[0] != vec_zero[0]

    def test_accumulate_encodes_step(self):
        vec1 = encode_node_attrs(PrimitiveType.ACCUMULATE.value, {"step_value": 1})
        vec9 = encode_node_attrs(PrimitiveType.ACCUMULATE.value, {"step_value": 9})
        assert vec1[0] < vec9[0]

    def test_nop_all_zeros(self):
        vec = encode_node_attrs(PrimitiveType.NOP.value, {})
        assert all(v == 0.0 for v in vec)

    def test_terminate_all_zeros(self):
        vec = encode_node_attrs(PrimitiveType.TERMINATE.value, {})
        assert all(v == 0.0 for v in vec)

    def test_values_approx_bounded(self):
        for prim in PrimitiveType:
            attrs = {"condition": "positive", "agg_fn": "sum", "transform_fn": "increment",
                     "constraint": "positive", "num_branches": 4, "step_value": 5,
                     "table": {0: 50, 1: 50}}
            vec = encode_node_attrs(prim.value, attrs)
            for v in vec:
                assert -0.1 <= v <= 1.1, f"prim={prim}, vec={vec}, v={v} out of range"

    def test_lookup_table_string_keys_handled(self):
        # JSON round-trip converts int keys to strings
        attrs = {"table": {"0": 80, "1": 20}}
        vec = encode_node_attrs(PrimitiveType.LOOKUP.value, attrs)
        assert len(vec) == ATTR_DIM


class TestNodeTokenizer:
    def test_output_shape_batched(self):
        tokenizer = NodeTokenizer(d_model=32)
        B, N = 4, 10
        primitive_ids = torch.randint(0, NUM_PRIMITIVES, (B, N))
        attr_vec = torch.zeros(B, N, ATTR_DIM)
        out = tokenizer(primitive_ids, attr_vec)
        assert out.shape == (B, N, 32)

    def test_output_shape_single(self):
        tokenizer = NodeTokenizer(d_model=64)
        N = 7
        primitive_ids = torch.randint(0, NUM_PRIMITIVES, (N,))
        attr_vec = torch.zeros(N, ATTR_DIM)
        out = tokenizer(primitive_ids, attr_vec)
        assert out.shape == (N, 64)

    def test_different_primitives_different_embeddings(self):
        tokenizer = NodeTokenizer(d_model=32)
        attr_vec = torch.zeros(1, ATTR_DIM)
        out0 = tokenizer(torch.tensor([0]), attr_vec)
        out1 = tokenizer(torch.tensor([1]), attr_vec)
        assert not torch.allclose(out0, out1)

    def test_gradients_flow(self):
        tokenizer = NodeTokenizer(d_model=16)
        primitive_ids = torch.randint(0, NUM_PRIMITIVES, (2, 5))
        attr_vec = torch.zeros(2, 5, ATTR_DIM)
        out = tokenizer(primitive_ids, attr_vec)
        loss = out.sum()
        loss.backward()
        for name, p in tokenizer.named_parameters():
            assert p.grad is not None, f"no grad for {name}"
