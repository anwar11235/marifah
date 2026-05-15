"""Tests for GraphDAGDataset.

Requires the tiny dataset to be generated before running.  If not present,
tests are skipped with an informative message.
"""

import json
import tempfile
from pathlib import Path

import pytest
import torch

from marifah.data.adapter.dataset import GraphDAGDataset
from marifah.data.adapter.tokenizer import NODE_FEAT_DIM
from marifah.data.adapter.positional import K_PE_DEFAULT


def _make_tiny_dataset(tmp_path: Path) -> Path:
    """Generate a tiny dataset into a temp directory."""
    from marifah.data.synthetic.generator import DagGenerator
    from marifah.data.synthetic.storage import write_split, write_manifest
    from marifah.data.synthetic.vertical_config import GeneratorConfig, SplitSizes
    from marifah.data.synthetic.vertical_config import _hash_config
    from marifah.data.synthetic.workflows import build_reserved_primitive_pairs

    cfg = GeneratorConfig(seed=999)
    cfg.split_sizes = SplitSizes(train=10, val=5, test_id=5, test_ood_size=3, test_ood_composition=3)
    cfg.config_hash = _hash_config(cfg)
    reserved = build_reserved_primitive_pairs(holdout_fraction=0.15, seed=0)

    gen = DagGenerator(cfg)
    records = gen.generate_split("val", 5, seed_offset=50_000)
    write_split(records, tmp_path, "val")
    return tmp_path / "val"


@pytest.fixture(scope="module")
def tiny_val_dir(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("tiny_ds")
    return _make_tiny_dataset(tmp)


class TestGraphDAGDataset:
    def test_len(self, tiny_val_dir):
        ds = GraphDAGDataset(tiny_val_dir)
        assert len(ds) > 0

    def test_getitem_keys(self, tiny_val_dir):
        ds = GraphDAGDataset(tiny_val_dir)
        item = ds[0]
        required_keys = {
            "num_nodes", "node_feat", "edges",
            "primitive_ids", "region_assignments", "primitive_assignments",
            "halt_step", "workflow_type_id", "execution_trace",
            "pos_encoding", "attention_mask",
        }
        assert required_keys.issubset(set(item.keys()))

    def test_node_feat_shape(self, tiny_val_dir):
        ds = GraphDAGDataset(tiny_val_dir)
        item = ds[0]
        nf = item["node_feat"]
        n = item["num_nodes"]
        import numpy as np
        assert isinstance(nf, np.ndarray)
        assert nf.shape == (n, NODE_FEAT_DIM)

    def test_pos_encoding_shape(self, tiny_val_dir):
        ds = GraphDAGDataset(tiny_val_dir, k_pe=4)
        item = ds[0]
        pe = item["pos_encoding"]
        n = item["num_nodes"]
        assert pe.shape == (n, 4)

    def test_attention_mask_shape(self, tiny_val_dir):
        ds = GraphDAGDataset(tiny_val_dir)
        item = ds[0]
        mask = item["attention_mask"]
        n = item["num_nodes"]
        assert mask.shape == (n, n)

    def test_attention_mask_diagonal(self, tiny_val_dir):
        ds = GraphDAGDataset(tiny_val_dir)
        for i in range(min(3, len(ds))):
            item = ds[i]
            mask = item["attention_mask"]
            n = item["num_nodes"]
            for j in range(n):
                assert mask[j, j].item() == 0.0, f"item {i}: self-loop mask[{j},{j}] != 0"

    def test_max_nodes_filter(self, tiny_val_dir):
        ds_large = GraphDAGDataset(tiny_val_dir, max_nodes=10_000)
        ds_tiny = GraphDAGDataset(tiny_val_dir, max_nodes=5)
        assert len(ds_tiny) <= len(ds_large)

    def test_halt_step_non_negative(self, tiny_val_dir):
        ds = GraphDAGDataset(tiny_val_dir)
        for i in range(len(ds)):
            assert ds[i]["halt_step"] >= 0
