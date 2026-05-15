"""Data pipeline: GraphDAGDataset -> DataLoaders for CORAL training.

Wraps the Session-3 adapter (GraphDAGDataset + collate_graphs) into training-ready
DataLoaders for all five dataset splits.  Pads each batch to a fixed max_nodes so the
CORAL model's seq_len stays constant across batches.
"""

from __future__ import annotations

import logging
from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import torch
from torch.utils.data import DataLoader

from marifah.data.adapter.batch_format import GraphBatch
from marifah.data.adapter.collate import collate_graphs
from marifah.data.adapter.attention_mask import pad_attention_masks
from marifah.training.config import TrainingConfig

logger = logging.getLogger(__name__)

_SPLITS = ["train", "val", "test_id", "test_ood_size", "test_ood_composition"]


def collate_graphs_padded(batch: List[Dict[str, Any]], max_nodes: int) -> GraphBatch:
    """Collate a batch of DAG dicts, then right-pad to a fixed max_nodes.

    The CORAL model requires a fixed seq_len, so every batch must be padded to
    the same length regardless of how many real nodes it contains.
    """
    gb = collate_graphs(batch)  # pads to batch's own max_nodes

    N = gb.max_nodes
    if N == max_nodes:
        return gb
    if N > max_nodes:
        raise ValueError(
            f"Batch max_nodes ({N}) exceeds the configured max_nodes ({max_nodes}). "
            "Increase max_nodes or lower the dataset max_nodes filter."
        )

    B = gb.batch_size
    pad = max_nodes - N

    # node_features: (B, N, feat_dim) → (B, max_nodes, feat_dim)
    node_features = torch.cat(
        [gb.node_features, torch.zeros(B, pad, gb.node_features.shape[-1])], dim=1
    )

    # attention_mask: (B, N, N) → (B, max_nodes, max_nodes)
    attention_mask = pad_attention_masks(
        [gb.attention_mask[b] for b in range(B)], max_nodes
    )

    # node_mask: (B, N) → (B, max_nodes)
    node_mask = torch.cat(
        [gb.node_mask, torch.zeros(B, pad, dtype=torch.bool)], dim=1
    )

    # pos_encoding: (B, N, K_pe) → (B, max_nodes, K_pe)
    pos_encoding = torch.cat(
        [gb.pos_encoding, torch.zeros(B, pad, gb.pos_encoding.shape[-1])], dim=1
    )

    # region_assignments: (B, N) → (B, max_nodes)  with -1 padding
    region_assignments = torch.cat(
        [gb.region_assignments, torch.full((B, pad), -1, dtype=torch.int64)], dim=1
    )

    # primitive_assignments: (B, N) → (B, max_nodes)  with -1 padding
    primitive_assignments = torch.cat(
        [gb.primitive_assignments, torch.full((B, pad), -1, dtype=torch.int64)], dim=1
    )

    return GraphBatch(
        node_features=node_features,
        attention_mask=attention_mask,
        node_mask=node_mask,
        pos_encoding=pos_encoding,
        workflow_type_id=gb.workflow_type_id,
        region_assignments=region_assignments,
        primitive_assignments=primitive_assignments,
        halt_step=gb.halt_step,
        execution_trace=gb.execution_trace,
        cycle_annotations=None,
    )


def _make_loader(
    split_dir: Path,
    config: TrainingConfig,
    shuffle: bool,
    drop_last: bool,
) -> Optional[DataLoader]:
    """Build a DataLoader for a single split directory."""
    from marifah.data.adapter.dataset import GraphDAGDataset

    if not split_dir.exists():
        logger.warning("Split directory not found, skipping: %s", split_dir)
        return None

    ds = GraphDAGDataset(
        split_dir=split_dir,
        max_nodes=config.model.max_nodes,
    )
    if len(ds) == 0:
        logger.warning("Empty dataset at %s, skipping", split_dir)
        return None

    collate_fn: Callable = partial(collate_graphs_padded, max_nodes=config.model.max_nodes)

    generator = torch.Generator()
    generator.manual_seed(config.experiment.seed)

    return DataLoader(
        ds,
        batch_size=config.training.batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
        drop_last=drop_last,
        generator=generator if shuffle else None,
    )


def build_data_loaders(config: TrainingConfig) -> Dict[str, Optional[DataLoader]]:
    """Build DataLoaders for all five dataset splits.

    Returns:
        dict mapping split name → DataLoader (or None if split dir absent)
        Keys: 'train', 'val', 'test_id', 'test_ood_size', 'test_ood_composition'
    """
    root = Path(config.data.dataset_root)
    loaders: Dict[str, Optional[DataLoader]] = {}

    for split in _SPLITS:
        split_dir = root / split
        shuffle = (split == "train")
        drop = config.data.drop_last if split == "train" else False
        loaders[split] = _make_loader(split_dir, config, shuffle=shuffle, drop_last=drop)
        if loaders[split] is not None:
            logger.info(
                "Built DataLoader for split=%s | size=%d | batch=%d",
                split,
                len(loaders[split].dataset),  # type: ignore[arg-type]
                config.training.batch_size,
            )

    return loaders
