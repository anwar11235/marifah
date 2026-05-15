"""GraphDAGDataset: torch Dataset wrapping parquet shard directories.

Loads all DAG records from a split directory (or single shard), precomputes
Laplacian PE and attention masks at init time, and serves individual DAG
dicts from __getitem__.

DAGs exceeding max_nodes are filtered out with explicit logging (never silently
dropped) per the architectural constraint in the session prompt §3.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from marifah.data.adapter.attention_mask import build_attention_mask, AttentionDirection
from marifah.data.adapter.positional import laplacian_pe_tensor, K_PE_DEFAULT
from marifah.data.adapter.tokenizer import encode_node_attrs, NODE_FEAT_DIM

logger = logging.getLogger(__name__)

_NEG1 = -1   # padding sentinel for label tensors


class GraphDAGDataset(Dataset):
    """torch Dataset wrapping a parquet split directory.

    Parameters
    ----------
    split_dir:
        Path to split directory (e.g., ``tiny/train``).  All ``shard_*.parquet``
        files are loaded.  Can also point to a single ``.parquet`` file.
    k_pe:
        Number of Laplacian eigenvectors for positional encoding.
    max_nodes:
        DAGs with more nodes than this are filtered out.  Filtered count is
        logged; no silent dropping.
    attention_direction:
        "directed" (default) or "bidirectional".
    precompute:
        If True (default), compute PE and attention masks at init time.
        Set False only for very large datasets where init-time precompute
        is impractical.
    """

    def __init__(
        self,
        split_dir: str | Path,
        k_pe: int = K_PE_DEFAULT,
        max_nodes: int = 512,
        attention_direction: AttentionDirection = "directed",
        precompute: bool = True,
    ) -> None:
        self.split_dir = Path(split_dir)
        self.k_pe = k_pe
        self.max_nodes = max_nodes
        self.attention_direction = attention_direction

        raw_records = self._load_parquet_records()
        self._records: List[Dict[str, Any]] = []
        filtered = 0

        for rec in raw_records:
            n = int(rec.get("num_nodes", 0)) or len(json.loads(rec["nodes"]) if isinstance(rec["nodes"], str) else rec["nodes"])
            if n > max_nodes:
                filtered += 1
                continue
            self._records.append(rec)

        if filtered:
            logger.warning(
                "Filtered %d DAGs exceeding max_nodes=%d out of %d total",
                filtered, max_nodes, filtered + len(self._records),
            )

        # Pre-compute masks and PE
        if precompute:
            self._precomputed: List[Dict[str, Any]] = [
                self._precompute_one(rec) for rec in self._records
            ]
        else:
            self._precomputed = [None] * len(self._records)  # type: ignore[list-item]

    # ------------------------------------------------------------------
    # Parquet loading
    # ------------------------------------------------------------------

    def _load_parquet_records(self) -> List[Dict[str, Any]]:
        import pyarrow.parquet as pq

        paths: List[Path] = []
        if self.split_dir.is_file() and self.split_dir.suffix == ".parquet":
            paths = [self.split_dir]
        elif self.split_dir.is_dir():
            paths = sorted(self.split_dir.glob("shard_*.parquet"))
            if not paths:
                # Maybe the dir IS the top-level dataset; look one level down
                paths = sorted(self.split_dir.glob("**/shard_*.parquet"))
        else:
            raise FileNotFoundError(f"Dataset path not found: {self.split_dir}")

        if not paths:
            raise FileNotFoundError(f"No shard_*.parquet files found in {self.split_dir}")

        rows: List[Dict[str, Any]] = []
        for p in paths:
            rows.extend(pq.read_table(p).to_pylist())
        return rows

    # ------------------------------------------------------------------
    # Pre-computation
    # ------------------------------------------------------------------

    def _parse_record(self, rec: Dict[str, Any]) -> Dict[str, Any]:
        """Decode JSON fields and extract typed per-node data."""
        nodes_raw = json.loads(rec["nodes"]) if isinstance(rec["nodes"], str) else rec["nodes"]
        edges_raw = json.loads(rec["edges"]) if isinstance(rec["edges"], str) else rec["edges"]
        ra_raw = json.loads(rec["region_assignments"]) if isinstance(rec["region_assignments"], str) else rec["region_assignments"]
        pa_raw = json.loads(rec["primitive_assignments"]) if isinstance(rec["primitive_assignments"], str) else rec["primitive_assignments"]
        trace_raw = json.loads(rec["execution_trace"]) if isinstance(rec["execution_trace"], str) else rec["execution_trace"]
        ood_raw = json.loads(rec["ood_flags"]) if isinstance(rec["ood_flags"], str) else rec["ood_flags"]

        # node_id → sorted index mapping (nodes are already sorted in generator output)
        nodes = nodes_raw
        edges = [(e["src"], e["dst"]) for e in edges_raw]
        num_nodes = len(nodes)

        # Build node feature vectors
        node_feat = np.zeros((num_nodes, NODE_FEAT_DIM), dtype=np.float32)
        primitive_ids = []
        for i, n in enumerate(nodes):
            prim = int(n["primitive"])
            attrs = n.get("attributes", {})
            # Fix JSON int-key round-trip for LOOKUP table (same fix as in primitives.py)
            if "table" in attrs:
                attrs = dict(attrs)
                attrs["table"] = {int(k): v for k, v in attrs["table"].items()}
            attr_vec = encode_node_attrs(prim, attrs)
            node_feat[i, 0] = float(prim)
            node_feat[i, 1:] = attr_vec
            primitive_ids.append(prim)

        region_assign = [r["pattern_id"] for r in ra_raw]

        return {
            "dag_id": rec["dag_id"],
            "workflow_type_id": int(rec["workflow_type_id"]),
            "split": rec.get("split", ""),
            "seed": int(rec.get("seed", 0)),
            "num_nodes": num_nodes,
            "node_feat": node_feat,                   # (N, NODE_FEAT_DIM)
            "edges": edges,                            # list of (src, dst) tuples
            "primitive_ids": primitive_ids,            # list of ints
            "region_assignments": region_assign,       # list of ints (pattern_id per node)
            "primitive_assignments": pa_raw,           # list of ints
            "halt_step": int(rec["halt_step"]),
            "execution_trace": trace_raw,
            "ood_flags": ood_raw,
        }

    def _precompute_one(self, rec: Dict[str, Any]) -> Dict[str, Any]:
        parsed = self._parse_record(rec)
        num_nodes = parsed["num_nodes"]
        edges = parsed["edges"]

        pe = laplacian_pe_tensor(edges, num_nodes, k=self.k_pe)           # (N, k_pe)
        mask = build_attention_mask(
            edges, num_nodes, direction=self.attention_direction
        )  # (N, N)

        return {**parsed, "pos_encoding": pe, "attention_mask": mask}

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if self._precomputed[idx] is not None:
            return self._precomputed[idx]
        # Lazy path (precompute=False)
        return self._precompute_one(self._records[idx])
