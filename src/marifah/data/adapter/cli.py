"""CLI entry point: python -m marifah.data.adapter.cli <command> [args]

Commands
--------
precompute-pe   One-time pass adding Laplacian PE as a column to each parquet shard.
inspect-batch   Load N random DAGs, collate, print tensor shapes and a sample.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path


def cmd_precompute_pe(args: argparse.Namespace) -> int:
    """Add 'pos_encoding' column to each parquet shard in the dataset."""
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq

    from marifah.data.adapter.positional import compute_laplacian_pe

    dataset_path = Path(args.dataset_path)
    if not dataset_path.exists():
        print(f"ERROR: dataset path not found: {dataset_path}", file=sys.stderr)
        return 1

    k_pe = args.k_pe
    shards = sorted(dataset_path.glob("**/shard_*.parquet"))
    if not shards:
        print(f"ERROR: no shard_*.parquet files found under {dataset_path}", file=sys.stderr)
        return 1

    total_processed = 0
    for shard in shards:
        table = pq.read_table(shard)
        rows = table.to_pylist()

        pe_values: list = []
        for row in rows:
            nodes_raw = json.loads(row["nodes"]) if isinstance(row["nodes"], str) else row["nodes"]
            edges_raw = json.loads(row["edges"]) if isinstance(row["edges"], str) else row["edges"]
            edges = [(e["src"], e["dst"]) for e in edges_raw]
            num_nodes = len(nodes_raw)
            pe = compute_laplacian_pe(edges, num_nodes, k=k_pe)  # (N, k_pe) float32
            pe_values.append(json.dumps(pe.tolist()))

        # Add or overwrite pos_encoding column
        pe_array = pa.array(pe_values, type=pa.string())
        if "pos_encoding" in table.schema.names:
            idx = table.schema.get_field_index("pos_encoding")
            table = table.set_column(idx, "pos_encoding", pe_array)
        else:
            table = table.append_column("pos_encoding", pe_array)

        pq.write_table(table, shard, compression="snappy")
        total_processed += len(rows)
        print(f"  Wrote PE to {shard.relative_to(dataset_path)} ({len(rows)} records)")

    print(f"\nprecompute-pe: {total_processed} DAGs processed, k_pe={k_pe}")
    return 0


def cmd_inspect_batch(args: argparse.Namespace) -> int:
    """Load batch_size random DAGs, collate, and print tensor shapes."""
    from torch.utils.data import DataLoader

    from marifah.data.adapter.dataset import GraphDAGDataset
    from marifah.data.adapter.collate import collate_graphs

    dataset_path = Path(args.dataset_path)
    if not dataset_path.exists():
        print(f"ERROR: dataset path not found: {dataset_path}", file=sys.stderr)
        return 1

    # Try to find a split directory
    split_dirs = ["train", "val", "test_id"]
    split_dir = dataset_path
    for s in split_dirs:
        if (dataset_path / s).exists():
            split_dir = dataset_path / s
            break

    print(f"Loading dataset from: {split_dir}")
    dataset = GraphDAGDataset(split_dir, k_pe=args.k_pe, max_nodes=args.max_nodes)
    print(f"  Total DAGs: {len(dataset)}")

    if len(dataset) == 0:
        print("ERROR: empty dataset", file=sys.stderr)
        return 1

    batch_size = min(args.batch_size, len(dataset))
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collate_graphs,
        shuffle=True,
    )
    batch = next(iter(loader))

    print(f"\nBatch (B={batch.batch_size}, N_max={batch.max_nodes}):")
    print(f"  node_features:       {tuple(batch.node_features.shape)}")
    print(f"  attention_mask:      {tuple(batch.attention_mask.shape)}")
    print(f"  node_mask:           {tuple(batch.node_mask.shape)}")
    print(f"  pos_encoding:        {tuple(batch.pos_encoding.shape)}")
    print(f"  workflow_type_id:    {tuple(batch.workflow_type_id.shape)}")
    print(f"  region_assignments:  {tuple(batch.region_assignments.shape)}")
    print(f"  primitive_assignments: {tuple(batch.primitive_assignments.shape)}")
    print(f"  halt_step:           {tuple(batch.halt_step.shape)}")
    print(f"  execution_trace:     list[{len(batch.execution_trace)} dicts]")

    # Sample: first DAG in batch
    nc0 = int(batch.node_mask[0].sum().item())
    print(f"\nSample (batch[0], {nc0} nodes):")
    print(f"  primitive_ids: {batch.primitive_ids[0, :nc0].tolist()}")
    print(f"  halt_step: {int(batch.halt_step[0].item())}")
    print(f"  workflow_type_id: {int(batch.workflow_type_id[0].item())}")

    # Mask sanity: check diagonal is 0.0 (self-attend)
    mask0 = batch.attention_mask[0, :nc0, :nc0]
    diag_ok = (mask0.diagonal() == 0.0).all().item()
    print(f"  attention_mask diagonal is 0.0 (self-loop): {diag_ok}")

    print("\ninspect-batch: OK")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="marifah.data.adapter.cli")
    sub = parser.add_subparsers(dest="command")

    p_pe = sub.add_parser("precompute-pe", help="Add Laplacian PE column to parquet shards")
    p_pe.add_argument("dataset_path", help="Path to dataset root or split directory")
    p_pe.add_argument("--k-pe", type=int, default=8)

    p_insp = sub.add_parser("inspect-batch", help="Load a batch and print shapes")
    p_insp.add_argument("dataset_path", help="Path to dataset root")
    p_insp.add_argument("--batch-size", type=int, default=8)
    p_insp.add_argument("--k-pe", type=int, default=8)
    p_insp.add_argument("--max-nodes", type=int, default=512)

    args = parser.parse_args(argv)

    if args.command == "precompute-pe":
        return cmd_precompute_pe(args)
    elif args.command == "inspect-batch":
        return cmd_inspect_batch(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
