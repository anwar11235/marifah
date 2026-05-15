"""§8 — Label schema: DAGRecord and label completeness audit."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple

from marifah.data.synthetic.executor import TraceStep


# ---------------------------------------------------------------------------
# DAGRecord: the on-disk representation of one generated DAG (§8.1/§8.3)
# ---------------------------------------------------------------------------

@dataclass
class NodeRecord:
    node_id: int
    primitive: int            # PrimitiveType value (int)
    primitive_name: str
    attributes: Dict[str, Any]


@dataclass
class EdgeRecord:
    src: int
    dst: int
    branch: Optional[int]     # None for unconditional edges


@dataclass
class RegionAssignment:
    node_id: int
    pattern_id: int
    pattern_instance_id: int


@dataclass
class DAGRecord:
    # Identifiers
    dag_id: str
    workflow_type_id: int         # 1–50
    split: str                    # train | val | test_id | test_ood_size | test_ood_composition
    seed: int

    # Graph structure
    nodes: List[NodeRecord]
    edges: List[EdgeRecord]

    # Labels (§8.1)
    region_assignments: List[RegionAssignment]
    primitive_assignments: List[int]          # per-node, indexed by position in `nodes`
    execution_trace: List[Dict[str, Any]]     # serialised TraceStep list
    halt_step: int
    ood_flags: Dict[str, bool]

    # Metadata
    num_nodes: int = 0
    num_edges: int = 0
    frequency_tier: str = ""

    def __post_init__(self) -> None:
        self.num_nodes = len(self.nodes)
        self.num_edges = len(self.edges)

    def to_parquet_row(self) -> Dict[str, Any]:
        """Flatten into a dict of JSON-serialisable types for parquet."""
        import json
        return {
            "dag_id": self.dag_id,
            "workflow_type_id": self.workflow_type_id,
            "split": self.split,
            "seed": self.seed,
            "nodes": json.dumps([{
                "node_id": n.node_id,
                "primitive": n.primitive,
                "primitive_name": n.primitive_name,
                "attributes": n.attributes,
            } for n in self.nodes]),
            "edges": json.dumps([{
                "src": e.src,
                "dst": e.dst,
                "branch": e.branch,
            } for e in self.edges]),
            "region_assignments": json.dumps([{
                "node_id": r.node_id,
                "pattern_id": r.pattern_id,
                "pattern_instance_id": r.pattern_instance_id,
            } for r in self.region_assignments]),
            "primitive_assignments": json.dumps(self.primitive_assignments),
            "execution_trace": json.dumps(self.execution_trace),
            "halt_step": self.halt_step,
            "ood_flags": json.dumps(self.ood_flags),
            "num_nodes": self.num_nodes,
            "num_edges": self.num_edges,
            "frequency_tier": self.frequency_tier,
        }


# ---------------------------------------------------------------------------
# Label completeness audit (§8.4)
# ---------------------------------------------------------------------------

class LabelIncompleteError(ValueError):
    pass


def audit_labels(record: DAGRecord) -> None:
    """Raise LabelIncompleteError if any label field is incomplete or inconsistent."""
    if not (1 <= record.workflow_type_id <= 50):
        raise LabelIncompleteError(f"workflow_type_id {record.workflow_type_id} out of range [1,50]")

    node_ids = {n.node_id for n in record.nodes}

    # region_assignments covers every node exactly once
    covered = {r.node_id for r in record.region_assignments}
    if covered != node_ids:
        missing = node_ids - covered
        extra = covered - node_ids
        raise LabelIncompleteError(
            f"region_assignments mismatch: missing {missing}, extra {extra}"
        )

    # primitive_assignments has one entry per node
    if len(record.primitive_assignments) != len(record.nodes):
        raise LabelIncompleteError(
            f"primitive_assignments length {len(record.primitive_assignments)} != "
            f"node count {len(record.nodes)}"
        )

    # halt_step present and consistent with trace
    if record.halt_step < 0:
        raise LabelIncompleteError("halt_step is negative (execution did not halt)")

    if record.execution_trace:
        last_step = record.execution_trace[-1].get("step", -1)
        # halt_step should equal the step of the terminate node
        # (or last step if no terminate was reached)
        if record.halt_step > last_step:
            raise LabelIncompleteError(
                f"halt_step {record.halt_step} > last trace step {last_step}"
            )
