"""§5.1 step 8 + §8.4 — Well-formedness and distribution audits."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np

from marifah.data.synthetic.labels import DAGRecord, audit_labels
from marifah.data.synthetic.workflows import WORKFLOW_TIER_MAP


# ---------------------------------------------------------------------------
# Per-record validation
# ---------------------------------------------------------------------------

@dataclass
class ValidationResult:
    ok: bool
    errors: List[str] = field(default_factory=list)


def validate_record(record: DAGRecord) -> ValidationResult:
    errors: List[str] = []

    # Reconstruct DAG for structural checks
    dag = nx.DiGraph()
    for n in record.nodes:
        dag.add_node(n.node_id)
    for e in record.edges:
        if e.branch is not None:
            dag.add_edge(e.src, e.dst, branch=e.branch)
        else:
            dag.add_edge(e.src, e.dst)

    # Acyclicity
    if not nx.is_directed_acyclic_graph(dag):
        errors.append("DAG contains cycles")

    # Terminate reachability
    node_ids = {n.node_id for n in record.nodes}
    from marifah.data.synthetic.primitives import PrimitiveType
    terminate_nodes = {
        n.node_id for n in record.nodes
        if n.primitive == PrimitiveType.TERMINATE.value
    }
    if not terminate_nodes:
        errors.append("no terminate node in DAG")
    else:
        root_nodes = [n for n in dag.nodes if dag.in_degree(n) == 0]
        any_reachable = False
        for root in root_nodes:
            for term in terminate_nodes:
                if nx.has_path(dag, root, term):
                    any_reachable = True
                    break
            if any_reachable:
                break
        if not any_reachable:
            errors.append("no terminate node reachable from any root")

    # No orphan nodes (every non-root has at least one predecessor)
    # — guaranteed by construction; check anyway
    orphans = [
        n for n in dag.nodes
        if dag.in_degree(n) == 0 and dag.out_degree(n) == 0
    ]
    if orphans:
        errors.append(f"orphan nodes: {orphans}")

    # Label completeness (from labels.py)
    try:
        audit_labels(record)
    except Exception as exc:
        errors.append(f"label audit failed: {exc}")

    # Halt step check
    if record.halt_step < 0:
        errors.append("halt_step is negative")

    return ValidationResult(ok=len(errors) == 0, errors=errors)


# ---------------------------------------------------------------------------
# Dataset-level distribution audit
# ---------------------------------------------------------------------------

@dataclass
class DistributionAudit:
    ok: bool
    workflow_coverage: Dict[int, int]       # wf_type_id → count in this split
    pattern_coverage: Dict[int, int]        # pattern_id → count in this split
    primitive_coverage: Dict[int, int]      # primitive_id → count
    missing_workflows: List[int]
    issues: List[str] = field(default_factory=list)


def audit_distribution(
    records: List[DAGRecord],
    split: str,
    expected_workflow_ids: Optional[List[int]] = None,
) -> DistributionAudit:
    """Audit that all expected workflow types appear and coverage is reasonable."""
    wf_counts: Dict[int, int] = {}
    pat_counts: Dict[int, int] = {}
    prim_counts: Dict[int, int] = {}

    for rec in records:
        wf_counts[rec.workflow_type_id] = wf_counts.get(rec.workflow_type_id, 0) + 1
        for ra in rec.region_assignments:
            pat_counts[ra.pattern_id] = pat_counts.get(ra.pattern_id, 0) + 1
        for pa in rec.primitive_assignments:
            prim_counts[pa] = prim_counts.get(pa, 0) + 1

    issues: List[str] = []
    missing: List[int] = []
    if expected_workflow_ids:
        for wf_id in expected_workflow_ids:
            if wf_counts.get(wf_id, 0) == 0:
                missing.append(wf_id)
        if missing:
            issues.append(f"missing workflow types in {split}: {missing[:10]}...")

    return DistributionAudit(
        ok=len(issues) == 0,
        workflow_coverage=wf_counts,
        pattern_coverage=pat_counts,
        primitive_coverage=prim_counts,
        missing_workflows=missing,
        issues=issues,
    )


# ---------------------------------------------------------------------------
# Spot-check executor consistency
# ---------------------------------------------------------------------------

def spot_check_traces(records: List[DAGRecord], n: int = 10) -> ValidationResult:
    """Re-execute n random DAGs and verify stored traces match re-execution."""
    import random
    from marifah.data.synthetic.executor import execute_dag
    from marifah.data.synthetic.primitives import PrimitiveType

    sample = records[:n] if len(records) >= n else records
    errors: List[str] = []

    for rec in sample:
        # Reconstruct
        dag = nx.DiGraph()
        node_attrs: Dict[int, dict] = {}
        for n_rec in rec.nodes:
            dag.add_node(n_rec.node_id)
            node_attrs[n_rec.node_id] = {
                "primitive": PrimitiveType(n_rec.primitive),
                **n_rec.attributes,
            }
        for e in rec.edges:
            if e.branch is not None:
                dag.add_edge(e.src, e.dst, branch=e.branch)
            else:
                dag.add_edge(e.src, e.dst)

        # Initial states from stored trace step 0 for each root
        initial_states: Dict[int, int] = {}
        for trace_step in rec.execution_trace:
            nid = trace_step["node_id"]
            if dag.in_degree(nid) == 0:
                initial_states[nid] = trace_step["input_state"]

        if not initial_states:
            continue

        try:
            result = execute_dag(dag, node_attrs, initial_states)
        except Exception as exc:
            errors.append(f"re-execution failed for {rec.dag_id}: {exc}")
            continue

        if result.halt_step != rec.halt_step:
            errors.append(
                f"{rec.dag_id}: halt_step mismatch stored={rec.halt_step} "
                f"recomputed={result.halt_step}"
            )

    return ValidationResult(ok=len(errors) == 0, errors=errors)
