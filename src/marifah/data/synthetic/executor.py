"""§6 — Reference executor: ground-truth oracle for DAG execution.

Deterministic state propagation in topological order.  Conditional / route
nodes direct state along exactly one outgoing branch; aggregate and compare
nodes consume all reachable predecessor states.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx

from marifah.data.synthetic.primitives import (
    PrimitiveResult,
    PrimitiveType,
    State,
    apply_primitive,
)


# ---------------------------------------------------------------------------
# Trace and result types (§6.2)
# ---------------------------------------------------------------------------

@dataclass
class TraceStep:
    step: int
    node_id: int
    primitive: str
    input_state: Any
    output_state: Any
    branch_taken: Optional[int]


@dataclass
class ExecutionResult:
    trace: List[TraceStep]
    halt_step: int                # 0-indexed step at which terminate fired
    final_state: Any
    halted: bool = True


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _is_branching(prim: PrimitiveType) -> bool:
    return prim in (PrimitiveType.CONDITIONAL, PrimitiveType.ROUTE)


def _multi_input(prim: PrimitiveType) -> bool:
    return prim in (PrimitiveType.AGGREGATE, PrimitiveType.COMPARE)


def _gather_inputs(
    dag: nx.DiGraph,
    node_id: int,
    node_outputs: Dict[int, State],
    branch_taken: Dict[int, int],
    node_attrs: Dict[int, dict],
) -> Optional[List[State]]:
    """Return the list of states arriving at node_id, or None if not reachable."""
    predecessors = list(dag.predecessors(node_id))
    if not predecessors:
        return None  # root — handled separately

    collected: List[State] = []
    for pred in predecessors:
        if pred not in node_outputs:
            continue  # predecessor was not executed (not reachable)

        edge_data = dag.get_edge_data(pred, node_id) or {}
        branch = edge_data.get("branch", None)

        if branch is not None:
            # This edge is a conditional / route branch edge.
            # Only carry state if this branch was taken.
            if branch_taken.get(pred) != branch:
                continue

        collected.append(node_outputs[pred])

    if not collected:
        return None  # not reachable in this execution

    return collected


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def execute_dag(
    dag: nx.DiGraph,
    node_attrs: Dict[int, dict],
    initial_states: Dict[int, State],
) -> ExecutionResult:
    """Execute a DAG and return the ground-truth execution trace.

    Parameters
    ----------
    dag:
        Directed acyclic graph.  Nodes are ints; edges may carry a 'branch'
        attribute for conditional / route outgoing edges.
    node_attrs:
        Per-node dict with at minimum a 'primitive' key (PrimitiveType).
    initial_states:
        State values for root nodes (nodes with in-degree 0).

    Returns
    -------
    ExecutionResult with full trace, halt step, and final state.
    """
    try:
        topo_order: List[int] = list(nx.topological_sort(dag))
    except nx.NetworkXUnfeasible as exc:
        raise ValueError("DAG contains cycles") from exc

    node_outputs: Dict[int, State] = {}
    branch_taken: Dict[int, int] = {}

    # Seed root nodes with initial states
    for node_id in topo_order:
        if dag.in_degree(node_id) == 0:
            if node_id in initial_states:
                node_outputs[node_id] = initial_states[node_id]
            # Roots without an initial state are treated as unreachable.

    trace: List[TraceStep] = []
    halt_step = -1
    final_state: State = 0

    for step, node_id in enumerate(topo_order):
        prim_type: PrimitiveType = node_attrs[node_id]["primitive"]
        attrs: dict = node_attrs[node_id]

        # Determine input state(s)
        if dag.in_degree(node_id) == 0:
            # Root node
            if node_id not in node_outputs:
                continue  # no initial state → skip
            raw_inputs = [node_outputs[node_id]]
        else:
            raw_inputs = _gather_inputs(dag, node_id, node_outputs, branch_taken, node_attrs)
            if raw_inputs is None:
                continue  # not reachable

        # Prepare input for this primitive
        if _multi_input(prim_type):
            input_for_prim: Any = raw_inputs
        else:
            input_for_prim = raw_inputs[0]

        # Apply primitive
        result: PrimitiveResult = apply_primitive(prim_type, input_for_prim, attrs)

        if result.branch_taken is not None:
            branch_taken[node_id] = result.branch_taken
            # Conditional / route nodes forward the input state to successors,
            # not the branch_id.  branch_id is an internal routing decision.
            node_outputs[node_id] = (
                input_for_prim if not isinstance(input_for_prim, list)
                else input_for_prim[0]
            )
        else:
            node_outputs[node_id] = result.output_state

        trace.append(
            TraceStep(
                step=step,
                node_id=node_id,
                primitive=prim_type.name.lower(),
                input_state=_serialize_state(input_for_prim),
                output_state=_serialize_state(result.output_state),
                branch_taken=result.branch_taken,
            )
        )

        if prim_type == PrimitiveType.TERMINATE:
            halt_step = step
            final_state = result.output_state
            break

    return ExecutionResult(
        trace=trace,
        halt_step=halt_step,
        final_state=final_state,
        halted=(halt_step >= 0),  # True only when TERMINATE node was reached
    )


def _serialize_state(state: Any) -> Any:
    """Make state JSON-serialisable (bool is a subtype of int — keep as-is)."""
    if isinstance(state, list):
        return [_serialize_state(s) for s in state]
    if isinstance(state, dict):
        return {str(k): _serialize_state(v) for k, v in state.items()}
    return state
