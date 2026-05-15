"""§3 — Sub-DAG pattern templates: 12 parametric pattern implementations.

Each pattern is an instantiable class that produces a networkx DiGraph
fragment with a defined entry node and one or more exit nodes.  The
generator wires patterns together at exit→entry boundaries.

Node IDs within each pattern are local (0-based).  The generator
renumbers them when assembling workflows.

Edge attribute 'branch' is only set on outgoing edges of conditional and
route nodes to indicate which branch the edge represents.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, List, Optional, Tuple

import networkx as nx
import numpy as np

from marifah.data.synthetic.primitives import (
    PrimitiveType,
    sample_attrs,
)


# ---------------------------------------------------------------------------
# Pattern result
# ---------------------------------------------------------------------------

@dataclass
class PatternInstance:
    dag: nx.DiGraph
    node_attrs: Dict[int, dict]       # node_id → {primitive, ...attrs}
    entry_node: int
    exit_nodes: List[int]
    pattern_id: int
    pattern_instance_id: int = 0
    size: int = 0

    def __post_init__(self) -> None:
        self.size = self.dag.number_of_nodes()


# ---------------------------------------------------------------------------
# Pattern base
# ---------------------------------------------------------------------------

class Pattern:
    pattern_id: int
    name: str
    min_size: int
    max_size: int
    # dominant primitive types (informational; used for coverage checks)
    dominant_primitives: Tuple[PrimitiveType, ...] = ()

    def instantiate(
        self, rng: np.random.Generator, instance_id: int = 0
    ) -> PatternInstance:
        raise NotImplementedError


def _make_node(
    dag: nx.DiGraph,
    node_attrs: dict,
    node_id: int,
    primitive: PrimitiveType,
    rng: np.random.Generator,
    **attr_kwargs: Any,
) -> None:
    attrs = {"primitive": primitive, **sample_attrs(primitive, rng, **attr_kwargs)}
    dag.add_node(node_id)
    node_attrs[node_id] = attrs


def _connect(dag: nx.DiGraph, src: int, dst: int, branch: Optional[int] = None) -> None:
    if branch is not None:
        dag.add_edge(src, dst, branch=branch)
    else:
        dag.add_edge(src, dst)


# ---------------------------------------------------------------------------
# 1. linear_chain
# ---------------------------------------------------------------------------

class LinearChain(Pattern):
    pattern_id = 0
    name = "linear_chain"
    min_size = 3
    max_size = 8
    dominant_primitives = (PrimitiveType.TRANSFORM, PrimitiveType.LOOKUP, PrimitiveType.NOP)

    def instantiate(self, rng: np.random.Generator, instance_id: int = 0) -> PatternInstance:
        size = int(rng.integers(self.min_size, self.max_size + 1))
        dag: nx.DiGraph = nx.DiGraph()
        node_attrs: Dict[int, dict] = {}
        candidates = [PrimitiveType.TRANSFORM, PrimitiveType.LOOKUP, PrimitiveType.NOP, PrimitiveType.ACCUMULATE]
        for i in range(size):
            prim = PrimitiveType(rng.choice([c.value for c in candidates]))
            _make_node(dag, node_attrs, i, prim, rng)
        for i in range(size - 1):
            _connect(dag, i, i + 1)
        return PatternInstance(dag, node_attrs, 0, [size - 1], self.pattern_id, instance_id)


# ---------------------------------------------------------------------------
# 2. conditional_fork
# ---------------------------------------------------------------------------

class ConditionalFork(Pattern):
    pattern_id = 1
    name = "conditional_fork"
    min_size = 4
    max_size = 8
    dominant_primitives = (PrimitiveType.CONDITIONAL, PrimitiveType.TRANSFORM)

    def instantiate(self, rng: np.random.Generator, instance_id: int = 0) -> PatternInstance:
        branch_len = int(rng.integers(1, 3))  # 1–2 nodes per branch
        size = 1 + 2 * branch_len
        size = max(self.min_size, min(size, self.max_size))
        dag: nx.DiGraph = nx.DiGraph()
        node_attrs: Dict[int, dict] = {}

        _make_node(dag, node_attrs, 0, PrimitiveType.CONDITIONAL, rng)

        candidates = [PrimitiveType.TRANSFORM, PrimitiveType.LOOKUP, PrimitiveType.NOP]
        next_id = 1
        branch0_nodes: List[int] = []
        for _ in range(branch_len):
            prim = PrimitiveType(rng.choice([c.value for c in candidates]))
            _make_node(dag, node_attrs, next_id, prim, rng)
            branch0_nodes.append(next_id)
            next_id += 1

        branch1_nodes: List[int] = []
        for _ in range(branch_len):
            prim = PrimitiveType(rng.choice([c.value for c in candidates]))
            _make_node(dag, node_attrs, next_id, prim, rng)
            branch1_nodes.append(next_id)
            next_id += 1

        _connect(dag, 0, branch0_nodes[0], branch=0)
        _connect(dag, 0, branch1_nodes[0], branch=1)
        for i in range(len(branch0_nodes) - 1):
            _connect(dag, branch0_nodes[i], branch0_nodes[i + 1])
        for i in range(len(branch1_nodes) - 1):
            _connect(dag, branch1_nodes[i], branch1_nodes[i + 1])

        exits = [branch0_nodes[-1], branch1_nodes[-1]]
        return PatternInstance(dag, node_attrs, 0, exits, self.pattern_id, instance_id)


# ---------------------------------------------------------------------------
# 3. fork_and_join
# ---------------------------------------------------------------------------

class ForkAndJoin(Pattern):
    pattern_id = 2
    name = "fork_and_join"
    min_size = 5
    max_size = 12
    dominant_primitives = (PrimitiveType.AGGREGATE, PrimitiveType.TRANSFORM, PrimitiveType.NOP)

    def instantiate(self, rng: np.random.Generator, instance_id: int = 0) -> PatternInstance:
        branch_len = int(rng.integers(1, 4))
        dag: nx.DiGraph = nx.DiGraph()
        node_attrs: Dict[int, dict] = {}

        # entry (nop) → two parallel chains → aggregate
        _make_node(dag, node_attrs, 0, PrimitiveType.NOP, rng)
        candidates = [PrimitiveType.TRANSFORM, PrimitiveType.ACCUMULATE, PrimitiveType.LOOKUP]
        next_id = 1
        chain_a: List[int] = []
        for _ in range(branch_len):
            prim = PrimitiveType(rng.choice([c.value for c in candidates]))
            _make_node(dag, node_attrs, next_id, prim, rng)
            chain_a.append(next_id)
            next_id += 1
        chain_b: List[int] = []
        for _ in range(branch_len):
            prim = PrimitiveType(rng.choice([c.value for c in candidates]))
            _make_node(dag, node_attrs, next_id, prim, rng)
            chain_b.append(next_id)
            next_id += 1

        agg_id = next_id
        _make_node(dag, node_attrs, agg_id, PrimitiveType.AGGREGATE, rng)

        _connect(dag, 0, chain_a[0])
        _connect(dag, 0, chain_b[0])
        for i in range(len(chain_a) - 1):
            _connect(dag, chain_a[i], chain_a[i + 1])
        for i in range(len(chain_b) - 1):
            _connect(dag, chain_b[i], chain_b[i + 1])
        _connect(dag, chain_a[-1], agg_id)
        _connect(dag, chain_b[-1], agg_id)

        return PatternInstance(dag, node_attrs, 0, [agg_id], self.pattern_id, instance_id)


# ---------------------------------------------------------------------------
# 4. sequential_validation
# ---------------------------------------------------------------------------

class SequentialValidation(Pattern):
    pattern_id = 3
    name = "sequential_validation"
    min_size = 4
    max_size = 10
    dominant_primitives = (PrimitiveType.VALIDATE, PrimitiveType.ROUTE, PrimitiveType.TERMINATE)

    def instantiate(self, rng: np.random.Generator, instance_id: int = 0) -> PatternInstance:
        n_stages = int(rng.integers(1, 3))
        dag: nx.DiGraph = nx.DiGraph()
        node_attrs: Dict[int, dict] = {}

        next_id = 0
        _make_node(dag, node_attrs, next_id, PrimitiveType.NOP, rng)
        entry = next_id
        next_id += 1

        prev = entry
        for stage in range(n_stages):
            val_id = next_id
            _make_node(dag, node_attrs, val_id, PrimitiveType.VALIDATE, rng)
            next_id += 1
            cond_id = next_id
            _make_node(dag, node_attrs, cond_id, PrimitiveType.CONDITIONAL, rng)
            next_id += 1
            fail_term_id = next_id
            _make_node(dag, node_attrs, fail_term_id, PrimitiveType.TERMINATE, rng)
            next_id += 1
            _connect(dag, prev, val_id)
            _connect(dag, val_id, cond_id)
            _connect(dag, cond_id, fail_term_id, branch=0)
            prev = cond_id

        # Success path
        success_transform_id = next_id
        _make_node(dag, node_attrs, success_transform_id, PrimitiveType.TRANSFORM, rng)
        next_id += 1
        _connect(dag, prev, success_transform_id, branch=1)

        return PatternInstance(
            dag, node_attrs, entry, [success_transform_id], self.pattern_id, instance_id
        )


# ---------------------------------------------------------------------------
# 5. hierarchical_aggregate
# ---------------------------------------------------------------------------

class HierarchicalAggregate(Pattern):
    pattern_id = 4
    name = "hierarchical_aggregate"
    min_size = 6
    max_size = 15
    dominant_primitives = (PrimitiveType.LOOKUP, PrimitiveType.AGGREGATE, PrimitiveType.TRANSFORM)

    def instantiate(self, rng: np.random.Generator, instance_id: int = 0) -> PatternInstance:
        n_lookups = int(rng.integers(2, 5))
        dag: nx.DiGraph = nx.DiGraph()
        node_attrs: Dict[int, dict] = {}

        lookup_ids: List[int] = []
        for i in range(n_lookups):
            _make_node(dag, node_attrs, i, PrimitiveType.LOOKUP, rng)
            lookup_ids.append(i)

        next_id = n_lookups
        agg_id = next_id
        _make_node(dag, node_attrs, agg_id, PrimitiveType.AGGREGATE, rng)
        next_id += 1
        for lid in lookup_ids:
            _connect(dag, lid, agg_id)

        transform_id = next_id
        _make_node(dag, node_attrs, transform_id, PrimitiveType.TRANSFORM, rng)
        next_id += 1
        _connect(dag, agg_id, transform_id)

        return PatternInstance(
            dag, node_attrs, lookup_ids[0], [transform_id], self.pattern_id, instance_id
        )


# ---------------------------------------------------------------------------
# 6. lookup_and_compare
# ---------------------------------------------------------------------------

class LookupAndCompare(Pattern):
    pattern_id = 5
    name = "lookup_and_compare"
    min_size = 4
    max_size = 7
    dominant_primitives = (PrimitiveType.LOOKUP, PrimitiveType.COMPARE, PrimitiveType.CONDITIONAL)

    def instantiate(self, rng: np.random.Generator, instance_id: int = 0) -> PatternInstance:
        dag: nx.DiGraph = nx.DiGraph()
        node_attrs: Dict[int, dict] = {}

        _make_node(dag, node_attrs, 0, PrimitiveType.NOP, rng)
        _make_node(dag, node_attrs, 1, PrimitiveType.LOOKUP, rng)
        _make_node(dag, node_attrs, 2, PrimitiveType.LOOKUP, rng)
        _make_node(dag, node_attrs, 3, PrimitiveType.COMPARE, rng)
        _make_node(dag, node_attrs, 4, PrimitiveType.CONDITIONAL, rng)
        _make_node(dag, node_attrs, 5, PrimitiveType.TRANSFORM, rng)
        _make_node(dag, node_attrs, 6, PrimitiveType.NOP, rng)

        _connect(dag, 0, 1)
        _connect(dag, 0, 2)
        _connect(dag, 1, 3)
        _connect(dag, 2, 3)
        _connect(dag, 3, 4)
        _connect(dag, 4, 5, branch=0)
        _connect(dag, 4, 6, branch=1)

        return PatternInstance(dag, node_attrs, 0, [5, 6], self.pattern_id, instance_id)


# ---------------------------------------------------------------------------
# 7. multi_way_route
# ---------------------------------------------------------------------------

class MultiWayRoute(Pattern):
    pattern_id = 6
    name = "multi_way_route"
    min_size = 4
    max_size = 10
    dominant_primitives = (PrimitiveType.ROUTE, PrimitiveType.TRANSFORM, PrimitiveType.NOP)

    def instantiate(self, rng: np.random.Generator, instance_id: int = 0) -> PatternInstance:
        num_branches = int(rng.integers(2, 5))
        dag: nx.DiGraph = nx.DiGraph()
        node_attrs: Dict[int, dict] = {}

        _make_node(dag, node_attrs, 0, PrimitiveType.NOP, rng)
        _make_node(dag, node_attrs, 1, PrimitiveType.ROUTE, rng, num_branches=num_branches)

        _connect(dag, 0, 1)

        exits: List[int] = []
        candidates = [PrimitiveType.TRANSFORM, PrimitiveType.ACCUMULATE, PrimitiveType.NOP]
        next_id = 2
        for branch in range(num_branches):
            prim = PrimitiveType(rng.choice([c.value for c in candidates]))
            _make_node(dag, node_attrs, next_id, prim, rng)
            _connect(dag, 1, next_id, branch=branch)
            exits.append(next_id)
            next_id += 1

        return PatternInstance(dag, node_attrs, 0, exits, self.pattern_id, instance_id)


# ---------------------------------------------------------------------------
# 8. validate_then_route
# ---------------------------------------------------------------------------

class ValidateThenRoute(Pattern):
    pattern_id = 7
    name = "validate_then_route"
    min_size = 5
    max_size = 10
    dominant_primitives = (PrimitiveType.VALIDATE, PrimitiveType.CONDITIONAL, PrimitiveType.ROUTE)

    def instantiate(self, rng: np.random.Generator, instance_id: int = 0) -> PatternInstance:
        dag: nx.DiGraph = nx.DiGraph()
        node_attrs: Dict[int, dict] = {}

        _make_node(dag, node_attrs, 0, PrimitiveType.NOP, rng)
        _make_node(dag, node_attrs, 1, PrimitiveType.VALIDATE, rng)
        _make_node(dag, node_attrs, 2, PrimitiveType.CONDITIONAL, rng)
        _make_node(dag, node_attrs, 3, PrimitiveType.TERMINATE, rng)   # validation fail

        num_branches = int(rng.integers(2, 4))
        _make_node(dag, node_attrs, 4, PrimitiveType.ROUTE, rng, num_branches=num_branches)

        _connect(dag, 0, 1)
        _connect(dag, 1, 2)
        _connect(dag, 2, 3, branch=0)    # fail → terminate
        _connect(dag, 2, 4, branch=1)    # pass → route

        exits: List[int] = []
        candidates = [PrimitiveType.TRANSFORM, PrimitiveType.NOP]
        next_id = 5
        for branch in range(num_branches):
            prim = PrimitiveType(rng.choice([c.value for c in candidates]))
            _make_node(dag, node_attrs, next_id, prim, rng)
            _connect(dag, 4, next_id, branch=branch)
            exits.append(next_id)
            next_id += 1

        return PatternInstance(dag, node_attrs, 0, exits, self.pattern_id, instance_id)


# ---------------------------------------------------------------------------
# 9. accumulate_path
# ---------------------------------------------------------------------------

class AccumulatePath(Pattern):
    pattern_id = 8
    name = "accumulate_path"
    min_size = 5
    max_size = 12
    dominant_primitives = (PrimitiveType.ACCUMULATE, PrimitiveType.TRANSFORM, PrimitiveType.TERMINATE)

    def instantiate(self, rng: np.random.Generator, instance_id: int = 0) -> PatternInstance:
        n_acc = int(rng.integers(2, 6))
        dag: nx.DiGraph = nx.DiGraph()
        node_attrs: Dict[int, dict] = {}

        _make_node(dag, node_attrs, 0, PrimitiveType.NOP, rng)
        next_id = 1
        prev = 0
        for _ in range(n_acc):
            _make_node(dag, node_attrs, next_id, PrimitiveType.ACCUMULATE, rng)
            _connect(dag, prev, next_id)
            prev = next_id
            next_id += 1

        term_id = next_id
        _make_node(dag, node_attrs, term_id, PrimitiveType.TERMINATE, rng)
        _connect(dag, prev, term_id)

        return PatternInstance(dag, node_attrs, 0, [term_id], self.pattern_id, instance_id)


# ---------------------------------------------------------------------------
# 10. branch_merge_resolve
# ---------------------------------------------------------------------------

class BranchMergeResolve(Pattern):
    pattern_id = 9
    name = "branch_merge_resolve"
    min_size = 8
    max_size = 18
    dominant_primitives = (PrimitiveType.CONDITIONAL, PrimitiveType.AGGREGATE, PrimitiveType.TRANSFORM)

    def instantiate(self, rng: np.random.Generator, instance_id: int = 0) -> PatternInstance:
        branch_len = int(rng.integers(1, 4))
        dag: nx.DiGraph = nx.DiGraph()
        node_attrs: Dict[int, dict] = {}

        _make_node(dag, node_attrs, 0, PrimitiveType.NOP, rng)
        _make_node(dag, node_attrs, 1, PrimitiveType.CONDITIONAL, rng)
        _connect(dag, 0, 1)

        candidates = [PrimitiveType.TRANSFORM, PrimitiveType.ACCUMULATE]
        next_id = 2
        chain_a: List[int] = []
        for _ in range(branch_len):
            prim = PrimitiveType(rng.choice([c.value for c in candidates]))
            _make_node(dag, node_attrs, next_id, prim, rng)
            chain_a.append(next_id)
            next_id += 1
        chain_b: List[int] = []
        for _ in range(branch_len):
            prim = PrimitiveType(rng.choice([c.value for c in candidates]))
            _make_node(dag, node_attrs, next_id, prim, rng)
            chain_b.append(next_id)
            next_id += 1

        _connect(dag, 1, chain_a[0], branch=0)
        _connect(dag, 1, chain_b[0], branch=1)
        for i in range(len(chain_a) - 1):
            _connect(dag, chain_a[i], chain_a[i + 1])
        for i in range(len(chain_b) - 1):
            _connect(dag, chain_b[i], chain_b[i + 1])

        agg_id = next_id
        _make_node(dag, node_attrs, agg_id, PrimitiveType.AGGREGATE, rng)
        _connect(dag, chain_a[-1], agg_id)
        _connect(dag, chain_b[-1], agg_id)
        next_id += 1

        resolve_id = next_id
        _make_node(dag, node_attrs, resolve_id, PrimitiveType.CONDITIONAL, rng)
        _connect(dag, agg_id, resolve_id)
        next_id += 1

        exit_a = next_id
        _make_node(dag, node_attrs, exit_a, PrimitiveType.TRANSFORM, rng)
        _connect(dag, resolve_id, exit_a, branch=0)
        next_id += 1
        exit_b = next_id
        _make_node(dag, node_attrs, exit_b, PrimitiveType.NOP, rng)
        _connect(dag, resolve_id, exit_b, branch=1)

        return PatternInstance(dag, node_attrs, 0, [exit_a, exit_b], self.pattern_id, instance_id)


# ---------------------------------------------------------------------------
# 11. lookup_aggregate_validate
# ---------------------------------------------------------------------------

class LookupAggregateValidate(Pattern):
    pattern_id = 10
    name = "lookup_aggregate_validate"
    min_size = 6
    max_size = 14
    dominant_primitives = (PrimitiveType.LOOKUP, PrimitiveType.AGGREGATE, PrimitiveType.VALIDATE)

    def instantiate(self, rng: np.random.Generator, instance_id: int = 0) -> PatternInstance:
        n_lookups = int(rng.integers(2, 5))
        dag: nx.DiGraph = nx.DiGraph()
        node_attrs: Dict[int, dict] = {}

        # Entry + n lookups
        _make_node(dag, node_attrs, 0, PrimitiveType.NOP, rng)
        lookup_ids: List[int] = []
        for i in range(n_lookups):
            nid = i + 1
            _make_node(dag, node_attrs, nid, PrimitiveType.LOOKUP, rng)
            _connect(dag, 0, nid)
            lookup_ids.append(nid)

        next_id = 1 + n_lookups
        agg_id = next_id
        _make_node(dag, node_attrs, agg_id, PrimitiveType.AGGREGATE, rng)
        for lid in lookup_ids:
            _connect(dag, lid, agg_id)
        next_id += 1

        val_id = next_id
        _make_node(dag, node_attrs, val_id, PrimitiveType.VALIDATE, rng)
        _connect(dag, agg_id, val_id)
        next_id += 1

        cond_id = next_id
        _make_node(dag, node_attrs, cond_id, PrimitiveType.CONDITIONAL, rng)
        _connect(dag, val_id, cond_id)
        next_id += 1

        pass_id = next_id
        _make_node(dag, node_attrs, pass_id, PrimitiveType.TRANSFORM, rng)
        _connect(dag, cond_id, pass_id, branch=1)
        next_id += 1

        fail_id = next_id
        _make_node(dag, node_attrs, fail_id, PrimitiveType.NOP, rng)
        _connect(dag, cond_id, fail_id, branch=0)

        return PatternInstance(dag, node_attrs, 0, [pass_id, fail_id], self.pattern_id, instance_id)


# ---------------------------------------------------------------------------
# 12. constrained_terminate
# ---------------------------------------------------------------------------

class ConstrainedTerminate(Pattern):
    pattern_id = 11
    name = "constrained_terminate"
    min_size = 5
    max_size = 10
    dominant_primitives = (PrimitiveType.VALIDATE, PrimitiveType.TRANSFORM, PrimitiveType.TERMINATE)

    def instantiate(self, rng: np.random.Generator, instance_id: int = 0) -> PatternInstance:
        dag: nx.DiGraph = nx.DiGraph()
        node_attrs: Dict[int, dict] = {}

        _make_node(dag, node_attrs, 0, PrimitiveType.NOP, rng)
        _make_node(dag, node_attrs, 1, PrimitiveType.VALIDATE, rng)
        _make_node(dag, node_attrs, 2, PrimitiveType.CONDITIONAL, rng)
        _make_node(dag, node_attrs, 3, PrimitiveType.TERMINATE, rng)      # invalid path
        _make_node(dag, node_attrs, 4, PrimitiveType.TRANSFORM, rng)
        _make_node(dag, node_attrs, 5, PrimitiveType.TERMINATE, rng)      # valid path

        _connect(dag, 0, 1)
        _connect(dag, 1, 2)
        _connect(dag, 2, 3, branch=0)   # fail → early terminate
        _connect(dag, 2, 4, branch=1)   # pass → transform
        _connect(dag, 4, 5)

        return PatternInstance(dag, node_attrs, 0, [3, 5], self.pattern_id, instance_id)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

ALL_PATTERNS: List[Pattern] = [
    LinearChain(),
    ConditionalFork(),
    ForkAndJoin(),
    SequentialValidation(),
    HierarchicalAggregate(),
    LookupAndCompare(),
    MultiWayRoute(),
    ValidateThenRoute(),
    AccumulatePath(),
    BranchMergeResolve(),
    LookupAggregateValidate(),
    ConstrainedTerminate(),
]

PATTERN_BY_ID: Dict[int, Pattern] = {p.pattern_id: p for p in ALL_PATTERNS}
PATTERN_BY_NAME: Dict[str, Pattern] = {p.name: p for p in ALL_PATTERNS}

NUM_PATTERNS = len(ALL_PATTERNS)
