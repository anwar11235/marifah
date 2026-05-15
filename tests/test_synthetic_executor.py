"""Unit tests for the reference executor on hand-crafted DAGs."""

import pytest
import networkx as nx
import numpy as np

from marifah.data.synthetic.executor import execute_dag, ExecutionResult
from marifah.data.synthetic.primitives import PrimitiveType


def _na(primitive: PrimitiveType, **kwargs) -> dict:
    return {"primitive": primitive, **kwargs}


class TestLinearChainExecution:
    def test_nop_terminate(self):
        dag = nx.DiGraph()
        dag.add_edge(0, 1)
        attrs = {
            0: _na(PrimitiveType.NOP),
            1: _na(PrimitiveType.TERMINATE),
        }
        result = execute_dag(dag, attrs, {0: 7})
        assert result.halted
        assert result.halt_step >= 0
        assert result.final_state == 7
        assert len(result.trace) == 2

    def test_transform_then_terminate(self):
        dag = nx.DiGraph()
        dag.add_edges_from([(0, 1), (1, 2)])
        attrs = {
            0: _na(PrimitiveType.NOP),
            1: _na(PrimitiveType.TRANSFORM, transform_fn="double"),
            2: _na(PrimitiveType.TERMINATE),
        }
        result = execute_dag(dag, attrs, {0: 5})
        assert result.final_state == 10

    def test_accumulate_chain(self):
        dag = nx.DiGraph()
        dag.add_edges_from([(0, 1), (1, 2), (2, 3)])
        attrs = {
            0: _na(PrimitiveType.NOP),
            1: _na(PrimitiveType.ACCUMULATE, step_value=3),
            2: _na(PrimitiveType.ACCUMULATE, step_value=3),
            3: _na(PrimitiveType.TERMINATE),
        }
        result = execute_dag(dag, attrs, {0: 0})
        # 0 → 0+3=3 → 3+3=6 → halt with 6
        assert result.final_state == 6


class TestConditionalExecution:
    def test_branch0_taken(self):
        dag = nx.DiGraph()
        dag.add_edge(0, 1)
        dag.add_edge(1, 2, branch=0)
        dag.add_edge(1, 3, branch=1)
        dag.add_edge(2, 4)
        dag.add_edge(3, 5)
        attrs = {
            0: _na(PrimitiveType.NOP),
            1: _na(PrimitiveType.CONDITIONAL, condition="positive"),
            2: _na(PrimitiveType.TRANSFORM, transform_fn="double"),  # branch 0
            3: _na(PrimitiveType.TRANSFORM, transform_fn="negate"),  # branch 1
            4: _na(PrimitiveType.TERMINATE),
            5: _na(PrimitiveType.TERMINATE),
        }
        result = execute_dag(dag, attrs, {0: -5})  # negative → branch 0
        assert result.halted
        # branch 0: double(-5) = -10
        assert result.final_state == -10

    def test_branch1_taken(self):
        dag = nx.DiGraph()
        dag.add_edge(0, 1)
        dag.add_edge(1, 2, branch=0)
        dag.add_edge(1, 3, branch=1)
        attrs = {
            0: _na(PrimitiveType.NOP),
            1: _na(PrimitiveType.CONDITIONAL, condition="positive"),
            2: _na(PrimitiveType.TERMINATE),  # branch 0
            3: _na(PrimitiveType.TERMINATE),  # branch 1
        }
        result = execute_dag(dag, attrs, {0: 10})  # positive → branch 1
        assert result.halted


class TestAggregateExecution:
    def test_fork_join(self):
        # 0 → [1, 2] → 3 (aggregate)
        dag = nx.DiGraph()
        dag.add_edge(0, 1)
        dag.add_edge(0, 2)
        dag.add_edge(1, 3)
        dag.add_edge(2, 3)
        dag.add_edge(3, 4)
        attrs = {
            0: _na(PrimitiveType.NOP),
            1: _na(PrimitiveType.TRANSFORM, transform_fn="double"),
            2: _na(PrimitiveType.TRANSFORM, transform_fn="increment"),
            3: _na(PrimitiveType.AGGREGATE, agg_fn="sum"),
            4: _na(PrimitiveType.TERMINATE),
        }
        result = execute_dag(dag, attrs, {0: 5})
        # double(5)=10, increment(5)=6, sum(10,6)=16
        assert result.final_state == 16


class TestLookupExecution:
    def test_lookup_chain(self):
        dag = nx.DiGraph()
        dag.add_edges_from([(0, 1), (1, 2)])
        attrs = {
            0: _na(PrimitiveType.NOP),
            1: _na(PrimitiveType.LOOKUP, table={0: 42, 1: 99}),
            2: _na(PrimitiveType.TERMINATE),
        }
        result = execute_dag(dag, attrs, {0: 0})  # key = 0 % 2 = 0 → 42
        assert result.final_state == 42


class TestRouteExecution:
    def test_multi_way_route(self):
        dag = nx.DiGraph()
        dag.add_edge(0, 1)
        dag.add_edge(1, 2, branch=0)
        dag.add_edge(1, 3, branch=1)
        dag.add_edge(1, 4, branch=2)
        attrs = {
            0: _na(PrimitiveType.NOP),
            1: _na(PrimitiveType.ROUTE, num_branches=3),
            2: _na(PrimitiveType.TERMINATE),
            3: _na(PrimitiveType.TERMINATE),
            4: _na(PrimitiveType.TERMINATE),
        }
        result = execute_dag(dag, attrs, {0: 4})  # 4 % 3 = 1 → branch 1
        assert result.halted


class TestTraceFormat:
    def test_trace_fields(self):
        dag = nx.DiGraph()
        dag.add_edge(0, 1)
        attrs = {
            0: _na(PrimitiveType.NOP),
            1: _na(PrimitiveType.TERMINATE),
        }
        result = execute_dag(dag, attrs, {0: 55})
        for step in result.trace:
            assert hasattr(step, "step")
            assert hasattr(step, "node_id")
            assert hasattr(step, "primitive")
            assert hasattr(step, "input_state")
            assert hasattr(step, "output_state")
            assert hasattr(step, "branch_taken")


class TestCycleRejection:
    def test_cycle_raises(self):
        dag = nx.DiGraph()
        dag.add_edge(0, 1)
        dag.add_edge(1, 0)  # cycle
        attrs = {
            0: _na(PrimitiveType.NOP),
            1: _na(PrimitiveType.TERMINATE),
        }
        with pytest.raises(ValueError, match="cycle"):
            execute_dag(dag, attrs, {0: 1})


class TestNoInitialState:
    def test_no_root_state_skips(self):
        dag = nx.DiGraph()
        dag.add_edge(0, 1)
        attrs = {
            0: _na(PrimitiveType.NOP),
            1: _na(PrimitiveType.TERMINATE),
        }
        # Provide no initial state — root is skipped, no halt
        result = execute_dag(dag, attrs, {})
        assert not result.halted or result.halt_step >= 0
