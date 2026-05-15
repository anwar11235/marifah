"""§5 — Main generation pipeline: steps 1–9 + multiprocessing support.

Single DAG generation path:
  1. Sample workflow type (weighted by frequency)
  2. Retrieve workflow composition spec
  3. Instantiate each pattern in the composition
  4. Primitives/attributes are instantiated inside patterns.instantiate()
  5. Connect patterns per workflow topology
  6. Generate initial state values
  7. Run reference executor
  8. Validate (well-formed, halts, labels complete)
  9. Emit DAGRecord

Determinism: given (config_hash, seed) the output is byte-identical.
"""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple

import networkx as nx
import numpy as np

from marifah.data.synthetic.executor import execute_dag, ExecutionResult
from marifah.data.synthetic.labels import (
    DAGRecord,
    EdgeRecord,
    NodeRecord,
    RegionAssignment,
    audit_labels,
)
from marifah.data.synthetic.patterns import PATTERN_BY_ID, PatternInstance
from marifah.data.synthetic.primitives import PrimitiveType, PRIMITIVE_NAMES
from marifah.data.synthetic.vertical_config import GeneratorConfig
from marifah.data.synthetic.workflows import (
    WORKFLOW_BY_ID,
    WORKFLOW_DEFINITIONS,
    WORKFLOW_INSTANCES_MAP,
    WORKFLOW_SAMPLING_WEIGHTS,
    WORKFLOW_TIER_MAP,
    WorkflowSpec,
)


# ---------------------------------------------------------------------------
# OOD helpers
# ---------------------------------------------------------------------------

def _cross_pattern_primitive_pairs(
    dag: nx.DiGraph,
    node_attrs: Dict[int, dict],
    region_info: List[Tuple[int, int, int]],
) -> Set[Tuple[int, int]]:
    """Return (src_prim, dst_prim) pairs only for edges that cross pattern instance boundaries.

    The OOD-composition holdout is about cross-pattern composition — which primitives
    appear adjacent BETWEEN pattern instances, not within a single pattern.  Checking
    all edges would reject almost every training DAG for large workflows.
    """
    inst_map: Dict[int, int] = {nid: inst_id for nid, _pat_id, inst_id in region_info}
    pairs: Set[Tuple[int, int]] = set()
    for src, dst in dag.edges():
        if inst_map.get(src) != inst_map.get(dst):
            src_p = int(node_attrs[src]["primitive"])
            dst_p = int(node_attrs[dst]["primitive"])
            pairs.add((src_p, dst_p))
    return pairs


def _contains_reserved_pair(
    pairs: Set[Tuple[int, int]], reserved: FrozenSet[Tuple[int, int]]
) -> bool:
    return bool(pairs & reserved)


# ---------------------------------------------------------------------------
# Workflow assembly (step 5)
# ---------------------------------------------------------------------------

def _assemble_workflow(
    workflow: WorkflowSpec,
    rng: np.random.Generator,
) -> Tuple[nx.DiGraph, Dict[int, dict], List[Tuple[int, int, int]]]:
    """Instantiate and wire all patterns in the workflow.

    Returns
    -------
    dag:
        Assembled DAG with global node IDs.
    global_node_attrs:
        Global node_id → attrs dict.
    region_info:
        List of (global_node_id, pattern_id, pattern_instance_id) for all nodes.
    """
    global_dag: nx.DiGraph = nx.DiGraph()
    global_node_attrs: Dict[int, dict] = {}
    region_info: List[Tuple[int, int, int]] = []
    offset = 0
    prev_exit_global: List[int] = []

    for inst_idx, pat_id in enumerate(workflow.pattern_sequence):
        pattern = PATTERN_BY_ID[pat_id]
        local_inst = pattern.instantiate(rng, instance_id=inst_idx)

        # Map local → global node IDs
        local_to_global: Dict[int, int] = {}
        for local_id in local_inst.dag.nodes():
            global_id = local_id + offset
            local_to_global[local_id] = global_id
            global_dag.add_node(global_id)
            global_node_attrs[global_id] = local_inst.node_attrs[local_id].copy()
            region_info.append((global_id, pat_id, inst_idx))

        for src_local, dst_local, edata in local_inst.dag.edges(data=True):
            src_g = local_to_global[src_local]
            dst_g = local_to_global[dst_local]
            branch = edata.get("branch", None)
            if branch is not None:
                global_dag.add_edge(src_g, dst_g, branch=branch)
            else:
                global_dag.add_edge(src_g, dst_g)

        global_entry = local_to_global[local_inst.entry_node]
        global_exits = [local_to_global[e] for e in local_inst.exit_nodes]

        # Wire previous pattern's exits to this pattern's entry
        if prev_exit_global:
            for prev_exit in prev_exit_global:
                global_dag.add_edge(prev_exit, global_entry)

        prev_exit_global = global_exits
        offset += local_inst.dag.number_of_nodes()

    # Guarantee every assembled DAG has a reachable TERMINATE node.
    # If the final pattern's exits are not already TERMINATE nodes, add one.
    has_terminate = any(
        v["primitive"] == PrimitiveType.TERMINATE
        for v in global_node_attrs.values()
    )
    if not has_terminate or prev_exit_global:
        # Check if any exit node is already a TERMINATE
        exits_are_terminate = all(
            global_node_attrs[e]["primitive"] == PrimitiveType.TERMINATE
            for e in prev_exit_global
        )
        if not exits_are_terminate:
            term_id = offset
            global_dag.add_node(term_id)
            global_node_attrs[term_id] = {
                "primitive": PrimitiveType.TERMINATE,
            }
            # Use the last pattern's pattern_id for the terminal region
            last_pat_id = workflow.pattern_sequence[-1] if workflow.pattern_sequence else 0
            region_info.append((term_id, last_pat_id, len(workflow.pattern_sequence)))
            for exit_node in prev_exit_global:
                global_dag.add_edge(exit_node, term_id)

    return global_dag, global_node_attrs, region_info


# ---------------------------------------------------------------------------
# Initial state generation (step 6)
# ---------------------------------------------------------------------------

def _generate_initial_states(
    dag: nx.DiGraph, rng: np.random.Generator
) -> Dict[int, Any]:
    """Assign initial states to root nodes."""
    initial: Dict[int, Any] = {}
    for node_id in dag.nodes():
        if dag.in_degree(node_id) == 0:
            initial[node_id] = int(rng.integers(1, 100))  # 1–99
    return initial


# ---------------------------------------------------------------------------
# OOD-size scaling (§7.3)
# ---------------------------------------------------------------------------

def _scale_dag_for_ood_size(
    workflow: WorkflowSpec,
    rng: np.random.Generator,
    scale_min: float,
    scale_max: float,
) -> Tuple[nx.DiGraph, Dict[int, dict], List[Tuple[int, int, int]]]:
    """Generate a larger DAG by repeating the workflow pattern sequence."""
    scale = rng.uniform(scale_min, scale_max)
    repeats = max(2, int(round(scale)))
    extended_spec = WorkflowSpec(
        workflow_type_id=workflow.workflow_type_id,
        pattern_sequence=workflow.pattern_sequence * repeats,
        size_tier=workflow.size_tier,
        frequency_tier=workflow.frequency_tier,
    )
    return _assemble_workflow(extended_spec, rng)


# ---------------------------------------------------------------------------
# Single DAG generation (steps 1–9)
# ---------------------------------------------------------------------------

MAX_RETRIES = 20


def generate_one(
    *,
    seed: int,
    split: str,
    config: GeneratorConfig,
    reserved_pairs: FrozenSet[Tuple[int, int]],
    workflow_type_id: Optional[int] = None,
    require_reserved_pair: bool = False,
    ood_size: bool = False,
) -> Optional[DAGRecord]:
    """Generate a single DAG record, or return None if all retries are exhausted.

    Parameters
    ----------
    seed:           Per-DAG seed (ensures determinism).
    split:          Which split this DAG belongs to.
    config:         Generator configuration.
    reserved_pairs: Primitive pairs reserved for OOD-composition split.
    workflow_type_id:
        If specified, always use this workflow.  Otherwise sample by weight.
    require_reserved_pair:
        If True, reject DAGs that don't contain at least one reserved pair
        (used for test_ood_composition split).
    ood_size:
        If True, generate an OOD-size DAG (scaled up).
    """
    rng = np.random.default_rng(seed)

    for attempt in range(MAX_RETRIES):
        attempt_rng = np.random.default_rng([seed, attempt])

        # Step 1–2: sample workflow
        if workflow_type_id is not None:
            wf_id = workflow_type_id
        else:
            wf_id = int(attempt_rng.choice(
                len(WORKFLOW_DEFINITIONS),
                p=WORKFLOW_SAMPLING_WEIGHTS,
            )) + 1
        workflow = WORKFLOW_BY_ID[wf_id]

        # Step 3–5: assemble workflow
        try:
            if ood_size:
                dag, node_attrs, region_info = _scale_dag_for_ood_size(
                    workflow, attempt_rng,
                    config.ood_size_scale_min,
                    config.ood_size_scale_max,
                )
            else:
                dag, node_attrs, region_info = _assemble_workflow(workflow, attempt_rng)
        except Exception:
            continue

        # Step 6: initial states
        initial_states = _generate_initial_states(dag, attempt_rng)

        # Step 7: execute
        try:
            result: ExecutionResult = execute_dag(dag, node_attrs, initial_states)
        except Exception:
            continue

        # Step 8: validate
        if not result.halted:
            continue
        if result.halt_step < 0:
            continue

        # Check cross-pattern adjacency for OOD-composition constraints
        pairs = _cross_pattern_primitive_pairs(dag, node_attrs, region_info)
        if split == "train" and _contains_reserved_pair(pairs, reserved_pairs):
            continue  # training DAGs must not contain reserved pairs
        if require_reserved_pair and not _contains_reserved_pair(pairs, reserved_pairs):
            continue  # OOD-composition DAGs must contain ≥ 1 reserved pair

        # DAG acyclicity (should be guaranteed by construction, but verify)
        if not nx.is_directed_acyclic_graph(dag):
            continue

        # Step 9: emit record
        dag_id = f"{split}_{seed:010d}_{attempt}"

        nodes_sorted = sorted(dag.nodes())
        node_records = [
            NodeRecord(
                node_id=nid,
                primitive=int(node_attrs[nid]["primitive"]),
                primitive_name=PRIMITIVE_NAMES[node_attrs[nid]["primitive"]],
                attributes={
                    k: v for k, v in node_attrs[nid].items() if k != "primitive"
                },
            )
            for nid in nodes_sorted
        ]

        edge_records = [
            EdgeRecord(src=src, dst=dst, branch=edata.get("branch", None))
            for src, dst, edata in dag.edges(data=True)
        ]

        region_map: Dict[int, Tuple[int, int]] = {
            nid: (pat_id, inst_id) for nid, pat_id, inst_id in region_info
        }
        region_assignments = [
            RegionAssignment(node_id=nid, pattern_id=region_map[nid][0],
                             pattern_instance_id=region_map[nid][1])
            for nid in nodes_sorted
        ]

        primitive_assignments = [int(node_attrs[nid]["primitive"]) for nid in nodes_sorted]

        trace_serialised = [
            {
                "step": ts.step,
                "node_id": ts.node_id,
                "primitive": ts.primitive,
                "input_state": ts.input_state,
                "output_state": ts.output_state,
                "branch_taken": ts.branch_taken,
            }
            for ts in result.trace
        ]

        ood_flags: Dict[str, bool] = {
            "is_ood_size": ood_size,
            "is_ood_composition": require_reserved_pair,
        }

        record = DAGRecord(
            dag_id=dag_id,
            workflow_type_id=wf_id,
            split=split,
            seed=seed,
            nodes=node_records,
            edges=edge_records,
            region_assignments=region_assignments,
            primitive_assignments=primitive_assignments,
            execution_trace=trace_serialised,
            halt_step=result.halt_step,
            ood_flags=ood_flags,
            frequency_tier=WORKFLOW_TIER_MAP.get(wf_id, ""),
        )

        try:
            audit_labels(record)
        except Exception:
            continue

        return record

    return None  # exhausted retries


# ---------------------------------------------------------------------------
# Worker function for multiprocessing (must be top-level for pickle)
# ---------------------------------------------------------------------------

@dataclass
class GenerationTask:
    seed: int
    split: str
    workflow_type_id: Optional[int]
    require_reserved_pair: bool
    ood_size: bool


def _worker(args: Tuple[GenerationTask, GeneratorConfig, FrozenSet]) -> Optional[DAGRecord]:
    task, config, reserved_pairs = args
    return generate_one(
        seed=task.seed,
        split=task.split,
        config=config,
        reserved_pairs=reserved_pairs,
        workflow_type_id=task.workflow_type_id,
        require_reserved_pair=task.require_reserved_pair,
        ood_size=task.ood_size,
    )


# ---------------------------------------------------------------------------
# DagGenerator: orchestrates split generation with multiprocessing
# ---------------------------------------------------------------------------

class DagGenerator:
    def __init__(self, config: GeneratorConfig) -> None:
        from marifah.data.synthetic.workflows import build_reserved_primitive_pairs
        self.config = config
        self.reserved_pairs: FrozenSet[Tuple[int, int]] = build_reserved_primitive_pairs(
            holdout_fraction=config.ood_holdout_fraction,
            seed=config.ood_holdout_seed,
        )

    def _build_train_tasks(self, seed_offset: int) -> List[GenerationTask]:
        """Build generation tasks for training split using the frequency distribution."""
        tasks: List[GenerationTask] = []
        current_seed = seed_offset
        for wf in WORKFLOW_DEFINITIONS:
            n = WORKFLOW_INSTANCES_MAP[wf.workflow_type_id]
            for _ in range(n):
                tasks.append(GenerationTask(
                    seed=current_seed,
                    split="train",
                    workflow_type_id=wf.workflow_type_id,
                    require_reserved_pair=False,
                    ood_size=False,
                ))
                current_seed += 1
        return tasks

    def generate_split(
        self,
        split: str,
        n: int,
        seed_offset: int,
        *,
        ood_size: bool = False,
        require_reserved_pair: bool = False,
        num_workers: int = 1,
    ) -> List[DAGRecord]:
        """Generate n DAG records for one split."""
        import multiprocessing as mp

        if split == "train":
            tasks = self._build_train_tasks(seed_offset)
            # Honour requested n by truncating or repeating
            tasks = (tasks * ((n // len(tasks)) + 1))[:n]
        else:
            tasks = [
                GenerationTask(
                    seed=seed_offset + i,
                    split=split,
                    workflow_type_id=None,
                    require_reserved_pair=require_reserved_pair,
                    ood_size=ood_size,
                )
                for i in range(n)
            ]

        args_list = [(t, self.config, self.reserved_pairs) for t in tasks]

        records: List[DAGRecord] = []
        if num_workers > 1:
            with mp.Pool(processes=num_workers) as pool:
                for rec in pool.imap_unordered(_worker, args_list, chunksize=64):
                    if rec is not None:
                        records.append(rec)
        else:
            for args in args_list:
                rec = _worker(args)
                if rec is not None:
                    records.append(rec)

        return records

    def benchmark_throughput(self, n: int = 200, num_workers: int = 1) -> float:
        """Return DAGs/sec over a sample of n DAGs."""
        t0 = time.time()
        records = self.generate_split(
            "val", n, seed_offset=999_000_000, num_workers=num_workers
        )
        elapsed = time.time() - t0
        return len(records) / max(elapsed, 1e-6)
