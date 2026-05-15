"""§4 — Workflow signatures and frequency distribution.

50 workflow signatures are defined deterministically at import time using a
fixed seed (42).  Each workflow is a sequence of pattern IDs assembled by
the generator.

Frequency tiers (§4.2) are hard-coded: 5 very-high, 10 high, 15 medium,
15 low, 5 very-low.  Workflow IDs 1–50 map to tiers in that order.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np

from marifah.data.synthetic.patterns import NUM_PATTERNS


# ---------------------------------------------------------------------------
# Frequency tier definitions (§4.2 — load-bearing for Claim A4)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FrequencyTier:
    name: str
    count: int           # number of workflow types in this tier
    instances: int       # training instances per workflow type

FREQUENCY_TIERS: List[FrequencyTier] = [
    FrequencyTier("very_high", count=5,  instances=100_000),
    FrequencyTier("high",      count=10, instances=20_000),
    FrequencyTier("medium",    count=15, instances=5_000),
    FrequencyTier("low",       count=15, instances=500),
    FrequencyTier("very_low",  count=5,  instances=50),
]

# workflow_type_id → tier name (1-indexed)
def _build_tier_map() -> Dict[int, str]:
    mapping: Dict[int, str] = {}
    wf_id = 1
    for tier in FREQUENCY_TIERS:
        for _ in range(tier.count):
            mapping[wf_id] = tier.name
            wf_id += 1
    return mapping

WORKFLOW_TIER_MAP: Dict[int, str] = _build_tier_map()
WORKFLOW_INSTANCES_MAP: Dict[int, int] = {
    wf_id: next(t.instances for t in FREQUENCY_TIERS if t.name == tier_name)
    for wf_id, tier_name in WORKFLOW_TIER_MAP.items()
}

# Sampling weights over workflow type IDs (used in generator step [1])
def _build_sampling_weights() -> np.ndarray:
    weights = np.array(
        [WORKFLOW_INSTANCES_MAP[wf_id] for wf_id in sorted(WORKFLOW_INSTANCES_MAP)],
        dtype=np.float64,
    )
    return weights / weights.sum()

WORKFLOW_SAMPLING_WEIGHTS: np.ndarray = _build_sampling_weights()


# ---------------------------------------------------------------------------
# Workflow spec
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class WorkflowSpec:
    workflow_type_id: int     # 1–50
    pattern_sequence: Tuple[int, ...]  # ordered pattern IDs (0-indexed)
    size_tier: str            # "simple" | "medium" | "complex"
    frequency_tier: str       # from FREQUENCY_TIERS


# ---------------------------------------------------------------------------
# Deterministic workflow definition (seed=42)
# ---------------------------------------------------------------------------

_N_WORKFLOWS = 50
_SIMPLE_RANGE  = range(0, 10)    # workflow indices 0–9  → IDs 1–10
_MEDIUM_RANGE  = range(10, 35)   # workflow indices 10–34 → IDs 11–35
_COMPLEX_RANGE = range(35, 50)   # workflow indices 35–49 → IDs 36–50

# Minimum patterns per tier
_MIN_PATTERNS = {"simple": 2, "medium": 3, "complex": 5}
_MAX_PATTERNS = {"simple": 3, "medium": 5, "complex": 8}


def _generate_workflow_definitions(seed: int = 42) -> List[WorkflowSpec]:
    rng = np.random.default_rng(seed)

    workflow_pattern_lists: List[List[int]] = [[] for _ in range(_N_WORKFLOWS)]

    # --- Phase 1: guarantee each pattern appears in ≥ 5 workflows ---
    for pat_id in range(NUM_PATTERNS):
        targets = rng.choice(_N_WORKFLOWS, size=5, replace=False).tolist()
        for wf_idx in targets:
            if pat_id not in workflow_pattern_lists[wf_idx]:
                workflow_pattern_lists[wf_idx].append(pat_id)

    # --- Phase 2: fill to target size per tier ---
    def _tier(idx: int) -> str:
        if idx in _SIMPLE_RANGE:
            return "simple"
        if idx in _MEDIUM_RANGE:
            return "medium"
        return "complex"

    for wf_idx in range(_N_WORKFLOWS):
        tier = _tier(wf_idx)
        min_p = _MIN_PATTERNS[tier]
        max_p = _MAX_PATTERNS[tier]
        target = int(rng.integers(min_p, max_p + 1))
        seen = set(workflow_pattern_lists[wf_idx])
        while len(workflow_pattern_lists[wf_idx]) < target:
            candidate = int(rng.integers(0, NUM_PATTERNS))
            if candidate not in seen:
                workflow_pattern_lists[wf_idx].append(candidate)
                seen.add(candidate)

    # --- Phase 3: enforce minimum size ---
    for wf_idx in range(_N_WORKFLOWS):
        tier = _tier(wf_idx)
        min_p = _MIN_PATTERNS[tier]
        seen = set(workflow_pattern_lists[wf_idx])
        while len(workflow_pattern_lists[wf_idx]) < min_p:
            candidate = int(rng.integers(0, NUM_PATTERNS))
            if candidate not in seen:
                workflow_pattern_lists[wf_idx].append(candidate)
                seen.add(candidate)

    # Build specs
    specs: List[WorkflowSpec] = []
    for wf_idx in range(_N_WORKFLOWS):
        wf_id = wf_idx + 1
        tier = _tier(wf_idx)
        freq_tier = WORKFLOW_TIER_MAP[wf_id]
        specs.append(WorkflowSpec(
            workflow_type_id=wf_id,
            pattern_sequence=tuple(workflow_pattern_lists[wf_idx]),
            size_tier=tier,
            frequency_tier=freq_tier,
        ))

    return specs


WORKFLOW_DEFINITIONS: List[WorkflowSpec] = _generate_workflow_definitions(seed=42)
WORKFLOW_BY_ID: Dict[int, WorkflowSpec] = {w.workflow_type_id: w for w in WORKFLOW_DEFINITIONS}


# ---------------------------------------------------------------------------
# Coverage validation (spec §4.3)
# ---------------------------------------------------------------------------

def validate_coverage() -> Tuple[bool, str]:
    """Return (ok, message).  Checks §4.3 coverage requirements."""
    pattern_wf_count: Dict[int, int] = {p: 0 for p in range(NUM_PATTERNS)}
    for wf in WORKFLOW_DEFINITIONS:
        for pat_id in wf.pattern_sequence:
            pattern_wf_count[pat_id] = pattern_wf_count.get(pat_id, 0) + 1

    failures: List[str] = []
    for pat_id, count in pattern_wf_count.items():
        if count < 5:
            failures.append(f"pattern {pat_id} appears in only {count} workflows (need ≥5)")

    # Each workflow uses ≥ 2 distinct patterns
    for wf in WORKFLOW_DEFINITIONS:
        if len(set(wf.pattern_sequence)) < 2:
            failures.append(f"workflow {wf.workflow_type_id} has < 2 distinct patterns")

    if failures:
        return False, "; ".join(failures)
    return True, "coverage ok"


# ---------------------------------------------------------------------------
# OOD composition: reserved primitive-pair set
# ---------------------------------------------------------------------------

def build_reserved_primitive_pairs(
    holdout_fraction: float = 0.15,
    seed: int = 0,
) -> frozenset:
    """Return a frozenset of (src_primitive_id, dst_primitive_id) pairs reserved
    for the OOD-composition split.  These pairs never appear as adjacent
    directed edges in training DAGs.
    """
    import itertools
    from marifah.data.synthetic.primitives import PrimitiveType, NUM_BASE_PRIMITIVES
    all_pairs = [
        (a, b)
        for a in range(NUM_BASE_PRIMITIVES)
        for b in range(NUM_BASE_PRIMITIVES)
    ]
    n_reserve = max(1, int(len(all_pairs) * holdout_fraction))
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(all_pairs), size=n_reserve, replace=False)
    return frozenset(all_pairs[i] for i in indices)
