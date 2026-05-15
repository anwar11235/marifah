"""§9 — Vertical config loader: YAML parsing, validation, augmentation slot."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import yaml

from marifah.data.synthetic.primitives import (
    NUM_BASE_PRIMITIVES,
    PrimitiveType,
    register_augmented_primitive,
)
from marifah.data.synthetic.patterns import NUM_PATTERNS
from marifah.data.synthetic.workflows import _N_WORKFLOWS


# ---------------------------------------------------------------------------
# Config dataclass
# ---------------------------------------------------------------------------

@dataclass
class SplitSizes:
    train: int = 800_000
    val: int = 10_000
    test_id: int = 10_000
    test_ood_size: int = 5_000
    test_ood_composition: int = 5_000


@dataclass
class GeneratorConfig:
    # Vocabulary
    primitive_names: List[str] = field(default_factory=lambda: [
        "conditional", "aggregate", "lookup", "compare", "transform",
        "validate", "route", "terminate", "accumulate", "nop",
    ])
    augmented_primitives: Dict[str, Any] = field(default_factory=dict)

    # Distribution
    allow_cycles: bool = False
    ood_holdout_fraction: float = 0.15
    ood_size_scale_min: float = 2.0
    ood_size_scale_max: float = 5.0
    ood_holdout_seed: int = 0

    # Splits
    split_sizes: SplitSizes = field(default_factory=SplitSizes)

    # Seeding
    seed: int = 0

    # Derived (populated after validation)
    config_hash: str = ""


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def load_config(path: str | Path) -> GeneratorConfig:
    """Load and validate a YAML generator config."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")

    with open(path) as fh:
        raw: dict = yaml.safe_load(fh) or {}

    cfg = GeneratorConfig()

    if "primitives" in raw:
        cfg.primitive_names = list(raw["primitives"])
    if "allow_cycles" in raw:
        cfg.allow_cycles = bool(raw["allow_cycles"])
    if "ood_holdout_fraction" in raw:
        cfg.ood_holdout_fraction = float(raw["ood_holdout_fraction"])
    if "ood_size_scale_min" in raw:
        cfg.ood_size_scale_min = float(raw["ood_size_scale_min"])
    if "ood_size_scale_max" in raw:
        cfg.ood_size_scale_max = float(raw["ood_size_scale_max"])
    if "ood_holdout_seed" in raw:
        cfg.ood_holdout_seed = int(raw["ood_holdout_seed"])
    if "seed" in raw:
        cfg.seed = int(raw["seed"])

    if "split_sizes" in raw:
        ss = raw["split_sizes"]
        cfg.split_sizes = SplitSizes(
            train=int(ss.get("train", cfg.split_sizes.train)),
            val=int(ss.get("val", cfg.split_sizes.val)),
            test_id=int(ss.get("test_id", cfg.split_sizes.test_id)),
            test_ood_size=int(ss.get("test_ood_size", cfg.split_sizes.test_ood_size)),
            test_ood_composition=int(ss.get("test_ood_composition", cfg.split_sizes.test_ood_composition)),
        )

    if "augmented_primitives" in raw:
        cfg.augmented_primitives = dict(raw["augmented_primitives"])

    _validate_config(cfg)
    cfg.config_hash = _hash_config(cfg)

    # Register any augmented primitives
    for name, _impl_ref in cfg.augmented_primitives.items():
        # Augmented primitive implementations loaded separately via plugin mechanism.
        # The config simply declares names; actual callables are registered by the
        # vertical's plugin module.
        pass

    return cfg


def _validate_config(cfg: GeneratorConfig) -> None:
    if not cfg.primitive_names:
        raise ValueError("primitive_names must be non-empty")
    if not (0 < cfg.ood_holdout_fraction < 1):
        raise ValueError("ood_holdout_fraction must be in (0, 1)")
    if cfg.ood_size_scale_min < 1:
        raise ValueError("ood_size_scale_min must be >= 1")
    if cfg.ood_size_scale_max < cfg.ood_size_scale_min:
        raise ValueError("ood_size_scale_max must be >= ood_size_scale_min")
    if cfg.allow_cycles:
        from marifah.data.synthetic.cyclic import CyclicNotImplementedError
        raise CyclicNotImplementedError(
            "allow_cycles=True is not yet implemented (cyclic mode deferred per session prompt §3)"
        )


def _hash_config(cfg: GeneratorConfig) -> str:
    import hashlib, json
    payload = json.dumps({
        "primitives": sorted(cfg.primitive_names),
        "allow_cycles": cfg.allow_cycles,
        "ood_holdout_fraction": cfg.ood_holdout_fraction,
        "ood_size_scale_min": cfg.ood_size_scale_min,
        "ood_size_scale_max": cfg.ood_size_scale_max,
        "seed": cfg.seed,
    }, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Tiny config convenience
# ---------------------------------------------------------------------------

def tiny_config(base: GeneratorConfig) -> GeneratorConfig:
    """Return a copy of cfg scaled down for tiny-dataset generation (~1K DAGs)."""
    import copy
    cfg = copy.deepcopy(base)
    scale = 1_000 / (
        base.split_sizes.train + base.split_sizes.val
        + base.split_sizes.test_id + base.split_sizes.test_ood_size
        + base.split_sizes.test_ood_composition
    )
    cfg.split_sizes = SplitSizes(
        train=max(100, int(base.split_sizes.train * scale)),
        val=max(20, int(base.split_sizes.val * scale)),
        test_id=max(20, int(base.split_sizes.test_id * scale)),
        test_ood_size=max(10, int(base.split_sizes.test_ood_size * scale)),
        test_ood_composition=max(10, int(base.split_sizes.test_ood_composition * scale)),
    )
    cfg.config_hash = _hash_config(cfg)
    return cfg
