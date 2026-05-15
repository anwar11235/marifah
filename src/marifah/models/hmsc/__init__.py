"""Hierarchical Multi-Scale Codebook (HMSC) — the Marifah mechanism (Recognition Cortex).

Three codebooks at three scales:
  - GlobalCodebook (G): workflow signatures, pooled carry, broadcast mode
  - RegionalCodebook (R): sub-DAG patterns, region-attention pooling, per-region modes
  - PerPositionCodebook (P): reasoning primitives, cross-attention per node
"""

from marifah.models.hmsc.global_codebook import GlobalCodebook
from marifah.models.hmsc.regional_codebook import RegionalCodebook
from marifah.models.hmsc.perposition_codebook import PerPositionCodebook
from marifah.models.hmsc.composition import HMSCComposition
from marifah.models.hmsc.auxiliary_heads import (
    GlobalAuxHead,
    RegionalAuxHead,
    PerPositionAuxHead,
    compute_aux_losses,
)
from marifah.models.hmsc.hmsc import HMSC

__all__ = [
    "GlobalCodebook",
    "RegionalCodebook",
    "PerPositionCodebook",
    "HMSCComposition",
    "GlobalAuxHead",
    "RegionalAuxHead",
    "PerPositionAuxHead",
    "compute_aux_losses",
    "HMSC",
]
