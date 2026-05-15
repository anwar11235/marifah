"""Node tokenizer: raw per-node data → fixed-dim embedding.

The tokenizer is a learnable nn.Module that lives at the adapter boundary.
The data loader pre-computes raw features (primitive IDs + attribute vectors);
the tokenizer embeds them at training time.

Node feature encoding
---------------------
Each node is represented by node_feat_dim = 1 + ATTR_DIM values packed into
the batch's node_features tensor:
  index 0          : primitive_id (int, cast to float in the tensor)
  indices 1..ATTR_DIM : attribute encoding (ATTR_DIM = 4 floats in [0, 1])

Attribute encoding per primitive
---------------------------------
CONDITIONAL  : vec[0] = condition_index / 4
AGGREGATE    : vec[0] = agg_fn_index / 4
LOOKUP       : vec[0] = mean(table values) / 100
COMPARE      : all zeros
TRANSFORM    : vec[0] = transform_fn_index / 6
VALIDATE     : vec[0] = constraint_index / 4
ROUTE        : vec[0] = num_branches / 8
TERMINATE    : all zeros
ACCUMULATE   : vec[0] = step_value / 10
NOP          : all zeros
"""

from __future__ import annotations

from typing import Any, Dict, List

import torch
import torch.nn as nn

from marifah.data.synthetic.primitives import PrimitiveType

NUM_PRIMITIVES = 10
ATTR_DIM = 4            # float dimensions per node for attribute encoding
NODE_FEAT_DIM = 1 + ATTR_DIM   # total raw node feature dim (primitive_id + attrs)

_CONDITION_INDEX = {c: i for i, c in enumerate(["positive", "non_negative", "even", "odd", "zero"])}
_AGGREGATE_INDEX = {c: i for i, c in enumerate(["sum", "count", "max", "min", "mean"])}
_TRANSFORM_INDEX = {c: i for i, c in enumerate(["increment", "decrement", "double", "halve", "negate", "absolute", "square"])}
_CONSTRAINT_INDEX = {c: i for i, c in enumerate(["positive", "non_negative", "even", "non_zero", "in_range_0_100"])}


def encode_node_attrs(primitive: int, attrs: Dict[str, Any]) -> List[float]:
    """Encode a node's attributes into an ATTR_DIM-dimensional float vector.

    All values are approximately in [0, 1] for stable training.
    """
    vec = [0.0] * ATTR_DIM
    p = PrimitiveType(primitive)

    if p == PrimitiveType.CONDITIONAL:
        cond = attrs.get("condition", "positive")
        vec[0] = _CONDITION_INDEX.get(cond, 0) / 4.0

    elif p == PrimitiveType.AGGREGATE:
        fn = attrs.get("agg_fn", "sum")
        vec[0] = _AGGREGATE_INDEX.get(fn, 0) / 4.0

    elif p == PrimitiveType.LOOKUP:
        table = attrs.get("table", {})
        if table:
            vals = [v for v in table.values() if isinstance(v, (int, float))]
            vec[0] = sum(vals) / (len(vals) * 100.0 + 1e-6) if vals else 0.0

    elif p == PrimitiveType.TRANSFORM:
        fn = attrs.get("transform_fn", "increment")
        vec[0] = _TRANSFORM_INDEX.get(fn, 0) / 6.0

    elif p == PrimitiveType.VALIDATE:
        con = attrs.get("constraint", "positive")
        vec[0] = _CONSTRAINT_INDEX.get(con, 0) / 4.0

    elif p == PrimitiveType.ROUTE:
        nb = attrs.get("num_branches", 2)
        vec[0] = float(nb) / 8.0

    elif p == PrimitiveType.ACCUMULATE:
        sv = attrs.get("step_value", 1)
        vec[0] = float(sv) / 10.0

    # COMPARE, TERMINATE, NOP: all zeros
    return vec


class NodeTokenizer(nn.Module):
    """Learnable module: (primitive_ids, attr_vec) → (B, N_max, d_model).

    Parameters
    ----------
    d_model:
        Output embedding dimension.
    num_primitives:
        Vocabulary size for primitive type embedding (default 10).
    """

    def __init__(self, d_model: int = 256, num_primitives: int = NUM_PRIMITIVES) -> None:
        super().__init__()
        self.d_model = d_model
        self.primitive_embedding = nn.Embedding(num_primitives, d_model)
        self.attr_proj = nn.Linear(ATTR_DIM, d_model, bias=False)
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.primitive_embedding.weight, std=0.02)
        nn.init.normal_(self.attr_proj.weight, std=0.02)

    def forward(
        self,
        primitive_ids: torch.Tensor,   # (B, N_max) int64  OR  (N_max,) int64
        attr_vec: torch.Tensor,        # (B, N_max, ATTR_DIM) float  OR  (N_max, ATTR_DIM)
    ) -> torch.Tensor:
        """Returns (B, N_max, d_model) or (N_max, d_model) depending on input rank."""
        prim_emb = self.primitive_embedding(primitive_ids)           # (..., d_model)
        attr_emb = self.attr_proj(attr_vec.to(prim_emb.dtype))      # (..., d_model)
        return prim_emb + attr_emb
