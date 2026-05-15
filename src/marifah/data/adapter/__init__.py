"""Graph adapter: bridges synthetic DAG generator output to CORAL forward pass.

Components
----------
batch_format    GraphBatch dataclass and tensor layout spec
tokenizer       NodeTokenizer — learnable module mapping per-node raw features to d_model
positional      Laplacian eigenvector positional encoding (computed at data-load time)
attention_mask  Edge-induced additive-bias attention mask construction
dataset         GraphDAGDataset — torch.utils.data.Dataset wrapping parquet shards
collate         collate_graphs — batch collation with padding to max-nodes-in-batch
cli             CLI entry points: precompute-pe, inspect-batch
"""

from marifah.data.adapter.batch_format import GraphBatch
from marifah.data.adapter.tokenizer import NodeTokenizer, encode_node_attrs, ATTR_DIM
from marifah.data.adapter.positional import compute_laplacian_pe, laplacian_pe_tensor
from marifah.data.adapter.attention_mask import build_attention_mask, pad_attention_masks
from marifah.data.adapter.dataset import GraphDAGDataset
from marifah.data.adapter.collate import collate_graphs

__all__ = [
    "GraphBatch",
    "NodeTokenizer",
    "encode_node_attrs",
    "ATTR_DIM",
    "compute_laplacian_pe",
    "laplacian_pe_tensor",
    "build_attention_mask",
    "pad_attention_masks",
    "GraphDAGDataset",
    "collate_graphs",
]
