"""Laplacian eigenvector positional encoding for DAG nodes.

Computed from the symmetric graph Laplacian L = D - A, where:
  - A is the undirected adjacency (DAG edges treated as undirected for PE)
  - D is the diagonal degree matrix

The top-K eigenvectors corresponding to the K smallest *non-zero* eigenvalues
(i.e. the Fiedler vectors and beyond) form the positional encoding.

Computed at data-loading time (not training time).  Can be precomputed and
cached via the CLI 'precompute-pe' command.

Reference: "Benchmarking Graph Neural Networks" (Dwivedi et al. 2020) §4.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import torch

K_PE_DEFAULT = 8


def _build_symmetric_laplacian(
    edges: List[Tuple[int, int]],
    num_nodes: int,
) -> "np.ndarray":
    """Build the dense symmetric Laplacian as a numpy array."""
    if num_nodes == 0:
        return np.zeros((0, 0), dtype=np.float32)

    A = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    for src, dst in edges:
        if src != dst and 0 <= src < num_nodes and 0 <= dst < num_nodes:
            A[src, dst] = 1.0
            A[dst, src] = 1.0

    # Clip: multiple edges should not inflate degree beyond 1
    A = np.clip(A, 0.0, 1.0)
    D = np.diag(A.sum(axis=1))
    return (D - A).astype(np.float32)


def compute_laplacian_pe(
    edges: List[Tuple[int, int]],
    num_nodes: int,
    k: int = K_PE_DEFAULT,
) -> np.ndarray:
    """Compute top-K Laplacian eigenvector positional encoding.

    Parameters
    ----------
    edges:
        List of (src, dst) directed edge pairs.
    num_nodes:
        Total node count.
    k:
        Number of eigenvectors to include.

    Returns
    -------
    pe : (num_nodes, k) float32 ndarray
        Padded with zeros when fewer than k non-zero eigenvectors exist.
    """
    pe = np.zeros((num_nodes, k), dtype=np.float32)

    if num_nodes <= 1:
        return pe

    L = _build_symmetric_laplacian(edges, num_nodes)

    try:
        if num_nodes <= 32:
            # Dense path: reliable for small graphs
            eigenvalues, eigenvectors = np.linalg.eigh(L)
        else:
            import scipy.sparse as sp
            import scipy.sparse.linalg as spla

            L_sparse = sp.csr_matrix(L)
            n_request = min(k + 2, num_nodes - 1)
            if n_request <= 0:
                return pe
            eigenvalues, eigenvectors = spla.eigsh(
                L_sparse, k=n_request, which="SM", tol=1e-6
            )
            idx = np.argsort(eigenvalues)
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]

        # Skip zero eigenvalue(s) — constant eigenvector(s) carry no positional information
        nonzero_idx = np.where(np.abs(eigenvalues) > 1e-5)[0]
        if len(nonzero_idx) == 0:
            return pe

        start = nonzero_idx[0]
        selected = eigenvectors[:, start : start + k].copy()  # (num_nodes, ≤k)

        # Sign-normalize: eigenvectors are only defined up to sign; fix convention so
        # that the first element with |value| > 1e-7 in each column is positive.
        # Without this, precompute=False (lazy, per-worker) can produce sign flips
        # relative to precompute=True (eager, single-threaded at init).
        for col in range(selected.shape[1]):
            col_vals = selected[:, col]
            nz = col_vals[np.abs(col_vals) > 1e-7]
            if len(nz) > 0 and nz[0] < 0:
                selected[:, col] = -col_vals

        actual = min(selected.shape[1], k)
        pe[:, :actual] = selected[:, :actual].astype(np.float32)

    except Exception:
        pass  # Return zero PE if eigendecomposition fails

    return pe


def laplacian_pe_tensor(
    edges: List[Tuple[int, int]],
    num_nodes: int,
    k: int = K_PE_DEFAULT,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """Compute Laplacian PE and return as (num_nodes, k) float32 Tensor."""
    pe = compute_laplacian_pe(edges, num_nodes, k)
    return torch.from_numpy(pe).to(device)
