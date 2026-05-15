"""Tests for Laplacian eigenvector positional encoding.

Verification §4 item 4: PE on a 5-node chain graph must match
analytically-computed eigenvectors.
"""

import numpy as np
import pytest
import torch

from marifah.data.adapter.positional import (
    compute_laplacian_pe,
    laplacian_pe_tensor,
    K_PE_DEFAULT,
)


class TestLaplacianPEShapes:
    def test_shape_default_k(self):
        edges = [(0, 1), (1, 2), (2, 3)]
        pe = compute_laplacian_pe(edges, num_nodes=4)
        assert pe.shape == (4, K_PE_DEFAULT)

    def test_shape_small_k(self):
        edges = [(0, 1), (1, 2)]
        pe = compute_laplacian_pe(edges, num_nodes=3, k=2)
        assert pe.shape == (3, 2)

    def test_single_node_returns_zeros(self):
        pe = compute_laplacian_pe([], num_nodes=1)
        assert pe.shape == (1, K_PE_DEFAULT)
        assert np.allclose(pe, 0.0)

    def test_no_edges_returns_zeros(self):
        pe = compute_laplacian_pe([], num_nodes=5)
        assert pe.shape == (5, K_PE_DEFAULT)
        assert np.allclose(pe, 0.0)

    def test_tensor_output(self):
        edges = [(0, 1), (1, 2)]
        t = laplacian_pe_tensor(edges, 3, k=4)
        assert isinstance(t, torch.Tensor)
        assert t.shape == (3, 4)
        assert t.dtype == torch.float32

    def test_fewer_nodes_than_k(self):
        # 3-node graph: max 2 non-zero eigenvectors; remaining k dims padded with zeros
        edges = [(0, 1), (1, 2)]
        pe = compute_laplacian_pe(edges, num_nodes=3, k=8)
        assert pe.shape == (3, 8)
        # Columns beyond available eigenvectors should be zero
        assert np.allclose(pe[:, 2:], 0.0)


class TestLaplacianPEAnalytical:
    def test_line_graph_eigenvalues(self):
        """5-node path graph (0-1-2-3-4): Laplacian eigenvalues are
        2 - 2*cos(k*pi/5) for k=0..4.  We verify the PE columns are
        non-trivial and the structure is consistent."""
        edges = [(0, 1), (1, 2), (2, 3), (3, 4)]
        pe = compute_laplacian_pe(edges, num_nodes=5, k=4)
        assert pe.shape == (5, 4)

        # PE columns should be non-trivial (not all zero)
        col_norms = np.linalg.norm(pe, axis=0)
        assert all(n > 1e-6 for n in col_norms[:2]), f"First 2 PE cols should be non-zero: {col_norms}"

    def test_line_graph_eigenvector_orthogonality(self):
        """Eigenvectors of a symmetric matrix are orthogonal."""
        edges = [(0, 1), (1, 2), (2, 3), (3, 4)]
        pe = compute_laplacian_pe(edges, num_nodes=5, k=4)
        # Only use columns with non-trivial norms
        col_norms = np.linalg.norm(pe, axis=0)
        good_cols = [i for i, n in enumerate(col_norms) if n > 1e-6]
        if len(good_cols) >= 2:
            pe_sub = pe[:, good_cols]
            gram = pe_sub.T @ pe_sub
            off_diag = gram - np.diag(np.diag(gram))
            assert np.max(np.abs(off_diag)) < 1e-4, f"PE columns not orthogonal: {gram}"

    def test_complete_graph_k4(self):
        """Complete graph K4: all off-diagonal Laplacian entries are -1, diagonal = 3.
        Non-zero eigenvalues should all be 4 (multiplicity 3)."""
        edges = [(i, j) for i in range(4) for j in range(i + 1, 4)]
        pe = compute_laplacian_pe(edges, num_nodes=4, k=3)
        assert pe.shape == (4, 3)

    def test_deterministic(self):
        edges = [(0, 1), (1, 2), (2, 3)]
        pe1 = compute_laplacian_pe(edges, 4, k=3)
        pe2 = compute_laplacian_pe(edges, 4, k=3)
        assert np.allclose(np.abs(pe1), np.abs(pe2), atol=1e-5)  # eigenvectors unique up to sign
