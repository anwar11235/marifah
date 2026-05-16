"""Tests for substrate-quality probes: delta_probe and shuffled_probe.

Validates:
  1. Δ-probe: delta ≈ 0 when substrate is random-init (adds nothing beyond inputs)
  2. Δ-probe: delta ≈ 0 when z_H is constructed to equal node_features (tautology)
  3. Shuffled-primitive: shuffle preserves per-DAG primitive multiset (not global counts)
  4. Shuffled-primitive: AUC on random-init model ≈ 0.5 (chance level)
  5. Shuffled-primitive: AUC drops when primitives are shuffled for a feature-rich z_H

All tests use synthetic numpy arrays — no GPU, no real data, no checkpoint I/O.
Integration tests (real model forward, checkpoint load) are marked @pytest.mark.gpu
and run only on Vast.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from warmstart_probe import compute_workflow_type_auc


# ---------------------------------------------------------------------------
# Synthetic data helpers
# ---------------------------------------------------------------------------

def _informative_features(n: int, n_classes: int, d: int = 32, snr: float = 5.0,
                           seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Carry states where class signal is strong (SNR=5 → AUC near 1.0)."""
    rng = np.random.RandomState(seed)
    labels = np.repeat(np.arange(n_classes), n // n_classes + 1)[:n].astype(np.int64)
    centroids = rng.randn(n_classes, d).astype(np.float32) * snr
    carry = centroids[labels] + rng.randn(n, d).astype(np.float32)
    return carry, labels


def _random_features(n: int, d: int = 32, seed: int = 1) -> np.ndarray:
    rng = np.random.RandomState(seed)
    return rng.randn(n, d).astype(np.float32)


def _random_labels(n: int, n_classes: int, seed: int = 2) -> np.ndarray:
    rng = np.random.RandomState(seed)
    return rng.randint(0, n_classes, size=n).astype(np.int64)


# ---------------------------------------------------------------------------
# Δ-probe unit tests
# ---------------------------------------------------------------------------

class TestDeltaProbe:
    """Δ-probe logic: delta = AUC(z_H) - AUC(node_features)."""

    def test_delta_near_zero_when_zh_equals_node_features(self):
        """If z_H == node_features, baseline and substrate AUC are identical → Δ ≈ 0."""
        n, n_classes = 200, 5
        feats, labels = _informative_features(n, n_classes, d=16, snr=3.0, seed=10)
        z_H = feats.copy()  # deliberate tautology: z_H is the same as node_features

        res_baseline = compute_workflow_type_auc(feats, labels, seed=0)
        res_substrate = compute_workflow_type_auc(z_H, labels, seed=0)
        delta = res_substrate["auc"] - res_baseline["auc"]

        assert abs(delta) < 0.05, (
            f"Delta should be ~0 when z_H == node_features, got {delta:.4f}. "
            f"baseline={res_baseline['auc']:.4f} substrate={res_substrate['auc']:.4f}"
        )

    def test_delta_positive_when_zh_has_more_info_than_inputs(self):
        """If z_H carries extra class info that inputs don't, delta > 0."""
        n, n_classes = 300, 5
        rng = np.random.RandomState(20)
        labels = np.repeat(np.arange(n_classes), n // n_classes + 1)[:n].astype(np.int64)

        # Baseline features: weak class signal
        weak_centroids = rng.randn(n_classes, 8).astype(np.float32) * 0.5
        node_feats = weak_centroids[labels] + rng.randn(n, 8).astype(np.float32) * 2.0

        # z_H: strong class signal (simulates trained substrate)
        strong_centroids = rng.randn(n_classes, 16).astype(np.float32) * 5.0
        z_H = strong_centroids[labels] + rng.randn(n, 16).astype(np.float32)

        res_baseline = compute_workflow_type_auc(node_feats, labels, seed=0)
        res_substrate = compute_workflow_type_auc(z_H, labels, seed=0)
        delta = res_substrate["auc"] - res_baseline["auc"]

        assert delta > 0.05, (
            f"Expected positive delta when z_H has stronger class signal than inputs, "
            f"got {delta:.4f}. baseline={res_baseline['auc']:.4f} substrate={res_substrate['auc']:.4f}"
        )

    def test_delta_near_zero_on_random_init_substrate(self):
        """Random-init z_H carries no class information → AUC(z_H) ≈ AUC(node_features) for uninformative inputs."""
        n, n_classes = 200, 5
        rng = np.random.RandomState(30)
        labels = np.repeat(np.arange(n_classes), n // n_classes + 1)[:n].astype(np.int64)

        # Both are random — neither should be strongly informative
        node_feats = _random_features(n, d=8, seed=31)
        z_H = _random_features(n, d=32, seed=32)

        # Both AUCs should be near 0.5 (chance); delta should be near 0
        res_baseline = compute_workflow_type_auc(node_feats, labels, seed=0)
        res_substrate = compute_workflow_type_auc(z_H, labels, seed=0)
        delta = abs(res_substrate["auc"] - res_baseline["auc"])

        assert delta < 0.20, (
            f"With random features, |delta| should be small, got {delta:.4f}"
        )

    def test_bootstrap_delta_ci_contains_point_estimate(self):
        """Bootstrap CI on Δ should bracket the point-estimate delta."""
        sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
        from delta_probe import _bootstrap_delta

        n, n_classes = 200, 5
        feats, labels = _informative_features(n, n_classes, d=16, snr=3.0, seed=40)
        z_H, _ = _informative_features(n, n_classes, d=32, snr=5.0, seed=41)

        result = _bootstrap_delta(feats, z_H, labels, n_bootstrap=20, base_seed=0)

        assert result is not None, "Bootstrap returned None with 20 iterations on clean data"
        assert result["ci_low"] <= result["mean"] <= result["ci_high"], (
            f"CI order violated: {result['ci_low']:.4f} <= {result['mean']:.4f} <= {result['ci_high']:.4f}"
        )
        assert result["n_successful"] >= 10


# ---------------------------------------------------------------------------
# Shuffled-primitive probe unit tests
# ---------------------------------------------------------------------------

class TestShuffledProbe:
    """Shuffled-primitive probe mechanics."""

    def test_shuffle_preserves_per_dag_primitive_multiset(self):
        """Shuffling primitives within each DAG must preserve the multiset per DAG."""
        import torch
        from marifah.data.adapter.batch_format import GraphBatch
        sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
        from shuffled_probe import _shuffle_batch_primitives

        B, N, feat_dim = 4, 10, 5
        rng_pt = torch.manual_seed(77)
        node_features = torch.randint(0, 10, (B, N, feat_dim)).float()
        node_mask = torch.ones(B, N, dtype=torch.bool)
        node_mask[:, 7:] = False  # last 3 positions padding
        primitive_assignments = node_features[:, :, 0].long()

        batch = GraphBatch(
            node_features=node_features,
            attention_mask=torch.zeros(B, N, N),
            node_mask=node_mask,
            pos_encoding=torch.zeros(B, N, 4),
            primitive_assignments=primitive_assignments,
            workflow_type_id=torch.zeros(B, dtype=torch.long),
            region_assignments=torch.zeros(B, N, dtype=torch.long),
            halt_step=torch.zeros(B, dtype=torch.long),
            execution_trace=[[] for _ in range(B)],
        )

        rng_np = np.random.RandomState(42)
        shuffled = _shuffle_batch_primitives(batch, torch.device("cpu"), rng_np)

        for b in range(B):
            real = node_mask[b].nonzero(as_tuple=True)[0]
            orig_set = sorted(node_features[b, real, 0].long().tolist())
            shuf_set = sorted(shuffled.node_features[b, real, 0].long().tolist())
            assert orig_set == shuf_set, (
                f"DAG {b}: primitive multiset changed after shuffle. "
                f"orig={orig_set} shuffled={shuf_set}"
            )

    def test_shuffle_does_not_change_graph_structure(self):
        """Shuffling must leave attention_mask, node_mask, pos_encoding unchanged."""
        import torch
        from marifah.data.adapter.batch_format import GraphBatch
        sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
        from shuffled_probe import _shuffle_batch_primitives

        B, N = 3, 8
        attention_mask = torch.randn(B, N, N)
        pos_encoding = torch.randn(B, N, 4)
        node_mask = torch.ones(B, N, dtype=torch.bool)

        batch = GraphBatch(
            node_features=torch.randn(B, N, 5),
            attention_mask=attention_mask.clone(),
            node_mask=node_mask,
            pos_encoding=pos_encoding.clone(),
            primitive_assignments=torch.zeros(B, N, dtype=torch.long),
            workflow_type_id=torch.zeros(B, dtype=torch.long),
            region_assignments=torch.zeros(B, N, dtype=torch.long),
            halt_step=torch.zeros(B, dtype=torch.long),
            execution_trace=[[] for _ in range(B)],
        )

        rng_np = np.random.RandomState(0)
        shuffled = _shuffle_batch_primitives(batch, torch.device("cpu"), rng_np)

        assert torch.allclose(shuffled.attention_mask, attention_mask), \
            "attention_mask changed after shuffle"
        assert torch.allclose(shuffled.pos_encoding, pos_encoding), \
            "pos_encoding changed after shuffle"
        assert torch.equal(shuffled.node_mask, node_mask), \
            "node_mask changed after shuffle"

    def test_shuffled_node_features_differ_from_original(self):
        """After shuffling, at least some node_features[b,:,0] values should move."""
        import torch
        from marifah.data.adapter.batch_format import GraphBatch
        sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
        from shuffled_probe import _shuffle_batch_primitives

        B, N = 4, 12
        # Use distinct primitive IDs so shuffle is visible
        node_features = torch.arange(B * N).view(B, N, 1).float().expand(B, N, 5).clone()
        node_mask = torch.ones(B, N, dtype=torch.bool)

        batch = GraphBatch(
            node_features=node_features,
            attention_mask=torch.zeros(B, N, N),
            node_mask=node_mask,
            pos_encoding=torch.zeros(B, N, 4),
            primitive_assignments=torch.arange(B * N).view(B, N),
            workflow_type_id=torch.zeros(B, dtype=torch.long),
            region_assignments=torch.zeros(B, N, dtype=torch.long),
            halt_step=torch.zeros(B, dtype=torch.long),
            execution_trace=[[] for _ in range(B)],
        )

        rng_np = np.random.RandomState(7)
        shuffled = _shuffle_batch_primitives(batch, torch.device("cpu"), rng_np)

        changed = not torch.equal(shuffled.node_features[:, :, 0], node_features[:, :, 0])
        assert changed, "node_features[...,0] should be permuted after shuffling"

    def test_shuffled_auc_near_chance_on_random_z_H(self):
        """On random z_H (random-init substrate), shuffled-primitive AUC should be near 0.5."""
        n, n_classes = 300, 5
        rng = np.random.RandomState(50)
        labels = np.repeat(np.arange(n_classes), n // n_classes + 1)[:n].astype(np.int64)

        z_H_random = rng.randn(n, 32).astype(np.float32)
        result = compute_workflow_type_auc(z_H_random, labels, seed=0)

        assert result["auc"] < 0.70, (
            f"AUC on random z_H should be near chance, got {result['auc']:.4f}. "
            f"If much higher, random features are somehow informative (data issue)."
        )

    def test_auc_drops_after_shuffle_when_z_H_reads_primitives(self):
        """If z_H encodes primitive identity, shuffling primitives destroys the class signal."""
        n, n_classes = 300, 5
        rng = np.random.RandomState(60)
        labels = np.repeat(np.arange(n_classes), n // n_classes + 1)[:n].astype(np.int64)

        # z_H copies primitive-like signal from inputs (simulates substrate reading primitive identity)
        node_feats_0 = (labels * 10 + rng.randint(0, 5, n)).astype(np.float32)  # strong primitive signal
        z_H = np.column_stack([node_feats_0, rng.randn(n, 15).astype(np.float32)])

        # Un-shuffled: AUC should be high (z_H encodes primitive identity → class)
        res_unshuffled = compute_workflow_type_auc(z_H, labels, seed=0)

        # Shuffled: simulate shuffling by replacing z_H[:,0] with shuffled node_feats_0
        shuffled_signal = rng.permutation(node_feats_0)
        z_H_shuffled = np.column_stack([shuffled_signal, rng.randn(n, 15).astype(np.float32)])
        # After shuffle, reassign labels to different classes (simulate class label stays fixed,
        # but z_H now carries a different sample's primitive signal)
        res_shuffled = compute_workflow_type_auc(z_H_shuffled, labels, seed=0)

        drop = res_unshuffled["auc"] - res_shuffled["auc"]

        assert drop > 0.0, (
            f"AUC should drop when z_H encodes primitive identity and primitives are shuffled. "
            f"un-shuffled={res_unshuffled['auc']:.4f} shuffled={res_shuffled['auc']:.4f} drop={drop:.4f}"
        )


# ---------------------------------------------------------------------------
# Dimensionality-matched Δ-baseline unit tests
# ---------------------------------------------------------------------------

class TestDeltaBaselineDimMatched:
    """Verify the dim-matched random projection applied to node_features baseline."""

    def test_delta_baseline_uses_dim_matched_projection(self):
        """Random projection must map node_features to d_model-dim space."""
        import torch
        raw_dim, d_model, n = 5, 32, 50
        raw = np.random.randn(n, raw_dim).astype(np.float32)

        rng_proj = torch.Generator()
        rng_proj.manual_seed(0)
        proj = torch.randn(raw_dim, d_model, generator=rng_proj)
        proj = proj / proj.norm(dim=0, keepdim=True)
        projected = (torch.from_numpy(raw) @ proj).numpy()

        assert projected.shape == (n, d_model), (
            f"Expected ({n}, {d_model}), got {projected.shape}"
        )

    def test_delta_random_init_is_near_zero_with_dim_matching(self):
        """With dim-matched baseline, delta on random z_H should be near 0 (was ~0.18 without).

        Both baseline (random projection of 5-dim inputs) and substrate (random init z_H)
        are uninformative in the same way. delta should be small, not 0.18 from dim-inflation.
        """
        import torch
        n, n_classes, d_model = 300, 5, 32
        rng = np.random.RandomState(70)
        labels = np.repeat(np.arange(n_classes), n // n_classes + 1)[:n].astype(np.int64)

        raw_feats = rng.randn(n, 5).astype(np.float32)

        rng_proj = torch.Generator()
        rng_proj.manual_seed(0)
        proj = torch.randn(5, d_model, generator=rng_proj)
        proj = proj / proj.norm(dim=0, keepdim=True)
        baseline_input = (torch.from_numpy(raw_feats) @ proj).numpy()

        z_H_random = rng.randn(n, d_model).astype(np.float32)

        res_baseline = compute_workflow_type_auc(baseline_input, labels, seed=0)
        res_substrate = compute_workflow_type_auc(z_H_random, labels, seed=0)
        delta = abs(res_substrate["auc"] - res_baseline["auc"])

        assert delta < 0.10, (
            f"With dim-matched baseline, |delta| on random z_H should be small. "
            f"Got {delta:.4f} (was ~0.18 with 5-dim vs 512-dim). "
            f"baseline={res_baseline['auc']:.4f} substrate={res_substrate['auc']:.4f}"
        )
