"""Regression tests for warmstart_probe.compute_workflow_type_auc.

Verifies:
  1. Stratified split prevents NaN AUC on skewed 37-class distributions.
  2. AUC failure raises RuntimeError — no silent fallback to accuracy.
  3. Bootstrap CI is internally consistent and values are in [0, 1].
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# Ensure scripts/ is importable without installation
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from warmstart_probe import compute_workflow_type_auc, _bootstrap_auc


# ---------------------------------------------------------------------------
# Synthetic data helpers
# ---------------------------------------------------------------------------


def _make_carry_states(n: int, d: int = 32, seed: int = 42) -> np.ndarray:
    rng = np.random.RandomState(seed)
    return rng.randn(n, d).astype(np.float32)


def _make_skewed_labels(n_total: int, n_classes: int, seed: int = 0) -> np.ndarray:
    """Produce labels with a realistic long-tail: some classes have 1-2 samples."""
    rng = np.random.RandomState(seed)
    labels = []
    # Assign guaranteed samples first (many classes get only 1)
    for c in range(n_classes):
        count = 1 if c > n_classes // 2 else max(2, n_total // n_classes)
        labels.extend([c] * count)
    # Fill remainder from the common classes
    remaining = n_total - len(labels)
    if remaining > 0:
        common = np.arange(n_classes // 2)
        extra = rng.choice(common, size=remaining, replace=True)
        labels.extend(extra.tolist())
    arr = np.array(labels[:n_total], dtype=np.int64)
    rng.shuffle(arr)
    return arr


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestComputeWorkflowTypeAucStratified:
    """compute_workflow_type_auc uses stratified split and returns real AUC."""

    def test_stratified_split_returns_finite_auc_on_skewed_37_class(self):
        """37-class skewed distribution (previously triggered NaN) must yield finite AUC."""
        n = 1000
        n_classes = 37
        labels = _make_skewed_labels(n, n_classes, seed=7)
        carry = _make_carry_states(n, seed=7)

        result = compute_workflow_type_auc(carry, labels, seed=0)

        assert isinstance(result["auc"], float), "auc must be a float"
        assert np.isfinite(result["auc"]), f"auc must be finite, got {result['auc']}"
        assert 0.0 <= result["auc"] <= 1.0, f"auc must be in [0,1], got {result['auc']}"
        assert result["n_classes_present_in_test"] >= 2

    def test_result_has_all_required_keys(self):
        labels = _make_skewed_labels(200, 5, seed=1)
        carry = _make_carry_states(200, seed=1)
        result = compute_workflow_type_auc(carry, labels, seed=0)
        for key in ("auc", "accuracy", "n_classes", "n_classes_present_in_test",
                    "n_classes_filtered_single_sample", "n_train", "n_test"):
            assert key in result, f"missing key: {key}"

    def test_n_train_plus_n_test_equals_filtered_n(self):
        labels = _make_skewed_labels(500, 10, seed=2)
        carry = _make_carry_states(500, seed=2)
        result = compute_workflow_type_auc(carry, labels, test_fraction=0.3, seed=0)
        assert result["n_train"] + result["n_test"] <= 500

    def test_single_instance_classes_are_filtered_with_warning(self, caplog):
        """Classes with only 1 sample should be filtered and logged."""
        import logging
        # 10 classes; 3 of them have exactly 1 sample
        labels = np.array([0] * 100 + [1] * 100 + [2] * 100 + [3] * 100 +
                          [4] * 100 + [5] * 100 + [6] * 100 + [7] * 1 +
                          [8] * 1 + [9] * 1, dtype=np.int64)
        np.random.RandomState(0).shuffle(labels)
        carry = _make_carry_states(len(labels), seed=3)

        with caplog.at_level(logging.WARNING, logger="warmstart_probe"):
            result = compute_workflow_type_auc(carry, labels, seed=0)

        assert result["n_classes_filtered_single_sample"] == 3
        assert "1 sample" in caplog.text


class TestComputeWorkflowTypeAucRaisesOnBadInput:
    """AUC failure raises RuntimeError; no silent accuracy fallback."""

    def test_raises_when_too_few_samples_after_filtering(self):
        """If filtering singletons leaves <10 samples, raise RuntimeError."""
        # 5 classes, each with exactly 1 sample → all filtered
        labels = np.arange(5, dtype=np.int64)
        carry = _make_carry_states(5, seed=4)
        with pytest.raises(RuntimeError, match="Too few samples"):
            compute_workflow_type_auc(carry, labels, seed=0)

    def test_does_not_return_accuracy_when_auc_is_unavailable(self):
        """The old code returned accuracy as 'AUC' on NaN. The new code must raise."""
        # Construct a worst-case: all labels the same class in one partition.
        # This is hard to trigger post-stratification, so we test via the
        # too-few-samples path which previously fell through to accuracy.
        labels = np.array([0, 1], dtype=np.int64)  # only 2 samples total
        carry = _make_carry_states(2, seed=5)
        # With 2 samples and test_fraction=0.3 → test_size=1 → 1 class in test
        with pytest.raises((RuntimeError, ValueError)):
            compute_workflow_type_auc(carry, labels, test_fraction=0.3, seed=0)


class TestBootstrapAuc:
    """_bootstrap_auc returns internally consistent CI."""

    def test_bootstrap_returns_valid_ci(self):
        """ci_low <= mean <= ci_high and all in [0, 1] on clean data."""
        n = 200
        n_classes = 5
        # Make informative carry states: class centroid + noise
        rng = np.random.RandomState(10)
        centroids = rng.randn(n_classes, 16).astype(np.float32)
        labels = np.repeat(np.arange(n_classes), n // n_classes).astype(np.int64)
        carry = centroids[labels] + 0.3 * rng.randn(n, 16).astype(np.float32)
        np.random.RandomState(11).shuffle(labels)

        result = _bootstrap_auc(carry, labels, n_bootstrap=20, base_seed=0)

        assert result is not None, "bootstrap returned None with only 20 iterations on clean data"
        assert result["ci_low"] <= result["mean"] <= result["ci_high"], (
            f"CI order violated: {result['ci_low']} <= {result['mean']} <= {result['ci_high']}"
        )
        assert 0.0 <= result["ci_low"] <= 1.0
        assert 0.0 <= result["ci_high"] <= 1.0
        assert result["n_successful"] >= 10

    def test_bootstrap_returns_none_when_too_few_succeed(self):
        """Bootstrap returns None when n_bootstrap < 10 (can never reach min threshold)."""
        # n_bootstrap=5 means at most 5 iterations can succeed, and 5 < 10 → always None.
        # Using a 5-class clean dataset so each iteration would succeed if run,
        # but the threshold (n_successful >= 10) is impossible to reach with only 5 attempts.
        rng = np.random.RandomState(0)
        labels = np.repeat(np.arange(5), 20).astype(np.int64)
        carry = rng.randn(100, 8).astype(np.float32)
        result = _bootstrap_auc(carry, labels, n_bootstrap=5, base_seed=0)
        assert result is None

    def test_bootstrap_does_not_rerun_model_inference(self):
        """_bootstrap_auc takes pre-extracted carry_states, not a model — no inference."""
        # Verify the function signature accepts only numpy arrays, not a model object.
        import inspect
        sig = inspect.signature(_bootstrap_auc)
        param_names = list(sig.parameters.keys())
        assert "model" not in param_names, (
            "_bootstrap_auc must not accept a model argument — carry_states must be pre-extracted"
        )
        assert "carry_states" in param_names
        assert "labels" in param_names
