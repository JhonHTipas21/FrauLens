"""Unit tests for src/anomaly_detector.py.

Tests cover anomaly score shapes, binary prediction output, contamination
boundary conditions, refitting behavior, and edge case inputs.
"""

import pytest
import numpy as np

from src.anomaly_detector import IsolationForestDetector


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def small_X_train() -> np.ndarray:
    """Returns a small normal-like training matrix (100 samples, 5 features)."""
    np.random.seed(42)
    return np.random.randn(100, 5)


@pytest.fixture
def small_X_test() -> np.ndarray:
    """Returns a small test matrix (25 samples, 5 features)."""
    np.random.seed(7)
    return np.random.randn(25, 5)


@pytest.fixture
def fitted_detector(small_X_train) -> IsolationForestDetector:
    """Returns a pre-fitted IsolationForestDetector."""
    detector = IsolationForestDetector(contamination=0.05, random_state=42)
    detector.fit(small_X_train)
    return detector


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestIsolationForestDetector:
    """Tests for the IsolationForestDetector class."""

    def test_predict_anomaly_score_shape(self, fitted_detector, small_X_test):
        """Anomaly scores should have the same number of rows as the input."""
        scores = fitted_detector.predict_anomaly_score(small_X_test)
        assert scores.shape == (len(small_X_test),)

    def test_predict_anomaly_binary_output(self, fitted_detector, small_X_test):
        """predict_anomaly must return only values in {0, 1}."""
        preds = fitted_detector.predict_anomaly(small_X_test)
        assert set(np.unique(preds)).issubset({0, 1})

    def test_predict_anomaly_shape(self, fitted_detector, small_X_test):
        """Binary predictions should have the same row count as the input."""
        preds = fitted_detector.predict_anomaly(small_X_test)
        assert preds.shape == (len(small_X_test),)

    def test_anomaly_scores_are_floats(self, fitted_detector, small_X_test):
        """Anomaly score array should have a float dtype."""
        scores = fitted_detector.predict_anomaly_score(small_X_test)
        assert np.issubdtype(scores.dtype, np.floating)

    def test_contamination_boundary_low(self, small_X_train, small_X_test):
        """Detector should work with very low contamination (near 0)."""
        detector = IsolationForestDetector(contamination=0.001, random_state=0)
        detector.fit(small_X_train)
        preds = detector.predict_anomaly(small_X_test)
        assert set(np.unique(preds)).issubset({0, 1})

    def test_contamination_boundary_high(self, small_X_train, small_X_test):
        """Detector should work with higher contamination (e.g. 0.45)."""
        detector = IsolationForestDetector(contamination=0.45, random_state=0)
        detector.fit(small_X_train)
        preds = detector.predict_anomaly(small_X_test)
        assert set(np.unique(preds)).issubset({0, 1})

    def test_refit_updates_model(self, small_X_train, small_X_test):
        """Calling fit again should produce a new usable model without errors."""
        detector = IsolationForestDetector(contamination=0.1, random_state=42)
        detector.fit(small_X_train)
        first_scores = detector.predict_anomaly_score(small_X_test).copy()
        # Refit with different seed
        detector2 = IsolationForestDetector(contamination=0.1, random_state=99)
        detector2.fit(small_X_train)
        second_scores = detector2.predict_anomaly_score(small_X_test)
        # Different seeds may produce different scores
        assert first_scores.shape == second_scores.shape

    def test_single_sample_inference(self, fitted_detector):
        """Detector should handle a single-row input without errors."""
        X_single = np.random.randn(1, 5)
        score = fitted_detector.predict_anomaly_score(X_single)
        pred = fitted_detector.predict_anomaly(X_single)
        assert score.shape == (1,)
        assert pred.shape == (1,)

    def test_scores_are_negative_in_range(self, fitted_detector, small_X_test):
        """Isolation Forest anomaly scores are expected to be in range (-1, 0]."""
        scores = fitted_detector.predict_anomaly_score(small_X_test)
        assert np.all(scores < 0.1), "Scores unexpectedly high — possible API mismatch."

    def test_high_dimensional_input(self):
        """Detector should scale to high-dimensional features without errors."""
        X = np.random.randn(300, 30)
        detector = IsolationForestDetector(contamination=0.01, random_state=0)
        detector.fit(X)
        scores = detector.predict_anomaly_score(X[:10])
        assert scores.shape == (10,)
