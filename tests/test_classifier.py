"""Unit tests for src/classifier.py.

Tests cover the FraudClassifier which is a hybrid model combining an
anomaly detector (phase 1) and a supervised classifier (phase 2).
"""

import pytest
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from src.anomaly_detector import IsolationForestDetector
from src.classifier import FraudClassifier


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def dummy_X() -> np.ndarray:
    """Returns a dummy feature matrix (50 samples, 5 features)."""
    np.random.seed(42)
    return np.random.randn(50, 5)


@pytest.fixture
def dummy_y() -> np.ndarray:
    """Returns a dummy target array with 5 fraud instances."""
    np.random.seed(42)
    y = np.zeros(50, dtype=int)
    y[:5] = 1
    np.random.shuffle(y)
    return y


@pytest.fixture
def anomaly_detector() -> IsolationForestDetector:
    """Returns an un-fitted IsolationForestDetector."""
    return IsolationForestDetector(contamination=0.1, random_state=42)


@pytest.fixture
def base_classifier() -> RandomForestClassifier:
    """Returns an un-fitted RandomForestClassifier as a stand-in for XGBoost."""
    return RandomForestClassifier(n_estimators=10, random_state=42)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestFraudClassifier:
    """Tests for the hybrid FraudClassifier."""

    def test_fit_and_predict(self, dummy_X, dummy_y, anomaly_detector, base_classifier):
        """FraudClassifier should fit correctly and output binary predictions."""
        hybrid = FraudClassifier(
            anomaly_detector=anomaly_detector, classifier_model=base_classifier
        )
        hybrid.fit(dummy_X, dummy_y)

        preds = hybrid.predict(dummy_X)
        assert preds.shape == (50,)
        assert set(np.unique(preds)).issubset({0, 1})

    def test_predict_proba(self, dummy_X, dummy_y, anomaly_detector, base_classifier):
        """predict_proba should return probabilities for both classes that sum to 1."""
        hybrid = FraudClassifier(
            anomaly_detector=anomaly_detector, classifier_model=base_classifier
        )
        hybrid.fit(dummy_X, dummy_y)

        probs = hybrid.predict_proba(dummy_X)
        assert probs.shape == (50, 2)
        assert np.all((probs >= 0) & (probs <= 1))
        np.testing.assert_allclose(probs.sum(axis=1), 1.0)

    def test_threshold_adjustment(
        self, dummy_X, dummy_y, anomaly_detector, base_classifier
    ):
        """Lowering the threshold should result in more positive predictions."""
        hybrid = FraudClassifier(
            anomaly_detector=anomaly_detector, classifier_model=base_classifier
        )
        hybrid.fit(dummy_X, dummy_y)

        preds_low = hybrid.predict(dummy_X, threshold=0.1)
        preds_high = hybrid.predict(dummy_X, threshold=0.9)

        assert preds_low.sum() >= preds_high.sum()

    def test_not_fitted_error(self, dummy_X, anomaly_detector, base_classifier):
        """Predicting without fitting should raise an error."""
        hybrid = FraudClassifier(
            anomaly_detector=anomaly_detector, classifier_model=base_classifier
        )
        # scikit-learn base models typically raise NotFittedError
        with pytest.raises(Exception):
            hybrid.predict(dummy_X)
