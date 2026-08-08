"""Unit tests for src/explainer.py.

Tests cover the SHAPExplainer and verify that individual and global explanations
are generated correctly and correctly formatted into domain objects.
"""

import pytest
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from src.anomaly_detector import IsolationForestDetector
from src.classifier import FraudClassifier
from src.explainer import SHAPExplainer
from src.interfaces import Explanation


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def dummy_X() -> np.ndarray:
    """Returns a dummy feature matrix (20 samples, 4 features)."""
    np.random.seed(42)
    return np.random.randn(20, 4)


@pytest.fixture
def dummy_y() -> np.ndarray:
    """Returns a dummy target array with binary labels."""
    np.random.seed(42)
    return np.random.choice([0, 1], size=20)


@pytest.fixture
def feature_names() -> list:
    """Returns a dummy list of feature names."""
    return ["Time", "Amount", "V1", "V2"]


@pytest.fixture
def fitted_hybrid_model(dummy_X, dummy_y) -> FraudClassifier:
    """Returns a pre-fitted hybrid FraudClassifier."""
    # We use RandomForest instead of XGBoost so SHAP's TreeExplainer doesn't 
    # complain about missing libraries or cause slow startup in tests.
    detector = IsolationForestDetector(contamination=0.1, random_state=42)
    base_model = RandomForestClassifier(n_estimators=5, max_depth=3, random_state=42)
    hybrid = FraudClassifier(anomaly_detector=detector, classifier_model=base_model)
    hybrid.fit(dummy_X, dummy_y)
    return hybrid


@pytest.fixture
def explainer(fitted_hybrid_model) -> SHAPExplainer:
    """Returns an initialized SHAPExplainer wrapped around the hybrid model."""
    return SHAPExplainer(fitted_hybrid_model)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestSHAPExplainer:
    """Tests for the SHAPExplainer wrapper."""

    def test_explain_instance_type(self, explainer, dummy_X, feature_names):
        """explain_instance should return an Explanation domain object."""
        exp = explainer.explain_instance(dummy_X[0], feature_names)
        assert isinstance(exp, Explanation)

    def test_explain_instance_feature_count(self, explainer, dummy_X, feature_names):
        """Explanation should contain all base features plus 'anomaly_score'."""
        exp = explainer.explain_instance(dummy_X[0], feature_names)
        expected_len = len(feature_names) + 1
        assert len(exp.features) == expected_len

    def test_explain_instance_anomaly_score_present(self, explainer, dummy_X, feature_names):
        """The 'anomaly_score' must be explicitly present in the explanation."""
        exp = explainer.explain_instance(dummy_X[0], feature_names)
        feature_names_in_exp = [f.feature_name for f in exp.features]
        assert "anomaly_score" in feature_names_in_exp

    def test_explain_global_dict_return(self, explainer, dummy_X, feature_names):
        """explain_global should return a dictionary of feature importances."""
        global_exp = explainer.explain_global(dummy_X[:5], feature_names)
        assert isinstance(global_exp, dict)

    def test_explain_global_keys(self, explainer, dummy_X, feature_names):
        """Global explanation should contain keys for all features + anomaly_score."""
        global_exp = explainer.explain_global(dummy_X[:5], feature_names)
        expected_keys = set(feature_names + ["anomaly_score"])
        assert set(global_exp.keys()) == expected_keys

    def test_explain_global_values_sorted(self, explainer, dummy_X, feature_names):
        """Global explanation values must be sorted in descending order of absolute importance."""
        global_exp = explainer.explain_global(dummy_X[:5], feature_names)
        values = list(global_exp.values())
        for i in range(len(values) - 1):
            assert values[i] >= values[i + 1]

    def test_prediction_score_matches(self, explainer, fitted_hybrid_model, dummy_X, feature_names):
        """The prediction_score in the Explanation object should match the model's actual predict_proba."""
        instance = dummy_X[0]
        # Explanation prediction score
        exp = explainer.explain_instance(instance, feature_names)
        
        # Actual model prediction score
        # Note: FraudClassifier expects a 2D array
        actual_prob = fitted_hybrid_model.predict_proba(instance.reshape(1, -1))[0, 1]
        
        np.testing.assert_allclose(exp.prediction_score, actual_prob, rtol=1e-5)
