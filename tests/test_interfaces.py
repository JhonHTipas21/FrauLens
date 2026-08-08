import pytest
from src.interfaces import (
    AnomalyScore, 
    FeatureExplanation, 
    Explanation, 
    BaseAnomalyDetector, 
    BaseExplainer
)

def test_dataclasses():
    """Verifies instantiation and attribute access of data contracts."""
    # Test AnomalyScore
    score = AnomalyScore(score=0.85, is_anomaly=True, metadata={"method": "iforest"})
    assert score.score == 0.85
    assert score.is_anomaly is True
    assert score.metadata["method"] == "iforest"

    # Test FeatureExplanation
    feat_exp = FeatureExplanation(feature_name="Amount", value=150.0, contribution=0.45)
    assert feat_exp.feature_name == "Amount"
    assert feat_exp.value == 150.0
    assert feat_exp.contribution == 0.45

    # Test Explanation
    exp = Explanation(base_value=0.1, prediction_score=0.75, features=[feat_exp])
    assert exp.base_value == 0.1
    assert exp.prediction_score == 0.75
    assert len(exp.features) == 1
    assert exp.features[0].feature_name == "Amount"

def test_cannot_instantiate_abstract_detector():
    """Verifies that BaseAnomalyDetector cannot be instantiated due to abstract methods."""
    with pytest.raises(TypeError):
        # BaseAnomalyDetector has abstract methods fit, predict_anomaly_score, predict_anomaly
        BaseAnomalyDetector()

def test_cannot_instantiate_abstract_explainer():
    """Verifies that BaseExplainer cannot be instantiated due to abstract methods."""
    with pytest.raises(TypeError):
        # BaseExplainer has abstract methods explain_instance, explain_global
        BaseExplainer()
