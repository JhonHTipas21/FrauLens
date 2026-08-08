import pytest
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from src.anomaly_detector import IsolationForestDetector
from src.classifier import FraudClassifier
from src.explainer import SHAPExplainer
from src.utils import DataPreprocessor, temporal_split, calculate_metrics

@pytest.fixture
def dummy_data():
    """Generates a dummy DataFrame with Time, V1-V5, Amount, and Class."""
    np.random.seed(42)
    n_samples = 200
    
    # 5 random features plus Time and Amount
    X = np.random.randn(n_samples, 5)
    time = np.linspace(0, 1000, n_samples)
    amount = np.random.exponential(scale=50, size=n_samples)
    
    # Extreme class imbalance: 5 fraud instances
    y = np.zeros(n_samples, dtype=int)
    y[np.random.choice(n_samples, 5, replace=False)] = 1
    
    data = {
        'Time': time,
        'Amount': amount,
        'Class': y
    }
    for i in range(5):
        data[f'V{i+1}'] = X[:, i]
        
    return pd.DataFrame(data)

def test_temporal_split(dummy_data):
    """Checks that the temporal split splits based on time and returns correct shapes."""
    train_df, test_df = temporal_split(dummy_data, time_col='Time', test_ratio=0.2)
    
    assert len(train_df) == 160
    assert len(test_df) == 40
    
    # Check chronological order
    assert train_df['Time'].max() < test_df['Time'].min()

def test_data_preprocessor(dummy_data):
    """Verifies that the preprocessor scales variables and splits features/targets."""
    preprocessor = DataPreprocessor()
    preprocessor.fit(dummy_data, target_col='Class')
    
    X, y = preprocessor.transform(dummy_data)
    
    # Features include Time, Amount, V1-V5 (7 columns total)
    assert X.shape == (200, 7)
    assert y.shape == (200,)
    
    # Original target remains binary
    assert np.array_equal(np.unique(y), [0, 1])

def test_isolation_forest_detector():
    """Verifies that the IsolationForestDetector outputs correct shapes and binary labels."""
    X_train = np.random.randn(100, 5)
    X_test = np.random.randn(20, 5)
    
    detector = IsolationForestDetector(contamination=0.1, random_state=42)
    detector.fit(X_train)
    
    scores = detector.predict_anomaly_score(X_test)
    preds = detector.predict_anomaly(X_test)
    
    assert scores.shape == (20,)
    assert preds.shape == (20,)
    # Verify predictions are binary 0/1
    assert set(np.unique(preds)).issubset({0, 1})

def test_fraud_classifier(dummy_data):
    """Verifies the hybrid FraudClassifier fits and predicts probabilities/labels correctly."""
    preprocessor = DataPreprocessor().fit(dummy_data, target_col='Class')
    X, y = preprocessor.transform(dummy_data)
    
    detector = IsolationForestDetector(contamination=0.05, random_state=42)
    
    # Use standard RandomForestClassifier for dummy testing to bypass XGBoost logging or warnings
    rf = RandomForestClassifier(n_estimators=10, random_state=42)
    
    hybrid = FraudClassifier(anomaly_detector=detector, classifier_model=rf)
    hybrid.fit(X, y)
    
    probs = hybrid.predict_proba(X)
    preds = hybrid.predict(X, threshold=0.5)
    
    assert probs.shape == (200, 2)
    assert preds.shape == (200,)
    assert set(np.unique(preds)).issubset({0, 1})

def test_calculate_metrics():
    """Checks the metrics calculation dictionary format and values."""
    y_true = np.array([0, 0, 0, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.6, 0.4, 0.9])
    
    # At threshold 0.5: 
    # y_pred = [0, 0, 1, 0, 1]
    # TP = 1 (idx 4), FP = 1 (idx 2), TN = 2 (idx 0, 1), FN = 1 (idx 3)
    # Precision = TP / (TP+FP) = 1/2 = 0.5
    # Recall = TP / (TP+FN) = 1/2 = 0.5
    # F1 = 0.5
    metrics = calculate_metrics(y_true, y_prob, threshold=0.5)
    
    assert metrics["precision"] == 0.5
    assert metrics["recall"] == 0.5
    assert metrics["f1"] == 0.5
    assert metrics["confusion_matrix"]["tp"] == 1
    assert metrics["confusion_matrix"]["fp"] == 1
    assert metrics["confusion_matrix"]["tn"] == 2
    assert metrics["confusion_matrix"]["fn"] == 1

def test_shap_explainer(dummy_data):
    """Verifies SHAP explainer extracts correct individual and global explanation formats."""
    preprocessor = DataPreprocessor().fit(dummy_data, target_col='Class')
    X, y = preprocessor.transform(dummy_data)
    
    detector = IsolationForestDetector(contamination=0.05, random_state=42)
    # Fit SHAP on the trained models. SHAP TreeExplainer requires tree-based model.
    # We use XGBoost or RandomForest here.
    import xgboost as xgb
    xgb_model = xgb.XGBClassifier(n_estimators=10, max_depth=3, random_state=42, eval_metric="logloss")
    
    hybrid = FraudClassifier(anomaly_detector=detector, classifier_model=xgb_model)
    hybrid.fit(X, y)
    
    explainer = SHAPExplainer(hybrid)
    
    # Explain single transaction
    feature_names = preprocessor.feature_cols
    inst_idx = 0
    explanation = explainer.explain_instance(X[inst_idx], feature_names)
    
    # Features in explanation should match (original features + anomaly_score)
    expected_num_features = len(feature_names) + 1
    assert len(explanation.features) == expected_num_features
    assert explanation.features[-1].feature_name == "anomaly_score"
    
    # Verify predictions score matches class probability
    assert explanation.prediction_score == pytest.approx(hybrid.predict_proba(X[inst_idx].reshape(1, -1))[0, 1], abs=1e-5)
    
    # Test global explanation
    global_exp = explainer.explain_global(X[:50], feature_names)
    assert len(global_exp) == expected_num_features
    assert "anomaly_score" in global_exp
    # Assert values are sorted descending
    values = list(global_exp.values())
    assert all(values[i] >= values[i+1] for i in range(len(values)-1))
