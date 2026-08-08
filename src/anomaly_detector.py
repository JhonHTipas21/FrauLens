import numpy as np
from sklearn.ensemble import IsolationForest
from typing import Any, Dict
from .interfaces import BaseAnomalyDetector

class IsolationForestDetector(BaseAnomalyDetector):
    """Isolation Forest implementation of BaseAnomalyDetector."""
    
    def __init__(self, random_state: int = 42, contamination: float = 0.01, **kwargs):
        """
        Initializes the Isolation Forest anomaly detector.
        
        Args:
            random_state: Random state for reproducibility.
            contamination: Expected proportion of outliers in the data.
            **kwargs: Extra parameters for IsolationForest.
        """
        self.model = IsolationForest(
            random_state=random_state, 
            contamination=contamination,
            n_jobs=-1,
            **kwargs
        )
        self.contamination = contamination

    def fit(self, X: np.ndarray) -> 'IsolationForestDetector':
        """Fits the Isolation Forest detector on normal data."""
        self.model.fit(X)
        return self

    def predict_anomaly_score(self, X: np.ndarray) -> np.ndarray:
        """
        Computes the anomaly score. 
        Higher scores mean more anomalous.
        We invert sklearn's decision_function so higher is anomalous.
        """
        # decision_function returns negative values for anomalies, positive for inliers
        # Invert it so higher scores represent more anomalous points
        raw_scores = self.model.decision_function(X)
        return -raw_scores

    def predict_anomaly(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts binary anomaly labels.
        1 = anomaly, 0 = normal.
        """
        preds = self.model.predict(X)
        # sklearn returns 1 for inliers, -1 for outliers
        return np.where(preds == -1, 1, 0)
