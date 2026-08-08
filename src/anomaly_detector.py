import numpy as np
from sklearn.ensemble import IsolationForest
from .interfaces import BaseAnomalyDetector

class IsolationForestDetector(BaseAnomalyDetector):
    """Isolation Forest implementation of BaseAnomalyDetector.
    
    This detector wraps scikit-learn's unsupervised IsolationForest model
    and aligns its predictions and anomaly scores with the BaseAnomalyDetector contract.
    """
    
    def __init__(self, random_state: int = 42, contamination: float = 0.01, **kwargs) -> None:
        """Initializes the Isolation Forest anomaly detector.
        
        Args:
            random_state: Seed to ensure deterministic reproducibility.
            contamination: Fraction of dataset estimated to be anomalous.
            **kwargs: Additional parameters passed to sklearn's IsolationForest.
        """
        self.model = IsolationForest(
            random_state=random_state, 
            contamination=contamination,
            n_jobs=-1,
            **kwargs
        )
        self.contamination = contamination

    def fit(self, X: np.ndarray) -> 'IsolationForestDetector':
        """Fits the Isolation Forest detector on train features.
        
        Args:
            X: Training dataset features of shape (n_samples, n_features).
            
        Returns:
            The fitted instance of IsolationForestDetector.
        """
        self.model.fit(X)
        return self

    def predict_anomaly_score(self, X: np.ndarray) -> np.ndarray:
        """Computes continuous anomaly scores where higher is more anomalous.
        
        This inverts scikit-learn's decision_function scores so that higher values
        represent outliers (anomalies).
        
        Args:
            X: Dataset features of shape (n_samples, n_features).
            
        Returns:
            1D array of shape (n_samples,) representing inverted anomaly scores.
        """
        # decision_function returns positive values for inliers, negative for outliers
        raw_scores = self.model.decision_function(X)
        return -raw_scores

    def predict_anomaly(self, X: np.ndarray) -> np.ndarray:
        """Predicts binary anomaly status labels (1 for anomaly, 0 for normal).
        
        Args:
            X: Dataset features of shape (n_samples, n_features).
            
        Returns:
            1D array of shape (n_samples,) containing binary classification indicators.
        """
        preds = self.model.predict(X)
        # Convert sklearn's inlier/outlier (1/-1) labels to standard binary (0/1) indicators
        return np.where(preds == -1, 1, 0)
