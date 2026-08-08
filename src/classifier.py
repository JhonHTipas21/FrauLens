import numpy as np
import xgboost as xgb
from typing import Any, Optional
from .interfaces import BaseAnomalyDetector

class FraudClassifier:
    """
    Hybrid classifier combining unsupervised anomaly detection with supervised classification.
    Implements Open/Closed Principle: can accept any BaseAnomalyDetector and any standard scikit-learn classifier.
    """
    
    def __init__(self, anomaly_detector: BaseAnomalyDetector, classifier_model: Optional[Any] = None):
        """
        Initializes the FraudClassifier.
        
        Args:
            anomaly_detector: An instance of a class implementing BaseAnomalyDetector.
            classifier_model: An optional scikit-learn compatible classifier (e.g., RandomForestClassifier, XGBClassifier).
                              If None, a default XGBClassifier is instantiated.
        """
        self.anomaly_detector = anomaly_detector
        
        if classifier_model is None:
            # Default to XGBoost with settings suited for tabular data
            self.classifier = xgb.XGBClassifier(
                max_depth=5,
                learning_rate=0.05,
                n_estimators=150,
                random_state=42,
                eval_metric="logloss",
                use_label_encoder=False
            )
        else:
            self.classifier = classifier_model
            
        self.anomaly_detector_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'FraudClassifier':
        """
        Fits the hybrid model.
        1. Fits the anomaly detector on X (unsupervised).
        2. Obtains anomaly scores.
        3. Appends anomaly scores as an extra feature.
        4. Trains the supervised classifier on the augmented feature set.
        """
        # Fit anomaly detector if it hasn't been fitted already
        if not self.anomaly_detector_fitted:
            self.anomaly_detector.fit(X)
            self.anomaly_detector_fitted = True
            
        # Extract anomaly scores
        anomaly_scores = self.anomaly_detector.predict_anomaly_score(X)
        
        # Append anomaly score to features (adds to the last column)
        X_augmented = np.column_stack([X, anomaly_scores])
        
        # Fit the supervised model
        self.classifier.fit(X_augmented, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts fraud probabilities.
        Returns a 2D array of shape (n_samples, 2) where the second column represents fraud probability.
        """
        anomaly_scores = self.anomaly_detector.predict_anomaly_score(X)
        X_augmented = np.column_stack([X, anomaly_scores])
        
        # Ensure the model is fitted
        if not hasattr(self.classifier, "classes_"):
            raise ValueError("The classifier model is not fitted yet.")
            
        return self.classifier.predict_proba(X_augmented)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """
        Predicts binary fraud labels based on probability threshold.
        Returns a 1D array of shape (n_samples,).
        """
        probs = self.predict_proba(X)[:, 1]
        return (probs >= threshold).astype(int)
