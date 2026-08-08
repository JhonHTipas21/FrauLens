import numpy as np
import xgboost as xgb
from typing import Any, Optional
from .interfaces import BaseAnomalyDetector

class FraudClassifier:
    """Hybrid classifier combining unsupervised anomaly detection with supervised classification.
    
    This classifier implements the Open/Closed Principle (OCP) by accepting
    any anomaly detector matching the BaseAnomalyDetector interface and any standard
    scikit-learn compatible classifier.
    
    Attributes:
        anomaly_detector: A detector that implements BaseAnomalyDetector interface.
        classifier: Underlying supervised classification model (XGBoost or other).
        anomaly_detector_fitted: Boolean flag tracking the fitting state of the detector.
    """
    
    def __init__(self, anomaly_detector: BaseAnomalyDetector, classifier_model: Optional[Any] = None) -> None:
        """Initializes the FraudClassifier.
        
        Args:
            anomaly_detector: An instance implementing BaseAnomalyDetector interface.
            classifier_model: An optional scikit-learn compatible classifier.
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
        """Fits the hybrid detection and classification pipeline.
        
        This method first fits the unsupervised anomaly detector on the raw features,
        computes continuous anomaly scores, appends them to the feature space, and
        finally trains the supervised classifier.
        
        Args:
            X: Raw dataset features of shape (n_samples, n_features).
            y: Binary target labels of shape (n_samples,).
            
        Returns:
            The fitted instance of the FraudClassifier.
        """
        # Fit anomaly detector if it hasn't been fitted already
        if not self.anomaly_detector_fitted:
            self.anomaly_detector.fit(X)
            self.anomaly_detector_fitted = True
            
        # Extract anomaly scores
        anomaly_scores = self.anomaly_detector.predict_anomaly_score(X)
        
        # Append anomaly score to features (adds as the last column)
        X_augmented = np.column_stack([X, anomaly_scores])
        
        # Fit the supervised model
        self.classifier.fit(X_augmented, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predicts fraud probability class distribution.
        
        Args:
            X: Raw dataset features of shape (n_samples, n_features).
            
        Returns:
            2D array of shape (n_samples, 2) where the second column represents the fraud probability.
            
        Raises:
            ValueError: If the underlying classifier model is not fitted yet.
        """
        anomaly_scores = self.anomaly_detector.predict_anomaly_score(X)
        X_augmented = np.column_stack([X, anomaly_scores])
        
        # Ensure the model is fitted
        if not hasattr(self.classifier, "classes_"):
            raise ValueError("The classifier model is not fitted yet.")
            
        return self.classifier.predict_proba(X_augmented)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Predicts binary fraud labels based on probability threshold.
        
        Args:
            X: Raw dataset features of shape (n_samples, n_features).
            threshold: Probability threshold above which a transaction is flagged as fraud.
            
        Returns:
            1D array of shape (n_samples,) containing binary classification labels (0 or 1).
        """
        probs = self.predict_proba(X)[:, 1]
        return (probs >= threshold).astype(int)
