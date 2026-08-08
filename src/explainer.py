import numpy as np
import shap
from typing import List, Dict, Any
from .interfaces import BaseExplainer, Explanation, FeatureExplanation
from .classifier import FraudClassifier

class SHAPExplainer(BaseExplainer):
    """
    SHAP TreeExplainer implementation of BaseExplainer.
    Provides explanations for individual predictions and global feature importances.
    """
    
    def __init__(self, fraud_classifier: FraudClassifier):
        """
        Initializes the SHAPExplainer.
        
        Args:
            fraud_classifier: A fitted FraudClassifier instance.
        """
        self.fraud_classifier = fraud_classifier
        # Initialize TreeExplainer on the underlying supervised tree-based model
        self.explainer = shap.TreeExplainer(self.fraud_classifier.classifier)

    def explain_instance(self, X_instance: np.ndarray, feature_names: List[str]) -> Explanation:
        """
        Explains a single transaction prediction.
        
        Args:
            X_instance: 1D or 2D array of shape (n_features,) or (1, n_features).
            feature_names: List of strings containing original feature names.
            
        Returns:
            Explanation object containing base value, prediction score, and feature-level attributions.
        """
        if len(X_instance.shape) == 1:
            X_instance = X_instance.reshape(1, -1)
            
        # Get the anomaly score from the detector to augment the instance features
        anomaly_score = self.fraud_classifier.anomaly_detector.predict_anomaly_score(X_instance)
        X_augmented = np.column_stack([X_instance, anomaly_score])
        
        # Calculate SHAP values
        shap_vals = self.explainer.shap_values(X_augmented)
        
        # Handle different output structures of shap_values
        # (e.g., binary classifications can return a list [class0_shap, class1_shap] or single array)
        if isinstance(shap_vals, list):
            # Take SHAP values for the positive class (class 1: fraud)
            shap_vals = shap_vals[1]
        elif len(shap_vals.shape) == 3:
            # Multi-class or binary classification returned as 3D array (samples, features, classes)
            shap_vals = shap_vals[:, :, 1]
            
        # expected_value represents the base prediction before feature contributions (log-odds)
        base_val = self.explainer.expected_value
        if isinstance(base_val, (list, np.ndarray)):
            base_val = base_val[1] if len(base_val) > 1 else base_val[0]
            
        # Original features + anomaly_score
        full_feature_names = feature_names + ["anomaly_score"]
        feat_values = X_augmented[0]
        instance_shap = shap_vals[0]
        
        # Build feature explanations list
        feat_explanations = []
        for i, name in enumerate(full_feature_names):
            feat_explanations.append(FeatureExplanation(
                feature_name=name,
                value=float(feat_values[i]),
                contribution=float(instance_shap[i])
            ))
            
        # Get the model prediction probability for the positive class
        pred_score = self.fraud_classifier.predict_proba(X_instance)[0, 1]
        
        return Explanation(
            base_value=float(base_val),
            prediction_score=float(pred_score),
            features=feat_explanations
        )

    def explain_global(self, X_samples: np.ndarray, feature_names: List[str]) -> Dict[str, float]:
        """
        Calculates global feature importances as the mean absolute SHAP value for each feature.
        
        Args:
            X_samples: 2D array of representative background transactions.
            feature_names: List of strings containing original feature names.
            
        Returns:
            Dictionary mapping feature names to their global importance score.
        """
        # Augment features with anomaly score
        anomaly_scores = self.fraud_classifier.anomaly_detector.predict_anomaly_score(X_samples)
        X_augmented = np.column_stack([X_samples, anomaly_scores])
        
        # Calculate SHAP values for all samples
        shap_vals = self.explainer.shap_values(X_augmented)
        
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[1]
        elif len(shap_vals.shape) == 3:
            shap_vals = shap_vals[:, :, 1]
            
        # Mean absolute SHAP values across samples
        mean_abs_shap = np.mean(np.abs(shap_vals), axis=0)
        
        full_feature_names = feature_names + ["anomaly_score"]
        
        # Map feature names to their global importance
        importance_dict = {
            full_feature_names[i]: float(mean_abs_shap[i])
            for i in range(len(full_feature_names))
        }
        
        # Sort by importance descending
        return dict(sorted(importance_dict.items(), key=lambda item: item[1], reverse=True))
