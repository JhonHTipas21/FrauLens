from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import numpy as np

@dataclass
class AnomalyScore:
    """Standardized representation of anomaly detection outputs.
    
    Attributes:
        score: Continuous score where higher means more anomalous.
        is_anomaly: Binary indicator of anomaly status.
        metadata: Optional dictionary with supplementary metadata.
    """
    score: float
    is_anomaly: bool
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class FeatureExplanation:
    """SHAP or LIME-equivalent explanation for a single feature.
    
    Attributes:
        feature_name: Name of the feature.
        value: Original feature value before scaling.
        contribution: Impact or contribution value (e.g. Shapley value).
    """
    feature_name: str
    value: float
    contribution: float

@dataclass
class Explanation:
    """Unified container for individual prediction explanations.
    
    This container is independent of the underlying explainer engine, enforcing DIP.
    
    Attributes:
        base_value: The expected value or reference prediction.
        prediction_score: Final predicted score (e.g., probability).
        features: List of individual feature attributions.
    """
    base_value: float
    prediction_score: float
    features: List[FeatureExplanation]

class BaseAnomalyDetector(ABC):
    """Abstract interface for all anomaly detectors (Liskov Substitution Principle)."""
    
    @abstractmethod
    def fit(self, X: np.ndarray) -> 'BaseAnomalyDetector':
        """Fits the anomaly detector on normal data.
        
        Args:
            X: Input features of shape (n_samples, n_features).
            
        Returns:
            The fitted instance of the detector.
        """
        pass
        
    @abstractmethod
    def predict_anomaly_score(self, X: np.ndarray) -> np.ndarray:
        """Computes continuous anomaly score.
        
        Args:
            X: Input features of shape (n_samples, n_features).
            
        Returns:
            1D array of shape (n_samples,) with continuous scores. Higher values mean more anomalous.
        """
        pass
        
    @abstractmethod
    def predict_anomaly(self, X: np.ndarray) -> np.ndarray:
        """Predicts binary anomaly labels.
        
        Args:
            X: Input features of shape (n_samples, n_features).
            
        Returns:
            1D array of shape (n_samples,) containing binary labels (1: anomaly, 0: normal).
        """
        pass

class BaseExplainer(ABC):
    """Abstract interface for explainability engines (Dependency Inversion Principle)."""
    
    @abstractmethod
    def explain_instance(self, X_instance: np.ndarray, feature_names: List[str]) -> Explanation:
        """Explains a single transaction prediction.
        
        Args:
            X_instance: Input features of shape (n_features,) or (1, n_features).
            feature_names: List of original feature names.
            
        Returns:
            Explanation object containing attributions.
        """
        pass
        
    @abstractmethod
    def explain_global(self, X_samples: np.ndarray, feature_names: List[str]) -> Dict[str, float]:
        """Calculates global feature importances.
        
        Args:
            X_samples: Representative background transactions of shape (n_samples, n_features).
            feature_names: List of original feature names.
            
        Returns:
            Dictionary mapping feature names to their global importance score.
        """
        pass
