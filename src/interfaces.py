from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Any
import numpy as np

@dataclass
class AnomalyScore:
    """Standardized representation of anomaly detection outputs."""
    score: float
    is_anomaly: bool
    metadata: Dict[str, Any] = None

@dataclass
class FeatureExplanation:
    """SHAP or LIME-equivalent explanation for a single feature."""
    feature_name: str
    value: float
    contribution: float  # Feature impact/Shapley value

@dataclass
class Explanation:
    """Unified container for individual prediction explanations, independent of explainer engine."""
    base_value: float
    prediction_score: float
    features: List[FeatureExplanation]

class BaseAnomalyDetector(ABC):
    """Abstract interface for all anomaly detectors (Liskov Substitution Principle)."""
    
    @abstractmethod
    def fit(self, X: np.ndarray) -> 'BaseAnomalyDetector':
        """Fits the anomaly detector on normal data."""
        pass
        
    @abstractmethod
    def predict_anomaly_score(self, X: np.ndarray) -> np.ndarray:
        """Computes continuous anomaly score (higher value = more anomalous)."""
        pass
        
    @abstractmethod
    def predict_anomaly(self, X: np.ndarray) -> np.ndarray:
        """Predicts binary anomaly labels (1: anomaly, 0: normal)."""
        pass

class BaseExplainer(ABC):
    """Abstract interface for explainability engines (SHAP, LIME, etc.) (Dependency Inversion Principle)."""
    
    @abstractmethod
    def explain_instance(self, X_instance: np.ndarray, feature_names: List[str]) -> Explanation:
        """Explains a single transaction prediction."""
        pass
        
    @abstractmethod
    def explain_global(self, X_samples: np.ndarray, feature_names: List[str]) -> Dict[str, float]:
        """Calculates global feature importances (e.g., mean absolute SHAP values)."""
        pass
