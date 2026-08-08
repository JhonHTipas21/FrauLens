import joblib
from pathlib import Path
from typing import Tuple, Dict, Any, Optional
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    precision_score, 
    recall_score, 
    f1_score, 
    precision_recall_curve, 
    auc, 
    confusion_matrix
)

def temporal_split(df: pd.DataFrame, time_col: str = 'Time', test_ratio: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Performs a temporal split on the dataset based on the time column.
    
    Args:
        df: Input DataFrame.
        time_col: Name of the time column.
        test_ratio: Ratio of data to use for testing (from the end of the timeline).
        
    Returns:
        A tuple (train_df, test_df) representing the earlier and later split respectively.
    """
    max_time = df[time_col].max()
    min_time = df[time_col].min()
    time_span = max_time - min_time
    
    threshold = min_time + time_span * (1.0 - test_ratio)
    
    train_df = df[df[time_col] < threshold].copy()
    test_df = df[df[time_col] >= threshold].copy()
    
    print(f"Temporal split threshold: {threshold:.1f} seconds.")
    print(f"Train set: {len(train_df)} records (Time range: {train_df[time_col].min()} - {train_df[time_col].max()})")
    print(f"Test set: {len(test_df)} records (Time range: {test_df[time_col].min()} - {test_df[time_col].max()})")
    
    return train_df, test_df

class DataPreprocessor:
    """Preprocess raw transaction data (StandardScaler for Amount and Time).
    
    Attributes:
        scaler: StandardScaler instance.
        feature_cols: List of column names used as input features.
        target_col: Name of the label column.
    """
    
    def __init__(self) -> None:
        """Initializes the DataPreprocessor."""
        self.scaler = StandardScaler()
        self.feature_cols: Optional[list] = None
        self.target_col: str = 'Class'

    def fit(self, df: pd.DataFrame, target_col: str = 'Class') -> 'DataPreprocessor':
        """Fits the preprocessor to the training dataframe.
        
        Args:
            df: Input training DataFrame.
            target_col: The target label column name.
            
        Returns:
            The fitted instance of DataPreprocessor.
        """
        self.target_col = target_col
        # Features are all columns except target
        self.feature_cols = [c for c in df.columns if c != target_col]
        
        # Fit scaler on 'Time' and 'Amount'
        cols_to_scale = [c for c in ['Time', 'Amount'] if c in df.columns]
        if cols_to_scale:
            self.scaler.fit(df[cols_to_scale])
            
        return self

    def transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Transforms the raw transaction DataFrame to features and labels.
        
        Args:
            df: Input DataFrame.
            
        Returns:
            A tuple (X, y) where X is a numpy array of scaled features and
            y is a numpy array of binary labels (or None if class target is missing).
        """
        df_scaled = df.copy()
        
        cols_to_scale = [c for c in ['Time', 'Amount'] if c in df_scaled.columns]
        if cols_to_scale:
            df_scaled[cols_to_scale] = self.scaler.transform(df_scaled[cols_to_scale])
            
        X = df_scaled[self.feature_cols].values
        
        y = None
        if self.target_col in df.columns:
            y = df[self.target_col].values
            
        return X, y

def calculate_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Dict[str, Any]:
    """Calculates key metrics for classification: Precision, Recall, F1-Score, and AUC-PR.
    
    Args:
        y_true: Ground truth labels.
        y_prob: Predicted positive class probabilities.
        threshold: Classification threshold.
        
    Returns:
        Dictionary of calculated metrics containing precision, recall, f1, auc_pr, and confusion_matrix details.
    """
    y_pred = (y_prob >= threshold).astype(int)
    
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    # Calculate Precision-Recall Curve and Area Under Curve (AUC-PR)
    precisions, recalls, thresholds_pr = precision_recall_curve(y_true, y_prob)
    auc_pr = auc(recalls, precisions)
    
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "auc_pr": float(auc_pr),
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp)
        },
        "threshold": float(threshold)
    }

def save_model_artifacts(artifacts: Dict[str, Any], filepath: Path) -> None:
    """Serializes and saves model pipeline components to a file.
    
    Args:
        artifacts: Dictionary containing fitted model components.
        filepath: Local file path where artifacts should be written.
    """
    filepath.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifacts, filepath)
    print(f"Model artifacts saved successfully at {filepath}")

def load_model_artifacts(filepath: Path) -> Dict[str, Any]:
    """Loads serialized model pipeline components from a file.
    
    Args:
        filepath: Local file path to read model artifacts from.
        
    Returns:
        Dictionary of loaded model artifacts.
        
    Raises:
        FileNotFoundError: If the artifact file does not exist.
    """
    if not filepath.exists():
        raise FileNotFoundError(f"Model artifacts not found at {filepath}")
    return joblib.load(filepath)
