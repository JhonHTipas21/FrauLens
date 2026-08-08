"""Unit tests for src/utils.py.

Tests cover the DataPreprocessor, temporal splitting logic, metric calculations,
and model serialization utilities.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path

from src.utils import (
    DataPreprocessor, 
    temporal_split, 
    calculate_metrics, 
    save_model_artifacts, 
    load_model_artifacts
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def dummy_data() -> pd.DataFrame:
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


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestTemporalSplit:
    """Tests for the temporal_split function."""

    def test_temporal_split_shapes(self, dummy_data):
        """Split should return DataFrames sized proportionally to the test_ratio."""
        train_df, test_df = temporal_split(dummy_data, time_col='Time', test_ratio=0.2)
        assert len(train_df) == 160
        assert len(test_df) == 40
        assert len(train_df) + len(test_df) == len(dummy_data)

    def test_chronological_order(self, dummy_data):
        """Train split must chronologically precede the test split."""
        train_df, test_df = temporal_split(dummy_data, time_col='Time', test_ratio=0.2)
        assert train_df['Time'].max() < test_df['Time'].min()


class TestDataPreprocessor:
    """Tests for the DataPreprocessor."""

    def test_preprocessor_output_shapes(self, dummy_data):
        """Transform must output properly shaped X matrices and y vectors."""
        preprocessor = DataPreprocessor()
        preprocessor.fit(dummy_data, target_col='Class')
        X, y = preprocessor.transform(dummy_data)
        
        # Features include Time, Amount, V1-V5 (7 columns total)
        assert X.shape == (200, 7)
        assert y.shape == (200,)
        
    def test_preprocessor_scaling(self, dummy_data):
        """StandardScaler should scale 'Time' and 'Amount' columns to zero mean and unit variance."""
        preprocessor = DataPreprocessor()
        preprocessor.fit(dummy_data, target_col='Class')
        X, _ = preprocessor.transform(dummy_data)
        
        # Determine index of Time and Amount in dummy dataset
        feature_cols = [c for c in dummy_data.columns if c != 'Class']
        time_idx = feature_cols.index('Time')
        amount_idx = feature_cols.index('Amount')
        
        assert np.isclose(X[:, time_idx].mean(), 0, atol=1e-7)
        assert np.isclose(X[:, time_idx].std(), 1, atol=1e-7)
        assert np.isclose(X[:, amount_idx].mean(), 0, atol=1e-7)
        assert np.isclose(X[:, amount_idx].std(), 1, atol=1e-7)

    def test_transform_without_target(self, dummy_data):
        """Transform should handle inference DataFrames lacking the target column."""
        preprocessor = DataPreprocessor()
        preprocessor.fit(dummy_data, target_col='Class')
        
        # Drop target column to simulate inference
        dummy_inference = dummy_data.drop(columns=['Class'])
        X, y = preprocessor.transform(dummy_inference)
        
        assert X.shape == (200, 7)
        assert y is None


class TestCalculateMetrics:
    """Tests for classification metric calculations."""

    def test_metrics_calculation_at_05(self):
        """Metrics should be calculated correctly based on threshold (TP, TN, FP, FN, etc)."""
        y_true = np.array([0, 0, 0, 1, 1])
        y_prob = np.array([0.1, 0.2, 0.6, 0.4, 0.9])
        
        # Threshold 0.5 -> y_pred = [0, 0, 1, 0, 1]
        # TP: idx 4, FP: idx 2, TN: idx 0, 1, FN: idx 3
        metrics = calculate_metrics(y_true, y_prob, threshold=0.5)
        
        assert metrics["precision"] == 0.5
        assert metrics["recall"] == 0.5
        assert metrics["f1"] == 0.5
        assert metrics["confusion_matrix"]["tp"] == 1
        assert metrics["confusion_matrix"]["fp"] == 1
        assert metrics["confusion_matrix"]["tn"] == 2
        assert metrics["confusion_matrix"]["fn"] == 1


class TestModelSerialization:
    """Tests for joblib-based serialization functions."""

    def test_save_and_load_artifacts(self, tmp_path):
        """Serialization functions should properly write and read back a dictionary payload."""
        artifacts = {"model_name": "test_model", "version": 1.0}
        filepath = tmp_path / "artifacts.joblib"
        
        save_model_artifacts(artifacts, filepath)
        assert filepath.exists()
        
        loaded = load_model_artifacts(filepath)
        assert loaded == artifacts

    def test_load_non_existent(self, tmp_path):
        """Loading a missing file should raise FileNotFoundError."""
        filepath = tmp_path / "does_not_exist.joblib"
        with pytest.raises(FileNotFoundError):
            load_model_artifacts(filepath)
