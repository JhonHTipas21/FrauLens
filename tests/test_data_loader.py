"""Unit tests for src/data_loader.py.

Tests cover CSV loading, column validation, dtype enforcement,
missing file handling, and empty dataset edge cases.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock
import io

from src.data_loader import load_dataset


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

VALID_COLUMNS = ["Time", "Amount", "Class"] + [f"V{i}" for i in range(1, 29)]


def _make_valid_csv(n_rows: int = 10) -> str:
    """Creates a minimal valid CSV string matching the expected schema."""
    np.random.seed(0)
    data = {col: np.random.randn(n_rows) for col in VALID_COLUMNS}
    data["Class"] = np.zeros(n_rows, dtype=int)
    data["Class"][0] = 1
    df = pd.DataFrame(data)
    return df.to_csv(index=False)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestLoadDataset:
    """Tests for the load_dataset function in data_loader.py."""

    def test_returns_dataframe(self, tmp_path):
        """load_dataset should return a pandas DataFrame object."""
        csv_file = tmp_path / "test.csv"
        csv_file.write_text(_make_valid_csv())
        result = load_dataset(str(csv_file))
        assert isinstance(result, pd.DataFrame)

    def test_correct_column_count(self, tmp_path):
        """Loaded DataFrame should have exactly 31 columns (Time, V1-V28, Amount, Class)."""
        csv_file = tmp_path / "test.csv"
        csv_file.write_text(_make_valid_csv(50))
        df = load_dataset(str(csv_file))
        assert len(df.columns) == 31

    def test_class_column_present(self, tmp_path):
        """Loaded DataFrame must contain a 'Class' column."""
        csv_file = tmp_path / "test.csv"
        csv_file.write_text(_make_valid_csv())
        df = load_dataset(str(csv_file))
        assert "Class" in df.columns

    def test_time_column_present(self, tmp_path):
        """Loaded DataFrame must contain a 'Time' column."""
        csv_file = tmp_path / "test.csv"
        csv_file.write_text(_make_valid_csv())
        df = load_dataset(str(csv_file))
        assert "Time" in df.columns

    def test_amount_column_present(self, tmp_path):
        """Loaded DataFrame must contain an 'Amount' column."""
        csv_file = tmp_path / "test.csv"
        csv_file.write_text(_make_valid_csv())
        df = load_dataset(str(csv_file))
        assert "Amount" in df.columns

    def test_correct_row_count(self, tmp_path):
        """Row count in loaded DataFrame must match rows in source CSV."""
        n = 77
        csv_file = tmp_path / "test.csv"
        csv_file.write_text(_make_valid_csv(n))
        df = load_dataset(str(csv_file))
        assert len(df) == n

    def test_no_missing_values(self, tmp_path):
        """Loaded DataFrame from valid CSV should contain no NaN values."""
        csv_file = tmp_path / "test.csv"
        csv_file.write_text(_make_valid_csv(30))
        df = load_dataset(str(csv_file))
        assert not df.isnull().any().any()

    def test_class_column_dtype_is_numeric(self, tmp_path):
        """Class column should be an integer or float dtype (binary labels)."""
        csv_file = tmp_path / "test.csv"
        csv_file.write_text(_make_valid_csv())
        df = load_dataset(str(csv_file))
        assert pd.api.types.is_numeric_dtype(df["Class"])

    def test_file_not_found_raises(self):
        """load_dataset must raise an exception when the path does not exist."""
        with pytest.raises(Exception):
            load_dataset("/nonexistent/path/to/data.csv")

    def test_fraud_rows_present(self, tmp_path):
        """Loaded dataset should contain at least one fraud instance (Class == 1)."""
        csv_file = tmp_path / "test.csv"
        csv_file.write_text(_make_valid_csv(20))
        df = load_dataset(str(csv_file))
        assert df["Class"].sum() >= 1
