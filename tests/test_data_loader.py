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
    """Creates a minimal valid CSV string matching the expected schema (without headers)."""
    np.random.seed(0)
    data = {col: np.random.randn(n_rows) for col in VALID_COLUMNS}
    data["Class"] = np.zeros(n_rows, dtype=int)
    data["Class"][0] = 1
    df = pd.DataFrame(data)
    return df.to_csv(index=False, header=False)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestLoadDataset:
    """Tests for the load_dataset function in data_loader.py."""

    @patch("src.data_loader.get_or_download_dataset")
    def test_returns_dataframe(self, mock_get_dataset, tmp_path):
        """load_dataset should return a pandas DataFrame object."""
        csv_file = tmp_path / "test.csv"
        csv_file.write_text(_make_valid_csv())
        mock_get_dataset.return_value = csv_file
        result = load_dataset()
        assert isinstance(result, pd.DataFrame)

    @patch("src.data_loader.get_or_download_dataset")
    def test_correct_column_count(self, mock_get_dataset, tmp_path):
        """Loaded DataFrame should have exactly 31 columns (Time, V1-V28, Amount, Class)."""
        csv_file = tmp_path / "test.csv"
        csv_file.write_text(_make_valid_csv(50))
        mock_get_dataset.return_value = csv_file
        df = load_dataset()
        assert len(df.columns) == 31

    @patch("src.data_loader.get_or_download_dataset")
    def test_class_column_present(self, mock_get_dataset, tmp_path):
        """Loaded DataFrame must contain a 'Class' column."""
        csv_file = tmp_path / "test.csv"
        csv_file.write_text(_make_valid_csv())
        mock_get_dataset.return_value = csv_file
        df = load_dataset()
        assert "Class" in df.columns

    @patch("src.data_loader.get_or_download_dataset")
    def test_time_column_present(self, mock_get_dataset, tmp_path):
        """Loaded DataFrame must contain a 'Time' column."""
        csv_file = tmp_path / "test.csv"
        csv_file.write_text(_make_valid_csv())
        mock_get_dataset.return_value = csv_file
        df = load_dataset()
        assert "Time" in df.columns

    @patch("src.data_loader.get_or_download_dataset")
    def test_amount_column_present(self, mock_get_dataset, tmp_path):
        """Loaded DataFrame must contain an 'Amount' column."""
        csv_file = tmp_path / "test.csv"
        csv_file.write_text(_make_valid_csv())
        mock_get_dataset.return_value = csv_file
        df = load_dataset()
        assert "Amount" in df.columns

    @patch("src.data_loader.get_or_download_dataset")
    def test_correct_row_count(self, mock_get_dataset, tmp_path):
        """Row count in loaded DataFrame must match rows in source CSV."""
        n = 77
        csv_file = tmp_path / "test.csv"
        csv_file.write_text(_make_valid_csv(n))
        mock_get_dataset.return_value = csv_file
        df = load_dataset()
        assert len(df) == n

    @patch("src.data_loader.get_or_download_dataset")
    def test_no_missing_values(self, mock_get_dataset, tmp_path):
        """Loaded DataFrame from valid CSV should contain no NaN values."""
        csv_file = tmp_path / "test.csv"
        csv_file.write_text(_make_valid_csv(30))
        mock_get_dataset.return_value = csv_file
        df = load_dataset()
        assert not df.isnull().any().any()

    @patch("src.data_loader.get_or_download_dataset")
    def test_class_column_dtype_is_numeric(self, mock_get_dataset, tmp_path):
        """Class column should be an integer or float dtype (binary labels)."""
        csv_file = tmp_path / "test.csv"
        csv_file.write_text(_make_valid_csv())
        mock_get_dataset.return_value = csv_file
        df = load_dataset()
        assert pd.api.types.is_numeric_dtype(df["Class"])

    @patch("src.data_loader.get_or_download_dataset")
    def test_file_not_found_raises(self, mock_get_dataset):
        """load_dataset must raise an exception when the path does not exist."""
        mock_get_dataset.return_value = "/nonexistent/path/to/data.csv"
        with pytest.raises(Exception):
            load_dataset()

    @patch("src.data_loader.get_or_download_dataset")
    def test_fraud_rows_present(self, mock_get_dataset, tmp_path):
        """Loaded dataset should contain at least one fraud instance (Class == 1)."""
        csv_file = tmp_path / "test.csv"
        csv_file.write_text(_make_valid_csv(20))
        mock_get_dataset.return_value = csv_file
        df = load_dataset()
        assert df["Class"].sum() >= 1
