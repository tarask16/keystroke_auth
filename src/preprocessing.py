"""Preprocessing utilities for Keystroke Auth.

This module prepares processed CMU keystroke features for machine learning.

Current step:
- load data/processed/cmu_features.csv;
- validate required metadata columns;
- detect feature columns;
- split dataset into:
  - X: numeric feature matrix;
  - y: user labels;
  - metadata: service columns.

Later steps will add:
- data cleaning;
- train/validation/test split;
- StandardScaler normalization;
- scaler saving;
- split metadata saving.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.config import CMU_FEATURES_FILE
from src.data_loader import PROCESSED_META_COLUMNS, get_processed_feature_columns


@dataclass(frozen=True)
class PreparedDataset:
    """Container for prepared dataset parts."""

    X: pd.DataFrame
    y: pd.Series
    metadata: pd.DataFrame


def load_processed_dataset(path: Path) -> pd.DataFrame:
    """Load processed CMU features dataset.

    Args:
        path: Path to data/processed/cmu_features.csv.

    Returns:
        Loaded DataFrame.

    Raises:
        FileNotFoundError: If file does not exist.
        ValueError: If file is empty or invalid.
    """
    path = path.resolve()

    if not path.exists():
        raise FileNotFoundError(f"Processed dataset not found: {path}")

    if not path.is_file():
        raise ValueError(f"Processed dataset path is not a file: {path}")

    df = pd.read_csv(path)

    if df.empty:
        raise ValueError(f"Processed dataset is empty: {path}")

    validate_processed_dataset(df)

    return df


def validate_processed_dataset(df: pd.DataFrame) -> None:
    """Validate processed dataset structure.

    Args:
        df: Processed DataFrame.

    Raises:
        ValueError: If required metadata or feature columns are missing.
    """
    missing_meta_columns = [column for column in PROCESSED_META_COLUMNS if column not in df.columns]

    if missing_meta_columns:
        raise ValueError(f"Missing processed metadata columns: {missing_meta_columns}")

    feature_columns = get_processed_feature_columns(df)

    if not feature_columns:
        raise ValueError("No feature columns found in processed dataset.")

    non_numeric_features = [
        column for column in feature_columns if not pd.api.types.is_numeric_dtype(df[column])
    ]

    if non_numeric_features:
        raise ValueError(f"Non-numeric feature columns found: {non_numeric_features}")


def prepare_features_and_labels(df: pd.DataFrame) -> PreparedDataset:
    """Split processed DataFrame into X, y and metadata.

    Args:
        df: Processed CMU DataFrame.

    Returns:
        PreparedDataset with feature matrix, labels and metadata.

    Raises:
        ValueError: If dataset is invalid or contains inconsistent indexes.
    """
    validate_processed_dataset(df)

    feature_columns = get_processed_feature_columns(df)

    metadata = df.loc[:, PROCESSED_META_COLUMNS].copy()
    X = df.loc[:, feature_columns].copy()
    y = df["user_id"].copy()

    if len(X) != len(y) or len(X) != len(metadata):
        raise ValueError(
            f"Inconsistent dataset lengths: X={len(X)}, y={len(y)}, metadata={len(metadata)}"
        )

    if y.nunique() < 2:
        raise ValueError("At least two users are required for classification.")

    return PreparedDataset(X=X, y=y, metadata=metadata)


def summarize_processed_dataset(df: pd.DataFrame) -> dict[str, int]:
    """Create a short summary for processed dataset.

    Args:
        df: Processed DataFrame.

    Returns:
        Dataset summary dictionary.
    """
    feature_columns = get_processed_feature_columns(df)

    return {
        "rows": int(len(df)),
        "users": int(df["user_id"].nunique()),
        "features": int(len(feature_columns)),
        "missing_values": int(df.isna().sum().sum()),
    }


def summarize_prepared_dataset(prepared: PreparedDataset) -> dict[str, int]:
    """Create summary for X, y and metadata.

    Args:
        prepared: Prepared dataset container.

    Returns:
        Summary dictionary.
    """
    return {
        "x_rows": int(prepared.X.shape[0]),
        "x_features": int(prepared.X.shape[1]),
        "y_rows": int(prepared.y.shape[0]),
        "metadata_rows": int(prepared.metadata.shape[0]),
        "users": int(prepared.y.nunique()),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    """Build command-line argument parser.

    Returns:
        Configured ArgumentParser.
    """
    parser = argparse.ArgumentParser(description="Preprocess CMU keystroke features.")

    parser.add_argument(
        "--input",
        type=Path,
        default=CMU_FEATURES_FILE,
        help="Path to processed CMU features CSV.",
    )

    return parser


def main() -> None:
    """CLI entry point."""
    parser = build_arg_parser()
    args = parser.parse_args()

    df = load_processed_dataset(args.input)
    dataset_summary = summarize_processed_dataset(df)

    prepared = prepare_features_and_labels(df)
    prepared_summary = summarize_prepared_dataset(prepared)

    print("Processed dataset loaded successfully.")
    print(f"Input: {args.input}")
    print(f"Rows: {dataset_summary['rows']}")
    print(f"Users: {dataset_summary['users']}")
    print(f"Features: {dataset_summary['features']}")
    print(f"Missing values: {dataset_summary['missing_values']}")

    print()
    print("Prepared dataset:")
    print(f"X shape: {prepared.X.shape}")
    print(f"y shape: {prepared.y.shape}")
    print(f"metadata shape: {prepared.metadata.shape}")
    print(f"Users in y: {prepared_summary['users']}")
    print(f"First 5 labels: {prepared.y.head().to_list()}")
    print(f"First 5 feature columns: {prepared.X.columns[:5].to_list()}")


if __name__ == "__main__":
    main()
