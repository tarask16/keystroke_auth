"""Preprocessing utilities for Keystroke Auth.

This module prepares processed CMU keystroke features for machine learning.

Current step:
- load data/processed/cmu_features.csv;
- validate required metadata columns;
- detect feature columns;
- split dataset into X, y and metadata;
- clean invalid values and simple outliers;
- create stratified train/validation/test split;
- scale features with StandardScaler fitted on train only;
- save fitted scaler to models/scaler.pkl;
- save split metadata to data/processed/cmu_split.json.

Important CMU detail:
UD.* features may be negative. This is normal for overlapping keystrokes:
the next key can be pressed before the previous key is released.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.config import CMU_FEATURES_FILE, RANDOM_SEED
from src.data_loader import PROCESSED_META_COLUMNS, get_processed_feature_columns


OUTLIER_LOWER_QUANTILE = 0.01
OUTLIER_UPPER_QUANTILE = 0.99
DEFAULT_TEST_SIZE = 0.2
DEFAULT_VALIDATION_SIZE = 0.2
DEFAULT_SCALER_OUTPUT = Path(__file__).resolve().parents[1] / "models" / "scaler.pkl"
DEFAULT_SPLIT_OUTPUT = Path(__file__).resolve().parents[1] / "data" / "processed" / "cmu_split.json"


@dataclass(frozen=True)
class PreparedDataset:
    """Container for prepared dataset parts."""

    X: pd.DataFrame
    y: pd.Series
    metadata: pd.DataFrame


@dataclass(frozen=True)
class CleaningReport:
    """Short report for dataset cleaning."""

    rows_before: int
    rows_after: int
    rows_removed: int
    missing_values_before: int
    missing_values_after: int
    infinite_values_before: int
    invalid_negative_rows_removed: int
    valid_negative_ud_values_kept: int
    clipped_values_count: int


@dataclass(frozen=True)
class TrainTestSplit:
    """Container for train/test split."""

    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    metadata_train: pd.DataFrame
    metadata_test: pd.DataFrame


@dataclass(frozen=True)
class TrainValidationTestSplit:
    """Container for train/validation/test split."""

    X_train: pd.DataFrame
    X_validation: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_validation: pd.Series
    y_test: pd.Series
    metadata_train: pd.DataFrame
    metadata_validation: pd.DataFrame
    metadata_test: pd.DataFrame


@dataclass(frozen=True)
class ScaledTrainValidationTestSplit:
    """Container for scaled train/validation/test split."""

    X_train: pd.DataFrame
    X_validation: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_validation: pd.Series
    y_test: pd.Series
    metadata_train: pd.DataFrame
    metadata_validation: pd.DataFrame
    metadata_test: pd.DataFrame
    scaler: StandardScaler


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


def clean_prepared_dataset(prepared: PreparedDataset) -> tuple[PreparedDataset, CleaningReport]:
    """Clean prepared dataset.

    Cleaning rules:
    - replace inf and -inf with NaN;
    - remove rows with NaN values;
    - remove rows with invalid negative H.* or DD.* values;
    - keep negative UD.* values because they are valid in CMU;
    - clip feature values by 1% and 99% quantiles.

    Args:
        prepared: Prepared dataset.

    Returns:
        Tuple with cleaned PreparedDataset and CleaningReport.

    Raises:
        ValueError: If all rows are removed during cleaning.
    """
    X = prepared.X.copy()
    y = prepared.y.copy()
    metadata = prepared.metadata.copy()

    rows_before = len(X)
    missing_values_before = int(X.isna().sum().sum())
    infinite_values_before = count_infinite_values(X)

    X = X.replace([np.inf, -np.inf], np.nan)

    valid_not_missing_mask = ~X.isna().any(axis=1)

    X = X.loc[valid_not_missing_mask].copy()
    y = y.loc[valid_not_missing_mask].copy()
    metadata = metadata.loc[valid_not_missing_mask].copy()

    invalid_negative_mask = build_invalid_negative_mask(X)
    invalid_negative_rows_removed = int(invalid_negative_mask.sum())

    valid_non_negative_mask = ~invalid_negative_mask

    X = X.loc[valid_non_negative_mask].copy()
    y = y.loc[valid_non_negative_mask].copy()
    metadata = metadata.loc[valid_non_negative_mask].copy()

    if X.empty:
        raise ValueError("All rows were removed during dataset cleaning.")

    valid_negative_ud_values_kept = count_negative_ud_values(X)

    X_clipped, clipped_values_count = clip_feature_outliers(X)

    X_clipped = X_clipped.reset_index(drop=True)
    y = y.reset_index(drop=True)
    metadata = metadata.reset_index(drop=True)

    cleaned = PreparedDataset(X=X_clipped, y=y, metadata=metadata)

    validate_prepared_dataset_consistency(cleaned)

    missing_values_after = int(cleaned.X.isna().sum().sum())
    rows_after = len(cleaned.X)

    report = CleaningReport(
        rows_before=rows_before,
        rows_after=rows_after,
        rows_removed=rows_before - rows_after,
        missing_values_before=missing_values_before,
        missing_values_after=missing_values_after,
        infinite_values_before=infinite_values_before,
        invalid_negative_rows_removed=invalid_negative_rows_removed,
        valid_negative_ud_values_kept=valid_negative_ud_values_kept,
        clipped_values_count=clipped_values_count,
    )

    return cleaned, report


def count_infinite_values(X: pd.DataFrame) -> int:
    """Count infinite values in feature matrix.

    Args:
        X: Feature matrix.

    Returns:
        Number of inf and -inf values.
    """
    values = X.to_numpy()
    return int(np.isinf(values).sum())


def build_invalid_negative_mask(X: pd.DataFrame) -> pd.Series:
    """Build row mask for invalid negative feature values.

    Negative UD.* values are valid for keystroke dynamics, because the next
    key can be pressed before the previous one is released. Negative H.* and
    DD.* values are treated as invalid for this MVP.

    Args:
        X: Feature matrix.

    Returns:
        Boolean Series where True means row should be removed.
    """
    mask = pd.Series(False, index=X.index)

    h_columns = [column for column in X.columns if column.startswith("H.")]
    dd_columns = [column for column in X.columns if column.startswith("DD.")]

    if h_columns:
        mask = mask | (X.loc[:, h_columns] < 0).any(axis=1)

    if dd_columns:
        mask = mask | (X.loc[:, dd_columns] < 0).any(axis=1)

    return mask


def count_negative_ud_values(X: pd.DataFrame) -> int:
    """Count valid negative UD.* values kept in the dataset.

    Args:
        X: Feature matrix.

    Returns:
        Number of negative UD.* values.
    """
    ud_columns = [column for column in X.columns if column.startswith("UD.")]

    if not ud_columns:
        return 0

    return int((X.loc[:, ud_columns] < 0).sum().sum())


def clip_feature_outliers(X: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Clip feature outliers by fixed quantile limits.

    Args:
        X: Feature matrix.

    Returns:
        Tuple with clipped feature matrix and number of changed values.
    """
    lower_bounds = X.quantile(OUTLIER_LOWER_QUANTILE)
    upper_bounds = X.quantile(OUTLIER_UPPER_QUANTILE)

    clipped = X.clip(lower=lower_bounds, upper=upper_bounds, axis=1)

    changed_values_count = int((clipped != X).sum().sum())

    return clipped, changed_values_count


def create_train_test_split(
    prepared: PreparedDataset,
    test_size: float = DEFAULT_TEST_SIZE,
    random_state: int = RANDOM_SEED,
) -> TrainTestSplit:
    """Create stratified train/test split.

    Args:
        prepared: Cleaned prepared dataset.
        test_size: Fraction of samples assigned to test split.
        random_state: Random seed for reproducibility.

    Returns:
        TrainTestSplit container.

    Raises:
        ValueError: If test_size is invalid or split is inconsistent.
    """
    validate_prepared_dataset_consistency(prepared)

    if not 0 < test_size < 1:
        raise ValueError(f"test_size must be between 0 and 1, got: {test_size}")

    train_index, test_index = train_test_split(
        prepared.X.index,
        test_size=test_size,
        random_state=random_state,
        stratify=prepared.y,
    )

    X_train = prepared.X.loc[train_index].reset_index(drop=True)
    X_test = prepared.X.loc[test_index].reset_index(drop=True)

    y_train = prepared.y.loc[train_index].reset_index(drop=True)
    y_test = prepared.y.loc[test_index].reset_index(drop=True)

    metadata_train = prepared.metadata.loc[train_index].reset_index(drop=True)
    metadata_test = prepared.metadata.loc[test_index].reset_index(drop=True)

    split = TrainTestSplit(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        metadata_train=metadata_train,
        metadata_test=metadata_test,
    )

    validate_train_test_split(split)

    return split


def create_train_validation_test_split(
    prepared: PreparedDataset,
    test_size: float = DEFAULT_TEST_SIZE,
    validation_size: float = DEFAULT_VALIDATION_SIZE,
    random_state: int = RANDOM_SEED,
) -> TrainValidationTestSplit:
    """Create stratified train/validation/test split.

    The dataset is first split into train_full/test. Then train_full is split
    into train/validation. With default values, the final distribution is:
    - train: 64%
    - validation: 16%
    - test: 20%

    Args:
        prepared: Cleaned prepared dataset.
        test_size: Fraction of all samples assigned to test split.
        validation_size: Fraction of train_full assigned to validation split.
        random_state: Random seed for reproducibility.

    Returns:
        TrainValidationTestSplit container.

    Raises:
        ValueError: If split parameters are invalid or inconsistent.
    """
    if not 0 < validation_size < 1:
        raise ValueError(f"validation_size must be between 0 and 1, got: {validation_size}")

    train_test = create_train_test_split(
        prepared=prepared,
        test_size=test_size,
        random_state=random_state,
    )

    train_index, validation_index = train_test_split(
        train_test.X_train.index,
        test_size=validation_size,
        random_state=random_state,
        stratify=train_test.y_train,
    )

    split = TrainValidationTestSplit(
        X_train=train_test.X_train.loc[train_index].reset_index(drop=True),
        X_validation=train_test.X_train.loc[validation_index].reset_index(drop=True),
        X_test=train_test.X_test,
        y_train=train_test.y_train.loc[train_index].reset_index(drop=True),
        y_validation=train_test.y_train.loc[validation_index].reset_index(drop=True),
        y_test=train_test.y_test,
        metadata_train=train_test.metadata_train.loc[train_index].reset_index(drop=True),
        metadata_validation=train_test.metadata_train.loc[validation_index].reset_index(drop=True),
        metadata_test=train_test.metadata_test,
    )

    validate_train_validation_test_split(split)

    return split


def validate_train_test_split(split: TrainTestSplit) -> None:
    """Validate train/test split consistency.

    Args:
        split: Train/test split container.

    Raises:
        ValueError: If split lengths are inconsistent or user classes are missing.
    """
    if len(split.X_train) != len(split.y_train):
        raise ValueError("Train split has inconsistent X/y lengths.")

    if len(split.X_train) != len(split.metadata_train):
        raise ValueError("Train split has inconsistent X/metadata lengths.")

    if len(split.X_test) != len(split.y_test):
        raise ValueError("Test split has inconsistent X/y lengths.")

    if len(split.X_test) != len(split.metadata_test):
        raise ValueError("Test split has inconsistent X/metadata lengths.")

    train_users = set(split.y_train.unique())
    test_users = set(split.y_test.unique())

    if train_users != test_users:
        missing_in_train = sorted(test_users - train_users)
        missing_in_test = sorted(train_users - test_users)
        raise ValueError(
            "Train/test split lost user classes. "
            f"Missing in train: {missing_in_train}. "
            f"Missing in test: {missing_in_test}."
        )


def validate_train_validation_test_split(split: TrainValidationTestSplit) -> None:
    """Validate train/validation/test split consistency.

    Args:
        split: Train/validation/test split container.

    Raises:
        ValueError: If split lengths are inconsistent or user classes are missing.
    """
    validate_split_part_lengths(split.X_train, split.y_train, split.metadata_train, "train")
    validate_split_part_lengths(
        split.X_validation,
        split.y_validation,
        split.metadata_validation,
        "validation",
    )
    validate_split_part_lengths(split.X_test, split.y_test, split.metadata_test, "test")

    train_users = set(split.y_train.unique())
    validation_users = set(split.y_validation.unique())
    test_users = set(split.y_test.unique())

    if train_users != validation_users or train_users != test_users:
        missing_in_train = sorted((validation_users | test_users) - train_users)
        missing_in_validation = sorted((train_users | test_users) - validation_users)
        missing_in_test = sorted((train_users | validation_users) - test_users)
        raise ValueError(
            "Train/validation/test split lost user classes. "
            f"Missing in train: {missing_in_train}. "
            f"Missing in validation: {missing_in_validation}. "
            f"Missing in test: {missing_in_test}."
        )


def validate_split_part_lengths(
    X: pd.DataFrame,
    y: pd.Series,
    metadata: pd.DataFrame,
    split_name: str,
) -> None:
    """Validate X/y/metadata lengths for one split part.

    Args:
        X: Feature matrix.
        y: User labels.
        metadata: Service metadata.
        split_name: Human-readable split name.

    Raises:
        ValueError: If lengths are inconsistent.
    """
    if len(X) != len(y):
        raise ValueError(f"{split_name} split has inconsistent X/y lengths.")

    if len(X) != len(metadata):
        raise ValueError(f"{split_name} split has inconsistent X/metadata lengths.")


def scale_train_validation_test_split(
    split: TrainValidationTestSplit,
) -> ScaledTrainValidationTestSplit:
    """Scale train/validation/test features using StandardScaler.

    The scaler is fitted only on X_train to avoid data leakage.
    Validation and test data are transformed with the already fitted scaler.

    Args:
        split: Train/validation/test split container.

    Returns:
        Scaled split container and fitted scaler.
    """
    scaler = StandardScaler()

    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(split.X_train),
        columns=split.X_train.columns,
    )
    X_validation_scaled = pd.DataFrame(
        scaler.transform(split.X_validation),
        columns=split.X_validation.columns,
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(split.X_test),
        columns=split.X_test.columns,
    )

    scaled = ScaledTrainValidationTestSplit(
        X_train=X_train_scaled,
        X_validation=X_validation_scaled,
        X_test=X_test_scaled,
        y_train=split.y_train.copy(),
        y_validation=split.y_validation.copy(),
        y_test=split.y_test.copy(),
        metadata_train=split.metadata_train.copy(),
        metadata_validation=split.metadata_validation.copy(),
        metadata_test=split.metadata_test.copy(),
        scaler=scaler,
    )

    validate_scaled_train_validation_test_split(scaled)

    return scaled


def validate_scaled_train_validation_test_split(
    split: ScaledTrainValidationTestSplit,
) -> None:
    """Validate scaled train/validation/test split.

    Args:
        split: Scaled train/validation/test split container.

    Raises:
        ValueError: If split lengths are inconsistent or scaled values are invalid.
    """
    validate_split_part_lengths(split.X_train, split.y_train, split.metadata_train, "train")
    validate_split_part_lengths(
        split.X_validation,
        split.y_validation,
        split.metadata_validation,
        "validation",
    )
    validate_split_part_lengths(split.X_test, split.y_test, split.metadata_test, "test")

    for split_name, X in (
        ("train", split.X_train),
        ("validation", split.X_validation),
        ("test", split.X_test),
    ):
        values = X.to_numpy()
        if np.isnan(values).any():
            raise ValueError(f"Scaled {split_name} features contain NaN values.")
        if np.isinf(values).any():
            raise ValueError(f"Scaled {split_name} features contain infinite values.")


def summarize_scaled_split(split: ScaledTrainValidationTestSplit) -> dict[str, float]:
    """Create summary for scaled train/validation/test split.

    Args:
        split: Scaled train/validation/test split container.

    Returns:
        Summary dictionary.
    """
    train_values = split.X_train.to_numpy()

    return {
        "train_rows": float(split.X_train.shape[0]),
        "validation_rows": float(split.X_validation.shape[0]),
        "test_rows": float(split.X_test.shape[0]),
        "features": float(split.X_train.shape[1]),
        "max_abs_train_mean": float(np.abs(train_values.mean(axis=0)).max()),
        "max_abs_train_std_delta": float(np.abs(train_values.std(axis=0) - 1.0).max()),
    }


def validate_prepared_dataset_consistency(prepared: PreparedDataset) -> None:
    """Validate consistency of X, y and metadata lengths.

    Args:
        prepared: Prepared dataset.

    Raises:
        ValueError: If lengths are inconsistent.
    """
    x_rows = len(prepared.X)
    y_rows = len(prepared.y)
    metadata_rows = len(prepared.metadata)

    if x_rows != y_rows or x_rows != metadata_rows:
        raise ValueError(
            "Inconsistent prepared dataset lengths after cleaning: "
            f"X={x_rows}, y={y_rows}, metadata={metadata_rows}"
        )


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


def summarize_train_test_split(split: TrainTestSplit) -> dict[str, int]:
    """Create train/test split summary.

    Args:
        split: Train/test split container.

    Returns:
        Summary dictionary.
    """
    return {
        "train_rows": int(split.X_train.shape[0]),
        "test_rows": int(split.X_test.shape[0]),
        "train_users": int(split.y_train.nunique()),
        "test_users": int(split.y_test.nunique()),
        "features": int(split.X_train.shape[1]),
    }


def summarize_train_validation_test_split(split: TrainValidationTestSplit) -> dict[str, int]:
    """Create train/validation/test split summary.

    Args:
        split: Train/validation/test split container.

    Returns:
        Summary dictionary.
    """
    return {
        "train_rows": int(split.X_train.shape[0]),
        "validation_rows": int(split.X_validation.shape[0]),
        "test_rows": int(split.X_test.shape[0]),
        "train_users": int(split.y_train.nunique()),
        "validation_users": int(split.y_validation.nunique()),
        "test_users": int(split.y_test.nunique()),
        "features": int(split.X_train.shape[1]),
    }


def save_scaler(scaler: StandardScaler, output_path: Path) -> None:
    """Save fitted StandardScaler to disk.

    Args:
        scaler: Fitted StandardScaler instance.
        output_path: Path to scaler.pkl.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, output_path)


def save_split_metadata(
    split: ScaledTrainValidationTestSplit,
    output_path: Path,
    test_size: float,
    validation_size: float,
    random_state: int,
) -> None:
    """Save train/validation/test split metadata to JSON.

    The JSON file stores sample identifiers for each split. It does not store
    feature values. This allows future training and evaluation scripts to
    reproduce the same split without recalculating random partitions.

    Args:
        split: Scaled train/validation/test split container.
        output_path: Path to cmu_split.json.
        test_size: Fraction of all samples assigned to the test split.
        validation_size: Fraction of train_full assigned to validation split.
        random_state: Random seed used for splitting.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = build_split_metadata_payload(
        split=split,
        test_size=test_size,
        validation_size=validation_size,
        random_state=random_state,
    )

    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def build_split_metadata_payload(
    split: ScaledTrainValidationTestSplit,
    test_size: float,
    validation_size: float,
    random_state: int,
) -> dict[str, object]:
    """Build JSON-serializable split metadata payload.

    Args:
        split: Scaled train/validation/test split container.
        test_size: Fraction of all samples assigned to the test split.
        validation_size: Fraction of train_full assigned to validation split.
        random_state: Random seed used for splitting.

    Returns:
        JSON-serializable dictionary.
    """
    feature_columns = split.X_train.columns.to_list()

    return {
        "dataset": "cmu_keystroke_dynamics",
        "random_state": random_state,
        "test_size": test_size,
        "validation_size": validation_size,
        "feature_columns": feature_columns,
        "counts": {
            "train": int(len(split.y_train)),
            "validation": int(len(split.y_validation)),
            "test": int(len(split.y_test)),
            "features": int(len(feature_columns)),
        },
        "users": {
            "train": sorted(split.y_train.unique().tolist()),
            "validation": sorted(split.y_validation.unique().tolist()),
            "test": sorted(split.y_test.unique().tolist()),
        },
        "splits": {
            "train": metadata_to_records(split.metadata_train),
            "validation": metadata_to_records(split.metadata_validation),
            "test": metadata_to_records(split.metadata_test),
        },
    }


def metadata_to_records(metadata: pd.DataFrame) -> list[dict[str, object]]:
    """Convert split metadata DataFrame to JSON records.

    Args:
        metadata: Split metadata with user_id, session_index, rep and sample_id.

    Returns:
        List of JSON-serializable records.
    """
    return metadata.loc[:, PROCESSED_META_COLUMNS].to_dict(orient="records")


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

    parser.add_argument(
        "--test-size",
        type=float,
        default=DEFAULT_TEST_SIZE,
        help="Fraction of samples assigned to the test split.",
    )

    parser.add_argument(
        "--validation-size",
        type=float,
        default=DEFAULT_VALIDATION_SIZE,
        help="Fraction of train_full samples assigned to the validation split.",
    )

    parser.add_argument(
        "--scaler-output",
        type=Path,
        default=DEFAULT_SCALER_OUTPUT,
        help="Path to save fitted StandardScaler as .pkl.",
    )

    parser.add_argument(
        "--split-output",
        type=Path,
        default=DEFAULT_SPLIT_OUTPUT,
        help="Path to save train/validation/test split metadata as JSON.",
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

    cleaned, cleaning_report = clean_prepared_dataset(prepared)
    cleaned_summary = summarize_prepared_dataset(cleaned)

    train_test = create_train_test_split(cleaned, test_size=args.test_size)
    split_summary = summarize_train_test_split(train_test)

    train_validation_test = create_train_validation_test_split(
        cleaned,
        test_size=args.test_size,
        validation_size=args.validation_size,
    )
    tvt_summary = summarize_train_validation_test_split(train_validation_test)

    scaled = scale_train_validation_test_split(train_validation_test)
    scaled_summary = summarize_scaled_split(scaled)

    save_scaler(scaled.scaler, args.scaler_output)
    save_split_metadata(
        split=scaled,
        output_path=args.split_output,
        test_size=args.test_size,
        validation_size=args.validation_size,
        random_state=RANDOM_SEED,
    )

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

    print()
    print("Cleaning report:")
    print(f"Rows before: {cleaning_report.rows_before}")
    print(f"Rows after: {cleaning_report.rows_after}")
    print(f"Rows removed: {cleaning_report.rows_removed}")
    print(f"Missing values before: {cleaning_report.missing_values_before}")
    print(f"Missing values after: {cleaning_report.missing_values_after}")
    print(f"Infinite values before: {cleaning_report.infinite_values_before}")
    print(f"Invalid negative rows removed: {cleaning_report.invalid_negative_rows_removed}")
    print(f"Valid negative UD values kept: {cleaning_report.valid_negative_ud_values_kept}")
    print(f"Clipped values: {cleaning_report.clipped_values_count}")

    print()
    print("Cleaned dataset:")
    print(f"X shape: {cleaned.X.shape}")
    print(f"y shape: {cleaned.y.shape}")
    print(f"metadata shape: {cleaned.metadata.shape}")
    print(f"Users in y: {cleaned_summary['users']}")

    print()
    print("Train/test split:")
    print(f"Train X shape: {train_test.X_train.shape}")
    print(f"Test X shape: {train_test.X_test.shape}")
    print(f"Train y shape: {train_test.y_train.shape}")
    print(f"Test y shape: {train_test.y_test.shape}")
    print(f"Train metadata shape: {train_test.metadata_train.shape}")
    print(f"Test metadata shape: {train_test.metadata_test.shape}")
    print(f"Train users: {split_summary['train_users']}")
    print(f"Test users: {split_summary['test_users']}")
    print(f"Features: {split_summary['features']}")

    print()
    print("Train/validation/test split:")
    print(f"Train X shape: {train_validation_test.X_train.shape}")
    print(f"Validation X shape: {train_validation_test.X_validation.shape}")
    print(f"Test X shape: {train_validation_test.X_test.shape}")
    print(f"Train y shape: {train_validation_test.y_train.shape}")
    print(f"Validation y shape: {train_validation_test.y_validation.shape}")
    print(f"Test y shape: {train_validation_test.y_test.shape}")
    print(f"Train metadata shape: {train_validation_test.metadata_train.shape}")
    print(f"Validation metadata shape: {train_validation_test.metadata_validation.shape}")
    print(f"Test metadata shape: {train_validation_test.metadata_test.shape}")
    print(f"Train users: {tvt_summary['train_users']}")
    print(f"Validation users: {tvt_summary['validation_users']}")
    print(f"Test users: {tvt_summary['test_users']}")
    print(f"Features: {tvt_summary['features']}")

    print()
    print("Scaled train/validation/test dataset:")
    print(f"Scaled train X shape: {scaled.X_train.shape}")
    print(f"Scaled validation X shape: {scaled.X_validation.shape}")
    print(f"Scaled test X shape: {scaled.X_test.shape}")
    print(f"Features: {int(scaled_summary['features'])}")
    print(f"Max abs train mean: {scaled_summary['max_abs_train_mean']:.10f}")
    print(f"Max abs train std delta: {scaled_summary['max_abs_train_std_delta']:.10f}")

    print()
    print("Scaler saved:")
    print(f"Path: {args.scaler_output}")

    print()
    print("Split metadata saved:")
    print(f"Path: {args.split_output}")
    print(f"Train samples: {len(scaled.metadata_train)}")
    print(f"Validation samples: {len(scaled.metadata_validation)}")
    print(f"Test samples: {len(scaled.metadata_test)}")


if __name__ == "__main__":
    main()
