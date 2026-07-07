"""CMU dataset loader for Keystroke Auth.

This module loads the CMU Keystroke Dynamics Benchmark dataset
from DSL-StrongPasswordData.csv and converts it into the internal
baseline format used by the project.

Expected raw CMU columns:
- subject
- sessionIndex
- rep
- timing feature columns with prefixes:
  - H.   hold / dwell time
  - DD.  press-press / down-down latency
  - UD.  release-press / up-down latency
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.config import CMU_FEATURES_FILE, CMU_RAW_FILE, DATASET_REPORT_FILE

RAW_META_COLUMNS = ["subject", "sessionIndex", "rep"]
PROCESSED_META_COLUMNS = ["user_id", "session_index", "rep", "sample_id"]
CMU_FEATURE_PREFIXES = ("H.", "DD.", "UD.")


@dataclass(frozen=True)
class DatasetInfo:
    """Short summary of the loaded dataset."""

    users_count: int
    samples_count: int
    features_count: int
    missing_values_count: int
    min_samples_per_user: int
    max_samples_per_user: int


def load_cmu_dataset(path: Path) -> pd.DataFrame:
    """Load raw CMU dataset from CSV.

    Args:
        path: Path to DSL-StrongPasswordData.csv.

    Returns:
        Raw CMU DataFrame.

    Raises:
        FileNotFoundError: If the dataset file does not exist.
        ValueError: If the dataset is empty or has invalid structure.
    """
    path = path.resolve()

    if not path.exists():
        raise FileNotFoundError(f"CMU dataset file not found: {path}")

    if not path.is_file():
        raise ValueError(f"CMU dataset path is not a file: {path}")

    df = pd.read_csv(path)

    if df.empty:
        raise ValueError(f"CMU dataset is empty: {path}")

    validate_cmu_dataset(df)

    return df


def validate_cmu_dataset(df: pd.DataFrame) -> None:
    """Validate raw CMU dataset structure.

    Args:
        df: Raw CMU DataFrame.

    Raises:
        ValueError: If required columns are missing or feature columns are absent.
    """
    missing_meta_columns = [column for column in RAW_META_COLUMNS if column not in df.columns]

    if missing_meta_columns:
        raise ValueError(f"Missing required CMU columns: {missing_meta_columns}")

    feature_columns = get_cmu_feature_columns(df)

    if not feature_columns:
        raise ValueError(
            "No CMU timing feature columns found. Expected columns starting with H., DD. or UD."
        )

    non_numeric_columns = [
        column for column in feature_columns if not pd.api.types.is_numeric_dtype(df[column])
    ]

    if non_numeric_columns:
        raise ValueError(f"Non-numeric feature columns found: {non_numeric_columns}")


def get_cmu_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return timing feature columns from CMU dataset.

    Args:
        df: Input DataFrame.

    Returns:
        List of CMU feature column names.
    """
    return [
        column
        for column in df.columns
        if isinstance(column, str) and column.startswith(CMU_FEATURE_PREFIXES)
    ]


def prepare_cmu_features(df: pd.DataFrame) -> pd.DataFrame:
    """Convert raw CMU dataset to internal baseline format.

    Output format:
    - user_id
    - session_index
    - rep
    - sample_id
    - H.* / DD.* / UD.* feature columns

    Args:
        df: Raw CMU DataFrame.

    Returns:
        Processed DataFrame ready for preprocessing and training.
    """
    validate_cmu_dataset(df)

    feature_columns = get_cmu_feature_columns(df)

    processed = df.loc[:, RAW_META_COLUMNS + feature_columns].copy()

    processed = processed.rename(
        columns={
            "subject": "user_id",
            "sessionIndex": "session_index",
        }
    )

    processed["user_id"] = processed["user_id"].astype(str)
    processed["session_index"] = processed["session_index"].astype(int)
    processed["rep"] = processed["rep"].astype(int)

    processed["sample_id"] = (
        processed["user_id"]
        + "_s"
        + processed["session_index"].astype(str).str.zfill(2)
        + "_r"
        + processed["rep"].astype(str).str.zfill(2)
    )

    ordered_columns = PROCESSED_META_COLUMNS + feature_columns
    processed = processed.loc[:, ordered_columns]

    return processed


def calculate_dataset_info(df: pd.DataFrame) -> DatasetInfo:
    """Calculate basic dataset statistics.

    Args:
        df: Processed CMU DataFrame.

    Returns:
        DatasetInfo summary.
    """
    feature_columns = get_processed_feature_columns(df)

    samples_per_user = df.groupby("user_id").size()

    return DatasetInfo(
        users_count=int(df["user_id"].nunique()),
        samples_count=int(len(df)),
        features_count=int(len(feature_columns)),
        missing_values_count=int(df.isna().sum().sum()),
        min_samples_per_user=int(samples_per_user.min()),
        max_samples_per_user=int(samples_per_user.max()),
    )


def get_processed_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return feature columns from processed dataset.

    Args:
        df: Processed DataFrame.

    Returns:
        List of feature column names.
    """
    return [column for column in df.columns if column not in PROCESSED_META_COLUMNS]


def save_processed_dataset(df: pd.DataFrame, output_path: Path) -> None:
    """Save processed dataset to CSV.

    Args:
        df: Processed DataFrame.
        output_path: Output CSV path.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)


def create_dataset_report(df: pd.DataFrame, output_path: Path) -> None:
    """Create Markdown dataset report.

    Args:
        df: Processed CMU DataFrame.
        output_path: Output Markdown path.
    """
    info = calculate_dataset_info(df)
    feature_columns = get_processed_feature_columns(df)

    h_features = [column for column in feature_columns if column.startswith("H.")]
    dd_features = [column for column in feature_columns if column.startswith("DD.")]
    ud_features = [column for column in feature_columns if column.startswith("UD.")]

    report = f"""# Dataset Report: CMU Keystroke Dynamics Benchmark

## Summary

| Metric | Value |
|---|---:|
| Users | {info.users_count} |
| Samples | {info.samples_count} |
| Features | {info.features_count} |
| Missing values | {info.missing_values_count} |
| Min samples per user | {info.min_samples_per_user} |
| Max samples per user | {info.max_samples_per_user} |

## Feature Groups

| Group | Description | Count |
|---|---|---:|
| H.* | Hold / dwell time | {len(h_features)} |
| DD.* | Press-press / down-down latency | {len(dd_features)} |
| UD.* | Release-press / up-down latency | {len(ud_features)} |

## Internal Format

The processed dataset uses the following metadata columns:

- user_id
- session_index
- rep
- sample_id

Feature columns are preserved from the original CMU dataset.

## First Feature Columns

{format_feature_preview(feature_columns)}
"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")


def format_feature_preview(feature_columns: list[str], limit: int = 20) -> str:
    """Format feature column preview for Markdown report.

    Args:
        feature_columns: List of feature columns.
        limit: Maximum number of columns to show.

    Returns:
        Markdown bullet list.
    """
    preview = feature_columns[:limit]

    if not preview:
        return "- No feature columns found."

    lines = [f"- {column}" for column in preview]

    if len(feature_columns) > limit:
        lines.append(f"- ... {len(feature_columns) - limit} more")

    return "\n".join(lines)


def run_loader(input_path: Path, output_path: Path, report_path: Path) -> pd.DataFrame:
    """Run full CMU loading pipeline.

    Args:
        input_path: Raw CMU CSV path.
        output_path: Processed CSV path.
        report_path: Dataset report path.

    Returns:
        Processed DataFrame.
    """
    raw_df = load_cmu_dataset(input_path)
    processed_df = prepare_cmu_features(raw_df)

    save_processed_dataset(processed_df, output_path)
    create_dataset_report(processed_df, report_path)

    return processed_df


def build_arg_parser() -> argparse.ArgumentParser:
    """Build command-line argument parser.

    Returns:
        Configured ArgumentParser.
    """
    parser = argparse.ArgumentParser(description="Load CMU Keystroke Dynamics Benchmark dataset.")

    parser.add_argument(
        "--input",
        type=Path,
        default=CMU_RAW_FILE,
        help="Path to raw DSL-StrongPasswordData.csv file.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=CMU_FEATURES_FILE,
        help="Path to save processed dataset CSV.",
    )

    parser.add_argument(
        "--report",
        type=Path,
        default=DATASET_REPORT_FILE,
        help="Path to save dataset Markdown report.",
    )

    return parser


def main() -> None:
    """CLI entry point."""
    parser = build_arg_parser()
    args = parser.parse_args()

    processed_df = run_loader(
        input_path=args.input,
        output_path=args.output,
        report_path=args.report,
    )

    info = calculate_dataset_info(processed_df)

    print("CMU dataset loaded successfully.")
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    print(f"Report: {args.report}")
    print(f"Users: {info.users_count}")
    print(f"Samples: {info.samples_count}")
    print(f"Features: {info.features_count}")
    print(f"Missing values: {info.missing_values_count}")


if __name__ == "__main__":
    main()
