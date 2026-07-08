"""Per-user authentication diagnostics for baseline v2 BatchNorm.

The module evaluates user-level FAR/FRR diagnostics for saved v2 artifacts:

    models/mlp_v2_batchnorm.keras
    models/scaler_v2_batchnorm.pkl
    models/label_encoder_v2_batchnorm.pkl
    models/auth_policy_v2_batchnorm.json

It writes:

    reports/per_user_auth_diagnostics_v2_batchnorm.csv
    reports/per_user_auth_diagnostics_v2_batchnorm.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow import keras

from src.config import CMU_FEATURES_FILE
from src.preprocessing import (
    TrainValidationTestSplit,
    clean_prepared_dataset,
    create_train_validation_test_split,
    load_processed_dataset,
    prepare_features_and_labels,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_NAME = "mlp_v2_batchnorm"

DEFAULT_MODEL_INPUT = PROJECT_ROOT / "models" / "mlp_v2_batchnorm.keras"
DEFAULT_SCALER_INPUT = PROJECT_ROOT / "models" / "scaler_v2_batchnorm.pkl"
DEFAULT_LABEL_ENCODER_INPUT = PROJECT_ROOT / "models" / "label_encoder_v2_batchnorm.pkl"
DEFAULT_AUTH_POLICY_INPUT = PROJECT_ROOT / "models" / "auth_policy_v2_batchnorm.json"

DEFAULT_OUTPUT_CSV = PROJECT_ROOT / "reports" / "per_user_auth_diagnostics_v2_batchnorm.csv"
DEFAULT_OUTPUT_MD = PROJECT_ROOT / "reports" / "per_user_auth_diagnostics_v2_batchnorm.md"


def load_unscaled_split(input_path: Path) -> TrainValidationTestSplit:
    """Load, clean and split the processed CMU dataset.

    Args:
        input_path: Path to processed CMU features CSV.

    Returns:
        Unscaled train/validation/test split.
    """
    df = load_processed_dataset(input_path)
    prepared = prepare_features_and_labels(df)
    cleaned, _cleaning_report = clean_prepared_dataset(prepared)
    return create_train_validation_test_split(cleaned)


def read_auth_policy(path: Path) -> dict[str, Any]:
    """Read authentication policy JSON.

    Args:
        path: Path to auth policy JSON.

    Returns:
        Authentication policy dictionary.

    Raises:
        FileNotFoundError: If file does not exist.
        ValueError: If auth_threshold is missing.
    """
    if not path.exists():
        raise FileNotFoundError(f"Authentication policy file not found: {path}")

    policy = json.loads(path.read_text(encoding="utf-8"))

    if "auth_threshold" not in policy:
        raise ValueError(f"auth_threshold is missing in policy: {path}")

    return policy


def load_artifacts(
    model_path: Path,
    scaler_path: Path,
    label_encoder_path: Path,
    auth_policy_path: Path,
) -> tuple[keras.Model, StandardScaler, LabelEncoder, dict[str, Any]]:
    """Load v2 model, scaler, label encoder and auth policy.

    Args:
        model_path: Path to saved model.
        scaler_path: Path to saved scaler.
        label_encoder_path: Path to label encoder.
        auth_policy_path: Path to auth policy.

    Returns:
        Tuple with model, scaler, label encoder and policy.

    Raises:
        FileNotFoundError: If any artifact is missing.
        TypeError: If scaler or label encoder has unexpected type.
    """
    for path in (model_path, scaler_path, label_encoder_path, auth_policy_path):
        if not path.exists():
            raise FileNotFoundError(f"Required artifact not found: {path}")

    model = keras.models.load_model(model_path)
    scaler = joblib.load(scaler_path)
    label_encoder = joblib.load(label_encoder_path)
    auth_policy = read_auth_policy(auth_policy_path)

    if not isinstance(scaler, StandardScaler):
        raise TypeError(f"Expected StandardScaler, got: {type(scaler)!r}")

    if not isinstance(label_encoder, LabelEncoder):
        raise TypeError(f"Expected LabelEncoder, got: {type(label_encoder)!r}")

    return model, scaler, label_encoder, auth_policy


def predict_test_probabilities(
    model: keras.Model,
    scaler: StandardScaler,
    split: TrainValidationTestSplit,
) -> np.ndarray:
    """Predict softmax probabilities for the test split.

    Args:
        model: Loaded Keras model.
        scaler: Loaded StandardScaler.
        split: Unscaled train/validation/test split.

    Returns:
        Probability matrix with shape ``(test_samples, users)``.
    """
    X_test_scaled = scaler.transform(split.X_test)
    return model.predict(X_test_scaled.astype(np.float32), verbose=0)


def calculate_per_user_diagnostics(
    probabilities: np.ndarray,
    true_user_ids: pd.Series,
    label_encoder: LabelEncoder,
    auth_threshold: float,
) -> pd.DataFrame:
    """Calculate per-user authentication diagnostics.

    For each user:
    - genuine scores are probabilities assigned to the true user for that user's
      genuine test samples;
    - impostor scores are probabilities assigned to this user for all samples
      belonging to other users.

    Args:
        probabilities: Softmax probability matrix.
        true_user_ids: True user IDs for test rows.
        label_encoder: Fitted label encoder.
        auth_threshold: Global authentication threshold.

    Returns:
        Per-user diagnostics DataFrame.
    """
    true_user_ids_array = true_user_ids.astype(str).to_numpy()
    rows: list[dict[str, Any]] = []

    for user_id in label_encoder.classes_:
        user_class_index = int(label_encoder.transform([user_id])[0])
        user_scores = probabilities[:, user_class_index]

        genuine_mask = true_user_ids_array == str(user_id)
        impostor_mask = ~genuine_mask

        genuine_scores = user_scores[genuine_mask]
        impostor_scores = user_scores[impostor_mask]

        genuine_accepts = int(np.sum(genuine_scores >= auth_threshold))
        genuine_trials = int(genuine_scores.size)
        genuine_rejects = genuine_trials - genuine_accepts
        frr = genuine_rejects / genuine_trials if genuine_trials else np.nan

        impostor_accepts = int(np.sum(impostor_scores >= auth_threshold))
        impostor_trials = int(impostor_scores.size)
        impostor_rejects = impostor_trials - impostor_accepts
        far = impostor_accepts / impostor_trials if impostor_trials else np.nan

        rows.append(
            {
                "user_id": str(user_id),
                "genuine_trials": genuine_trials,
                "genuine_accepts": genuine_accepts,
                "genuine_rejects": genuine_rejects,
                "frr": float(frr),
                "impostor_trials": impostor_trials,
                "impostor_accepts": impostor_accepts,
                "impostor_rejects": impostor_rejects,
                "far": float(far),
                "genuine_score_min": score_stat(genuine_scores, "min"),
                "genuine_score_p05": score_stat(genuine_scores, "p05"),
                "genuine_score_median": score_stat(genuine_scores, "median"),
                "genuine_score_mean": score_stat(genuine_scores, "mean"),
                "genuine_score_p95": score_stat(genuine_scores, "p95"),
                "impostor_score_min": score_stat(impostor_scores, "min"),
                "impostor_score_p95": score_stat(impostor_scores, "p95"),
                "impostor_score_p99": score_stat(impostor_scores, "p99"),
                "impostor_score_max": score_stat(impostor_scores, "max"),
                "auth_threshold": auth_threshold,
            }
        )

    return pd.DataFrame(rows).sort_values(
        ["frr", "far", "user_id"],
        ascending=[False, False, True],
    )


def score_stat(scores: np.ndarray, stat_name: str) -> float:
    """Calculate score statistic.

    Args:
        scores: Score vector.
        stat_name: Statistic name.

    Returns:
        Statistic value.
    """
    if scores.size == 0:
        return float("nan")

    if stat_name == "min":
        return float(np.min(scores))
    if stat_name == "p05":
        return float(np.quantile(scores, 0.05))
    if stat_name == "median":
        return float(np.median(scores))
    if stat_name == "mean":
        return float(np.mean(scores))
    if stat_name == "p95":
        return float(np.quantile(scores, 0.95))
    if stat_name == "p99":
        return float(np.quantile(scores, 0.99))
    if stat_name == "max":
        return float(np.max(scores))

    raise ValueError(f"Unsupported score statistic: {stat_name}")


def calculate_global_metrics(per_user_df: pd.DataFrame) -> dict[str, float | int]:
    """Calculate global FAR/FRR from per-user diagnostics.

    Args:
        per_user_df: Per-user diagnostics DataFrame.

    Returns:
        Global metrics dictionary.
    """
    genuine_trials = int(per_user_df["genuine_trials"].sum())
    genuine_rejects = int(per_user_df["genuine_rejects"].sum())
    impostor_trials = int(per_user_df["impostor_trials"].sum())
    impostor_accepts = int(per_user_df["impostor_accepts"].sum())

    return {
        "users": int(len(per_user_df)),
        "genuine_trials": genuine_trials,
        "genuine_rejects": genuine_rejects,
        "global_frr": genuine_rejects / genuine_trials,
        "impostor_trials": impostor_trials,
        "impostor_accepts": impostor_accepts,
        "global_far": impostor_accepts / impostor_trials,
        "max_user_frr": float(per_user_df["frr"].max()),
        "max_user_far": float(per_user_df["far"].max()),
        "mean_user_frr": float(per_user_df["frr"].mean()),
        "mean_user_far": float(per_user_df["far"].mean()),
    }


def save_reports(
    per_user_df: pd.DataFrame,
    global_metrics: dict[str, float | int],
    auth_policy: dict[str, Any],
    output_csv: Path,
    output_md: Path,
) -> None:
    """Save per-user diagnostics CSV and Markdown report.

    Args:
        per_user_df: Per-user diagnostics table.
        global_metrics: Global metrics.
        auth_policy: Authentication policy.
        output_csv: CSV output path.
        output_md: Markdown output path.
    """
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)

    per_user_df.to_csv(output_csv, index=False)

    worst_by_frr = per_user_df.sort_values(
        ["frr", "genuine_rejects"],
        ascending=[False, False],
    ).head(10)
    worst_by_far = per_user_df.sort_values(
        ["far", "impostor_accepts"],
        ascending=[False, False],
    ).head(10)

    report_lines = [
        "# Per-user FAR/FRR diagnostics for baseline v2 BatchNorm",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Model | `{MODEL_NAME}` |",
        f"| Users | {global_metrics['users']} |",
        f"| Auth threshold | {format_float(auth_policy.get('auth_threshold'))} |",
        f"| Policy target FAR | {format_percent(auth_policy.get('target_far'))} |",
        f"| Policy actual FAR | {format_percent(auth_policy.get('actual_far'))} |",
        f"| Policy actual FRR | {format_percent(auth_policy.get('actual_frr'))} |",
        f"| Global FAR from per-user table | {format_percent(global_metrics['global_far'])} |",
        f"| Global FRR from per-user table | {format_percent(global_metrics['global_frr'])} |",
        f"| Max user FAR | {format_percent(global_metrics['max_user_far'])} |",
        f"| Max user FRR | {format_percent(global_metrics['max_user_frr'])} |",
        "",
        "## Worst users by FRR",
        "",
        dataframe_to_markdown(
            worst_by_frr[
                [
                    "user_id",
                    "genuine_trials",
                    "genuine_rejects",
                    "frr",
                    "genuine_score_median",
                    "genuine_score_mean",
                    "auth_threshold",
                ]
            ]
        ),
        "",
        "## Worst users by FAR",
        "",
        dataframe_to_markdown(
            worst_by_far[
                [
                    "user_id",
                    "impostor_trials",
                    "impostor_accepts",
                    "far",
                    "impostor_score_p99",
                    "impostor_score_max",
                    "auth_threshold",
                ]
            ]
        ),
        "",
        "## Full per-user diagnostics",
        "",
        dataframe_to_markdown(per_user_df),
        "",
    ]

    output_md.write_text("\n".join(report_lines), encoding="utf-8")


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    """Convert DataFrame to Markdown without optional tabulate dependency.

    Args:
        df: Source DataFrame.

    Returns:
        Markdown table.
    """
    if df.empty:
        return "_No data._"

    columns = list(df.columns)
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join("---" for _column in columns) + " |"

    rows = []
    for _index, row in df.iterrows():
        values = [format_markdown_cell(row[column]) for column in columns]
        rows.append("| " + " | ".join(values) + " |")

    return "\n".join([header, separator, *rows])


def format_markdown_cell(value: object) -> str:
    """Format Markdown table cell.

    Args:
        value: Source value.

    Returns:
        Formatted string.
    """
    if isinstance(value, float):
        return f"{value:.6f}"

    return str(value).replace("|", "\\|")


def format_float(value: Any, digits: int = 6) -> str:
    """Format float-like value.

    Args:
        value: Source value.
        digits: Number of decimal digits.

    Returns:
        Formatted string.
    """
    if value is None:
        return "N/A"

    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def format_percent(value: Any, digits: int = 2) -> str:
    """Format fraction as percent.

    Args:
        value: Source fraction.
        digits: Number of decimal digits.

    Returns:
        Formatted percent string.
    """
    if value is None:
        return "N/A"

    try:
        return f"{float(value) * 100:.{digits}f}%"
    except (TypeError, ValueError):
        return str(value)


def build_arg_parser() -> argparse.ArgumentParser:
    """Build command-line argument parser.

    Returns:
        Configured parser.
    """
    parser = argparse.ArgumentParser(
        description="Calculate per-user FAR/FRR diagnostics for baseline v2."
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=CMU_FEATURES_FILE,
        help="Path to processed CMU features CSV.",
    )
    parser.add_argument(
        "--model-input",
        type=Path,
        default=DEFAULT_MODEL_INPUT,
        help="Path to saved v2 Keras model.",
    )
    parser.add_argument(
        "--scaler-input",
        type=Path,
        default=DEFAULT_SCALER_INPUT,
        help="Path to saved v2 StandardScaler.",
    )
    parser.add_argument(
        "--label-encoder-input",
        type=Path,
        default=DEFAULT_LABEL_ENCODER_INPUT,
        help="Path to saved v2 LabelEncoder.",
    )
    parser.add_argument(
        "--auth-policy-input",
        type=Path,
        default=DEFAULT_AUTH_POLICY_INPUT,
        help="Path to v2 authentication policy JSON.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=DEFAULT_OUTPUT_CSV,
        help="Path to output per-user diagnostics CSV.",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=DEFAULT_OUTPUT_MD,
        help="Path to output per-user diagnostics Markdown.",
    )

    return parser


def print_summary(
    per_user_df: pd.DataFrame,
    global_metrics: dict[str, float | int],
    auth_policy: dict[str, Any],
    output_csv: Path,
    output_md: Path,
) -> None:
    """Print final diagnostics summary.

    Args:
        per_user_df: Per-user diagnostics table.
        global_metrics: Global metrics.
        auth_policy: Authentication policy.
        output_csv: CSV output path.
        output_md: Markdown output path.
    """
    worst_by_frr = per_user_df.sort_values(
        ["frr", "genuine_rejects"],
        ascending=[False, False],
    ).head(10)
    worst_by_far = per_user_df.sort_values(
        ["far", "impostor_accepts"],
        ascending=[False, False],
    ).head(10)

    print("Per-user authentication diagnostics finished.")
    print()
    print("Global diagnostics:")
    print(f"Users: {global_metrics['users']}")
    print(f"Auth threshold: {format_float(auth_policy.get('auth_threshold'))}")
    print(f"Policy actual FAR: {format_float(auth_policy.get('actual_far'))}")
    print(f"Policy actual FRR: {format_float(auth_policy.get('actual_frr'))}")
    print(f"Global FAR from per-user table: {global_metrics['global_far']:.6f}")
    print(f"Global FRR from per-user table: {global_metrics['global_frr']:.6f}")
    print(f"Max user FAR: {global_metrics['max_user_far']:.6f}")
    print(f"Max user FRR: {global_metrics['max_user_frr']:.6f}")
    print(f"Mean user FAR: {global_metrics['mean_user_far']:.6f}")
    print(f"Mean user FRR: {global_metrics['mean_user_frr']:.6f}")
    print()
    print("Worst users by FRR:")
    print(
        worst_by_frr[
            [
                "user_id",
                "genuine_trials",
                "genuine_rejects",
                "frr",
                "genuine_score_median",
                "genuine_score_mean",
            ]
        ].to_string(index=False)
    )
    print()
    print("Worst users by FAR:")
    print(
        worst_by_far[
            [
                "user_id",
                "impostor_trials",
                "impostor_accepts",
                "far",
                "impostor_score_p99",
                "impostor_score_max",
            ]
        ].to_string(index=False)
    )
    print()
    print("Reports saved:")
    print(f"CSV path: {output_csv}")
    print(f"Markdown path: {output_md}")


def main() -> None:
    """CLI entry point."""
    parser = build_arg_parser()
    args = parser.parse_args()

    tf.get_logger().setLevel("ERROR")

    model, scaler, label_encoder, auth_policy = load_artifacts(
        model_path=args.model_input,
        scaler_path=args.scaler_input,
        label_encoder_path=args.label_encoder_input,
        auth_policy_path=args.auth_policy_input,
    )
    split = load_unscaled_split(args.input)

    auth_threshold = float(auth_policy["auth_threshold"])
    probabilities = predict_test_probabilities(
        model=model,
        scaler=scaler,
        split=split,
    )
    per_user_df = calculate_per_user_diagnostics(
        probabilities=probabilities,
        true_user_ids=split.y_test,
        label_encoder=label_encoder,
        auth_threshold=auth_threshold,
    )
    global_metrics = calculate_global_metrics(per_user_df)

    save_reports(
        per_user_df=per_user_df,
        global_metrics=global_metrics,
        auth_policy=auth_policy,
        output_csv=args.output_csv,
        output_md=args.output_md,
    )
    print_summary(
        per_user_df=per_user_df,
        global_metrics=global_metrics,
        auth_policy=auth_policy,
        output_csv=args.output_csv,
        output_md=args.output_md,
    )


if __name__ == "__main__":
    main()
