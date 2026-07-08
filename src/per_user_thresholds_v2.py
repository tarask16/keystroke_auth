"""Per-user authentication thresholds for baseline v2 BatchNorm.

The module selects an individual authentication threshold for every claimed user
using that user's impostor-score distribution and a target FAR.

It writes:

    reports/per_user_thresholds_v2_batchnorm.csv
    reports/per_user_thresholds_v2_batchnorm.md

The module does not overwrite models/auth_policy_v2_batchnorm.json. It only
creates an experimental threshold table for analysis.
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

DEFAULT_OUTPUT_CSV = PROJECT_ROOT / "reports" / "per_user_thresholds_v2_batchnorm.csv"
DEFAULT_OUTPUT_MD = PROJECT_ROOT / "reports" / "per_user_thresholds_v2_batchnorm.md"

DEFAULT_TARGET_FAR = 0.01
HIGH_FAR_MULTIPLIER = 1.5
HIGH_FRR_LIMIT = 0.05


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
    """Load model, scaler, label encoder and auth policy.

    Args:
        model_path: Path to saved model.
        scaler_path: Path to saved scaler.
        label_encoder_path: Path to label encoder.
        auth_policy_path: Path to auth policy.

    Returns:
        Tuple with model, scaler, label encoder and policy.
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
        Probability matrix.
    """
    X_test_scaled = scaler.transform(split.X_test)
    return model.predict(X_test_scaled.astype(np.float32), verbose=0)


def select_threshold_for_target_far(
    impostor_scores: np.ndarray,
    target_far: float,
) -> float:
    """Select threshold for target FAR from impostor scores.

    Args:
        impostor_scores: Impostor scores for one claimed user.
        target_far: Desired FAR.

    Returns:
        Selected threshold.
    """
    if not 0 < target_far < 1:
        raise ValueError(f"target_far must be between 0 and 1, got: {target_far}")

    sorted_scores = np.sort(impostor_scores)
    quantile_index = int(np.ceil((1.0 - target_far) * len(sorted_scores))) - 1
    quantile_index = max(0, min(quantile_index, len(sorted_scores) - 1))

    return float(sorted_scores[quantile_index])


def calculate_threshold_table(
    probabilities: np.ndarray,
    true_user_ids: pd.Series,
    label_encoder: LabelEncoder,
    global_threshold: float,
    target_far: float,
) -> pd.DataFrame:
    """Calculate global-vs-individual threshold diagnostics.

    Args:
        probabilities: Softmax probability matrix.
        true_user_ids: True user IDs for test rows.
        label_encoder: Fitted label encoder.
        global_threshold: Current common auth threshold.
        target_far: Target FAR for per-user thresholds.

    Returns:
        Per-user threshold diagnostics table.
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

        individual_threshold = select_threshold_for_target_far(
            impostor_scores=impostor_scores,
            target_far=target_far,
        )

        global_metrics = calculate_user_metrics(
            genuine_scores=genuine_scores,
            impostor_scores=impostor_scores,
            threshold=global_threshold,
        )
        individual_metrics = calculate_user_metrics(
            genuine_scores=genuine_scores,
            impostor_scores=impostor_scores,
            threshold=individual_threshold,
        )

        rows.append(
            {
                "user_id": str(user_id),
                "target_far": target_far,
                "global_threshold": global_threshold,
                "individual_threshold": individual_threshold,
                "threshold_delta": individual_threshold - global_threshold,
                "global_far": global_metrics["far"],
                "global_frr": global_metrics["frr"],
                "global_genuine_rejects": global_metrics["genuine_rejects"],
                "global_impostor_accepts": global_metrics["impostor_accepts"],
                "individual_far": individual_metrics["far"],
                "individual_frr": individual_metrics["frr"],
                "individual_genuine_rejects": individual_metrics["genuine_rejects"],
                "individual_impostor_accepts": individual_metrics["impostor_accepts"],
                "genuine_trials": global_metrics["genuine_trials"],
                "impostor_trials": global_metrics["impostor_trials"],
                "genuine_score_median": score_stat(genuine_scores, "median"),
                "genuine_score_mean": score_stat(genuine_scores, "mean"),
                "impostor_score_p99": score_stat(impostor_scores, "p99"),
                "impostor_score_max": score_stat(impostor_scores, "max"),
                "diagnostic_group": classify_user_threshold_need(
                    global_far=global_metrics["far"],
                    global_frr=global_metrics["frr"],
                    target_far=target_far,
                ),
            }
        )

    return pd.DataFrame(rows).sort_values(
        ["diagnostic_group", "global_far", "global_frr", "user_id"],
        ascending=[True, False, False, True],
    )


def calculate_user_metrics(
    genuine_scores: np.ndarray,
    impostor_scores: np.ndarray,
    threshold: float,
) -> dict[str, float | int]:
    """Calculate one-user FAR/FRR metrics at selected threshold.

    Args:
        genuine_scores: Genuine scores for this claimed user.
        impostor_scores: Impostor scores for this claimed user.
        threshold: Authentication threshold.

    Returns:
        Metrics dictionary.
    """
    genuine_accepts = int(np.sum(genuine_scores >= threshold))
    genuine_trials = int(genuine_scores.size)
    genuine_rejects = genuine_trials - genuine_accepts

    impostor_accepts = int(np.sum(impostor_scores >= threshold))
    impostor_trials = int(impostor_scores.size)
    impostor_rejects = impostor_trials - impostor_accepts

    return {
        "genuine_trials": genuine_trials,
        "genuine_rejects": genuine_rejects,
        "frr": genuine_rejects / genuine_trials if genuine_trials else float("nan"),
        "impostor_trials": impostor_trials,
        "impostor_accepts": impostor_accepts,
        "impostor_rejects": impostor_rejects,
        "far": impostor_accepts / impostor_trials if impostor_trials else float("nan"),
    }


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

    if stat_name == "median":
        return float(np.median(scores))
    if stat_name == "mean":
        return float(np.mean(scores))
    if stat_name == "p99":
        return float(np.quantile(scores, 0.99))
    if stat_name == "max":
        return float(np.max(scores))

    raise ValueError(f"Unsupported score statistic: {stat_name}")


def classify_user_threshold_need(
    global_far: float,
    global_frr: float,
    target_far: float,
) -> str:
    """Classify user by threshold problem type.

    Args:
        global_far: User FAR under global threshold.
        global_frr: User FRR under global threshold.
        target_far: Target FAR.

    Returns:
        Diagnostic group.
    """
    high_far = global_far > target_far * HIGH_FAR_MULTIPLIER
    high_frr = global_frr > HIGH_FRR_LIMIT

    if high_far and high_frr:
        return "high_far_and_high_frr"
    if high_far:
        return "high_far"
    if high_frr:
        return "high_frr"

    return "acceptable"


def calculate_summary(threshold_df: pd.DataFrame) -> dict[str, float | int]:
    """Calculate global summary for global and individual thresholds.

    Args:
        threshold_df: Per-user threshold table.

    Returns:
        Summary dictionary.
    """
    global_genuine_rejects = int(threshold_df["global_genuine_rejects"].sum())
    individual_genuine_rejects = int(threshold_df["individual_genuine_rejects"].sum())
    global_impostor_accepts = int(threshold_df["global_impostor_accepts"].sum())
    individual_impostor_accepts = int(threshold_df["individual_impostor_accepts"].sum())

    genuine_trials = int(threshold_df["genuine_trials"].sum())
    impostor_trials = int(threshold_df["impostor_trials"].sum())

    return {
        "users": int(len(threshold_df)),
        "genuine_trials": genuine_trials,
        "impostor_trials": impostor_trials,
        "global_genuine_rejects": global_genuine_rejects,
        "individual_genuine_rejects": individual_genuine_rejects,
        "global_frr": global_genuine_rejects / genuine_trials,
        "individual_frr": individual_genuine_rejects / genuine_trials,
        "global_impostor_accepts": global_impostor_accepts,
        "individual_impostor_accepts": individual_impostor_accepts,
        "global_far": global_impostor_accepts / impostor_trials,
        "individual_far": individual_impostor_accepts / impostor_trials,
        "mean_global_far": float(threshold_df["global_far"].mean()),
        "mean_individual_far": float(threshold_df["individual_far"].mean()),
        "mean_global_frr": float(threshold_df["global_frr"].mean()),
        "mean_individual_frr": float(threshold_df["individual_frr"].mean()),
        "max_individual_far": float(threshold_df["individual_far"].max()),
        "max_individual_frr": float(threshold_df["individual_frr"].max()),
    }


def save_reports(
    threshold_df: pd.DataFrame,
    summary: dict[str, float | int],
    output_csv: Path,
    output_md: Path,
) -> None:
    """Save per-user threshold reports.

    Args:
        threshold_df: Per-user threshold table.
        summary: Global summary dictionary.
        output_csv: CSV output path.
        output_md: Markdown output path.
    """
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)

    threshold_df.to_csv(output_csv, index=False)

    high_far_users = threshold_df[
        threshold_df["diagnostic_group"].isin(["high_far", "high_far_and_high_frr"])
    ].sort_values("global_far", ascending=False)
    high_frr_users = threshold_df[
        threshold_df["diagnostic_group"].isin(["high_frr", "high_far_and_high_frr"])
    ].sort_values("global_frr", ascending=False)

    report_lines = [
        "# Per-user threshold diagnostics for baseline v2 BatchNorm",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Users | {summary['users']} |",
        f"| Global FAR | {format_percent(summary['global_far'])} |",
        f"| Individual-threshold FAR | {format_percent(summary['individual_far'])} |",
        f"| Global FRR | {format_percent(summary['global_frr'])} |",
        f"| Individual-threshold FRR | {format_percent(summary['individual_frr'])} |",
        f"| Global false rejects | {summary['global_genuine_rejects']} |",
        f"| Individual-threshold false rejects | {summary['individual_genuine_rejects']} |",
        f"| Global false accepts | {summary['global_impostor_accepts']} |",
        f"| Individual-threshold false accepts | {summary['individual_impostor_accepts']} |",
        "",
        "## Users with high FAR under global threshold",
        "",
        dataframe_to_markdown(high_far_users.head(15)),
        "",
        "## Users with high FRR under global threshold",
        "",
        dataframe_to_markdown(high_frr_users.head(15)),
        "",
        "## Full threshold table",
        "",
        dataframe_to_markdown(threshold_df),
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
        Formatted cell.
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
        Formatted percent.
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
        description="Calculate per-user authentication thresholds for baseline v2."
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
        "--target-far",
        type=float,
        default=DEFAULT_TARGET_FAR,
        help="Target FAR for each individual user threshold.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=DEFAULT_OUTPUT_CSV,
        help="Path to output per-user threshold CSV.",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=DEFAULT_OUTPUT_MD,
        help="Path to output per-user threshold Markdown.",
    )

    return parser


def print_summary(
    threshold_df: pd.DataFrame,
    summary: dict[str, float | int],
    output_csv: Path,
    output_md: Path,
) -> None:
    """Print final summary.

    Args:
        threshold_df: Per-user threshold table.
        summary: Global summary dictionary.
        output_csv: CSV output path.
        output_md: Markdown output path.
    """
    high_far_users = threshold_df[
        threshold_df["diagnostic_group"].isin(["high_far", "high_far_and_high_frr"])
    ].sort_values("global_far", ascending=False)
    high_frr_users = threshold_df[
        threshold_df["diagnostic_group"].isin(["high_frr", "high_far_and_high_frr"])
    ].sort_values("global_frr", ascending=False)

    display_columns = [
        "user_id",
        "global_threshold",
        "individual_threshold",
        "threshold_delta",
        "global_far",
        "individual_far",
        "global_frr",
        "individual_frr",
        "diagnostic_group",
    ]

    print("Per-user threshold diagnostics finished.")
    print()
    print("Threshold summary:")
    print(f"Users: {summary['users']}")
    print(f"Global FAR: {summary['global_far']:.6f}")
    print(f"Individual-threshold FAR: {summary['individual_far']:.6f}")
    print(f"Global FRR: {summary['global_frr']:.6f}")
    print(f"Individual-threshold FRR: {summary['individual_frr']:.6f}")
    print(f"Global false rejects: {summary['global_genuine_rejects']}")
    print(f"Individual-threshold false rejects: {summary['individual_genuine_rejects']}")
    print(f"Global false accepts: {summary['global_impostor_accepts']}")
    print(f"Individual-threshold false accepts: {summary['individual_impostor_accepts']}")
    print()
    print("Users with high FAR under global threshold:")
    print(high_far_users.loc[:, display_columns].head(15).to_string(index=False))
    print()
    print("Users with high FRR under global threshold:")
    print(high_frr_users.loc[:, display_columns].head(15).to_string(index=False))
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

    probabilities = predict_test_probabilities(
        model=model,
        scaler=scaler,
        split=split,
    )

    threshold_df = calculate_threshold_table(
        probabilities=probabilities,
        true_user_ids=split.y_test,
        label_encoder=label_encoder,
        global_threshold=float(auth_policy["auth_threshold"]),
        target_far=args.target_far,
    )
    summary = calculate_summary(threshold_df)

    save_reports(
        threshold_df=threshold_df,
        summary=summary,
        output_csv=args.output_csv,
        output_md=args.output_md,
    )
    print_summary(
        threshold_df=threshold_df,
        summary=summary,
        output_csv=args.output_csv,
        output_md=args.output_md,
    )


if __name__ == "__main__":
    main()
