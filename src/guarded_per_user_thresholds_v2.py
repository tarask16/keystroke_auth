"""Guarded per-user threshold policy for baseline v2 BatchNorm.

This module compares three authentication policies:

1. global threshold
   - one common threshold from models/auth_policy_v2_batchnorm.json;

2. naive per-user thresholds
   - every user gets an individual threshold selected on validation impostor
     scores for target FAR;

3. guarded per-user thresholds
   - global threshold remains the default;
   - a user threshold is raised only when validation FAR is high;
   - the raised threshold is accepted only if validation FRR stays below a
     configured limit.

Thresholds are selected on validation split and evaluated on test split.

Outputs:
    reports/guarded_per_user_thresholds_v2_batchnorm.csv
    reports/guarded_per_user_thresholds_v2_batchnorm_summary.csv
    reports/guarded_per_user_thresholds_v2_batchnorm.md
    models/guarded_auth_policy_v2_batchnorm.json
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
POLICY_NAME = "guarded_per_user_thresholds_v2_batchnorm"

DEFAULT_MODEL_INPUT = PROJECT_ROOT / "models" / "mlp_v2_batchnorm.keras"
DEFAULT_SCALER_INPUT = PROJECT_ROOT / "models" / "scaler_v2_batchnorm.pkl"
DEFAULT_LABEL_ENCODER_INPUT = PROJECT_ROOT / "models" / "label_encoder_v2_batchnorm.pkl"
DEFAULT_AUTH_POLICY_INPUT = PROJECT_ROOT / "models" / "auth_policy_v2_batchnorm.json"

DEFAULT_OUTPUT_CSV = PROJECT_ROOT / "reports" / "guarded_per_user_thresholds_v2_batchnorm.csv"
DEFAULT_SUMMARY_CSV = (
    PROJECT_ROOT / "reports" / "guarded_per_user_thresholds_v2_batchnorm_summary.csv"
)
DEFAULT_OUTPUT_MD = PROJECT_ROOT / "reports" / "guarded_per_user_thresholds_v2_batchnorm.md"
DEFAULT_POLICY_OUTPUT = PROJECT_ROOT / "models" / "guarded_auth_policy_v2_batchnorm.json"

DEFAULT_TARGET_USER_FAR = 0.015
DEFAULT_HIGH_FAR_LIMIT = 0.015
DEFAULT_MAX_VALIDATION_FRR = 0.075


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
        Model, scaler, label encoder and auth policy.
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


def predict_probabilities(
    model: keras.Model,
    scaler: StandardScaler,
    X: pd.DataFrame,
) -> np.ndarray:
    """Scale features and predict softmax probabilities.

    Args:
        model: Loaded Keras model.
        scaler: Loaded StandardScaler.
        X: Unscaled feature matrix.

    Returns:
        Probability matrix.
    """
    X_scaled = scaler.transform(X)
    return model.predict(X_scaled.astype(np.float32), verbose=0)


def select_threshold_for_target_far(
    impostor_scores: np.ndarray,
    target_far: float,
) -> float:
    """Select threshold for target FAR from impostor scores.

    Args:
        impostor_scores: Impostor scores.
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


def get_user_scores(
    probabilities: np.ndarray,
    true_user_ids: pd.Series,
    label_encoder: LabelEncoder,
    user_id: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract genuine and impostor scores for one claimed user.

    Args:
        probabilities: Probability matrix.
        true_user_ids: True user IDs for rows.
        label_encoder: Fitted label encoder.
        user_id: Claimed user ID.

    Returns:
        Genuine scores and impostor scores.
    """
    user_class_index = int(label_encoder.transform([user_id])[0])
    user_scores = probabilities[:, user_class_index]
    true_user_ids_array = true_user_ids.astype(str).to_numpy()

    genuine_mask = true_user_ids_array == str(user_id)
    impostor_mask = ~genuine_mask

    return user_scores[genuine_mask], user_scores[impostor_mask]


def calculate_user_metrics(
    genuine_scores: np.ndarray,
    impostor_scores: np.ndarray,
    threshold: float,
) -> dict[str, float | int]:
    """Calculate FAR/FRR for one user and one threshold.

    Args:
        genuine_scores: Genuine scores.
        impostor_scores: Impostor scores.
        threshold: Authentication threshold.

    Returns:
        Metrics dictionary.
    """
    genuine_trials = int(genuine_scores.size)
    genuine_accepts = int(np.sum(genuine_scores >= threshold))
    genuine_rejects = genuine_trials - genuine_accepts

    impostor_trials = int(impostor_scores.size)
    impostor_accepts = int(np.sum(impostor_scores >= threshold))
    impostor_rejects = impostor_trials - impostor_accepts

    return {
        "genuine_trials": genuine_trials,
        "genuine_accepts": genuine_accepts,
        "genuine_rejects": genuine_rejects,
        "frr": genuine_rejects / genuine_trials if genuine_trials else float("nan"),
        "impostor_trials": impostor_trials,
        "impostor_accepts": impostor_accepts,
        "impostor_rejects": impostor_rejects,
        "far": impostor_accepts / impostor_trials if impostor_trials else float("nan"),
    }


def choose_guarded_threshold(
    global_threshold: float,
    candidate_threshold: float,
    validation_global_far: float,
    validation_candidate_far: float,
    validation_candidate_frr: float,
    high_far_limit: float,
    max_validation_frr: float,
) -> tuple[float, bool, str]:
    """Choose guarded user threshold.

    Args:
        global_threshold: Common threshold from global policy.
        candidate_threshold: Candidate per-user threshold.
        validation_global_far: User FAR on validation under global threshold.
        validation_candidate_far: User FAR on validation under candidate threshold.
        validation_candidate_frr: User FRR on validation under candidate threshold.
        high_far_limit: FAR level that triggers threshold hardening.
        max_validation_frr: Maximum allowed validation FRR after hardening.

    Returns:
        Guarded threshold, whether it was applied, and reason.
    """
    if validation_global_far <= high_far_limit:
        return global_threshold, False, "global_far_not_high"

    if candidate_threshold <= global_threshold:
        return global_threshold, False, "candidate_not_stricter"

    if validation_candidate_far >= validation_global_far:
        return global_threshold, False, "candidate_does_not_reduce_far"

    if validation_candidate_frr > max_validation_frr:
        return global_threshold, False, "candidate_frr_too_high"

    return candidate_threshold, True, "applied"


def build_threshold_table(
    validation_probabilities: np.ndarray,
    test_probabilities: np.ndarray,
    validation_user_ids: pd.Series,
    test_user_ids: pd.Series,
    label_encoder: LabelEncoder,
    global_threshold: float,
    target_user_far: float,
    high_far_limit: float,
    max_validation_frr: float,
) -> pd.DataFrame:
    """Build validation-calibrated threshold table and test evaluation.

    Args:
        validation_probabilities: Validation probability matrix.
        test_probabilities: Test probability matrix.
        validation_user_ids: Validation true user IDs.
        test_user_ids: Test true user IDs.
        label_encoder: Fitted label encoder.
        global_threshold: Common threshold.
        target_user_far: Target FAR for candidate thresholds.
        high_far_limit: FAR level that triggers guarded hardening.
        max_validation_frr: Maximum validation FRR allowed for guarded threshold.

    Returns:
        Per-user threshold table.
    """
    rows: list[dict[str, Any]] = []

    for user_id in label_encoder.classes_:
        user_id_str = str(user_id)

        validation_genuine_scores, validation_impostor_scores = get_user_scores(
            probabilities=validation_probabilities,
            true_user_ids=validation_user_ids,
            label_encoder=label_encoder,
            user_id=user_id_str,
        )
        test_genuine_scores, test_impostor_scores = get_user_scores(
            probabilities=test_probabilities,
            true_user_ids=test_user_ids,
            label_encoder=label_encoder,
            user_id=user_id_str,
        )

        candidate_threshold = select_threshold_for_target_far(
            impostor_scores=validation_impostor_scores,
            target_far=target_user_far,
        )

        validation_global = calculate_user_metrics(
            genuine_scores=validation_genuine_scores,
            impostor_scores=validation_impostor_scores,
            threshold=global_threshold,
        )
        validation_candidate = calculate_user_metrics(
            genuine_scores=validation_genuine_scores,
            impostor_scores=validation_impostor_scores,
            threshold=candidate_threshold,
        )

        guarded_threshold, guarded_applied, guarded_reason = choose_guarded_threshold(
            global_threshold=global_threshold,
            candidate_threshold=candidate_threshold,
            validation_global_far=float(validation_global["far"]),
            validation_candidate_far=float(validation_candidate["far"]),
            validation_candidate_frr=float(validation_candidate["frr"]),
            high_far_limit=high_far_limit,
            max_validation_frr=max_validation_frr,
        )

        validation_guarded = calculate_user_metrics(
            genuine_scores=validation_genuine_scores,
            impostor_scores=validation_impostor_scores,
            threshold=guarded_threshold,
        )

        test_global = calculate_user_metrics(
            genuine_scores=test_genuine_scores,
            impostor_scores=test_impostor_scores,
            threshold=global_threshold,
        )
        test_naive = calculate_user_metrics(
            genuine_scores=test_genuine_scores,
            impostor_scores=test_impostor_scores,
            threshold=candidate_threshold,
        )
        test_guarded = calculate_user_metrics(
            genuine_scores=test_genuine_scores,
            impostor_scores=test_impostor_scores,
            threshold=guarded_threshold,
        )

        rows.append(
            {
                "user_id": user_id_str,
                "global_threshold": global_threshold,
                "candidate_threshold": candidate_threshold,
                "guarded_threshold": guarded_threshold,
                "guarded_applied": guarded_applied,
                "guarded_reason": guarded_reason,
                "threshold_delta": guarded_threshold - global_threshold,
                "validation_global_far": validation_global["far"],
                "validation_global_frr": validation_global["frr"],
                "validation_candidate_far": validation_candidate["far"],
                "validation_candidate_frr": validation_candidate["frr"],
                "validation_guarded_far": validation_guarded["far"],
                "validation_guarded_frr": validation_guarded["frr"],
                "test_global_far": test_global["far"],
                "test_global_frr": test_global["frr"],
                "test_naive_far": test_naive["far"],
                "test_naive_frr": test_naive["frr"],
                "test_guarded_far": test_guarded["far"],
                "test_guarded_frr": test_guarded["frr"],
                "test_global_genuine_rejects": test_global["genuine_rejects"],
                "test_naive_genuine_rejects": test_naive["genuine_rejects"],
                "test_guarded_genuine_rejects": test_guarded["genuine_rejects"],
                "test_global_impostor_accepts": test_global["impostor_accepts"],
                "test_naive_impostor_accepts": test_naive["impostor_accepts"],
                "test_guarded_impostor_accepts": test_guarded["impostor_accepts"],
                "genuine_trials": test_global["genuine_trials"],
                "impostor_trials": test_global["impostor_trials"],
            }
        )

    return pd.DataFrame(rows).sort_values(
        ["guarded_applied", "test_global_far", "test_global_frr", "user_id"],
        ascending=[False, False, False, True],
    )


def summarize_policy(
    threshold_df: pd.DataFrame,
    policy_prefix: str,
) -> dict[str, float | int | str]:
    """Summarize a policy from per-user metrics.

    Args:
        threshold_df: Threshold table.
        policy_prefix: Policy prefix: global, naive or guarded.

    Returns:
        Summary dictionary.
    """
    genuine_trials = int(threshold_df["genuine_trials"].sum())
    impostor_trials = int(threshold_df["impostor_trials"].sum())
    genuine_rejects = int(threshold_df[f"test_{policy_prefix}_genuine_rejects"].sum())
    impostor_accepts = int(threshold_df[f"test_{policy_prefix}_impostor_accepts"].sum())

    return {
        "policy": policy_prefix,
        "genuine_trials": genuine_trials,
        "impostor_trials": impostor_trials,
        "false_rejects": genuine_rejects,
        "false_accepts": impostor_accepts,
        "frr": genuine_rejects / genuine_trials,
        "far": impostor_accepts / impostor_trials,
        "max_user_frr": float(threshold_df[f"test_{policy_prefix}_frr"].max()),
        "max_user_far": float(threshold_df[f"test_{policy_prefix}_far"].max()),
        "mean_user_frr": float(threshold_df[f"test_{policy_prefix}_frr"].mean()),
        "mean_user_far": float(threshold_df[f"test_{policy_prefix}_far"].mean()),
    }


def build_summary_table(threshold_df: pd.DataFrame) -> pd.DataFrame:
    """Build summary table for global, naive and guarded policies.

    Args:
        threshold_df: Per-user threshold table.

    Returns:
        Policy summary table.
    """
    return pd.DataFrame(
        [
            summarize_policy(threshold_df, "global"),
            summarize_policy(threshold_df, "naive"),
            summarize_policy(threshold_df, "guarded"),
        ]
    )


def build_guarded_policy_json(
    threshold_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    auth_policy: dict[str, Any],
    target_user_far: float,
    high_far_limit: float,
    max_validation_frr: float,
) -> dict[str, Any]:
    """Build guarded policy JSON.

    Args:
        threshold_df: Per-user threshold table.
        summary_df: Summary table.
        auth_policy: Source global auth policy.
        target_user_far: Target FAR for candidate thresholds.
        high_far_limit: FAR trigger for guarded hardening.
        max_validation_frr: Max validation FRR for guarded hardening.

    Returns:
        Guarded policy dictionary.
    """
    guarded_summary = summary_df[summary_df["policy"] == "guarded"].iloc[0]
    thresholds = {
        str(row["user_id"]): float(row["guarded_threshold"])
        for _index, row in threshold_df.iterrows()
    }
    applied_users = [
        str(row["user_id"])
        for _index, row in threshold_df[threshold_df["guarded_applied"]].iterrows()
    ]

    return {
        "policy_name": POLICY_NAME,
        "model_name": MODEL_NAME,
        "score_type": auth_policy.get("score_type"),
        "base_global_threshold": float(auth_policy["auth_threshold"]),
        "target_user_far": target_user_far,
        "high_far_limit": high_far_limit,
        "max_validation_frr": max_validation_frr,
        "threshold_selection_split": "validation",
        "evaluation_split": "test",
        "thresholds": thresholds,
        "guarded_applied_users": applied_users,
        "guarded_applied_users_count": len(applied_users),
        "test_far": float(guarded_summary["far"]),
        "test_frr": float(guarded_summary["frr"]),
        "test_false_accepts": int(guarded_summary["false_accepts"]),
        "test_false_rejects": int(guarded_summary["false_rejects"]),
    }


def save_reports(
    threshold_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    guarded_policy: dict[str, Any],
    output_csv: Path,
    summary_csv: Path,
    output_md: Path,
    policy_output: Path,
) -> None:
    """Save CSV, Markdown and JSON policy outputs.

    Args:
        threshold_df: Per-user threshold table.
        summary_df: Policy summary table.
        guarded_policy: Guarded policy JSON dictionary.
        output_csv: Per-user CSV output path.
        summary_csv: Summary CSV output path.
        output_md: Markdown output path.
        policy_output: Guarded policy JSON output path.
    """
    for output_path in (output_csv, summary_csv, output_md, policy_output):
        output_path.parent.mkdir(parents=True, exist_ok=True)

    threshold_df.to_csv(output_csv, index=False)
    summary_df.to_csv(summary_csv, index=False)
    policy_output.write_text(
        json.dumps(guarded_policy, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    applied_users = threshold_df[threshold_df["guarded_applied"]].copy()
    rejected_candidates = threshold_df[~threshold_df["guarded_applied"]].copy()

    report_lines = [
        "# Guarded per-user threshold policy for baseline v2 BatchNorm",
        "",
        "## Policy summary on test split",
        "",
        dataframe_to_markdown(summary_df),
        "",
        "## Users where guarded threshold was applied",
        "",
        dataframe_to_markdown(
            applied_users[
                [
                    "user_id",
                    "global_threshold",
                    "candidate_threshold",
                    "guarded_threshold",
                    "threshold_delta",
                    "validation_global_far",
                    "validation_candidate_far",
                    "validation_candidate_frr",
                    "test_global_far",
                    "test_guarded_far",
                    "test_global_frr",
                    "test_guarded_frr",
                ]
            ]
        ),
        "",
        "## Rejected candidate thresholds",
        "",
        dataframe_to_markdown(
            rejected_candidates[
                [
                    "user_id",
                    "guarded_reason",
                    "global_threshold",
                    "candidate_threshold",
                    "validation_global_far",
                    "validation_candidate_frr",
                    "test_global_far",
                    "test_naive_far",
                    "test_global_frr",
                    "test_naive_frr",
                ]
            ]
        ),
        "",
        "## Full per-user threshold table",
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


def build_arg_parser() -> argparse.ArgumentParser:
    """Build command-line argument parser.

    Returns:
        Configured parser.
    """
    parser = argparse.ArgumentParser(
        description="Build guarded per-user threshold policy for baseline v2."
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
        "--target-user-far",
        type=float,
        default=DEFAULT_TARGET_USER_FAR,
        help="Target FAR for candidate per-user thresholds.",
    )
    parser.add_argument(
        "--high-far-limit",
        type=float,
        default=DEFAULT_HIGH_FAR_LIMIT,
        help="Validation FAR level that triggers guarded threshold hardening.",
    )
    parser.add_argument(
        "--max-validation-frr",
        type=float,
        default=DEFAULT_MAX_VALIDATION_FRR,
        help="Maximum allowed validation FRR for guarded threshold hardening.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=DEFAULT_OUTPUT_CSV,
        help="Path to output per-user threshold CSV.",
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=DEFAULT_SUMMARY_CSV,
        help="Path to output summary CSV.",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=DEFAULT_OUTPUT_MD,
        help="Path to output Markdown report.",
    )
    parser.add_argument(
        "--policy-output",
        type=Path,
        default=DEFAULT_POLICY_OUTPUT,
        help="Path to output guarded policy JSON.",
    )

    return parser


def print_summary(
    threshold_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    output_csv: Path,
    summary_csv: Path,
    output_md: Path,
    policy_output: Path,
) -> None:
    """Print final summary.

    Args:
        threshold_df: Per-user threshold table.
        summary_df: Policy summary table.
        output_csv: Per-user CSV output path.
        summary_csv: Summary CSV output path.
        output_md: Markdown output path.
        policy_output: Guarded policy JSON output path.
    """
    applied_users = threshold_df[threshold_df["guarded_applied"]]
    rejected_candidates = threshold_df[~threshold_df["guarded_applied"]]

    print("Guarded per-user threshold policy finished.")
    print()
    print("Policy summary on test split:")
    print(summary_df.to_string(index=False))
    print()
    print("Guarded thresholds applied:")
    print(f"Users: {len(applied_users)}")
    if applied_users.empty:
        print("No guarded thresholds were applied.")
    else:
        print(
            applied_users[
                [
                    "user_id",
                    "global_threshold",
                    "candidate_threshold",
                    "guarded_threshold",
                    "validation_global_far",
                    "validation_candidate_far",
                    "validation_candidate_frr",
                    "test_global_far",
                    "test_guarded_far",
                    "test_global_frr",
                    "test_guarded_frr",
                ]
            ].to_string(index=False)
        )
    print()
    print("Rejected candidate threshold reasons:")
    print(rejected_candidates["guarded_reason"].value_counts().to_string())
    print()
    print("Reports saved:")
    print(f"Per-user CSV path: {output_csv}")
    print(f"Summary CSV path: {summary_csv}")
    print(f"Markdown path: {output_md}")
    print(f"Policy JSON path: {policy_output}")


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

    validation_probabilities = predict_probabilities(
        model=model,
        scaler=scaler,
        X=split.X_validation,
    )
    test_probabilities = predict_probabilities(
        model=model,
        scaler=scaler,
        X=split.X_test,
    )

    threshold_df = build_threshold_table(
        validation_probabilities=validation_probabilities,
        test_probabilities=test_probabilities,
        validation_user_ids=split.y_validation,
        test_user_ids=split.y_test,
        label_encoder=label_encoder,
        global_threshold=float(auth_policy["auth_threshold"]),
        target_user_far=args.target_user_far,
        high_far_limit=args.high_far_limit,
        max_validation_frr=args.max_validation_frr,
    )
    summary_df = build_summary_table(threshold_df)
    guarded_policy = build_guarded_policy_json(
        threshold_df=threshold_df,
        summary_df=summary_df,
        auth_policy=auth_policy,
        target_user_far=args.target_user_far,
        high_far_limit=args.high_far_limit,
        max_validation_frr=args.max_validation_frr,
    )

    save_reports(
        threshold_df=threshold_df,
        summary_df=summary_df,
        guarded_policy=guarded_policy,
        output_csv=args.output_csv,
        summary_csv=args.summary_csv,
        output_md=args.output_md,
        policy_output=args.policy_output,
    )
    print_summary(
        threshold_df=threshold_df,
        summary_df=summary_df,
        output_csv=args.output_csv,
        summary_csv=args.summary_csv,
        output_md=args.output_md,
        policy_output=args.policy_output,
    )


if __name__ == "__main__":
    main()
