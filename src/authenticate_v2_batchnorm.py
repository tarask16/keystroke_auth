"""Authentication entry point for Keystroke Auth MLP v2 BatchNorm.

This module verifies saved v2 artifacts without overwriting the original
baseline authentication module.

Default artifacts:
- models/mlp_v2_batchnorm.keras
- models/scaler_v2_batchnorm.pkl
- models/label_encoder_v2_batchnorm.pkl
- models/auth_policy_v2_batchnorm.json

Supported modes:
- one-sample authentication;
- batch authentication smoke-test on the full test split.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
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
DEFAULT_BATCH_REPORT_OUTPUT = (
    PROJECT_ROOT / "reports" / "authentication_batch_report_v2_batchnorm.csv"
)
DEFAULT_SAMPLE_INDEX = 0


@dataclass(frozen=True)
class AuthenticationArtifacts:
    """Container for loaded v2 authentication artifacts."""

    model: keras.Model
    scaler: StandardScaler
    label_encoder: LabelEncoder
    auth_policy: dict[str, Any]


@dataclass(frozen=True)
class SelectedSample:
    """Container for selected test sample."""

    X: pd.DataFrame
    metadata: pd.Series
    true_user_id: str
    test_index: int


@dataclass(frozen=True)
class AuthenticationResult:
    """Authentication result for one claimed user and one sample."""

    accepted: bool
    claimed_user_id: str
    true_user_id: str
    predicted_user_id: str
    sample_id: str
    sample_index: int
    claimed_user_score: float
    predicted_user_score: float
    auth_threshold: float
    target_far: float | None
    actual_far: float | None
    actual_frr: float | None
    eer: float | None


@dataclass(frozen=True)
class BatchAuthenticationReport:
    """Batch authentication smoke-test report."""

    genuine_trials: int
    genuine_accepts: int
    genuine_rejects: int
    empirical_frr: float
    impostor_trials: int
    impostor_accepts: int
    impostor_rejects: int
    empirical_far: float
    auth_threshold: float
    target_far: float | None
    policy_actual_far: float | None
    policy_actual_frr: float | None
    policy_eer: float | None
    predicted_classes: int


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


def load_authentication_artifacts(
    model_path: Path,
    scaler_path: Path,
    label_encoder_path: Path,
    auth_policy_path: Path,
) -> AuthenticationArtifacts:
    """Load saved v2 authentication artifacts.

    Args:
        model_path: Path to saved Keras model.
        scaler_path: Path to saved StandardScaler.
        label_encoder_path: Path to saved LabelEncoder.
        auth_policy_path: Path to authentication policy JSON.

    Returns:
        Loaded authentication artifacts.

    Raises:
        FileNotFoundError: If any artifact is missing.
        TypeError: If scaler or label encoder has unexpected type.
        ValueError: If policy does not contain auth_threshold.
    """
    for path in (model_path, scaler_path, label_encoder_path, auth_policy_path):
        if not path.exists():
            raise FileNotFoundError(f"Required authentication artifact not found: {path}")

    model = keras.models.load_model(model_path)
    scaler = joblib.load(scaler_path)
    label_encoder = joblib.load(label_encoder_path)
    auth_policy = json.loads(auth_policy_path.read_text(encoding="utf-8"))

    if not isinstance(scaler, StandardScaler):
        raise TypeError(f"Expected StandardScaler, got: {type(scaler)!r}")

    if not isinstance(label_encoder, LabelEncoder):
        raise TypeError(f"Expected LabelEncoder, got: {type(label_encoder)!r}")

    if "auth_threshold" not in auth_policy:
        raise ValueError(f"auth_threshold is missing in auth policy: {auth_policy_path}")

    return AuthenticationArtifacts(
        model=model,
        scaler=scaler,
        label_encoder=label_encoder,
        auth_policy=auth_policy,
    )


def select_test_sample(
    split: TrainValidationTestSplit,
    sample_index: int,
    sample_id: str | None,
) -> SelectedSample:
    """Select one sample from the test split.

    Args:
        split: Unscaled train/validation/test split.
        sample_index: Fallback row index in test split.
        sample_id: Optional sample_id to locate in test metadata.

    Returns:
        Selected test sample.

    Raises:
        IndexError: If sample_index is out of range.
        ValueError: If sample_id is absent in test metadata.
    """
    if sample_id is not None:
        sample_id_values = split.metadata_test["sample_id"].astype(str)
        matching_indices = sample_id_values.index[sample_id_values == str(sample_id)].to_list()

        if not matching_indices:
            raise ValueError(f"sample_id not found in test split: {sample_id}")

        selected_index = int(matching_indices[0])
    else:
        if sample_index < 0 or sample_index >= len(split.X_test):
            raise IndexError(
                "sample_index is out of test split range: "
                f"{sample_index}; valid range: 0..{len(split.X_test) - 1}"
            )
        selected_index = sample_index

    X = split.X_test.iloc[[selected_index]].copy()
    metadata = split.metadata_test.iloc[selected_index].copy()
    true_user_id = str(split.y_test.iloc[selected_index])

    return SelectedSample(
        X=X,
        metadata=metadata,
        true_user_id=true_user_id,
        test_index=selected_index,
    )


def authenticate_sample(
    artifacts: AuthenticationArtifacts,
    sample: SelectedSample,
    claimed_user_id: str | None,
) -> AuthenticationResult:
    """Authenticate one selected sample against one claimed user.

    Args:
        artifacts: Loaded model, scaler, label encoder and auth policy.
        sample: Selected test sample.
        claimed_user_id: Claimed user ID. If None, true_user_id is used.

    Returns:
        Authentication result.

    Raises:
        ValueError: If claimed_user_id is unknown to the label encoder.
    """
    effective_claimed_user_id = claimed_user_id or sample.true_user_id
    known_users = set(artifacts.label_encoder.classes_.tolist())

    if effective_claimed_user_id not in known_users:
        raise ValueError(
            "Unknown claimed_user_id: "
            f"{effective_claimed_user_id}. Known examples: "
            f"{artifacts.label_encoder.classes_[:5].tolist()}..."
        )

    probabilities = predict_probabilities(artifacts, sample.X)[0]

    claimed_class_index = int(artifacts.label_encoder.transform([effective_claimed_user_id])[0])
    predicted_class_index = int(np.argmax(probabilities))
    predicted_user_id = str(artifacts.label_encoder.inverse_transform([predicted_class_index])[0])

    claimed_user_score = float(probabilities[claimed_class_index])
    predicted_user_score = float(probabilities[predicted_class_index])
    auth_threshold = float(artifacts.auth_policy["auth_threshold"])

    return AuthenticationResult(
        accepted=claimed_user_score >= auth_threshold,
        claimed_user_id=effective_claimed_user_id,
        true_user_id=sample.true_user_id,
        predicted_user_id=predicted_user_id,
        sample_id=str(sample.metadata["sample_id"]),
        sample_index=sample.test_index,
        claimed_user_score=claimed_user_score,
        predicted_user_score=predicted_user_score,
        auth_threshold=auth_threshold,
        target_far=optional_float(artifacts.auth_policy.get("target_far")),
        actual_far=optional_float(artifacts.auth_policy.get("actual_far")),
        actual_frr=optional_float(artifacts.auth_policy.get("actual_frr")),
        eer=optional_float(artifacts.auth_policy.get("eer")),
    )


def predict_probabilities(
    artifacts: AuthenticationArtifacts,
    X: pd.DataFrame,
) -> np.ndarray:
    """Scale features and predict class probabilities.

    Args:
        artifacts: Loaded authentication artifacts.
        X: Unscaled feature matrix.

    Returns:
        Probability matrix with shape ``(samples, users)``.
    """
    X_scaled = artifacts.scaler.transform(X)
    return artifacts.model.predict(
        X_scaled.astype(np.float32),
        verbose=0,
    )


def run_batch_authentication_test(
    artifacts: AuthenticationArtifacts,
    split: TrainValidationTestSplit,
) -> BatchAuthenticationReport:
    """Run batch authentication smoke-test on the whole test split.

    Genuine trials:
        claim = true user for each test sample.

    Impostor trials:
        claim = every other known user for each test sample.

    Args:
        artifacts: Loaded authentication artifacts.
        split: Unscaled train/validation/test split.

    Returns:
        Batch authentication report.
    """
    probabilities = predict_probabilities(artifacts, split.X_test)
    true_class_indices = artifacts.label_encoder.transform(split.y_test)

    auth_threshold = float(artifacts.auth_policy["auth_threshold"])

    row_indices = np.arange(len(split.y_test))
    genuine_scores = probabilities[row_indices, true_class_indices]
    genuine_accepts = int((genuine_scores >= auth_threshold).sum())
    genuine_trials = int(len(genuine_scores))
    genuine_rejects = genuine_trials - genuine_accepts
    empirical_frr = genuine_rejects / genuine_trials

    impostor_mask = np.ones_like(probabilities, dtype=bool)
    impostor_mask[row_indices, true_class_indices] = False
    impostor_scores = probabilities[impostor_mask]

    impostor_accepts = int((impostor_scores >= auth_threshold).sum())
    impostor_trials = int(impostor_scores.size)
    impostor_rejects = impostor_trials - impostor_accepts
    empirical_far = impostor_accepts / impostor_trials

    predicted_classes = int(np.unique(np.argmax(probabilities, axis=1)).size)

    return BatchAuthenticationReport(
        genuine_trials=genuine_trials,
        genuine_accepts=genuine_accepts,
        genuine_rejects=genuine_rejects,
        empirical_frr=float(empirical_frr),
        impostor_trials=impostor_trials,
        impostor_accepts=impostor_accepts,
        impostor_rejects=impostor_rejects,
        empirical_far=float(empirical_far),
        auth_threshold=auth_threshold,
        target_far=optional_float(artifacts.auth_policy.get("target_far")),
        policy_actual_far=optional_float(artifacts.auth_policy.get("actual_far")),
        policy_actual_frr=optional_float(artifacts.auth_policy.get("actual_frr")),
        policy_eer=optional_float(artifacts.auth_policy.get("eer")),
        predicted_classes=predicted_classes,
    )


def save_batch_report(report: BatchAuthenticationReport, output_path: Path) -> None:
    """Save batch authentication report as CSV.

    Args:
        report: Batch authentication report.
        output_path: Output CSV path.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    report_df = pd.DataFrame(
        [
            {
                "model_name": MODEL_NAME,
                "genuine_trials": report.genuine_trials,
                "genuine_accepts": report.genuine_accepts,
                "genuine_rejects": report.genuine_rejects,
                "empirical_frr": report.empirical_frr,
                "impostor_trials": report.impostor_trials,
                "impostor_accepts": report.impostor_accepts,
                "impostor_rejects": report.impostor_rejects,
                "empirical_far": report.empirical_far,
                "auth_threshold": report.auth_threshold,
                "target_far": report.target_far,
                "policy_actual_far": report.policy_actual_far,
                "policy_actual_frr": report.policy_actual_frr,
                "policy_eer": report.policy_eer,
                "predicted_classes": report.predicted_classes,
            }
        ]
    )
    report_df.to_csv(output_path, index=False)


def optional_float(value: Any) -> float | None:
    """Convert optional JSON value to float.

    Args:
        value: JSON value.

    Returns:
        Converted float or None.
    """
    if value is None:
        return None

    return float(value)


def build_arg_parser() -> argparse.ArgumentParser:
    """Build command-line argument parser.

    Returns:
        Configured ArgumentParser.
    """
    parser = argparse.ArgumentParser(
        description="Authenticate CMU test samples with saved MLP v2 BatchNorm.",
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
        "--claimed-user-id",
        type=str,
        default=None,
        help=(
            "Claimed user ID. If omitted, the selected sample's true user_id "
            "is used for a genuine authentication attempt."
        ),
    )
    parser.add_argument(
        "--sample-index",
        type=int,
        default=DEFAULT_SAMPLE_INDEX,
        help="Row index inside the test split used when --sample-id is omitted.",
    )
    parser.add_argument(
        "--sample-id",
        type=str,
        default=None,
        help="Optional exact sample_id from test metadata.",
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Run batch authentication smoke-test on the full test split.",
    )
    parser.add_argument(
        "--batch-report-output",
        type=Path,
        default=DEFAULT_BATCH_REPORT_OUTPUT,
        help="Path to output v2 batch authentication report CSV.",
    )

    return parser


def main() -> None:
    """CLI entry point."""
    parser = build_arg_parser()
    args = parser.parse_args()

    artifacts = load_authentication_artifacts(
        model_path=args.model_input,
        scaler_path=args.scaler_input,
        label_encoder_path=args.label_encoder_input,
        auth_policy_path=args.auth_policy_input,
    )
    split = load_unscaled_split(args.input)

    print("Authentication v2 finished.")
    print(f"Model name: {MODEL_NAME}")
    print(f"Input: {args.input}")
    print(f"Model input: {args.model_input}")
    print(f"Scaler input: {args.scaler_input}")
    print(f"Label encoder input: {args.label_encoder_input}")
    print(f"Auth policy input: {args.auth_policy_input}")

    if args.batch:
        report = run_batch_authentication_test(artifacts=artifacts, split=split)
        save_batch_report(report=report, output_path=args.batch_report_output)
        print_batch_authentication_report(report=report)
        print()
        print("V2 batch report saved:")
        print(f"Path: {args.batch_report_output}")
        return

    sample = select_test_sample(
        split=split,
        sample_index=args.sample_index,
        sample_id=args.sample_id,
    )
    result = authenticate_sample(
        artifacts=artifacts,
        sample=sample,
        claimed_user_id=args.claimed_user_id,
    )
    print_single_authentication_result(result)


def print_single_authentication_result(result: AuthenticationResult) -> None:
    """Print one-sample authentication result.

    Args:
        result: One-sample authentication result.
    """
    print()
    print("Authentication request:")
    print(f"Sample index: {result.sample_index}")
    print(f"Sample ID: {result.sample_id}")
    print(f"True user ID: {result.true_user_id}")
    print(f"Claimed user ID: {result.claimed_user_id}")
    print(f"Predicted user ID: {result.predicted_user_id}")
    print()
    print("Authentication decision:")
    print(f"Claimed user score: {result.claimed_user_score:.6f}")
    print(f"Predicted user score: {result.predicted_user_score:.6f}")
    print(f"Auth threshold: {result.auth_threshold:.6f}")
    print(f"Decision: {'ACCEPT' if result.accepted else 'REJECT'}")
    print()
    print("Authentication policy:")
    print(f"Target FAR: {format_optional_float(result.target_far)}")
    print(f"Actual FAR: {format_optional_float(result.actual_far)}")
    print(f"Actual FRR: {format_optional_float(result.actual_frr)}")
    print(f"EER: {format_optional_float(result.eer)}")


def print_batch_authentication_report(report: BatchAuthenticationReport) -> None:
    """Print batch authentication report.

    Args:
        report: Batch authentication report.
    """
    print()
    print("V2 batch authentication test:")
    print(f"Genuine trials: {report.genuine_trials}")
    print(f"Genuine accepts: {report.genuine_accepts}")
    print(f"Genuine rejects: {report.genuine_rejects}")
    print(f"Empirical FRR: {report.empirical_frr:.6f}")
    print(f"Impostor trials: {report.impostor_trials}")
    print(f"Impostor accepts: {report.impostor_accepts}")
    print(f"Impostor rejects: {report.impostor_rejects}")
    print(f"Empirical FAR: {report.empirical_far:.6f}")
    print(f"Auth threshold: {report.auth_threshold:.6f}")
    print(f"Target FAR: {format_optional_float(report.target_far)}")
    print(f"Policy actual FAR: {format_optional_float(report.policy_actual_far)}")
    print(f"Policy actual FRR: {format_optional_float(report.policy_actual_frr)}")
    print(f"Policy EER: {format_optional_float(report.policy_eer)}")
    print(f"Predicted classes: {report.predicted_classes}")


def format_optional_float(value: float | None) -> str:
    """Format optional float for CLI output.

    Args:
        value: Value to format.

    Returns:
        Formatted value or N/A.
    """
    if value is None:
        return "N/A"

    return f"{value:.6f}"


if __name__ == "__main__":
    tf.get_logger().setLevel("ERROR")
    main()
