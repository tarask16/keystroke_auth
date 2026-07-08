"""Authentication CLI for baseline v2 with guarded per-user thresholds.

This module uses the saved v2 BatchNorm model and guarded threshold policy:

    models/mlp_v2_batchnorm.keras
    models/scaler_v2_batchnorm.pkl
    models/label_encoder_v2_batchnorm.pkl
    models/guarded_auth_policy_v2_batchnorm.json

The policy contains:

    thresholds[user_id] -> authentication threshold

Supported modes:
- one-sample authentication;
- batch authentication smoke-test on the full test split.

Outputs:
    reports/authentication_batch_report_v2_guarded.csv
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

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
DEFAULT_GUARDED_POLICY_INPUT = PROJECT_ROOT / "models" / "guarded_auth_policy_v2_batchnorm.json"
DEFAULT_BATCH_REPORT_OUTPUT = (
    PROJECT_ROOT / "reports" / "authentication_batch_report_v2_guarded.csv"
)

DEFAULT_SAMPLE_INDEX = 0


@dataclass(frozen=True)
class GuardedAuthenticationArtifacts:
    """Container for loaded guarded authentication artifacts."""

    model: keras.Model
    scaler: StandardScaler
    label_encoder: LabelEncoder
    guarded_policy: dict[str, Any]
    thresholds: dict[str, float]
    fallback_threshold: float


@dataclass(frozen=True)
class SelectedSample:
    """Container for one selected test sample."""

    X: pd.DataFrame
    metadata: pd.Series
    true_user_id: str
    test_index: int


@dataclass(frozen=True)
class GuardedAuthenticationResult:
    """One-sample guarded authentication result."""

    accepted: bool
    claimed_user_id: str
    true_user_id: str
    predicted_user_id: str
    sample_id: str
    sample_index: int
    claimed_user_score: float
    predicted_user_score: float
    user_threshold: float
    fallback_threshold: float
    threshold_source: str
    policy_name: str


@dataclass(frozen=True)
class GuardedBatchAuthenticationReport:
    """Batch authentication report for guarded policy."""

    model_name: str
    policy_name: str
    genuine_trials: int
    genuine_accepts: int
    genuine_rejects: int
    empirical_frr: float
    impostor_trials: int
    impostor_accepts: int
    impostor_rejects: int
    empirical_far: float
    thresholds_count: int
    fallback_threshold: float
    guarded_applied_users_count: int
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


def load_guarded_artifacts(
    model_path: Path,
    scaler_path: Path,
    label_encoder_path: Path,
    guarded_policy_path: Path,
) -> GuardedAuthenticationArtifacts:
    """Load model, scaler, label encoder and guarded policy.

    Args:
        model_path: Path to Keras model.
        scaler_path: Path to StandardScaler.
        label_encoder_path: Path to LabelEncoder.
        guarded_policy_path: Path to guarded policy JSON.

    Returns:
        Loaded guarded authentication artifacts.

    Raises:
        FileNotFoundError: If any artifact is missing.
        TypeError: If scaler or label encoder type is invalid.
        ValueError: If guarded policy is malformed.
    """
    for path in (model_path, scaler_path, label_encoder_path, guarded_policy_path):
        if not path.exists():
            raise FileNotFoundError(f"Required artifact not found: {path}")

    model = keras.models.load_model(model_path)
    scaler = joblib.load(scaler_path)
    label_encoder = joblib.load(label_encoder_path)
    guarded_policy = json.loads(guarded_policy_path.read_text(encoding="utf-8"))

    if not isinstance(scaler, StandardScaler):
        raise TypeError(f"Expected StandardScaler, got: {type(scaler)!r}")

    if not isinstance(label_encoder, LabelEncoder):
        raise TypeError(f"Expected LabelEncoder, got: {type(label_encoder)!r}")

    thresholds_raw = guarded_policy.get("thresholds")
    if not isinstance(thresholds_raw, dict):
        raise ValueError("Guarded policy must contain thresholds dictionary.")

    fallback_threshold = extract_fallback_threshold(guarded_policy)
    thresholds = {str(user_id): float(threshold) for user_id, threshold in thresholds_raw.items()}

    return GuardedAuthenticationArtifacts(
        model=model,
        scaler=scaler,
        label_encoder=label_encoder,
        guarded_policy=guarded_policy,
        thresholds=thresholds,
        fallback_threshold=fallback_threshold,
    )


def extract_fallback_threshold(guarded_policy: dict[str, Any]) -> float:
    """Extract fallback global threshold from guarded policy.

    Args:
        guarded_policy: Guarded policy dictionary.

    Returns:
        Fallback threshold.

    Raises:
        ValueError: If fallback threshold is absent.
    """
    for key in ("base_global_threshold", "auth_threshold"):
        if key in guarded_policy:
            return float(guarded_policy[key])

    raise ValueError("Guarded policy must contain base_global_threshold or auth_threshold.")


def get_threshold_for_user(
    artifacts: GuardedAuthenticationArtifacts,
    user_id: str,
) -> tuple[float, str]:
    """Get authentication threshold for claimed user.

    Args:
        artifacts: Loaded guarded artifacts.
        user_id: Claimed user ID.

    Returns:
        Threshold and threshold source label.
    """
    if user_id in artifacts.thresholds:
        return artifacts.thresholds[user_id], "guarded_policy.thresholds"

    return artifacts.fallback_threshold, "fallback_global_threshold"


def select_test_sample(
    split: TrainValidationTestSplit,
    sample_index: int,
    sample_id: str | None,
) -> SelectedSample:
    """Select one sample from the test split.

    Args:
        split: Unscaled train/validation/test split.
        sample_index: Fallback row index inside test split.
        sample_id: Optional exact sample ID.

    Returns:
        Selected sample.

    Raises:
        IndexError: If sample_index is invalid.
        ValueError: If sample_id is not found.
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

    return SelectedSample(
        X=split.X_test.iloc[[selected_index]].copy(),
        metadata=split.metadata_test.iloc[selected_index].copy(),
        true_user_id=str(split.y_test.iloc[selected_index]),
        test_index=selected_index,
    )


def predict_probabilities(
    artifacts: GuardedAuthenticationArtifacts,
    X: pd.DataFrame,
) -> np.ndarray:
    """Scale features and predict softmax probabilities.

    Args:
        artifacts: Loaded guarded artifacts.
        X: Unscaled feature matrix.

    Returns:
        Probability matrix.
    """
    X_scaled = artifacts.scaler.transform(X)
    return artifacts.model.predict(
        X_scaled.astype(np.float32),
        verbose=0,
    )


def authenticate_sample(
    artifacts: GuardedAuthenticationArtifacts,
    sample: SelectedSample,
    claimed_user_id: str | None,
) -> GuardedAuthenticationResult:
    """Authenticate one sample with guarded per-user threshold.

    Args:
        artifacts: Loaded guarded artifacts.
        sample: Selected sample.
        claimed_user_id: Claimed user ID. If None, true user is used.

    Returns:
        Guarded authentication result.

    Raises:
        ValueError: If claimed user is unknown.
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

    user_threshold, threshold_source = get_threshold_for_user(
        artifacts=artifacts,
        user_id=effective_claimed_user_id,
    )
    claimed_user_score = float(probabilities[claimed_class_index])
    predicted_user_score = float(probabilities[predicted_class_index])

    return GuardedAuthenticationResult(
        accepted=claimed_user_score >= user_threshold,
        claimed_user_id=effective_claimed_user_id,
        true_user_id=sample.true_user_id,
        predicted_user_id=predicted_user_id,
        sample_id=str(sample.metadata["sample_id"]),
        sample_index=sample.test_index,
        claimed_user_score=claimed_user_score,
        predicted_user_score=predicted_user_score,
        user_threshold=user_threshold,
        fallback_threshold=artifacts.fallback_threshold,
        threshold_source=threshold_source,
        policy_name=str(artifacts.guarded_policy.get("policy_name", POLICY_NAME)),
    )


def run_batch_authentication_test(
    artifacts: GuardedAuthenticationArtifacts,
    split: TrainValidationTestSplit,
) -> GuardedBatchAuthenticationReport:
    """Run batch authentication test with guarded per-user thresholds.

    Genuine trials:
        claim = true user for each test sample.

    Impostor trials:
        claim = every other known user for each test sample.

    Args:
        artifacts: Loaded guarded artifacts.
        split: Unscaled train/validation/test split.

    Returns:
        Guarded batch authentication report.
    """
    probabilities = predict_probabilities(artifacts, split.X_test)
    true_class_indices = artifacts.label_encoder.transform(split.y_test)

    threshold_vector = build_threshold_vector(artifacts)
    row_indices = np.arange(len(split.y_test))

    genuine_scores = probabilities[row_indices, true_class_indices]
    genuine_thresholds = threshold_vector[true_class_indices]
    genuine_accepts = int(np.sum(genuine_scores >= genuine_thresholds))
    genuine_trials = int(genuine_scores.size)
    genuine_rejects = genuine_trials - genuine_accepts

    impostor_mask = np.ones_like(probabilities, dtype=bool)
    impostor_mask[row_indices, true_class_indices] = False

    threshold_matrix = np.broadcast_to(threshold_vector, probabilities.shape)
    impostor_accepts = int(np.sum(probabilities[impostor_mask] >= threshold_matrix[impostor_mask]))
    impostor_trials = int(np.sum(impostor_mask))
    impostor_rejects = impostor_trials - impostor_accepts

    predicted_classes = int(np.unique(np.argmax(probabilities, axis=1)).size)
    applied_users = artifacts.guarded_policy.get("guarded_applied_users", [])

    return GuardedBatchAuthenticationReport(
        model_name=MODEL_NAME,
        policy_name=str(artifacts.guarded_policy.get("policy_name", POLICY_NAME)),
        genuine_trials=genuine_trials,
        genuine_accepts=genuine_accepts,
        genuine_rejects=genuine_rejects,
        empirical_frr=genuine_rejects / genuine_trials,
        impostor_trials=impostor_trials,
        impostor_accepts=impostor_accepts,
        impostor_rejects=impostor_rejects,
        empirical_far=impostor_accepts / impostor_trials,
        thresholds_count=len(artifacts.thresholds),
        fallback_threshold=artifacts.fallback_threshold,
        guarded_applied_users_count=len(applied_users),
        predicted_classes=predicted_classes,
    )


def build_threshold_vector(artifacts: GuardedAuthenticationArtifacts) -> np.ndarray:
    """Build threshold vector aligned with label encoder classes.

    Args:
        artifacts: Loaded guarded artifacts.

    Returns:
        Threshold vector with one threshold per class.
    """
    return np.array(
        [
            get_threshold_for_user(artifacts, str(user_id))[0]
            for user_id in artifacts.label_encoder.classes_
        ],
        dtype=np.float32,
    )


def save_batch_report(
    report: GuardedBatchAuthenticationReport,
    output_path: Path,
) -> None:
    """Save guarded batch report as CSV.

    Args:
        report: Batch report.
        output_path: Output CSV path.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([asdict(report)]).to_csv(output_path, index=False)


def build_arg_parser() -> argparse.ArgumentParser:
    """Build command-line argument parser.

    Returns:
        Configured parser.
    """
    parser = argparse.ArgumentParser(
        description="Authenticate CMU samples with guarded v2 BatchNorm policy.",
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
        "--guarded-policy-input",
        type=Path,
        default=DEFAULT_GUARDED_POLICY_INPUT,
        help="Path to guarded v2 authentication policy JSON.",
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
        help="Path to output guarded batch authentication report CSV.",
    )

    return parser


def main() -> None:
    """CLI entry point."""
    parser = build_arg_parser()
    args = parser.parse_args()

    tf.get_logger().setLevel("ERROR")

    artifacts = load_guarded_artifacts(
        model_path=args.model_input,
        scaler_path=args.scaler_input,
        label_encoder_path=args.label_encoder_input,
        guarded_policy_path=args.guarded_policy_input,
    )
    split = load_unscaled_split(args.input)

    print("Guarded authentication v2 finished.")
    print(f"Model name: {MODEL_NAME}")
    print(f"Policy name: {artifacts.guarded_policy.get('policy_name', POLICY_NAME)}")
    print(f"Input: {args.input}")
    print(f"Model input: {args.model_input}")
    print(f"Scaler input: {args.scaler_input}")
    print(f"Label encoder input: {args.label_encoder_input}")
    print(f"Guarded policy input: {args.guarded_policy_input}")

    if args.batch:
        report = run_batch_authentication_test(artifacts=artifacts, split=split)
        save_batch_report(report=report, output_path=args.batch_report_output)
        print_batch_report(report=report)
        print()
        print("Guarded batch report saved:")
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
    print_single_result(result)


def print_single_result(result: GuardedAuthenticationResult) -> None:
    """Print one-sample guarded authentication result.

    Args:
        result: Authentication result.
    """
    print()
    print("Authentication request:")
    print(f"Sample index: {result.sample_index}")
    print(f"Sample ID: {result.sample_id}")
    print(f"True user ID: {result.true_user_id}")
    print(f"Claimed user ID: {result.claimed_user_id}")
    print(f"Predicted user ID: {result.predicted_user_id}")
    print()
    print("Guarded authentication decision:")
    print(f"Claimed user score: {result.claimed_user_score:.6f}")
    print(f"Predicted user score: {result.predicted_user_score:.6f}")
    print(f"User threshold: {result.user_threshold:.6f}")
    print(f"Fallback threshold: {result.fallback_threshold:.6f}")
    print(f"Threshold source: {result.threshold_source}")
    print(f"Decision: {'ACCEPT' if result.accepted else 'REJECT'}")
    print()
    print("Guarded policy:")
    print(f"Policy name: {result.policy_name}")


def print_batch_report(report: GuardedBatchAuthenticationReport) -> None:
    """Print guarded batch report.

    Args:
        report: Batch report.
    """
    print()
    print("Guarded v2 batch authentication test:")
    print(f"Genuine trials: {report.genuine_trials}")
    print(f"Genuine accepts: {report.genuine_accepts}")
    print(f"Genuine rejects: {report.genuine_rejects}")
    print(f"Empirical FRR: {report.empirical_frr:.6f}")
    print(f"Impostor trials: {report.impostor_trials}")
    print(f"Impostor accepts: {report.impostor_accepts}")
    print(f"Impostor rejects: {report.impostor_rejects}")
    print(f"Empirical FAR: {report.empirical_far:.6f}")
    print(f"Thresholds count: {report.thresholds_count}")
    print(f"Fallback threshold: {report.fallback_threshold:.6f}")
    print(f"Guarded applied users count: {report.guarded_applied_users_count}")
    print(f"Predicted classes: {report.predicted_classes}")


if __name__ == "__main__":
    main()
