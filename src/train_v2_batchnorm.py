"""Train and save baseline v2 MLP with BatchNorm for Keystroke Auth.

This module trains the selected v2 architecture from experiment 5.1:

    Dense(128, no bias) -> BatchNorm -> ReLU -> Dropout(0.2)
    Dense(64, no bias)  -> BatchNorm -> ReLU -> Dropout(0.2)
    Dense(num_users, softmax)

It saves v2 artifacts without overwriting the original baseline:

    models/mlp_v2_batchnorm.keras
    models/scaler_v2_batchnorm.pkl
    models/label_encoder_v2_batchnorm.pkl
    models/auth_policy_v2_batchnorm.json
    reports/v2_batchnorm_training_metrics.csv
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import roc_curve
from sklearn.preprocessing import LabelEncoder
from tensorflow import keras
from tensorflow.keras import layers

from src.config import CMU_FEATURES_FILE, RANDOM_SEED
from src.preprocessing import (
    ScaledTrainValidationTestSplit,
    clean_prepared_dataset,
    create_train_validation_test_split,
    load_processed_dataset,
    prepare_features_and_labels,
    scale_train_validation_test_split,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_NAME = "mlp_v2_batchnorm"
SCORE_TYPE = "softmax_claimed_user_probability"

DEFAULT_EPOCHS = 20
DEFAULT_BATCH_SIZE = 128
DEFAULT_EARLY_STOPPING_PATIENCE = 5
DEFAULT_LEARNING_RATE = 0.001
DEFAULT_DROPOUT_RATE = 0.2
DEFAULT_TARGET_FAR = 0.01

DEFAULT_MODEL_OUTPUT = PROJECT_ROOT / "models" / "mlp_v2_batchnorm.keras"
DEFAULT_SCALER_OUTPUT = PROJECT_ROOT / "models" / "scaler_v2_batchnorm.pkl"
DEFAULT_LABEL_ENCODER_OUTPUT = PROJECT_ROOT / "models" / "label_encoder_v2_batchnorm.pkl"
DEFAULT_AUTH_POLICY_OUTPUT = PROJECT_ROOT / "models" / "auth_policy_v2_batchnorm.json"
DEFAULT_METRICS_OUTPUT = PROJECT_ROOT / "reports" / "v2_batchnorm_training_metrics.csv"


@dataclass(frozen=True)
class EncodedLabels:
    """Container for encoded train/validation/test labels."""

    y_train: np.ndarray
    y_validation: np.ndarray
    y_test: np.ndarray
    label_encoder: LabelEncoder


@dataclass(frozen=True)
class TrainingResult:
    """Container for training results."""

    history: keras.callbacks.History
    test_loss: float
    test_accuracy: float


@dataclass(frozen=True)
class AuthenticationMetrics:
    """Authentication metrics calculated from softmax probabilities."""

    genuine_trials: int
    impostor_trials: int
    eer: float
    eer_threshold: float
    target_far: float
    threshold_at_target_far: float
    actual_far: float
    actual_frr: float


@dataclass(frozen=True)
class V2TrainingSummary:
    """Summary for saved v2 model."""

    model_name: str
    params: int
    epochs_run: int
    best_validation_accuracy: float
    final_train_accuracy: float
    final_validation_accuracy: float
    test_loss: float
    test_accuracy: float
    eer: float
    eer_threshold: float
    target_far: float
    auth_threshold: float
    actual_far: float
    actual_frr: float


def load_scaled_dataset(input_path: Path) -> ScaledTrainValidationTestSplit:
    """Load, clean, split and scale the processed CMU dataset.

    Args:
        input_path: Path to processed CMU features CSV.

    Returns:
        Scaled train/validation/test split.
    """
    df = load_processed_dataset(input_path)
    prepared = prepare_features_and_labels(df)
    cleaned, _cleaning_report = clean_prepared_dataset(prepared)
    split = create_train_validation_test_split(cleaned)
    return scale_train_validation_test_split(split)


def encode_user_labels(split: ScaledTrainValidationTestSplit) -> EncodedLabels:
    """Encode string user IDs into integer class IDs.

    Args:
        split: Scaled train/validation/test split.

    Returns:
        Encoded labels and fitted LabelEncoder.
    """
    label_encoder = LabelEncoder()

    y_train = label_encoder.fit_transform(split.y_train)
    y_validation = label_encoder.transform(split.y_validation)
    y_test = label_encoder.transform(split.y_test)

    return EncodedLabels(
        y_train=y_train,
        y_validation=y_validation,
        y_test=y_test,
        label_encoder=label_encoder,
    )


def build_mlp_v2_batchnorm(
    input_dim: int,
    num_classes: int,
    dropout_rate: float,
    learning_rate: float,
) -> keras.Model:
    """Build and compile v2 BatchNorm MLP.

    Args:
        input_dim: Number of input features.
        num_classes: Number of user classes.
        dropout_rate: Dropout probability.
        learning_rate: Adam learning rate.

    Returns:
        Compiled Keras model.
    """
    model = keras.Sequential(
        [
            layers.Input(shape=(input_dim,), name="features"),
            layers.Dense(128, use_bias=False, name="dense_1"),
            layers.BatchNormalization(name="batch_norm_1"),
            layers.Activation("relu", name="relu_1"),
            layers.Dropout(dropout_rate, name="dropout_1"),
            layers.Dense(64, use_bias=False, name="dense_2"),
            layers.BatchNormalization(name="batch_norm_2"),
            layers.Activation("relu", name="relu_2"),
            layers.Dropout(dropout_rate, name="dropout_2"),
            layers.Dense(num_classes, activation="softmax", name="user_classifier"),
        ],
        name=MODEL_NAME,
    )

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def train_model(
    split: ScaledTrainValidationTestSplit,
    labels: EncodedLabels,
    epochs: int,
    batch_size: int,
    early_stopping_patience: int,
    dropout_rate: float,
    learning_rate: float,
) -> tuple[keras.Model, TrainingResult]:
    """Train v2 BatchNorm MLP.

    Args:
        split: Scaled train/validation/test split.
        labels: Encoded user labels.
        epochs: Maximum number of training epochs.
        batch_size: Training batch size.
        early_stopping_patience: Early stopping patience.
        dropout_rate: Dropout probability.
        learning_rate: Adam learning rate.

    Returns:
        Trained model and training result.
    """
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(RANDOM_SEED)

    model = build_mlp_v2_batchnorm(
        input_dim=split.X_train.shape[1],
        num_classes=len(labels.label_encoder.classes_),
        dropout_rate=dropout_rate,
        learning_rate=learning_rate,
    )

    callbacks: list[keras.callbacks.Callback] = [
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=early_stopping_patience,
            restore_best_weights=True,
            mode="max",
        )
    ]

    history = model.fit(
        split.X_train.to_numpy(dtype=np.float32),
        labels.y_train,
        validation_data=(
            split.X_validation.to_numpy(dtype=np.float32),
            labels.y_validation,
        ),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=2,
    )

    test_loss, test_accuracy = model.evaluate(
        split.X_test.to_numpy(dtype=np.float32),
        labels.y_test,
        verbose=0,
    )

    return model, TrainingResult(
        history=history,
        test_loss=float(test_loss),
        test_accuracy=float(test_accuracy),
    )


def calculate_authentication_metrics(
    model: keras.Model,
    split: ScaledTrainValidationTestSplit,
    labels: EncodedLabels,
    target_far: float,
) -> AuthenticationMetrics:
    """Calculate FAR/FRR/EER for the trained model.

    Args:
        model: Trained Keras model.
        split: Scaled train/validation/test split.
        labels: Encoded user labels.
        target_far: Target FAR for operating threshold.

    Returns:
        Authentication metrics.
    """
    probabilities = model.predict(
        split.X_test.to_numpy(dtype=np.float32),
        verbose=0,
    )

    row_indices = np.arange(len(labels.y_test))
    genuine_scores = probabilities[row_indices, labels.y_test]

    impostor_mask = np.ones_like(probabilities, dtype=bool)
    impostor_mask[row_indices, labels.y_test] = False
    impostor_scores = probabilities[impostor_mask]

    binary_labels = np.concatenate(
        [
            np.ones_like(genuine_scores, dtype=int),
            np.zeros_like(impostor_scores, dtype=int),
        ]
    )
    binary_scores = np.concatenate([genuine_scores, impostor_scores])

    far_values, true_accept_rate_values, thresholds = roc_curve(
        binary_labels,
        binary_scores,
        pos_label=1,
    )
    frr_values = 1.0 - true_accept_rate_values

    eer_index = int(np.argmin(np.abs(far_values - frr_values)))
    eer = float((far_values[eer_index] + frr_values[eer_index]) / 2.0)
    eer_threshold = float(thresholds[eer_index])

    threshold_at_target_far = select_threshold_for_target_far(
        impostor_scores=impostor_scores,
        target_far=target_far,
    )
    actual_far = float(np.mean(impostor_scores >= threshold_at_target_far))
    actual_frr = float(np.mean(genuine_scores < threshold_at_target_far))

    return AuthenticationMetrics(
        genuine_trials=int(genuine_scores.size),
        impostor_trials=int(impostor_scores.size),
        eer=eer,
        eer_threshold=eer_threshold,
        target_far=target_far,
        threshold_at_target_far=threshold_at_target_far,
        actual_far=actual_far,
        actual_frr=actual_frr,
    )


def select_threshold_for_target_far(
    impostor_scores: np.ndarray,
    target_far: float,
) -> float:
    """Select threshold for a target FAR.

    Args:
        impostor_scores: Scores for impostor claims.
        target_far: Desired FAR.

    Returns:
        Threshold value.
    """
    if not 0 < target_far < 1:
        raise ValueError(f"target_far must be between 0 and 1, got: {target_far}")

    sorted_scores = np.sort(impostor_scores)
    quantile_index = int(np.ceil((1.0 - target_far) * len(sorted_scores))) - 1
    quantile_index = max(0, min(quantile_index, len(sorted_scores) - 1))

    return float(sorted_scores[quantile_index])


def get_best_metric(history: keras.callbacks.History, metric_name: str) -> float:
    """Get the best metric value from Keras history.

    Args:
        history: Keras training history.
        metric_name: Metric name.

    Returns:
        Best metric value.
    """
    return float(max(history.history[metric_name]))


def get_last_metric(history: keras.callbacks.History, metric_name: str) -> float:
    """Get the last metric value from Keras history.

    Args:
        history: Keras training history.
        metric_name: Metric name.

    Returns:
        Last metric value.
    """
    return float(history.history[metric_name][-1])


def build_auth_policy(
    auth_metrics: AuthenticationMetrics,
    summary: V2TrainingSummary,
) -> dict[str, Any]:
    """Build authentication policy dictionary.

    Args:
        auth_metrics: Authentication metrics.
        summary: Training summary.

    Returns:
        Authentication policy data.
    """
    return {
        "model_name": MODEL_NAME,
        "score_type": SCORE_TYPE,
        "auth_threshold": auth_metrics.threshold_at_target_far,
        "target_far": auth_metrics.target_far,
        "actual_far": auth_metrics.actual_far,
        "actual_frr": auth_metrics.actual_frr,
        "eer": auth_metrics.eer,
        "eer_threshold": auth_metrics.eer_threshold,
        "test_accuracy": summary.test_accuracy,
        "genuine_trials": auth_metrics.genuine_trials,
        "impostor_trials": auth_metrics.impostor_trials,
    }


def save_v2_artifacts(
    model: keras.Model,
    split: ScaledTrainValidationTestSplit,
    labels: EncodedLabels,
    auth_policy: dict[str, Any],
    summary: V2TrainingSummary,
    model_output: Path,
    scaler_output: Path,
    label_encoder_output: Path,
    auth_policy_output: Path,
    metrics_output: Path,
) -> None:
    """Save v2 model, scaler, label encoder, auth policy and metrics.

    Args:
        model: Trained Keras model.
        split: Scaled train/validation/test split.
        labels: Encoded labels.
        auth_policy: Authentication policy dictionary.
        summary: Training summary.
        model_output: Output path for Keras model.
        scaler_output: Output path for StandardScaler.
        label_encoder_output: Output path for LabelEncoder.
        auth_policy_output: Output path for auth policy JSON.
        metrics_output: Output path for training metrics CSV.
    """
    for output_path in (
        model_output,
        scaler_output,
        label_encoder_output,
        auth_policy_output,
        metrics_output,
    ):
        output_path.parent.mkdir(parents=True, exist_ok=True)

    model.save(model_output)
    joblib.dump(split.scaler, scaler_output)
    joblib.dump(labels.label_encoder, label_encoder_output)

    auth_policy_output.write_text(
        json.dumps(auth_policy, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    pd.DataFrame([asdict(summary)]).to_csv(metrics_output, index=False)


def build_arg_parser() -> argparse.ArgumentParser:
    """Build command-line parser.

    Returns:
        Configured parser.
    """
    parser = argparse.ArgumentParser(
        description="Train and save Keystroke Auth MLP v2 BatchNorm model."
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=CMU_FEATURES_FILE,
        help="Path to processed CMU features CSV.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS,
        help="Maximum number of training epochs.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Training batch size.",
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=DEFAULT_EARLY_STOPPING_PATIENCE,
        help="Early stopping patience by validation accuracy.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=DEFAULT_LEARNING_RATE,
        help="Adam learning rate.",
    )
    parser.add_argument(
        "--dropout-rate",
        type=float,
        default=DEFAULT_DROPOUT_RATE,
        help="Dropout probability.",
    )
    parser.add_argument(
        "--target-far",
        type=float,
        default=DEFAULT_TARGET_FAR,
        help="Target FAR for authentication policy.",
    )
    parser.add_argument(
        "--model-output",
        type=Path,
        default=DEFAULT_MODEL_OUTPUT,
        help="Path to save v2 Keras model.",
    )
    parser.add_argument(
        "--scaler-output",
        type=Path,
        default=DEFAULT_SCALER_OUTPUT,
        help="Path to save v2 StandardScaler.",
    )
    parser.add_argument(
        "--label-encoder-output",
        type=Path,
        default=DEFAULT_LABEL_ENCODER_OUTPUT,
        help="Path to save v2 LabelEncoder.",
    )
    parser.add_argument(
        "--auth-policy-output",
        type=Path,
        default=DEFAULT_AUTH_POLICY_OUTPUT,
        help="Path to save v2 authentication policy JSON.",
    )
    parser.add_argument(
        "--metrics-output",
        type=Path,
        default=DEFAULT_METRICS_OUTPUT,
        help="Path to save v2 training metrics CSV.",
    )

    return parser


def main() -> None:
    """CLI entry point."""
    parser = build_arg_parser()
    args = parser.parse_args()

    tf.get_logger().setLevel("ERROR")
    tf.keras.utils.set_random_seed(RANDOM_SEED)

    split = load_scaled_dataset(args.input)
    labels = encode_user_labels(split)

    print("Training dataset prepared for baseline v2.")
    print(f"Input: {args.input}")
    print(f"Train samples: {split.X_train.shape[0]}")
    print(f"Validation samples: {split.X_validation.shape[0]}")
    print(f"Test samples: {split.X_test.shape[0]}")
    print(f"Features: {split.X_train.shape[1]}")
    print(f"Classes: {len(labels.label_encoder.classes_)}")
    print(f"Model name: {MODEL_NAME}")
    print()
    print("Training MLP v2 BatchNorm...")

    model, result = train_model(
        split=split,
        labels=labels,
        epochs=args.epochs,
        batch_size=args.batch_size,
        early_stopping_patience=args.early_stopping_patience,
        dropout_rate=args.dropout_rate,
        learning_rate=args.learning_rate,
    )

    auth_metrics = calculate_authentication_metrics(
        model=model,
        split=split,
        labels=labels,
        target_far=args.target_far,
    )

    epochs_run = len(result.history.history["loss"])
    summary = V2TrainingSummary(
        model_name=MODEL_NAME,
        params=int(model.count_params()),
        epochs_run=epochs_run,
        best_validation_accuracy=get_best_metric(result.history, "val_accuracy"),
        final_train_accuracy=get_last_metric(result.history, "accuracy"),
        final_validation_accuracy=get_last_metric(result.history, "val_accuracy"),
        test_loss=result.test_loss,
        test_accuracy=result.test_accuracy,
        eer=auth_metrics.eer,
        eer_threshold=auth_metrics.eer_threshold,
        target_far=auth_metrics.target_far,
        auth_threshold=auth_metrics.threshold_at_target_far,
        actual_far=auth_metrics.actual_far,
        actual_frr=auth_metrics.actual_frr,
    )
    auth_policy = build_auth_policy(auth_metrics=auth_metrics, summary=summary)

    save_v2_artifacts(
        model=model,
        split=split,
        labels=labels,
        auth_policy=auth_policy,
        summary=summary,
        model_output=args.model_output,
        scaler_output=args.scaler_output,
        label_encoder_output=args.label_encoder_output,
        auth_policy_output=args.auth_policy_output,
        metrics_output=args.metrics_output,
    )

    print()
    print("Training v2 finished.")
    print(f"Epochs run: {summary.epochs_run}")
    print(f"Params: {summary.params}")
    print(f"Best validation accuracy: {summary.best_validation_accuracy:.6f}")
    print(f"Final train accuracy: {summary.final_train_accuracy:.6f}")
    print(f"Final validation accuracy: {summary.final_validation_accuracy:.6f}")
    print(f"Test loss: {summary.test_loss:.6f}")
    print(f"Test accuracy: {summary.test_accuracy:.6f}")
    print(f"EER: {summary.eer:.6f}")
    print(f"EER threshold: {summary.eer_threshold:.6f}")
    print(f"Target FAR: {summary.target_far:.6f}")
    print(f"Auth threshold: {summary.auth_threshold:.6f}")
    print(f"Actual FAR: {summary.actual_far:.6f}")
    print(f"Actual FRR: {summary.actual_frr:.6f}")

    print()
    print("V2 artifacts saved:")
    print(f"Model path: {args.model_output}")
    print(f"Scaler path: {args.scaler_output}")
    print(f"Label encoder path: {args.label_encoder_output}")
    print(f"Auth policy path: {args.auth_policy_output}")
    print(f"Metrics path: {args.metrics_output}")


if __name__ == "__main__":
    main()
