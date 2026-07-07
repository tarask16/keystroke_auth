"""Training entry point for Keystroke Auth baseline MLP.

Current step:
- load processed CMU features;
- reuse preprocessing pipeline;
- encode user labels;
- train baseline MLP classifier;
- print train/validation/test metrics;
- save trained model and label encoder.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from tensorflow import keras

from src.config import CMU_FEATURES_FILE, RANDOM_SEED
from src.models import (
    DEFAULT_DROPOUT_RATE,
    DEFAULT_HIDDEN_UNITS,
    DEFAULT_LEARNING_RATE,
    build_compiled_mlp_baseline,
)
from src.preprocessing import (
    ScaledTrainValidationTestSplit,
    clean_prepared_dataset,
    create_train_validation_test_split,
    load_processed_dataset,
    prepare_features_and_labels,
    scale_train_validation_test_split,
)


DEFAULT_EPOCHS = 20
DEFAULT_BATCH_SIZE = 128
DEFAULT_EARLY_STOPPING_PATIENCE = 5
DEFAULT_MODEL_OUTPUT = Path(__file__).resolve().parents[1] / "models" / "mlp_baseline.keras"
DEFAULT_LABEL_ENCODER_OUTPUT = Path(__file__).resolve().parents[1] / "models" / "label_encoder.pkl"


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

    The label encoder is fitted only on train labels. Validation and test labels
    are transformed with the fitted encoder.

    Args:
        split: Scaled train/validation/test split.

    Returns:
        Encoded labels and fitted LabelEncoder.

    Raises:
        ValueError: If validation or test contains unknown users.
    """
    label_encoder = LabelEncoder()

    y_train = label_encoder.fit_transform(split.y_train)
    y_validation = label_encoder.transform(split.y_validation)
    y_test = label_encoder.transform(split.y_test)

    validate_encoded_labels(
        y_train=y_train,
        y_validation=y_validation,
        y_test=y_test,
        num_classes=len(label_encoder.classes_),
    )

    return EncodedLabels(
        y_train=y_train,
        y_validation=y_validation,
        y_test=y_test,
        label_encoder=label_encoder,
    )


def validate_encoded_labels(
    y_train: np.ndarray,
    y_validation: np.ndarray,
    y_test: np.ndarray,
    num_classes: int,
) -> None:
    """Validate encoded labels.

    Args:
        y_train: Encoded train labels.
        y_validation: Encoded validation labels.
        y_test: Encoded test labels.
        num_classes: Number of user classes.

    Raises:
        ValueError: If encoded labels are inconsistent.
    """
    if num_classes < 2:
        raise ValueError(f"At least two classes are required, got: {num_classes}")

    for split_name, y in (
        ("train", y_train),
        ("validation", y_validation),
        ("test", y_test),
    ):
        if y.size == 0:
            raise ValueError(f"{split_name} labels are empty.")

        if y.min() < 0 or y.max() >= num_classes:
            raise ValueError(
                f"{split_name} labels are outside valid range: "
                f"min={y.min()}, max={y.max()}, classes={num_classes}"
            )


def train_model(
    split: ScaledTrainValidationTestSplit,
    labels: EncodedLabels,
    hidden_units: int,
    dropout_rate: float,
    learning_rate: float,
    epochs: int,
    batch_size: int,
    early_stopping_patience: int,
) -> tuple[keras.Model, TrainingResult]:
    """Train baseline MLP classifier.

    Args:
        split: Scaled train/validation/test split.
        labels: Encoded user labels.
        hidden_units: Number of neurons in hidden dense layer.
        dropout_rate: Dropout probability.
        learning_rate: Adam learning rate.
        epochs: Maximum number of training epochs.
        batch_size: Training batch size.
        early_stopping_patience: Early stopping patience.

    Returns:
        Trained model and training result.
    """
    input_dim = split.X_train.shape[1]
    num_classes = len(labels.label_encoder.classes_)

    model = build_compiled_mlp_baseline(
        input_dim=input_dim,
        num_classes=num_classes,
        hidden_units=hidden_units,
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

    result = TrainingResult(
        history=history,
        test_loss=float(test_loss),
        test_accuracy=float(test_accuracy),
    )

    return model, result


def get_best_metric(history: keras.callbacks.History, metric_name: str) -> float:
    """Get the best value for a metric from Keras history.

    Args:
        history: Keras training history.
        metric_name: Metric key, for example val_accuracy.

    Returns:
        Best metric value.

    Raises:
        ValueError: If metric is missing.
    """
    values = history.history.get(metric_name)

    if not values:
        raise ValueError(f"Metric not found in training history: {metric_name}")

    return float(max(values))


def get_last_metric(history: keras.callbacks.History, metric_name: str) -> float:
    """Get the last value for a metric from Keras history.

    Args:
        history: Keras training history.
        metric_name: Metric key, for example accuracy.

    Returns:
        Last metric value.

    Raises:
        ValueError: If metric is missing.
    """
    values = history.history.get(metric_name)

    if not values:
        raise ValueError(f"Metric not found in training history: {metric_name}")

    return float(values[-1])


def save_training_artifacts(
    model: keras.Model,
    label_encoder: LabelEncoder,
    model_output: Path,
    label_encoder_output: Path,
) -> None:
    """Save trained model and label encoder.

    Args:
        model: Trained Keras model.
        label_encoder: Fitted LabelEncoder for user_id to class index mapping.
        model_output: Path to .keras model file.
        label_encoder_output: Path to label_encoder.pkl file.
    """
    model_output.parent.mkdir(parents=True, exist_ok=True)
    label_encoder_output.parent.mkdir(parents=True, exist_ok=True)

    model.save(model_output)
    joblib.dump(label_encoder, label_encoder_output)


def build_arg_parser() -> argparse.ArgumentParser:
    """Build command-line argument parser.

    Returns:
        Configured ArgumentParser.
    """
    parser = argparse.ArgumentParser(description="Train Keystroke Auth MLP baseline.")

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
        "--hidden-units",
        type=int,
        default=DEFAULT_HIDDEN_UNITS,
        help="Number of neurons in the hidden dense layer.",
    )
    parser.add_argument(
        "--dropout-rate",
        type=float,
        default=DEFAULT_DROPOUT_RATE,
        help="Dropout probability.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=DEFAULT_LEARNING_RATE,
        help="Adam optimizer learning rate.",
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=DEFAULT_EARLY_STOPPING_PATIENCE,
        help="Early stopping patience by validation accuracy.",
    )
    parser.add_argument(
        "--model-output",
        type=Path,
        default=DEFAULT_MODEL_OUTPUT,
        help="Path to save trained Keras model.",
    )
    parser.add_argument(
        "--label-encoder-output",
        type=Path,
        default=DEFAULT_LABEL_ENCODER_OUTPUT,
        help="Path to save fitted label encoder.",
    )

    return parser


def main() -> None:
    """CLI entry point."""
    parser = build_arg_parser()
    args = parser.parse_args()

    tf.keras.utils.set_random_seed(RANDOM_SEED)

    split = load_scaled_dataset(args.input)
    labels = encode_user_labels(split)

    print("Training dataset prepared.")
    print(f"Input: {args.input}")
    print(f"Train samples: {split.X_train.shape[0]}")
    print(f"Validation samples: {split.X_validation.shape[0]}")
    print(f"Test samples: {split.X_test.shape[0]}")
    print(f"Features: {split.X_train.shape[1]}")
    print(f"Classes: {len(labels.label_encoder.classes_)}")
    print(f"First 5 classes: {labels.label_encoder.classes_[:5].tolist()}")
    print()
    print("Training MLP baseline...")

    model, result = train_model(
        split=split,
        labels=labels,
        hidden_units=args.hidden_units,
        dropout_rate=args.dropout_rate,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        batch_size=args.batch_size,
        early_stopping_patience=args.early_stopping_patience,
    )

    epochs_run = len(result.history.history["loss"])
    best_validation_accuracy = get_best_metric(result.history, "val_accuracy")
    final_train_accuracy = get_last_metric(result.history, "accuracy")
    final_validation_accuracy = get_last_metric(result.history, "val_accuracy")

    print()
    print("Training finished.")
    print(f"Epochs run: {epochs_run}")
    print(f"Best validation accuracy: {best_validation_accuracy:.4f}")
    print(f"Final train accuracy: {final_train_accuracy:.4f}")
    print(f"Final validation accuracy: {final_validation_accuracy:.4f}")
    print(f"Test loss: {result.test_loss:.4f}")
    print(f"Test accuracy: {result.test_accuracy:.4f}")

    save_training_artifacts(
        model=model,
        label_encoder=labels.label_encoder,
        model_output=args.model_output,
        label_encoder_output=args.label_encoder_output,
    )

    print()
    print("Training artifacts saved:")
    print(f"Model path: {args.model_output}")
    print(f"Label encoder path: {args.label_encoder_output}")


if __name__ == "__main__":
    main()
