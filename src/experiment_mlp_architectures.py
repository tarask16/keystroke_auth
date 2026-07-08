"""Compare several MLP architectures for Keystroke Auth.

This experiment does not overwrite the current baseline model artifacts.
It trains several candidate MLP classifiers on the same processed CMU split and
writes comparison reports to:

    reports/mlp_architecture_comparison.csv
    reports/mlp_architecture_comparison.md
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path

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
DEFAULT_OUTPUT_CSV = PROJECT_ROOT / "reports" / "mlp_architecture_comparison.csv"
DEFAULT_OUTPUT_MD = PROJECT_ROOT / "reports" / "mlp_architecture_comparison.md"

DEFAULT_EPOCHS = 20
DEFAULT_BATCH_SIZE = 128
DEFAULT_EARLY_STOPPING_PATIENCE = 5
DEFAULT_LEARNING_RATE = 0.001
DEFAULT_TARGET_FAR = 0.01


@dataclass(frozen=True)
class ArchitectureSpec:
    """MLP architecture specification."""

    name: str
    hidden_layers: tuple[int, ...]
    dropout_rate: float
    batch_norm: bool


@dataclass(frozen=True)
class EncodedLabels:
    """Container for encoded train/validation/test labels."""

    y_train: np.ndarray
    y_validation: np.ndarray
    y_test: np.ndarray
    label_encoder: LabelEncoder


@dataclass(frozen=True)
class AuthenticationMetrics:
    """Authentication metrics calculated from softmax probabilities."""

    eer: float
    eer_threshold: float
    target_far: float
    threshold_at_target_far: float
    actual_far: float
    actual_frr: float


@dataclass(frozen=True)
class ArchitectureResult:
    """Result of one architecture experiment."""

    architecture: str
    hidden_layers: str
    dropout_rate: float
    batch_norm: bool
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
    threshold_at_target_far: float
    actual_far: float
    actual_frr: float


def get_default_architectures() -> list[ArchitectureSpec]:
    """Return default architecture candidates.

    Returns:
        List of architecture specifications.
    """
    return [
        ArchitectureSpec(
            name="mlp_64",
            hidden_layers=(64,),
            dropout_rate=0.2,
            batch_norm=False,
        ),
        ArchitectureSpec(
            name="mlp_128_64",
            hidden_layers=(128, 64),
            dropout_rate=0.2,
            batch_norm=False,
        ),
        ArchitectureSpec(
            name="mlp_256_128",
            hidden_layers=(256, 128),
            dropout_rate=0.3,
            batch_norm=False,
        ),
        ArchitectureSpec(
            name="mlp_128_64_dropout",
            hidden_layers=(128, 64),
            dropout_rate=0.35,
            batch_norm=False,
        ),
        ArchitectureSpec(
            name="mlp_128_64_batchnorm",
            hidden_layers=(128, 64),
            dropout_rate=0.2,
            batch_norm=True,
        ),
    ]


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


def build_mlp_from_spec(
    spec: ArchitectureSpec,
    input_dim: int,
    num_classes: int,
    learning_rate: float,
) -> keras.Model:
    """Build and compile MLP model from architecture specification.

    Args:
        spec: Architecture specification.
        input_dim: Number of input features.
        num_classes: Number of output classes.
        learning_rate: Adam learning rate.

    Returns:
        Compiled Keras model.
    """
    model_layers: list[keras.layers.Layer] = [layers.Input(shape=(input_dim,), name="features")]

    for layer_index, units in enumerate(spec.hidden_layers, start=1):
        if spec.batch_norm:
            model_layers.extend(
                [
                    layers.Dense(
                        units,
                        use_bias=False,
                        name=f"dense_{layer_index}",
                    ),
                    layers.BatchNormalization(name=f"batch_norm_{layer_index}"),
                    layers.Activation("relu", name=f"relu_{layer_index}"),
                ]
            )
        else:
            model_layers.append(
                layers.Dense(
                    units,
                    activation="relu",
                    name=f"dense_{layer_index}",
                )
            )

        if spec.dropout_rate > 0:
            model_layers.append(
                layers.Dropout(
                    spec.dropout_rate,
                    name=f"dropout_{layer_index}",
                )
            )

    model_layers.append(
        layers.Dense(
            num_classes,
            activation="softmax",
            name="user_classifier",
        )
    )

    model = keras.Sequential(model_layers, name=spec.name)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def train_architecture(
    spec: ArchitectureSpec,
    split: ScaledTrainValidationTestSplit,
    labels: EncodedLabels,
    epochs: int,
    batch_size: int,
    early_stopping_patience: int,
    learning_rate: float,
    target_far: float,
) -> ArchitectureResult:
    """Train and evaluate one architecture.

    Args:
        spec: Architecture specification.
        split: Scaled train/validation/test split.
        labels: Encoded labels.
        epochs: Maximum number of epochs.
        batch_size: Batch size.
        early_stopping_patience: Early stopping patience.
        learning_rate: Adam learning rate.
        target_far: Target FAR for threshold diagnostics.

    Returns:
        Architecture experiment result.
    """
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(RANDOM_SEED)

    model = build_mlp_from_spec(
        spec=spec,
        input_dim=split.X_train.shape[1],
        num_classes=len(labels.label_encoder.classes_),
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
        verbose=0,
    )

    test_loss, test_accuracy = model.evaluate(
        split.X_test.to_numpy(dtype=np.float32),
        labels.y_test,
        verbose=0,
    )

    probabilities = model.predict(
        split.X_test.to_numpy(dtype=np.float32),
        verbose=0,
    )
    auth_metrics = calculate_authentication_metrics(
        probabilities=probabilities,
        true_class_indices=labels.y_test,
        target_far=target_far,
    )

    return ArchitectureResult(
        architecture=spec.name,
        hidden_layers="-".join(str(units) for units in spec.hidden_layers),
        dropout_rate=spec.dropout_rate,
        batch_norm=spec.batch_norm,
        params=int(model.count_params()),
        epochs_run=len(history.history["loss"]),
        best_validation_accuracy=max_metric(history, "val_accuracy"),
        final_train_accuracy=last_metric(history, "accuracy"),
        final_validation_accuracy=last_metric(history, "val_accuracy"),
        test_loss=float(test_loss),
        test_accuracy=float(test_accuracy),
        eer=auth_metrics.eer,
        eer_threshold=auth_metrics.eer_threshold,
        target_far=auth_metrics.target_far,
        threshold_at_target_far=auth_metrics.threshold_at_target_far,
        actual_far=auth_metrics.actual_far,
        actual_frr=auth_metrics.actual_frr,
    )


def calculate_authentication_metrics(
    probabilities: np.ndarray,
    true_class_indices: np.ndarray,
    target_far: float,
) -> AuthenticationMetrics:
    """Calculate FAR/FRR/EER from classifier probabilities.

    Args:
        probabilities: Softmax probability matrix.
        true_class_indices: True class index for each row.
        target_far: Target FAR for operating-point threshold.

    Returns:
        Authentication metrics.
    """
    row_indices = np.arange(len(true_class_indices))
    genuine_scores = probabilities[row_indices, true_class_indices]

    impostor_mask = np.ones_like(probabilities, dtype=bool)
    impostor_mask[row_indices, true_class_indices] = False
    impostor_scores = probabilities[impostor_mask]

    labels = np.concatenate(
        [
            np.ones_like(genuine_scores, dtype=int),
            np.zeros_like(impostor_scores, dtype=int),
        ]
    )
    scores = np.concatenate([genuine_scores, impostor_scores])

    far_values, true_accept_rate_values, thresholds = roc_curve(
        labels,
        scores,
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


def max_metric(history: keras.callbacks.History, metric_name: str) -> float:
    """Get maximum metric value from training history.

    Args:
        history: Keras history.
        metric_name: Metric name.

    Returns:
        Maximum metric value.
    """
    return float(max(history.history[metric_name]))


def last_metric(history: keras.callbacks.History, metric_name: str) -> float:
    """Get last metric value from training history.

    Args:
        history: Keras history.
        metric_name: Metric name.

    Returns:
        Last metric value.
    """
    return float(history.history[metric_name][-1])


def run_architecture_comparison(
    split: ScaledTrainValidationTestSplit,
    labels: EncodedLabels,
    epochs: int,
    batch_size: int,
    early_stopping_patience: int,
    learning_rate: float,
    target_far: float,
) -> pd.DataFrame:
    """Run comparison for default architecture candidates.

    Args:
        split: Scaled train/validation/test split.
        labels: Encoded labels.
        epochs: Maximum epochs.
        batch_size: Batch size.
        early_stopping_patience: Early stopping patience.
        learning_rate: Adam learning rate.
        target_far: Target FAR.

    Returns:
        Result DataFrame.
    """
    results: list[ArchitectureResult] = []

    for spec in get_default_architectures():
        print(f"Training architecture: {spec.name}")

        result = train_architecture(
            spec=spec,
            split=split,
            labels=labels,
            epochs=epochs,
            batch_size=batch_size,
            early_stopping_patience=early_stopping_patience,
            learning_rate=learning_rate,
            target_far=target_far,
        )

        results.append(result)

        print(
            "Result: "
            f"test_accuracy={result.test_accuracy:.4f}, "
            f"eer={result.eer:.6f}, "
            f"far={result.actual_far:.6f}, "
            f"frr={result.actual_frr:.6f}"
        )

    return pd.DataFrame([asdict(result) for result in results])


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    """Convert DataFrame to a simple Markdown table without optional dependencies.

    Args:
        df: Source DataFrame.

    Returns:
        Markdown table string.
    """
    if df.empty:
        return "_No data._"

    columns = list(df.columns)
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"

    rows = []
    for _index, row in df.iterrows():
        values = [format_markdown_cell(row[column]) for column in columns]
        rows.append("| " + " | ".join(values) + " |")

    return "\n".join([header, separator, *rows])


def format_markdown_cell(value: object) -> str:
    """Format one Markdown table cell.

    Args:
        value: Cell value.

    Returns:
        Formatted cell text.
    """
    if isinstance(value, float):
        return f"{value:.6f}"

    return str(value).replace("|", "\\|")


def save_comparison_reports(
    result_df: pd.DataFrame,
    output_csv: Path,
    output_md: Path,
) -> None:
    """Save architecture comparison reports.

    Args:
        result_df: Result DataFrame.
        output_csv: Output CSV path.
        output_md: Output Markdown path.
    """
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)

    result_df.to_csv(output_csv, index=False)

    best_by_eer = result_df.sort_values("eer", ascending=True).iloc[0]
    best_by_test_accuracy = result_df.sort_values("test_accuracy", ascending=False).iloc[0]

    report_lines = [
        "# MLP Architecture Comparison",
        "",
        "## Summary",
        "",
        f"- best by EER: `{best_by_eer['architecture']}`;",
        f"- best EER: `{best_by_eer['eer']:.6f}`;",
        f"- best by test accuracy: `{best_by_test_accuracy['architecture']}`;",
        f"- best test accuracy: `{best_by_test_accuracy['test_accuracy']:.6f}`.",
        "",
        "## Results",
        "",
        dataframe_to_markdown(result_df),
        "",
    ]

    output_md.write_text("\n".join(report_lines), encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    """Build command-line argument parser.

    Returns:
        Configured parser.
    """
    parser = argparse.ArgumentParser(description="Compare MLP architectures.")

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
        help="Maximum number of epochs per architecture.",
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
        "--target-far",
        type=float,
        default=DEFAULT_TARGET_FAR,
        help="Target FAR for operating-point threshold comparison.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=DEFAULT_OUTPUT_CSV,
        help="Path to output comparison CSV.",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=DEFAULT_OUTPUT_MD,
        help="Path to output comparison Markdown report.",
    )

    return parser


def print_final_summary(result_df: pd.DataFrame, output_csv: Path, output_md: Path) -> None:
    """Print final experiment summary.

    Args:
        result_df: Result DataFrame.
        output_csv: Output CSV path.
        output_md: Output Markdown path.
    """
    best_by_eer = result_df.sort_values("eer", ascending=True).iloc[0]
    best_by_test_accuracy = result_df.sort_values("test_accuracy", ascending=False).iloc[0]

    display_columns = [
        "architecture",
        "params",
        "epochs_run",
        "best_validation_accuracy",
        "test_accuracy",
        "eer",
        "actual_far",
        "actual_frr",
        "threshold_at_target_far",
    ]

    print()
    print("Architecture comparison finished.")
    print()
    print("Architecture comparison:")
    print(result_df.loc[:, display_columns].to_string(index=False))
    print()
    print("Best by EER:")
    print(f"Architecture: {best_by_eer['architecture']}")
    print(f"Test accuracy: {best_by_eer['test_accuracy']:.6f}")
    print(f"EER: {best_by_eer['eer']:.6f}")
    print(f"FAR at target: {best_by_eer['actual_far']:.6f}")
    print(f"FRR at target: {best_by_eer['actual_frr']:.6f}")
    print(f"Threshold at target FAR: {best_by_eer['threshold_at_target_far']:.6f}")
    print()
    print("Best by test accuracy:")
    print(f"Architecture: {best_by_test_accuracy['architecture']}")
    print(f"Test accuracy: {best_by_test_accuracy['test_accuracy']:.6f}")
    print(f"EER: {best_by_test_accuracy['eer']:.6f}")
    print()
    print("Reports saved:")
    print(f"CSV path: {output_csv}")
    print(f"Markdown path: {output_md}")


def main() -> None:
    """CLI entry point."""
    parser = build_arg_parser()
    args = parser.parse_args()

    tf.get_logger().setLevel("ERROR")
    tf.keras.utils.set_random_seed(RANDOM_SEED)

    split = load_scaled_dataset(args.input)
    labels = encode_user_labels(split)

    print("Architecture comparison dataset prepared.")
    print(f"Input: {args.input}")
    print(f"Train samples: {split.X_train.shape[0]}")
    print(f"Validation samples: {split.X_validation.shape[0]}")
    print(f"Test samples: {split.X_test.shape[0]}")
    print(f"Features: {split.X_train.shape[1]}")
    print(f"Classes: {len(labels.label_encoder.classes_)}")
    print()

    result_df = run_architecture_comparison(
        split=split,
        labels=labels,
        epochs=args.epochs,
        batch_size=args.batch_size,
        early_stopping_patience=args.early_stopping_patience,
        learning_rate=args.learning_rate,
        target_far=args.target_far,
    )

    save_comparison_reports(
        result_df=result_df,
        output_csv=args.output_csv,
        output_md=args.output_md,
    )
    print_final_summary(
        result_df=result_df,
        output_csv=args.output_csv,
        output_md=args.output_md,
    )


if __name__ == "__main__":
    main()
