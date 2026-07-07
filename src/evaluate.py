"""Evaluation entry point for Keystroke Auth baseline MLP.

Current step:
- load processed CMU features;
- reuse preprocessing pipeline;
- load saved scaler, model and label encoder;
- evaluate saved model on validation and test splits;
- print extended classification diagnostics;
- calculate authentication-oriented FAR, FRR and EER diagnostics;
- save reports to CSV.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow import keras

from src.config import CMU_FEATURES_FILE
from src.preprocessing import (
    DEFAULT_SCALER_OUTPUT,
    ScaledTrainValidationTestSplit,
    TrainValidationTestSplit,
    clean_prepared_dataset,
    create_train_validation_test_split,
    load_processed_dataset,
    prepare_features_and_labels,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_INPUT = PROJECT_ROOT / "models" / "mlp_baseline.keras"
DEFAULT_LABEL_ENCODER_INPUT = PROJECT_ROOT / "models" / "label_encoder.pkl"
DEFAULT_CLASSIFICATION_REPORT_OUTPUT = PROJECT_ROOT / "reports" / "classification_report.csv"
DEFAULT_CONFUSION_MATRIX_OUTPUT = PROJECT_ROOT / "reports" / "confusion_matrix.csv"
DEFAULT_AUTHENTICATION_METRICS_OUTPUT = PROJECT_ROOT / "reports" / "authentication_metrics.csv"
DEFAULT_AUTHENTICATION_SCORE_SUMMARY_OUTPUT = (
    PROJECT_ROOT / "reports" / "authentication_score_summary.csv"
)
DEFAULT_TOP_N = 10
DEFAULT_THRESHOLD_POINTS = 1001
REFERENCE_THRESHOLD = 0.5


@dataclass(frozen=True)
class EvaluationArtifacts:
    """Container for loaded evaluation artifacts."""

    model: keras.Model
    scaler: StandardScaler
    label_encoder: LabelEncoder


@dataclass(frozen=True)
class EncodedEvaluationLabels:
    """Container for encoded validation/test labels."""

    y_validation: np.ndarray
    y_test: np.ndarray


@dataclass(frozen=True)
class EvaluationResult:
    """Container for evaluation metrics and predictions."""

    validation_loss: float
    validation_accuracy: float
    test_loss: float
    test_accuracy: float
    y_test_probabilities: np.ndarray
    y_test_predicted: np.ndarray
    y_test_predicted_user_ids: np.ndarray


@dataclass(frozen=True)
class ClassificationDiagnostics:
    """Container for classification diagnostics."""

    report: pd.DataFrame
    confusion_matrix: pd.DataFrame
    worst_users_by_recall: pd.DataFrame
    top_confusions: pd.DataFrame


@dataclass(frozen=True)
class AuthenticationDiagnostics:
    """Container for authentication-oriented diagnostics."""

    genuine_trials: int
    impostor_trials: int
    eer: float
    eer_threshold: float
    far_at_eer: float
    frr_at_eer: float
    far_at_reference_threshold: float
    frr_at_reference_threshold: float
    threshold_table: pd.DataFrame
    score_summary: pd.DataFrame


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


def load_evaluation_artifacts(
    model_path: Path,
    scaler_path: Path,
    label_encoder_path: Path,
) -> EvaluationArtifacts:
    """Load saved model, scaler and label encoder.

    Args:
        model_path: Path to saved Keras model.
        scaler_path: Path to saved StandardScaler.
        label_encoder_path: Path to saved LabelEncoder.

    Returns:
        Loaded evaluation artifacts.

    Raises:
        FileNotFoundError: If any artifact is missing.
        TypeError: If scaler or label encoder has an unexpected type.
    """
    for path in (model_path, scaler_path, label_encoder_path):
        if not path.exists():
            raise FileNotFoundError(f"Required evaluation artifact not found: {path}")

    model = keras.models.load_model(model_path)
    scaler = joblib.load(scaler_path)
    label_encoder = joblib.load(label_encoder_path)

    if not isinstance(scaler, StandardScaler):
        raise TypeError(f"Expected StandardScaler, got: {type(scaler)!r}")

    if not isinstance(label_encoder, LabelEncoder):
        raise TypeError(f"Expected LabelEncoder, got: {type(label_encoder)!r}")

    return EvaluationArtifacts(
        model=model,
        scaler=scaler,
        label_encoder=label_encoder,
    )


def scale_split_with_loaded_scaler(
    split: TrainValidationTestSplit,
    scaler: StandardScaler,
) -> ScaledTrainValidationTestSplit:
    """Scale split with an already fitted StandardScaler.

    Args:
        split: Unscaled train/validation/test split.
        scaler: Fitted StandardScaler loaded from models/scaler.pkl.

    Returns:
        Scaled train/validation/test split.
    """
    X_train_scaled = pd.DataFrame(
        scaler.transform(split.X_train),
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

    return ScaledTrainValidationTestSplit(
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


def encode_evaluation_labels(
    split: ScaledTrainValidationTestSplit,
    label_encoder: LabelEncoder,
) -> EncodedEvaluationLabels:
    """Encode validation/test user labels with saved LabelEncoder.

    Args:
        split: Scaled train/validation/test split.
        label_encoder: Fitted LabelEncoder loaded from models/label_encoder.pkl.

    Returns:
        Encoded validation/test labels.
    """
    return EncodedEvaluationLabels(
        y_validation=label_encoder.transform(split.y_validation),
        y_test=label_encoder.transform(split.y_test),
    )


def evaluate_saved_model(
    artifacts: EvaluationArtifacts,
    split: ScaledTrainValidationTestSplit,
    labels: EncodedEvaluationLabels,
) -> EvaluationResult:
    """Evaluate saved model on validation and test splits.

    Args:
        artifacts: Loaded model, scaler and label encoder.
        split: Scaled train/validation/test split.
        labels: Encoded validation/test labels.

    Returns:
        Evaluation metrics and test predictions.
    """
    validation_loss, validation_accuracy = artifacts.model.evaluate(
        split.X_validation.to_numpy(dtype=np.float32),
        labels.y_validation,
        verbose=0,
    )
    test_loss, test_accuracy = artifacts.model.evaluate(
        split.X_test.to_numpy(dtype=np.float32),
        labels.y_test,
        verbose=0,
    )

    y_test_probabilities = artifacts.model.predict(
        split.X_test.to_numpy(dtype=np.float32),
        verbose=0,
    )
    y_test_predicted = np.argmax(y_test_probabilities, axis=1)
    y_test_predicted_user_ids = artifacts.label_encoder.inverse_transform(y_test_predicted)

    return EvaluationResult(
        validation_loss=float(validation_loss),
        validation_accuracy=float(validation_accuracy),
        test_loss=float(test_loss),
        test_accuracy=float(test_accuracy),
        y_test_probabilities=y_test_probabilities,
        y_test_predicted=y_test_predicted,
        y_test_predicted_user_ids=y_test_predicted_user_ids,
    )


def build_classification_diagnostics(
    y_true: np.ndarray,
    y_predicted: np.ndarray,
    class_names: np.ndarray,
    top_n: int,
) -> ClassificationDiagnostics:
    """Build classification report, confusion matrix and error summaries.

    Args:
        y_true: Encoded true test labels.
        y_predicted: Encoded predicted test labels.
        class_names: User IDs in encoded class order.
        top_n: Number of weakest users and confusions to display.

    Returns:
        Classification diagnostics container.
    """
    labels = np.arange(len(class_names))

    report_dict = classification_report(
        y_true,
        y_predicted,
        labels=labels,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    report = pd.DataFrame(report_dict).transpose()

    matrix = confusion_matrix(y_true, y_predicted, labels=labels)
    confusion = pd.DataFrame(matrix, index=class_names, columns=class_names)

    user_rows = report.loc[class_names, ["precision", "recall", "f1-score", "support"]]
    worst_users = user_rows.sort_values(
        by=["recall", "f1-score", "support"],
        ascending=[True, True, False],
    ).head(top_n)

    top_confusions = build_top_confusions(
        confusion=confusion,
        class_names=class_names,
        top_n=top_n,
    )

    return ClassificationDiagnostics(
        report=report,
        confusion_matrix=confusion,
        worst_users_by_recall=worst_users,
        top_confusions=top_confusions,
    )


def build_top_confusions(
    confusion: pd.DataFrame,
    class_names: np.ndarray,
    top_n: int,
) -> pd.DataFrame:
    """Build table of most frequent wrong true/predicted user pairs.

    Args:
        confusion: Confusion matrix DataFrame.
        class_names: User IDs in encoded class order.
        top_n: Number of rows to return.

    Returns:
        DataFrame with true_user_id, predicted_user_id and count.
    """
    rows: list[dict[str, object]] = []

    for true_user_id in class_names:
        for predicted_user_id in class_names:
            if true_user_id == predicted_user_id:
                continue

            count = int(confusion.loc[true_user_id, predicted_user_id])
            if count > 0:
                rows.append(
                    {
                        "true_user_id": true_user_id,
                        "predicted_user_id": predicted_user_id,
                        "count": count,
                    }
                )

    if not rows:
        return pd.DataFrame(columns=["true_user_id", "predicted_user_id", "count"])

    return pd.DataFrame(rows).sort_values(by="count", ascending=False).head(top_n)


def build_authentication_diagnostics(
    y_true: np.ndarray,
    y_probabilities: np.ndarray,
    threshold_points: int,
) -> AuthenticationDiagnostics:
    """Build FAR, FRR and EER diagnostics from softmax scores.

    For this baseline, authentication score is the model probability assigned to
    the claimed user class. Genuine trials use the true user as the claimed user.
    Impostor trials use each test sample against all other user classes.

    Args:
        y_true: Encoded true test labels.
        y_probabilities: Softmax probabilities with shape (samples, classes).
        threshold_points: Number of thresholds between 0 and 1.

    Returns:
        Authentication diagnostics container.

    Raises:
        ValueError: If inputs have invalid shape.
    """
    validate_authentication_inputs(y_true, y_probabilities, threshold_points)

    row_indices = np.arange(y_true.shape[0])
    genuine_scores = y_probabilities[row_indices, y_true]

    impostor_mask = np.ones_like(y_probabilities, dtype=bool)
    impostor_mask[row_indices, y_true] = False
    impostor_scores = y_probabilities[impostor_mask]

    thresholds = np.linspace(0.0, 1.0, threshold_points)
    threshold_table = calculate_threshold_metrics(
        genuine_scores=genuine_scores,
        impostor_scores=impostor_scores,
        thresholds=thresholds,
    )

    eer_row_index = (threshold_table["far"] - threshold_table["frr"]).abs().idxmin()
    eer_row = threshold_table.loc[eer_row_index]

    reference_row = calculate_single_threshold_metrics(
        genuine_scores=genuine_scores,
        impostor_scores=impostor_scores,
        threshold=REFERENCE_THRESHOLD,
    )

    score_summary = build_score_summary(
        genuine_scores=genuine_scores,
        impostor_scores=impostor_scores,
    )

    return AuthenticationDiagnostics(
        genuine_trials=int(genuine_scores.shape[0]),
        impostor_trials=int(impostor_scores.shape[0]),
        eer=float((eer_row["far"] + eer_row["frr"]) / 2.0),
        eer_threshold=float(eer_row["threshold"]),
        far_at_eer=float(eer_row["far"]),
        frr_at_eer=float(eer_row["frr"]),
        far_at_reference_threshold=float(reference_row["far"]),
        frr_at_reference_threshold=float(reference_row["frr"]),
        threshold_table=threshold_table,
        score_summary=score_summary,
    )


def validate_authentication_inputs(
    y_true: np.ndarray,
    y_probabilities: np.ndarray,
    threshold_points: int,
) -> None:
    """Validate inputs for authentication diagnostics.

    Args:
        y_true: Encoded true labels.
        y_probabilities: Softmax probabilities.
        threshold_points: Number of thresholds.

    Raises:
        ValueError: If values are inconsistent.
    """
    if y_probabilities.ndim != 2:
        raise ValueError(f"y_probabilities must be a 2D array, got shape: {y_probabilities.shape}")

    if y_true.ndim != 1:
        raise ValueError(f"y_true must be a 1D array, got shape: {y_true.shape}")

    if y_true.shape[0] != y_probabilities.shape[0]:
        raise ValueError(
            "y_true and y_probabilities have inconsistent sample counts: "
            f"{y_true.shape[0]} != {y_probabilities.shape[0]}"
        )

    if threshold_points < 2:
        raise ValueError(f"threshold_points must be >= 2, got: {threshold_points}")


def calculate_threshold_metrics(
    genuine_scores: np.ndarray,
    impostor_scores: np.ndarray,
    thresholds: np.ndarray,
) -> pd.DataFrame:
    """Calculate FAR and FRR for a set of thresholds.

    Args:
        genuine_scores: Scores for correct user claims.
        impostor_scores: Scores for wrong user claims.
        thresholds: Decision thresholds.

    Returns:
        DataFrame with threshold, FAR and FRR.
    """
    genuine_sorted = np.sort(genuine_scores)
    impostor_sorted = np.sort(impostor_scores)

    false_rejects = np.searchsorted(genuine_sorted, thresholds, side="left")
    false_accepts = impostor_scores.shape[0] - np.searchsorted(
        impostor_sorted,
        thresholds,
        side="left",
    )

    frr = false_rejects / genuine_scores.shape[0]
    far = false_accepts / impostor_scores.shape[0]

    return pd.DataFrame(
        {
            "threshold": thresholds,
            "far": far,
            "frr": frr,
            "abs_far_frr_delta": np.abs(far - frr),
        }
    )


def calculate_single_threshold_metrics(
    genuine_scores: np.ndarray,
    impostor_scores: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    """Calculate FAR and FRR for one threshold.

    Args:
        genuine_scores: Scores for correct user claims.
        impostor_scores: Scores for wrong user claims.
        threshold: Decision threshold.

    Returns:
        Dictionary with threshold, FAR and FRR.
    """
    far = float(np.mean(impostor_scores >= threshold))
    frr = float(np.mean(genuine_scores < threshold))

    return {
        "threshold": threshold,
        "far": far,
        "frr": frr,
    }


def build_score_summary(
    genuine_scores: np.ndarray,
    impostor_scores: np.ndarray,
) -> pd.DataFrame:
    """Build score distribution summary for genuine and impostor trials.

    Args:
        genuine_scores: Scores for correct user claims.
        impostor_scores: Scores for wrong user claims.

    Returns:
        DataFrame with distribution statistics.
    """
    return pd.DataFrame(
        [
            summarize_scores("genuine", genuine_scores),
            summarize_scores("impostor", impostor_scores),
        ]
    )


def summarize_scores(score_type: str, scores: np.ndarray) -> dict[str, float | str]:
    """Summarize score distribution.

    Args:
        score_type: Score group name.
        scores: Score values.

    Returns:
        Summary row.
    """
    return {
        "score_type": score_type,
        "count": float(scores.shape[0]),
        "min": float(np.min(scores)),
        "p01": float(np.quantile(scores, 0.01)),
        "p05": float(np.quantile(scores, 0.05)),
        "median": float(np.median(scores)),
        "mean": float(np.mean(scores)),
        "p95": float(np.quantile(scores, 0.95)),
        "p99": float(np.quantile(scores, 0.99)),
        "max": float(np.max(scores)),
    }


def save_diagnostics(
    classification_diagnostics: ClassificationDiagnostics,
    authentication_diagnostics: AuthenticationDiagnostics,
    classification_report_output: Path,
    confusion_matrix_output: Path,
    authentication_metrics_output: Path,
    authentication_score_summary_output: Path,
) -> None:
    """Save classification and authentication diagnostics to CSV files.

    Args:
        classification_diagnostics: Classification diagnostics.
        authentication_diagnostics: Authentication diagnostics.
        classification_report_output: Output CSV path for classification report.
        confusion_matrix_output: Output CSV path for confusion matrix.
        authentication_metrics_output: Output CSV path for FAR/FRR table.
        authentication_score_summary_output: Output CSV path for score summary.
    """
    for output_path in (
        classification_report_output,
        confusion_matrix_output,
        authentication_metrics_output,
        authentication_score_summary_output,
    ):
        output_path.parent.mkdir(parents=True, exist_ok=True)

    classification_diagnostics.report.to_csv(classification_report_output, index=True)
    classification_diagnostics.confusion_matrix.to_csv(
        confusion_matrix_output,
        index=True,
    )
    authentication_diagnostics.threshold_table.to_csv(
        authentication_metrics_output,
        index=False,
    )
    authentication_diagnostics.score_summary.to_csv(
        authentication_score_summary_output,
        index=False,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    """Build command-line argument parser.

    Returns:
        Configured ArgumentParser.
    """
    parser = argparse.ArgumentParser(description="Evaluate Keystroke Auth MLP baseline.")

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
        help="Path to saved Keras model.",
    )
    parser.add_argument(
        "--scaler-input",
        type=Path,
        default=DEFAULT_SCALER_OUTPUT,
        help="Path to saved StandardScaler.",
    )
    parser.add_argument(
        "--label-encoder-input",
        type=Path,
        default=DEFAULT_LABEL_ENCODER_INPUT,
        help="Path to saved LabelEncoder.",
    )
    parser.add_argument(
        "--classification-report-output",
        type=Path,
        default=DEFAULT_CLASSIFICATION_REPORT_OUTPUT,
        help="Path to output classification report CSV.",
    )
    parser.add_argument(
        "--confusion-matrix-output",
        type=Path,
        default=DEFAULT_CONFUSION_MATRIX_OUTPUT,
        help="Path to output confusion matrix CSV.",
    )
    parser.add_argument(
        "--authentication-metrics-output",
        type=Path,
        default=DEFAULT_AUTHENTICATION_METRICS_OUTPUT,
        help="Path to output authentication metrics CSV.",
    )
    parser.add_argument(
        "--authentication-score-summary-output",
        type=Path,
        default=DEFAULT_AUTHENTICATION_SCORE_SUMMARY_OUTPUT,
        help="Path to output authentication score summary CSV.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=DEFAULT_TOP_N,
        help="Number of weakest users and confusions to print.",
    )
    parser.add_argument(
        "--threshold-points",
        type=int,
        default=DEFAULT_THRESHOLD_POINTS,
        help="Number of threshold points for FAR/FRR/EER calculation.",
    )

    return parser


def main() -> None:
    """CLI entry point."""
    parser = build_arg_parser()
    args = parser.parse_args()

    artifacts = load_evaluation_artifacts(
        model_path=args.model_input,
        scaler_path=args.scaler_input,
        label_encoder_path=args.label_encoder_input,
    )
    unscaled_split = load_unscaled_split(args.input)
    split = scale_split_with_loaded_scaler(unscaled_split, artifacts.scaler)
    labels = encode_evaluation_labels(split, artifacts.label_encoder)

    print("Evaluation dataset prepared.")
    print(f"Input: {args.input}")
    print(f"Model input: {args.model_input}")
    print(f"Scaler input: {args.scaler_input}")
    print(f"Label encoder input: {args.label_encoder_input}")
    print(f"Validation samples: {split.X_validation.shape[0]}")
    print(f"Test samples: {split.X_test.shape[0]}")
    print(f"Features: {split.X_test.shape[1]}")
    print(f"Classes: {len(artifacts.label_encoder.classes_)}")
    print()
    print("Evaluating saved MLP baseline...")

    result = evaluate_saved_model(
        artifacts=artifacts,
        split=split,
        labels=labels,
    )
    classification_diagnostics = build_classification_diagnostics(
        y_true=labels.y_test,
        y_predicted=result.y_test_predicted,
        class_names=artifacts.label_encoder.classes_,
        top_n=args.top_n,
    )
    authentication_diagnostics = build_authentication_diagnostics(
        y_true=labels.y_test,
        y_probabilities=result.y_test_probabilities,
        threshold_points=args.threshold_points,
    )
    save_diagnostics(
        classification_diagnostics=classification_diagnostics,
        authentication_diagnostics=authentication_diagnostics,
        classification_report_output=args.classification_report_output,
        confusion_matrix_output=args.confusion_matrix_output,
        authentication_metrics_output=args.authentication_metrics_output,
        authentication_score_summary_output=args.authentication_score_summary_output,
    )

    true_user_ids = split.y_test.reset_index(drop=True).head().to_list()
    predicted_user_ids = result.y_test_predicted_user_ids[:5].tolist()
    macro_avg = classification_diagnostics.report.loc["macro avg"]
    weighted_avg = classification_diagnostics.report.loc["weighted avg"]

    print()
    print("Evaluation finished.")
    print(f"Validation loss: {result.validation_loss:.4f}")
    print(f"Validation accuracy: {result.validation_accuracy:.4f}")
    print(f"Test loss: {result.test_loss:.4f}")
    print(f"Test accuracy: {result.test_accuracy:.4f}")
    print(f"Predictions shape: {result.y_test_predicted.shape}")
    print(f"Predicted classes: {len(np.unique(result.y_test_predicted))}")
    print(f"First 5 true labels: {true_user_ids}")
    print(f"First 5 predicted labels: {predicted_user_ids}")

    print()
    print("Classification diagnostics:")
    print(f"Macro precision: {macro_avg['precision']:.4f}")
    print(f"Macro recall: {macro_avg['recall']:.4f}")
    print(f"Macro F1: {macro_avg['f1-score']:.4f}")
    print(f"Weighted precision: {weighted_avg['precision']:.4f}")
    print(f"Weighted recall: {weighted_avg['recall']:.4f}")
    print(f"Weighted F1: {weighted_avg['f1-score']:.4f}")

    print()
    print("Worst users by recall:")
    print(classification_diagnostics.worst_users_by_recall.to_string())

    print()
    print("Top confusions:")
    print(classification_diagnostics.top_confusions.to_string(index=False))

    print()
    print("Authentication diagnostics:")
    print(f"Genuine trials: {authentication_diagnostics.genuine_trials}")
    print(f"Impostor trials: {authentication_diagnostics.impostor_trials}")
    print(f"EER: {authentication_diagnostics.eer:.6f}")
    print(f"EER threshold: {authentication_diagnostics.eer_threshold:.6f}")
    print(f"FAR at EER threshold: {authentication_diagnostics.far_at_eer:.6f}")
    print(f"FRR at EER threshold: {authentication_diagnostics.frr_at_eer:.6f}")
    print(
        f"FAR at threshold {REFERENCE_THRESHOLD:.2f}: "
        f"{authentication_diagnostics.far_at_reference_threshold:.6f}"
    )
    print(
        f"FRR at threshold {REFERENCE_THRESHOLD:.2f}: "
        f"{authentication_diagnostics.frr_at_reference_threshold:.6f}"
    )

    print()
    print("Authentication score summary:")
    print(authentication_diagnostics.score_summary.to_string(index=False))

    print()
    print("Diagnostics saved:")
    print(f"Classification report path: {args.classification_report_output}")
    print(f"Confusion matrix path: {args.confusion_matrix_output}")
    print(f"Authentication metrics path: {args.authentication_metrics_output}")
    print(f"Authentication score summary path: {args.authentication_score_summary_output}")


if __name__ == "__main__":
    tf.get_logger().setLevel("ERROR")
    main()
