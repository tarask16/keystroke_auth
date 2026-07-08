"""Обучение Siamese encoder-а через contrastive loss.

Скрипт использует pair-наборы, созданные задачей 7.1:
`metric_pairs_train.npz` и `metric_pairs_validation.npz`.

Результат задачи 7.2:
- `models/embedding_model/siamese_model.keras`;
- `models/embedding_model/siamese_encoder.keras`;
- `reports/embedding_model/siamese_training_metrics.csv`.
"""

from __future__ import annotations

import argparse
import contextlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from src.embedding_model.siamese_model import build_siamese_model

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "processed" / "embedding_model"
DEFAULT_MODELS_DIR = PROJECT_ROOT / "models" / "embedding_model"
DEFAULT_REPORTS_DIR = PROJECT_ROOT / "reports" / "embedding_model"

DEFAULT_TRAIN_PAIRS_FILE = DEFAULT_DATA_DIR / "metric_pairs_train.npz"
DEFAULT_VALIDATION_PAIRS_FILE = DEFAULT_DATA_DIR / "metric_pairs_validation.npz"
DEFAULT_MODEL_FILE = DEFAULT_MODELS_DIR / "siamese_model.keras"
DEFAULT_ENCODER_FILE = DEFAULT_MODELS_DIR / "siamese_encoder.keras"
DEFAULT_METRICS_FILE = DEFAULT_REPORTS_DIR / "siamese_training_metrics.csv"


@dataclass(frozen=True)
class PairArrays:
    """Массивы pair dataset для Siamese training."""

    x_a: np.ndarray
    x_b: np.ndarray
    y: np.ndarray

    @property
    def input_dim(self) -> int:
        """Вернуть размерность входных признаков."""
        return int(self.x_a.shape[1])


@dataclass(frozen=True)
class TrainingArtifacts:
    """Пути к сохранённым артефактам обучения."""

    model_path: Path
    encoder_path: Path
    metrics_path: Path


def set_reproducibility(random_state: int) -> None:
    """Настроить воспроизводимость обучения Siamese-модели."""
    np.random.seed(random_state)
    tf.keras.utils.set_random_seed(random_state)

    with contextlib.suppress(Exception):
        tf.config.experimental.enable_op_determinism()


def load_pair_dataset(path: Path) -> PairArrays:
    """Загрузить `.npz` pair dataset.

    Args:
        path: Путь к `metric_pairs_*.npz`.

    Returns:
        PairArrays с массивами `x_a`, `x_b`, `y`.

    Raises:
        FileNotFoundError: Если файл не найден.
        ValueError: Если структура файла некорректна.
    """
    path = path.resolve()

    if not path.exists():
        raise FileNotFoundError(f"Pair dataset not found: {path}")

    with np.load(path, allow_pickle=False) as data:
        required_keys = {"x_a", "x_b", "y"}
        missing_keys = sorted(required_keys - set(data.files))

        if missing_keys:
            raise ValueError(f"Pair dataset has no required arrays: {missing_keys}")

        x_a = data["x_a"].astype(np.float32)
        x_b = data["x_b"].astype(np.float32)
        y = data["y"].astype(np.float32)

    validate_pair_arrays(x_a=x_a, x_b=x_b, y=y, source_path=path)

    return PairArrays(x_a=x_a, x_b=x_b, y=y)


def validate_pair_arrays(
    x_a: np.ndarray,
    x_b: np.ndarray,
    y: np.ndarray,
    source_path: Path,
) -> None:
    """Проверить форму и метки pair dataset."""
    if x_a.ndim != 2 or x_b.ndim != 2:
        raise ValueError(f"Pair arrays must be 2D: {source_path}")

    if x_a.shape != x_b.shape:
        raise ValueError(f"x_a and x_b shapes must be equal: {x_a.shape} vs {x_b.shape}")

    if y.ndim != 1:
        raise ValueError(f"Pair labels must be 1D: {source_path}")

    if x_a.shape[0] != y.shape[0]:
        raise ValueError(f"Number of pairs and labels mismatch: {source_path}")

    unique_labels = set(np.unique(y).tolist())
    if not unique_labels.issubset({0.0, 1.0}):
        raise ValueError(f"Pair labels must contain only 0.0 and 1.0: {unique_labels}")

    if x_a.shape[0] == 0:
        raise ValueError(f"Pair dataset is empty: {source_path}")


def save_training_metrics(history: tf.keras.callbacks.History, output_path: Path) -> None:
    """Сохранить историю обучения в CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_df = pd.DataFrame(history.history)
    metrics_df.insert(0, "epoch", np.arange(1, len(metrics_df) + 1, dtype=int))
    metrics_df.to_csv(output_path, index=False)


def run_training(
    train_pairs_path: Path,
    validation_pairs_path: Path,
    model_output_path: Path,
    encoder_output_path: Path,
    metrics_output_path: Path,
    embedding_dim: int,
    hidden_units: int,
    dropout_rate: float,
    margin: float,
    learning_rate: float,
    epochs: int,
    batch_size: int,
    patience: int,
    random_state: int,
    embedding_activation: str | None,
    l2_normalize: bool,
) -> TrainingArtifacts:
    """Выполнить полный цикл обучения Siamese-модели."""
    set_reproducibility(random_state)

    train_pairs = load_pair_dataset(train_pairs_path)
    validation_pairs = load_pair_dataset(validation_pairs_path)

    if train_pairs.input_dim != validation_pairs.input_dim:
        raise ValueError(
            "Train and validation feature dimensions differ: "
            f"{train_pairs.input_dim} vs {validation_pairs.input_dim}"
        )

    model, encoder = build_siamese_model(
        input_dim=train_pairs.input_dim,
        embedding_dim=embedding_dim,
        hidden_units=hidden_units,
        dropout_rate=dropout_rate,
        margin=margin,
        learning_rate=learning_rate,
        embedding_activation=embedding_activation,
        l2_normalize=l2_normalize,
    )

    callbacks: list[tf.keras.callbacks.Callback] = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=patience,
            restore_best_weights=True,
        )
    ]

    history = model.fit(
        x=[train_pairs.x_a, train_pairs.x_b],
        y=train_pairs.y,
        validation_data=([validation_pairs.x_a, validation_pairs.x_b], validation_pairs.y),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=2,
    )

    model_output_path.parent.mkdir(parents=True, exist_ok=True)
    encoder_output_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(model_output_path)
    encoder.save(encoder_output_path)
    save_training_metrics(history, metrics_output_path)

    print("Siamese metric-learning training completed successfully.")
    print(f"Train pairs: {train_pairs.y.shape[0]}")
    print(f"Validation pairs: {validation_pairs.y.shape[0]}")
    print(f"Feature dimension: {train_pairs.input_dim}")
    print(f"Embedding dimension: {embedding_dim}")
    print(f"Margin: {margin}")
    print(f"Model output: {model_output_path}")
    print(f"Encoder output: {encoder_output_path}")
    print(f"Training metrics: {metrics_output_path}")

    return TrainingArtifacts(
        model_path=model_output_path,
        encoder_path=encoder_output_path,
        metrics_path=metrics_output_path,
    )


def normalize_embedding_activation(value: str) -> str | None:
    """Преобразовать CLI-значение активации embedding-слоя."""
    if value == "linear":
        return None
    return value


def build_arg_parser() -> argparse.ArgumentParser:
    """Создать CLI parser для обучения Siamese-модели."""
    parser = argparse.ArgumentParser(description="Train Siamese encoder with contrastive loss.")
    parser.add_argument("--train-pairs", type=Path, default=DEFAULT_TRAIN_PAIRS_FILE)
    parser.add_argument("--validation-pairs", type=Path, default=DEFAULT_VALIDATION_PAIRS_FILE)
    parser.add_argument("--model-output", type=Path, default=DEFAULT_MODEL_FILE)
    parser.add_argument("--encoder-output", type=Path, default=DEFAULT_ENCODER_FILE)
    parser.add_argument("--metrics-output", type=Path, default=DEFAULT_METRICS_FILE)
    parser.add_argument("--embedding-dim", type=int, default=32)
    parser.add_argument("--hidden-units", type=int, default=128)
    parser.add_argument("--dropout-rate", type=float, default=0.2)
    parser.add_argument("--margin", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--embedding-activation", choices=["linear", "relu"], default="linear")
    parser.add_argument(
        "--no-l2-normalize",
        action="store_true",
        help="Disable L2 normalization of embedding vectors.",
    )
    return parser


def main() -> None:
    """CLI entry point."""
    parser = build_arg_parser()
    args = parser.parse_args()

    run_training(
        train_pairs_path=args.train_pairs,
        validation_pairs_path=args.validation_pairs,
        model_output_path=args.model_output,
        encoder_output_path=args.encoder_output,
        metrics_output_path=args.metrics_output,
        embedding_dim=args.embedding_dim,
        hidden_units=args.hidden_units,
        dropout_rate=args.dropout_rate,
        margin=args.margin,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        batch_size=args.batch_size,
        patience=args.patience,
        random_state=args.random_state,
        embedding_activation=normalize_embedding_activation(args.embedding_activation),
        l2_normalize=not args.no_l2_normalize,
    )


if __name__ == "__main__":
    main()
