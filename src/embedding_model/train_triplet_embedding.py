"""Обучение Triplet encoder-а через triplet loss.

Скрипт использует triplet-наборы, созданные задачей 7.1:
`metric_triplets_train.npz` и `metric_triplets_validation.npz`.

Результат задачи 7.3:
- `models/embedding_model/triplet_model.keras`;
- `models/embedding_model/triplet_encoder.keras`;
- `reports/embedding_model/triplet_training_metrics.csv`.
"""

from __future__ import annotations

import argparse
import contextlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from src.embedding_model.triplet_model import build_triplet_model

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "processed" / "embedding_model"
DEFAULT_MODELS_DIR = PROJECT_ROOT / "models" / "embedding_model"
DEFAULT_REPORTS_DIR = PROJECT_ROOT / "reports" / "embedding_model"

DEFAULT_TRAIN_TRIPLETS_FILE = DEFAULT_DATA_DIR / "metric_triplets_train.npz"
DEFAULT_VALIDATION_TRIPLETS_FILE = DEFAULT_DATA_DIR / "metric_triplets_validation.npz"
DEFAULT_MODEL_FILE = DEFAULT_MODELS_DIR / "triplet_model.keras"
DEFAULT_ENCODER_FILE = DEFAULT_MODELS_DIR / "triplet_encoder.keras"
DEFAULT_METRICS_FILE = DEFAULT_REPORTS_DIR / "triplet_training_metrics.csv"


@dataclass(frozen=True)
class TripletArrays:
    """Массивы triplet dataset для Triplet training."""

    anchor: np.ndarray
    positive: np.ndarray
    negative: np.ndarray

    @property
    def input_dim(self) -> int:
        """Вернуть размерность входных признаков."""
        return int(self.anchor.shape[1])

    @property
    def dummy_labels(self) -> np.ndarray:
        """Вернуть фиктивные метки для Keras loss-интерфейса."""
        return np.zeros((self.anchor.shape[0],), dtype=np.float32)


@dataclass(frozen=True)
class TrainingArtifacts:
    """Пути к сохранённым артефактам обучения."""

    model_path: Path
    encoder_path: Path
    metrics_path: Path


def set_reproducibility(random_state: int) -> None:
    """Зафиксировать seed для воспроизводимого обучения."""
    np.random.seed(random_state)
    tf.keras.utils.set_random_seed(random_state)

    with contextlib.suppress(Exception):
        tf.config.experimental.enable_op_determinism()


def load_triplet_dataset(path: Path) -> TripletArrays:
    """Загрузить `.npz` triplet dataset.

    Args:
        path: Путь к `metric_triplets_*.npz`.

    Returns:
        TripletArrays с массивами `anchor`, `positive`, `negative`.

    Raises:
        FileNotFoundError: Если файл не найден.
        ValueError: Если структура файла некорректна.
    """
    path = path.resolve()

    if not path.exists():
        raise FileNotFoundError(f"Triplet dataset not found: {path}")

    with np.load(path, allow_pickle=False) as data:
        required_keys = {"anchor", "positive", "negative"}
        missing_keys = sorted(required_keys - set(data.files))

        if missing_keys:
            raise ValueError(f"Triplet dataset has no required arrays: {missing_keys}")

        anchor = data["anchor"].astype(np.float32)
        positive = data["positive"].astype(np.float32)
        negative = data["negative"].astype(np.float32)

    validate_triplet_arrays(
        anchor=anchor,
        positive=positive,
        negative=negative,
        source_path=path,
    )

    return TripletArrays(anchor=anchor, positive=positive, negative=negative)


def validate_triplet_arrays(
    anchor: np.ndarray,
    positive: np.ndarray,
    negative: np.ndarray,
    source_path: Path,
) -> None:
    """Проверить форму triplet dataset."""
    if anchor.ndim != 2 or positive.ndim != 2 or negative.ndim != 2:
        raise ValueError(f"Triplet arrays must be 2D: {source_path}")

    if anchor.shape != positive.shape or anchor.shape != negative.shape:
        raise ValueError(
            "Triplet array shapes must be equal: "
            f"anchor={anchor.shape}, positive={positive.shape}, negative={negative.shape}"
        )

    if anchor.shape[0] == 0:
        raise ValueError(f"Triplet dataset is empty: {source_path}")


def save_training_metrics(history: tf.keras.callbacks.History, output_path: Path) -> None:
    """Сохранить историю обучения в CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_df = pd.DataFrame(history.history)
    metrics_df.insert(0, "epoch", np.arange(1, len(metrics_df) + 1, dtype=int))
    metrics_df.to_csv(output_path, index=False)


def run_training(
    train_triplets_path: Path,
    validation_triplets_path: Path,
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
    """Выполнить полный цикл обучения Triplet-модели."""
    set_reproducibility(random_state)

    train_triplets = load_triplet_dataset(train_triplets_path)
    validation_triplets = load_triplet_dataset(validation_triplets_path)

    if train_triplets.input_dim != validation_triplets.input_dim:
        raise ValueError(
            "Train and validation feature dimensions differ: "
            f"{train_triplets.input_dim} vs {validation_triplets.input_dim}"
        )

    model, encoder = build_triplet_model(
        input_dim=train_triplets.input_dim,
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
        x=[train_triplets.anchor, train_triplets.positive, train_triplets.negative],
        y=train_triplets.dummy_labels,
        validation_data=(
            [
                validation_triplets.anchor,
                validation_triplets.positive,
                validation_triplets.negative,
            ],
            validation_triplets.dummy_labels,
        ),
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

    print("Triplet metric-learning training completed successfully.")
    print(f"Train triplets: {train_triplets.anchor.shape[0]}")
    print(f"Validation triplets: {validation_triplets.anchor.shape[0]}")
    print(f"Feature dimension: {train_triplets.input_dim}")
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
    """Создать CLI parser для обучения Triplet-модели."""
    parser = argparse.ArgumentParser(description="Train encoder with triplet loss.")
    parser.add_argument("--train-triplets", type=Path, default=DEFAULT_TRAIN_TRIPLETS_FILE)
    parser.add_argument(
        "--validation-triplets",
        type=Path,
        default=DEFAULT_VALIDATION_TRIPLETS_FILE,
    )
    parser.add_argument("--model-output", type=Path, default=DEFAULT_MODEL_FILE)
    parser.add_argument("--encoder-output", type=Path, default=DEFAULT_ENCODER_FILE)
    parser.add_argument("--metrics-output", type=Path, default=DEFAULT_METRICS_FILE)
    parser.add_argument("--embedding-dim", type=int, default=32)
    parser.add_argument("--hidden-units", type=int, default=128)
    parser.add_argument("--dropout-rate", type=float, default=0.2)
    parser.add_argument("--margin", type=float, default=0.2)
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
        train_triplets_path=args.train_triplets,
        validation_triplets_path=args.validation_triplets,
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
