"""Генерация triplet-наборов для metric-learning обучения encoder-а.

Каждая triplet-запись содержит:
- anchor: образец заявленного пользователя;
- positive: другой образец того же пользователя;
- negative: образец другого пользователя.

На текущей задаче реализован базовый random negative mining.
Semi-hard и hard negative mining требуют обученного encoder-а.
Они будут добавлены после первичного Triplet baseline.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.embedding_model.generate_metric_learning_pairs import (
    DEFAULT_FEATURES_FILE,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_SCALER_FILE,
    add_deterministic_split,
    get_feature_columns,
    load_feature_table,
    scale_train_validation,
)


@dataclass(frozen=True)
class TripletDataset:
    """Контейнер с массивами triplet samples."""

    anchor: np.ndarray
    positive: np.ndarray
    negative: np.ndarray
    anchor_user_id: np.ndarray
    positive_user_id: np.ndarray
    negative_user_id: np.ndarray
    anchor_sample_id: np.ndarray
    positive_sample_id: np.ndarray
    negative_sample_id: np.ndarray


def create_triplet_dataset(
    df: pd.DataFrame,
    feature_columns: list[str],
    triplets_per_user: int,
    rng: np.random.Generator,
) -> TripletDataset:
    """Сформировать triplet-набор с random negative mining.

    Args:
        df: Нормализованная таблица одного split-а.
        feature_columns: Список признаков.
        triplets_per_user: Число triplet-записей на user_id.
        rng: Генератор случайных чисел NumPy.

    Returns:
        TripletDataset с anchor/positive/negative массивами.

    Raises:
        ValueError: Если пользователей или samples недостаточно.
    """
    if triplets_per_user <= 0:
        raise ValueError("triplets_per_user must be positive.")

    users = sorted(df["user_id"].unique().tolist())

    if len(users) < 2:
        raise ValueError("At least two users are required for triplet generation.")

    grouped_indices = {
        user_id: df.index[df["user_id"] == user_id].to_numpy(dtype=np.int64) for user_id in users
    }

    for user_id, indices in grouped_indices.items():
        if len(indices) < 2:
            raise ValueError(f"User {user_id} has less than two samples.")

    anchor_rows: list[np.ndarray] = []
    positive_rows: list[np.ndarray] = []
    negative_rows: list[np.ndarray] = []
    anchor_user_ids: list[str] = []
    positive_user_ids: list[str] = []
    negative_user_ids: list[str] = []
    anchor_sample_ids: list[str] = []
    positive_sample_ids: list[str] = []
    negative_sample_ids: list[str] = []

    for anchor_user_id in users:
        anchor_indices = grouped_indices[anchor_user_id]
        negative_users = [candidate for candidate in users if candidate != anchor_user_id]

        for _ in range(triplets_per_user):
            anchor_index, positive_index = rng.choice(anchor_indices, size=2, replace=False)
            negative_user_id = str(rng.choice(negative_users))
            negative_index = int(rng.choice(grouped_indices[negative_user_id]))

            append_triplet(
                df=df,
                feature_columns=feature_columns,
                anchor_index=int(anchor_index),
                positive_index=int(positive_index),
                negative_index=negative_index,
                anchor_rows=anchor_rows,
                positive_rows=positive_rows,
                negative_rows=negative_rows,
                anchor_user_ids=anchor_user_ids,
                positive_user_ids=positive_user_ids,
                negative_user_ids=negative_user_ids,
                anchor_sample_ids=anchor_sample_ids,
                positive_sample_ids=positive_sample_ids,
                negative_sample_ids=negative_sample_ids,
            )

    dataset = TripletDataset(
        anchor=np.vstack(anchor_rows).astype(np.float32),
        positive=np.vstack(positive_rows).astype(np.float32),
        negative=np.vstack(negative_rows).astype(np.float32),
        anchor_user_id=np.asarray(anchor_user_ids, dtype=str),
        positive_user_id=np.asarray(positive_user_ids, dtype=str),
        negative_user_id=np.asarray(negative_user_ids, dtype=str),
        anchor_sample_id=np.asarray(anchor_sample_ids, dtype=str),
        positive_sample_id=np.asarray(positive_sample_ids, dtype=str),
        negative_sample_id=np.asarray(negative_sample_ids, dtype=str),
    )

    return shuffle_triplet_dataset(dataset, rng)


def append_triplet(
    df: pd.DataFrame,
    feature_columns: list[str],
    anchor_index: int,
    positive_index: int,
    negative_index: int,
    anchor_rows: list[np.ndarray],
    positive_rows: list[np.ndarray],
    negative_rows: list[np.ndarray],
    anchor_user_ids: list[str],
    positive_user_ids: list[str],
    negative_user_ids: list[str],
    anchor_sample_ids: list[str],
    positive_sample_ids: list[str],
    negative_sample_ids: list[str],
) -> None:
    """Добавить одну triplet-запись в накапливаемые списки."""
    anchor_row = df.loc[anchor_index]
    positive_row = df.loc[positive_index]
    negative_row = df.loc[negative_index]

    anchor_rows.append(anchor_row.loc[feature_columns].to_numpy(dtype=np.float32))
    positive_rows.append(positive_row.loc[feature_columns].to_numpy(dtype=np.float32))
    negative_rows.append(negative_row.loc[feature_columns].to_numpy(dtype=np.float32))

    anchor_user_ids.append(str(anchor_row["user_id"]))
    positive_user_ids.append(str(positive_row["user_id"]))
    negative_user_ids.append(str(negative_row["user_id"]))

    anchor_sample_ids.append(str(anchor_row["sample_id"]))
    positive_sample_ids.append(str(positive_row["sample_id"]))
    negative_sample_ids.append(str(negative_row["sample_id"]))


def shuffle_triplet_dataset(dataset: TripletDataset, rng: np.random.Generator) -> TripletDataset:
    """Перемешать triplet-записи общей перестановкой."""
    permutation = rng.permutation(dataset.anchor.shape[0])

    return TripletDataset(
        anchor=dataset.anchor[permutation],
        positive=dataset.positive[permutation],
        negative=dataset.negative[permutation],
        anchor_user_id=dataset.anchor_user_id[permutation],
        positive_user_id=dataset.positive_user_id[permutation],
        negative_user_id=dataset.negative_user_id[permutation],
        anchor_sample_id=dataset.anchor_sample_id[permutation],
        positive_sample_id=dataset.positive_sample_id[permutation],
        negative_sample_id=dataset.negative_sample_id[permutation],
    )


def save_triplet_dataset(dataset: TripletDataset, output_path: Path) -> None:
    """Сохранить triplet-набор в `.npz` формате."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        anchor=dataset.anchor,
        positive=dataset.positive,
        negative=dataset.negative,
        anchor_user_id=dataset.anchor_user_id,
        positive_user_id=dataset.positive_user_id,
        negative_user_id=dataset.negative_user_id,
        anchor_sample_id=dataset.anchor_sample_id,
        positive_sample_id=dataset.positive_sample_id,
        negative_sample_id=dataset.negative_sample_id,
    )


def run_generation(
    input_path: Path,
    output_dir: Path,
    scaler_output_path: Path,
    train_triplets_per_user: int,
    validation_triplets_per_user: int,
    random_state: int,
    train_fraction: float,
    validation_fraction: float,
) -> None:
    """Выполнить полный цикл генерации train/validation triplets."""
    rng = np.random.default_rng(random_state)
    df = load_feature_table(input_path)
    df = add_deterministic_split(
        df=df,
        train_fraction=train_fraction,
        validation_fraction=validation_fraction,
    )
    feature_columns = get_feature_columns(df)
    split_tables = scale_train_validation(df, feature_columns, scaler_output_path)

    train_triplets = create_triplet_dataset(
        df=split_tables.train,
        feature_columns=split_tables.feature_columns,
        triplets_per_user=train_triplets_per_user,
        rng=rng,
    )
    validation_triplets = create_triplet_dataset(
        df=split_tables.validation,
        feature_columns=split_tables.feature_columns,
        triplets_per_user=validation_triplets_per_user,
        rng=rng,
    )

    train_output = output_dir / "metric_triplets_train.npz"
    validation_output = output_dir / "metric_triplets_validation.npz"
    save_triplet_dataset(train_triplets, train_output)
    save_triplet_dataset(validation_triplets, validation_output)

    print("Metric-learning triplets generated successfully.")
    print(f"Train triplets: {train_triplets.anchor.shape[0]}")
    print(f"Validation triplets: {validation_triplets.anchor.shape[0]}")
    print(f"Feature dimension: {train_triplets.anchor.shape[1]}")
    print(f"Scaler: {scaler_output_path}")
    print(f"Train output: {train_output}")
    print(f"Validation output: {validation_output}")


def build_arg_parser() -> argparse.ArgumentParser:
    """Создать CLI parser для генератора triplets."""
    parser = argparse.ArgumentParser(description="Generate triplets for metric-learning training.")
    parser.add_argument("--input", type=Path, default=DEFAULT_FEATURES_FILE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--scaler-output", type=Path, default=DEFAULT_SCALER_FILE)
    parser.add_argument("--train-triplets-per-user", type=int, default=1000)
    parser.add_argument("--validation-triplets-per-user", type=int, default=250)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--train-fraction", type=float, default=0.64)
    parser.add_argument("--validation-fraction", type=float, default=0.16)
    return parser


def main() -> None:
    """CLI entry point."""
    parser = build_arg_parser()
    args = parser.parse_args()

    run_generation(
        input_path=args.input,
        output_dir=args.output_dir,
        scaler_output_path=args.scaler_output,
        train_triplets_per_user=args.train_triplets_per_user,
        validation_triplets_per_user=args.validation_triplets_per_user,
        random_state=args.random_state,
        train_fraction=args.train_fraction,
        validation_fraction=args.validation_fraction,
    )


if __name__ == "__main__":
    main()
