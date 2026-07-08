"""Генерация genuine/impostor пар для Siamese metric-learning обучения.

Модуль строит сбалансированные пары образцов.

Типы пар:
- genuine: оба образца принадлежат одному пользователю;
- impostor: образцы принадлежат разным пользователям.

Test split не используется.
Это исключает утечку данных в обучение
и последующий подбор порогов.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FEATURES_FILE = PROJECT_ROOT / "data" / "processed" / "cmu_features.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "embedding_model"
DEFAULT_SCALER_FILE = PROJECT_ROOT / "models" / "embedding_model" / "metric_learning_scaler.pkl"

META_COLUMNS = {"user_id", "session_index", "rep", "sample_id", "split"}
FEATURE_PREFIXES = ("H.", "DD.", "UD.")
TRAIN_SPLIT = "train"
VALIDATION_SPLIT = "validation"
TEST_SPLIT = "test"


@dataclass(frozen=True)
class PairDataset:
    """Контейнер с массивами пар для Siamese-модели."""

    x_a: np.ndarray
    x_b: np.ndarray
    y: np.ndarray
    user_id_a: np.ndarray
    user_id_b: np.ndarray
    sample_id_a: np.ndarray
    sample_id_b: np.ndarray


@dataclass(frozen=True)
class SplitTables:
    """Train/validation таблицы после нормализации."""

    train: pd.DataFrame
    validation: pd.DataFrame
    feature_columns: list[str]


def load_feature_table(path: Path) -> pd.DataFrame:
    """Загрузить таблицу признаков CMU."

    Args:
        path: Путь к `data/processed/cmu_features.csv`.

    Returns:
        DataFrame с метаданными и числовыми признаками.

    Raises:
        FileNotFoundError: Если CSV-файл не найден.
        ValueError: Если таблица пуста или некорректна.
    """
    path = path.resolve()

    if not path.exists():
        raise FileNotFoundError(f"Feature table not found: {path}")

    df = pd.read_csv(path)

    if df.empty:
        raise ValueError(f"Feature table is empty: {path}")

    required_columns = {"user_id", "sample_id"}
    missing_columns = sorted(required_columns - set(df.columns))

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    feature_columns = get_feature_columns(df)

    if not feature_columns:
        raise ValueError("No timing feature columns found in feature table.")

    return df


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Вернуть список признаков клавиатурной динамики.

    Args:
        df: Таблица признаков.

    Returns:
        Список колонок H.*, DD.*, UD.*.
        Если таких колонок нет, возвращаются
        числовые колонки без метаданных.
    """
    prefixed_columns = [
        column
        for column in df.columns
        if isinstance(column, str) and column.startswith(FEATURE_PREFIXES)
    ]

    if prefixed_columns:
        return prefixed_columns

    return [
        column
        for column in df.columns
        if column not in META_COLUMNS and pd.api.types.is_numeric_dtype(df[column])
    ]


def add_deterministic_split(
    df: pd.DataFrame,
    train_fraction: float = 0.64,
    validation_fraction: float = 0.16,
) -> pd.DataFrame:
    """Добавить per-user split при отсутствии колонки `split`.

    Разбиение выполняется внутри каждого пользователя.
    Для CMU 400 samples/user значения 0.64/0.16 дают:
    256 train, 64 validation и 80 test samples.

    Args:
        df: Таблица признаков.
        train_fraction: Доля train-образцов внутри user_id.
        validation_fraction: Доля validation-образцов внутри user_id.

    Returns:
        Копия DataFrame с колонкой `split`.

    Raises:
        ValueError: Если доли разбиения некорректны.
    """
    if train_fraction <= 0 or validation_fraction <= 0:
        raise ValueError("Split fractions must be positive.")

    if train_fraction + validation_fraction >= 1.0:
        raise ValueError("Train + validation fractions must be less than 1.0.")

    if "split" in df.columns:
        return df.copy()

    sort_columns = [
        column
        for column in ["user_id", "session_index", "rep", "sample_id"]
        if column in df.columns
    ]
    result = df.sort_values(sort_columns).copy() if sort_columns else df.copy()
    result["split"] = TEST_SPLIT

    for _, user_indices in result.groupby("user_id", sort=False).groups.items():
        ordered_indices = list(user_indices)
        samples_count = len(ordered_indices)
        train_count = int(samples_count * train_fraction)
        validation_count = int(samples_count * validation_fraction)
        train_indices = ordered_indices[:train_count]
        validation_indices = ordered_indices[train_count : train_count + validation_count]

        result.loc[train_indices, "split"] = TRAIN_SPLIT
        result.loc[validation_indices, "split"] = VALIDATION_SPLIT

    return result


def scale_train_validation(
    df: pd.DataFrame,
    feature_columns: list[str],
    scaler_output_path: Path,
) -> SplitTables:
    """Нормализовать train/validation без использования test split.

    StandardScaler обучается только на train split.
    Validation split используется только через transform.
    Test split в этом генераторе не применяется.

    Args:
        df: Таблица признаков с колонкой `split`.
        feature_columns: Список признаков.
        scaler_output_path: Путь для сохранения scaler-а.

    Returns:
        Нормализованные train/validation таблицы.

    Raises:
        ValueError: Если отсутствуют train или validation samples.
    """
    train_df = df[df["split"] == TRAIN_SPLIT].copy().reset_index(drop=True)
    validation_df = df[df["split"] == VALIDATION_SPLIT].copy().reset_index(drop=True)

    if train_df.empty:
        raise ValueError("Train split is empty.")

    if validation_df.empty:
        raise ValueError("Validation split is empty.")

    scaler = StandardScaler()
    scaler.fit(train_df.loc[:, feature_columns].to_numpy(dtype=np.float32))

    train_df.loc[:, feature_columns] = scaler.transform(
        train_df.loc[:, feature_columns].to_numpy(dtype=np.float32)
    )
    validation_df.loc[:, feature_columns] = scaler.transform(
        validation_df.loc[:, feature_columns].to_numpy(dtype=np.float32)
    )

    scaler_output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, scaler_output_path)

    return SplitTables(train=train_df, validation=validation_df, feature_columns=feature_columns)


def create_pair_dataset(
    df: pd.DataFrame,
    feature_columns: list[str],
    pairs_per_user: int,
    rng: np.random.Generator,
) -> PairDataset:
    """Сформировать сбалансированный набор genuine/impostor пар.

    Для каждого user_id создаётся заданное число genuine-пар
    и столько же impostor-пар. Это снижает риск смещения
    contrastive loss в сторону более частого класса.

    Args:
        df: Нормализованная таблица одного split-а.
        feature_columns: Список признаков.
        pairs_per_user: Число genuine/impostor пар на user_id.
        rng: Генератор случайных чисел NumPy.

    Returns:
        PairDataset с массивами x_a, x_b и метками y.

    Raises:
        ValueError: Если samples недостаточно для генерации пар.
    """
    if pairs_per_user <= 0:
        raise ValueError("pairs_per_user must be positive.")

    users = sorted(df["user_id"].unique().tolist())

    if len(users) < 2:
        raise ValueError("At least two users are required for impostor pair generation.")

    grouped_indices = {
        user_id: df.index[df["user_id"] == user_id].to_numpy(dtype=np.int64) for user_id in users
    }

    for user_id, indices in grouped_indices.items():
        if len(indices) < 2:
            raise ValueError(f"User {user_id} has less than two samples.")

    x_a_rows: list[np.ndarray] = []
    x_b_rows: list[np.ndarray] = []
    labels: list[float] = []
    user_id_a: list[str] = []
    user_id_b: list[str] = []
    sample_id_a: list[str] = []
    sample_id_b: list[str] = []

    for user_id in users:
        current_indices = grouped_indices[user_id]
        negative_users = [candidate for candidate in users if candidate != user_id]

        for _ in range(pairs_per_user):
            first_index, second_index = rng.choice(current_indices, size=2, replace=False)
            append_pair(
                df=df,
                feature_columns=feature_columns,
                first_index=int(first_index),
                second_index=int(second_index),
                label=1.0,
                x_a_rows=x_a_rows,
                x_b_rows=x_b_rows,
                labels=labels,
                user_id_a=user_id_a,
                user_id_b=user_id_b,
                sample_id_a=sample_id_a,
                sample_id_b=sample_id_b,
            )

            negative_user = str(rng.choice(negative_users))
            impostor_first_index = int(rng.choice(current_indices))
            impostor_second_index = int(rng.choice(grouped_indices[negative_user]))
            append_pair(
                df=df,
                feature_columns=feature_columns,
                first_index=impostor_first_index,
                second_index=impostor_second_index,
                label=0.0,
                x_a_rows=x_a_rows,
                x_b_rows=x_b_rows,
                labels=labels,
                user_id_a=user_id_a,
                user_id_b=user_id_b,
                sample_id_a=sample_id_a,
                sample_id_b=sample_id_b,
            )

    dataset = PairDataset(
        x_a=np.vstack(x_a_rows).astype(np.float32),
        x_b=np.vstack(x_b_rows).astype(np.float32),
        y=np.asarray(labels, dtype=np.float32),
        user_id_a=np.asarray(user_id_a, dtype=str),
        user_id_b=np.asarray(user_id_b, dtype=str),
        sample_id_a=np.asarray(sample_id_a, dtype=str),
        sample_id_b=np.asarray(sample_id_b, dtype=str),
    )

    return shuffle_pair_dataset(dataset, rng)


def append_pair(
    df: pd.DataFrame,
    feature_columns: list[str],
    first_index: int,
    second_index: int,
    label: float,
    x_a_rows: list[np.ndarray],
    x_b_rows: list[np.ndarray],
    labels: list[float],
    user_id_a: list[str],
    user_id_b: list[str],
    sample_id_a: list[str],
    sample_id_b: list[str],
) -> None:
    """Добавить одну пару в накапливаемые списки."""
    first_row = df.loc[first_index]
    second_row = df.loc[second_index]

    x_a_rows.append(first_row.loc[feature_columns].to_numpy(dtype=np.float32))
    x_b_rows.append(second_row.loc[feature_columns].to_numpy(dtype=np.float32))
    labels.append(label)
    user_id_a.append(str(first_row["user_id"]))
    user_id_b.append(str(second_row["user_id"]))
    sample_id_a.append(str(first_row["sample_id"]))
    sample_id_b.append(str(second_row["sample_id"]))


def shuffle_pair_dataset(dataset: PairDataset, rng: np.random.Generator) -> PairDataset:
    """Перемешать пары общей перестановкой."""
    permutation = rng.permutation(dataset.y.shape[0])

    return PairDataset(
        x_a=dataset.x_a[permutation],
        x_b=dataset.x_b[permutation],
        y=dataset.y[permutation],
        user_id_a=dataset.user_id_a[permutation],
        user_id_b=dataset.user_id_b[permutation],
        sample_id_a=dataset.sample_id_a[permutation],
        sample_id_b=dataset.sample_id_b[permutation],
    )


def save_pair_dataset(dataset: PairDataset, output_path: Path) -> None:
    """Сохранить пары в `.npz` формате."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        x_a=dataset.x_a,
        x_b=dataset.x_b,
        y=dataset.y,
        user_id_a=dataset.user_id_a,
        user_id_b=dataset.user_id_b,
        sample_id_a=dataset.sample_id_a,
        sample_id_b=dataset.sample_id_b,
    )


def run_generation(
    input_path: Path,
    output_dir: Path,
    scaler_output_path: Path,
    train_pairs_per_user: int,
    validation_pairs_per_user: int,
    random_state: int,
    train_fraction: float,
    validation_fraction: float,
) -> None:
    """Выполнить полный цикл генерации train/validation пар."""
    rng = np.random.default_rng(random_state)
    df = load_feature_table(input_path)
    df = add_deterministic_split(
        df=df,
        train_fraction=train_fraction,
        validation_fraction=validation_fraction,
    )
    feature_columns = get_feature_columns(df)
    split_tables = scale_train_validation(df, feature_columns, scaler_output_path)

    train_pairs = create_pair_dataset(
        df=split_tables.train,
        feature_columns=split_tables.feature_columns,
        pairs_per_user=train_pairs_per_user,
        rng=rng,
    )
    validation_pairs = create_pair_dataset(
        df=split_tables.validation,
        feature_columns=split_tables.feature_columns,
        pairs_per_user=validation_pairs_per_user,
        rng=rng,
    )

    train_output = output_dir / "metric_pairs_train.npz"
    validation_output = output_dir / "metric_pairs_validation.npz"
    save_pair_dataset(train_pairs, train_output)
    save_pair_dataset(validation_pairs, validation_output)

    print("Metric-learning pairs generated successfully.")
    print(f"Train pairs: {train_pairs.y.shape[0]}")
    print(f"Validation pairs: {validation_pairs.y.shape[0]}")
    print(f"Train genuine pairs: {int(train_pairs.y.sum())}")
    print(f"Train impostor pairs: {int((train_pairs.y == 0).sum())}")
    print(f"Feature dimension: {train_pairs.x_a.shape[1]}")
    print(f"Scaler: {scaler_output_path}")
    print(f"Train output: {train_output}")
    print(f"Validation output: {validation_output}")


def build_arg_parser() -> argparse.ArgumentParser:
    """Создать CLI parser для генератора пар."""
    parser = argparse.ArgumentParser(
        description="Generate balanced genuine/impostor pairs for Siamese training."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_FEATURES_FILE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--scaler-output", type=Path, default=DEFAULT_SCALER_FILE)
    parser.add_argument("--train-pairs-per-user", type=int, default=1000)
    parser.add_argument("--validation-pairs-per-user", type=int, default=250)
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
        train_pairs_per_user=args.train_pairs_per_user,
        validation_pairs_per_user=args.validation_pairs_per_user,
        random_state=args.random_state,
        train_fraction=args.train_fraction,
        validation_fraction=args.validation_fraction,
    )


if __name__ == "__main__":
    main()
