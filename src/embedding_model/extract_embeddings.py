"""Извлечение embedding-векторов для этапа 6 проекта Keystroke Auth.

Файл размещается в каталоге:

    src/embedding_model/extract_embeddings.py

Назначение файла:

    1. Загрузить исходную таблицу временных признаков.
    2. Загрузить split-файл, созданный при обучении embedding-классификатора.
    3. Загрузить scaler, обученный только на train split.
    4. Загрузить encoder, извлечённый из embedding-классификатора.
    5. Преобразовать train, validation и test samples в embedding-векторы.
    6. Сохранить embedding-векторы в отдельные CSV-файлы.

Выходные файлы:

    data/processed/embedding_model/embeddings_train.csv
    data/processed/embedding_model/embeddings_validation.csv
    data/processed/embedding_model/embeddings_test.csv

Методологическое правило:

    На этом шаге test split не используется для подбора порогов.
    Скрипт только извлекает embedding-векторы и сохраняет их для будущей
    диагностики расстояний и финальной оценки.

Терминология:

- ``embedding-вектор`` — компактное числовое представление одной попытки ввода;
- ``encoder`` — часть обученной модели от входного слоя до embedding-слоя;
- ``train split`` — данные для формирования базовых шаблонов и обучения;
- ``validation split`` — данные для подбора порогов;
- ``test split`` — данные только для финальной оценки качества.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import joblib
import numpy as np
import pandas as pd
from tensorflow import keras

from src.embedding_model.embedding import EmbeddingModelPaths, extract_embeddings

# Имена выходных CSV-файлов с embedding-векторами.
TRAIN_EMBEDDINGS_FILENAME: Final[str] = "embeddings_train.csv"
VALIDATION_EMBEDDINGS_FILENAME: Final[str] = "embeddings_validation.csv"
TEST_EMBEDDINGS_FILENAME: Final[str] = "embeddings_test.csv"

# Возможные имена колонки с номером сессии в разных версиях подготовленного CSV.
SESSION_COLUMN_CANDIDATES: Final[tuple[str, ...]] = (
    "session_id",
    "session",
    "sessionIndex",
    "sessionindex",
    "session_index",
)

# Возможные имена колонки с номером попытки ввода.
REPETITION_COLUMN_CANDIDATES: Final[tuple[str, ...]] = (
    "rep",
    "repetition",
    "repetition_id",
    "rep_id",
)

# Возможные имена явного идентификатора образца.
SAMPLE_ID_COLUMN_CANDIDATES: Final[tuple[str, ...]] = (
    "sample_id",
    "sample",
    "id",
)


@dataclass(frozen=True)
class EmbeddingExtractionConfig:
    """Параметры извлечения embedding-векторов."""

    dataset_path: Path
    batch_size: int = 256


@dataclass(frozen=True)
class SplitPayload:
    """Данные из split-файла, созданного при обучении модели."""

    label_column: str
    feature_columns: list[str]
    train_indices: list[int]
    validation_indices: list[int]
    test_indices: list[int]


def load_feature_table(dataset_path: Path) -> pd.DataFrame:
    """Загрузить таблицу временных признаков клавиатурного почерка.

    Аргументы:
        dataset_path: Путь к CSV-файлу с подготовленными признаками.

    Возвращает:
        DataFrame с временными признаками и идентификатором пользователя.

    Исключения:
        FileNotFoundError: Возникает, если CSV-файл не найден.
        ValueError: Возникает, если таблица пуста.
    """
    if not dataset_path.exists():
        raise FileNotFoundError(f"Файл датасета не найден: {dataset_path}")

    dataframe = pd.read_csv(dataset_path)

    if dataframe.empty:
        raise ValueError(f"Файл датасета пуст: {dataset_path}")

    return dataframe


def load_split_payload(split_path: Path) -> SplitPayload:
    """Загрузить split-файл этапа 6.

    Аргументы:
        split_path: Путь к JSON-файлу с индексами train, validation и test.

    Возвращает:
        Структуру с колонкой пользователя, списком признаков и индексами split-ов.

    Исключения:
        FileNotFoundError: Возникает, если split-файл не найден.
        ValueError: Возникает, если split-файл не содержит обязательных полей.
    """
    if not split_path.exists():
        raise FileNotFoundError(f"Split-файл не найден: {split_path}")

    payload = json.loads(split_path.read_text(encoding="utf-8"))

    required_fields = {
        "label_column",
        "feature_columns",
        "train_indices",
        "validation_indices",
        "test_indices",
    }
    missing_fields = required_fields - set(payload)

    if missing_fields:
        raise ValueError(
            f"Split-файл не содержит обязательные поля: {', '.join(sorted(missing_fields))}."
        )

    return SplitPayload(
        label_column=str(payload["label_column"]),
        feature_columns=list(payload["feature_columns"]),
        train_indices=[int(index) for index in payload["train_indices"]],
        validation_indices=[int(index) for index in payload["validation_indices"]],
        test_indices=[int(index) for index in payload["test_indices"]],
    )


def validate_required_columns(dataframe: pd.DataFrame, split_payload: SplitPayload) -> None:
    """Проверить наличие колонок пользователя и временных признаков.

    Аргументы:
        dataframe: Исходная таблица признаков.
        split_payload: Данные из split-файла.

    Исключения:
        ValueError: Возникает, если в таблице отсутствуют необходимые колонки.
    """
    required_columns = {split_payload.label_column, *split_payload.feature_columns}
    missing_columns = required_columns - set(dataframe.columns)

    if missing_columns:
        raise ValueError(
            "В таблице признаков отсутствуют необходимые колонки: "
            f"{', '.join(sorted(missing_columns))}."
        )


def load_encoder_and_scaler(paths: EmbeddingModelPaths) -> tuple[keras.Model, object]:
    """Загрузить encoder и scaler embedding-этапа.

    Аргументы:
        paths: Пути к артефактам embedding-этапа.

    Возвращает:
        Кортеж из Keras-модели encoder и объекта StandardScaler.

    Исключения:
        FileNotFoundError: Возникает, если encoder или scaler не найдены.
    """
    if not paths.encoder_path.exists():
        raise FileNotFoundError(f"Файл encoder не найден: {paths.encoder_path}")

    if not paths.scaler_path.exists():
        raise FileNotFoundError(f"Файл scaler не найден: {paths.scaler_path}")

    encoder = keras.models.load_model(paths.encoder_path)
    scaler = joblib.load(paths.scaler_path)

    return encoder, scaler


def find_first_existing_column(
    dataframe: pd.DataFrame,
    candidates: tuple[str, ...],
) -> str | None:
    """Найти первую существующую колонку из списка кандидатов."""
    for column in candidates:
        if column in dataframe.columns:
            return column

    return None


def build_metadata_frame(
    dataframe: pd.DataFrame,
    row_indices: list[int],
    label_column: str,
) -> pd.DataFrame:
    """Сформировать служебные колонки для CSV с embedding-векторами.

    Аргументы:
        dataframe: Исходная таблица признаков.
        row_indices: Индексы строк текущего split.
        label_column: Имя колонки пользователя.

    Возвращает:
        DataFrame со служебными колонками ``sample_id``, ``source_index``,
        ``user_id``, ``session_id`` и ``repetition_id``.
    """
    split_frame = dataframe.iloc[row_indices].copy()

    sample_id_column = find_first_existing_column(split_frame, SAMPLE_ID_COLUMN_CANDIDATES)
    session_column = find_first_existing_column(split_frame, SESSION_COLUMN_CANDIDATES)
    repetition_column = find_first_existing_column(split_frame, REPETITION_COLUMN_CANDIDATES)

    sample_ids = row_indices if sample_id_column is None else split_frame[sample_id_column].tolist()

    if session_column is None:
        session_ids = [None] * len(split_frame)
    else:
        session_ids = split_frame[session_column].tolist()

    if repetition_column is None:
        repetition_ids = [None] * len(split_frame)
    else:
        repetition_ids = split_frame[repetition_column].tolist()

    return pd.DataFrame(
        {
            "sample_id": sample_ids,
            "source_index": row_indices,
            "user_id": split_frame[label_column].astype(str).tolist(),
            "session_id": session_ids,
            "repetition_id": repetition_ids,
        }
    )


def build_embeddings_frame(
    dataframe: pd.DataFrame,
    row_indices: list[int],
    split_payload: SplitPayload,
    encoder: keras.Model,
    scaler: object,
    batch_size: int,
) -> pd.DataFrame:
    """Построить DataFrame с embedding-векторами для одного split.

    Аргументы:
        dataframe: Исходная таблица признаков.
        row_indices: Индексы строк текущего split.
        split_payload: Данные split-файла.
        encoder: Загруженная Keras-модель encoder.
        scaler: Загруженный StandardScaler.
        batch_size: Размер пакета при вычислении embedding-векторов.

    Возвращает:
        DataFrame со служебными колонками и embedding-координатами.
    """
    x_raw = dataframe.iloc[row_indices][split_payload.feature_columns]
    x_raw = x_raw.replace([np.inf, -np.inf], np.nan)

    if x_raw.isna().any().any():
        raise ValueError("В признаках обнаружены NaN или бесконечные значения.")

    x_scaled = scaler.transform(x_raw.to_numpy(dtype=np.float32)).astype(np.float32)
    embedding_matrix = extract_embeddings(encoder, x_scaled, batch_size=batch_size)

    metadata_frame = build_metadata_frame(
        dataframe=dataframe,
        row_indices=row_indices,
        label_column=split_payload.label_column,
    )

    embedding_columns = [f"embedding_{index:02d}" for index in range(embedding_matrix.shape[1])]
    embeddings_frame = pd.DataFrame(embedding_matrix, columns=embedding_columns)

    return pd.concat(
        [metadata_frame.reset_index(drop=True), embeddings_frame.reset_index(drop=True)],
        axis=1,
    )


def save_embeddings(
    dataframe: pd.DataFrame,
    split_payload: SplitPayload,
    encoder: keras.Model,
    scaler: object,
    paths: EmbeddingModelPaths,
    batch_size: int,
) -> dict[str, Path]:
    """Сохранить embedding-векторы для train, validation и test.

    Аргументы:
        dataframe: Исходная таблица признаков.
        split_payload: Данные split-файла.
        encoder: Загруженная Keras-модель encoder.
        scaler: Загруженный StandardScaler.
        paths: Пути к артефактам embedding-этапа.
        batch_size: Размер пакета при вычислении embedding-векторов.

    Возвращает:
        Словарь с путями к созданным CSV-файлам.
    """
    paths.ensure_directories()

    split_definitions = {
        "train": (split_payload.train_indices, TRAIN_EMBEDDINGS_FILENAME),
        "validation": (split_payload.validation_indices, VALIDATION_EMBEDDINGS_FILENAME),
        "test": (split_payload.test_indices, TEST_EMBEDDINGS_FILENAME),
    }

    output_paths: dict[str, Path] = {}

    for split_name, (row_indices, filename) in split_definitions.items():
        embeddings_frame = build_embeddings_frame(
            dataframe=dataframe,
            row_indices=row_indices,
            split_payload=split_payload,
            encoder=encoder,
            scaler=scaler,
            batch_size=batch_size,
        )

        output_path = paths.data_processed_dir / filename
        embeddings_frame.to_csv(output_path, index=False)
        output_paths[split_name] = output_path

        print(
            f"{split_name}: сохранено {len(embeddings_frame)} samples, "
            f"размерность embedding-вектора: "
            f"{count_embedding_columns(embeddings_frame)}"
        )

    return output_paths


def count_embedding_columns(dataframe: pd.DataFrame) -> int:
    """Посчитать количество embedding-координат в таблице."""
    return len([column for column in dataframe.columns if column.startswith("embedding_")])


def parse_args() -> argparse.Namespace:
    """Разобрать аргументы командной строки."""
    default_paths = EmbeddingModelPaths.from_source_dir()
    default_dataset_path = default_paths.project_root / "data" / "processed" / "cmu_features.csv"

    parser = argparse.ArgumentParser(
        description="Извлечение embedding-векторов для train, validation и test.",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=default_dataset_path,
        help="Путь к CSV-файлу с подготовленными признаками CMU.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Размер пакета при вычислении embedding-векторов.",
    )

    return parser.parse_args()


def main() -> None:
    """Выполнить извлечение embedding-векторов для всех split-ов."""
    args = parse_args()

    config = EmbeddingExtractionConfig(
        dataset_path=args.dataset,
        batch_size=args.batch_size,
    )

    paths = EmbeddingModelPaths.from_source_dir()
    split_path = paths.data_processed_dir / "cmu_embedding_split.json"

    dataframe = load_feature_table(config.dataset_path)
    split_payload = load_split_payload(split_path)
    validate_required_columns(dataframe, split_payload)

    encoder, scaler = load_encoder_and_scaler(paths)

    output_paths = save_embeddings(
        dataframe=dataframe,
        split_payload=split_payload,
        encoder=encoder,
        scaler=scaler,
        paths=paths,
        batch_size=config.batch_size,
    )

    print("Извлечение embedding-векторов завершено.")
    print(f"Использовано признаков: {len(split_payload.feature_columns)}")
    print(f"Колонка пользователя: {split_payload.label_column}")
    print(f"Train embeddings: {output_paths['train']}")
    print(f"Validation embeddings: {output_paths['validation']}")
    print(f"Test embeddings: {output_paths['test']}")


if __name__ == "__main__":
    main()
