"""Обучение embedding-классификатора для этапа 6 проекта Keystroke Auth.

Файл размещается в каталоге:

    src/embedding_model/train_embedding_classifier.py

Назначение файла:

    1. Загрузить подготовленный датасет с признаками клавиатурного почерка.
    2. Выбрать только временные признаки клавиатурного почерка.
    3. Исключить служебные числовые колонки: индекс CSV, номер строки,
       номер сессии, номер повторения и другие метаданные.
    4. Разделить данные на train, validation и test без утечки test split.
    5. Обучить embedding-классификатор.
    6. Извлечь и сохранить encoder.
    7. Сохранить scaler, label encoder, список признаков и метрики качества.
    8. Сохранить split-файл для воспроизводимости следующих задач этапа 6.

Важно:

    Для CMU Keystroke Dynamics Benchmark ожидается 31 временной признак.
    Если выбранное количество признаков отличается от 31, обучение
    останавливается, чтобы не допустить попадания служебной колонки в модель.

Артефакты сохраняются в существующих каталогах проекта:

    models/embedding_model/
    reports/embedding_model/
    data/processed/embedding_model/

Терминология:

- ``embedding-классификатор`` — нейросетевая модель, обучаемая как
  многоклассовый классификатор пользователей, но содержащая выделенный
  embedding-слой;
- ``encoder`` — часть embedding-классификатора от входного слоя до
  embedding-слоя включительно;
- ``временные признаки`` — признаки клавиатурной динамики: hold time,
  keydown-keydown latency, keyup-keydown latency и другие интервальные
  характеристики набора;
- ``validation split`` — часть данных для контроля обучения и подбора
  будущих порогов аутентификации;
- ``test split`` — часть данных только для финальной оценки качества.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Final

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow import keras

from src.embedding_model.embedding import (
    DEFAULT_EMBEDDING_DIM,
    DEFAULT_LEARNING_RATE,
    DEFAULT_RANDOM_SEED,
    EmbeddingModelPaths,
    build_embedding_classifier,
    build_encoder_from_classifier,
)

# Для текущего этапа целевой датасет — CMU Keystroke Dynamics Benchmark.
# В подготовленной таблице CMU должно быть 31 временное значение на попытку ввода.
DEFAULT_EXPECTED_FEATURE_COUNT: Final[int] = 31

# Префиксы штатных временных признаков CMU benchmark.
# H.*  — длительность удержания клавиши.
# DD.* — интервал между нажатиями двух клавиш.
# UD.* — интервал между отпусканием первой клавиши и нажатием следующей.
CMU_TIMING_PREFIXES: Final[tuple[str, ...]] = ("H.", "DD.", "UD.")

# Дополнительные префиксы оставлены для будущих датасетов и расширенных признаков.
# Они не мешают CMU, потому что в CMU обычно присутствуют только H.*, DD.* и UD.*.
EXTENDED_TIMING_PREFIXES: Final[tuple[str, ...]] = (
    "PP.",
    "RR.",
    "RP.",
    "PR.",
    "DU.",
    "UU.",
)

# Колонки, которые не должны попадать во входной вектор модели.
# Они описывают пользователя, сессию, номер попытки или служебный идентификатор.
META_COLUMNS: Final[set[str]] = {
    "subject",
    "subject_id",
    "user",
    "user_id",
    "participant",
    "participant_id",
    "session",
    "session_id",
    "sessionindex",
    "sessionIndex",
    "session_index",
    "session_idx",
    "rep",
    "repetition",
    "repetition_id",
    "rep_id",
    "sample_id",
    "sample",
    "id",
    "label",
    "target",
}

# Частые названия служебных колонок, возникающих при сохранении DataFrame в CSV.
# Такие колонки могут быть числовыми и ошибочно попадать в признаки.
INDEX_LIKE_COLUMNS: Final[set[str]] = {
    "",
    "index",
    "level_0",
    "row",
    "row_id",
    "row_index",
    "unnamed: 0",
    "unnamed: 0.1",
}

# Наиболее вероятные имена колонки пользователя в подготовленных CMU-файлах.
LABEL_COLUMN_CANDIDATES: Final[tuple[str, ...]] = (
    "user_id",
    "subject",
    "subject_id",
    "user",
    "participant_id",
    "participant",
    "label",
    "target",
)


@dataclass(frozen=True)
class TrainingConfig:
    """Параметры обучения embedding-классификатора."""

    dataset_path: Path
    random_seed: int = DEFAULT_RANDOM_SEED
    test_size: float = 0.20
    validation_size_from_trainval: float = 0.20
    embedding_dim: int = DEFAULT_EMBEDDING_DIM
    learning_rate: float = DEFAULT_LEARNING_RATE
    dropout_rate: float = 0.2
    batch_size: int = 64
    epochs: int = 120
    patience: int = 15
    expected_feature_count: int = DEFAULT_EXPECTED_FEATURE_COUNT
    allow_feature_count_mismatch: bool = False


@dataclass(frozen=True)
class SplitData:
    """Данные после разделения на train, validation и test."""

    x_train: np.ndarray
    x_validation: np.ndarray
    x_test: np.ndarray
    y_train: np.ndarray
    y_validation: np.ndarray
    y_test: np.ndarray
    train_indices: list[int]
    validation_indices: list[int]
    test_indices: list[int]


def set_reproducibility(random_seed: int) -> None:
    """Зафиксировать генераторы случайных чисел для воспроизводимости."""
    random.seed(random_seed)
    np.random.seed(random_seed)
    tf.keras.utils.set_random_seed(random_seed)


def load_feature_table(dataset_path: Path) -> pd.DataFrame:
    """Загрузить таблицу признаков клавиатурного почерка.

    Аргументы:
        dataset_path: Путь к CSV-файлу с подготовленными признаками.

    Возвращает:
        DataFrame с признаками и идентификатором пользователя.

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


def detect_label_column(dataframe: pd.DataFrame) -> str:
    """Определить колонку с идентификатором пользователя.

    Аргументы:
        dataframe: Таблица признаков.

    Возвращает:
        Имя колонки, содержащей идентификатор пользователя.

    Исключения:
        ValueError: Возникает, если подходящая колонка не найдена.
    """
    for column in LABEL_COLUMN_CANDIDATES:
        if column in dataframe.columns:
            return column

    raise ValueError(
        "Не найдена колонка пользователя. Ожидалась одна из колонок: "
        f"{', '.join(LABEL_COLUMN_CANDIDATES)}."
    )


def normalize_column_name(column: str) -> str:
    """Привести имя колонки к виду для устойчивого сравнения."""
    return column.strip().lower()


def is_index_like_column(column: str) -> bool:
    """Проверить, похожа ли колонка на служебный индекс CSV."""
    normalized = normalize_column_name(column)

    if normalized in INDEX_LIKE_COLUMNS:
        return True

    return normalized.startswith("unnamed:")


def is_metadata_column(column: str, label_column: str) -> bool:
    """Проверить, является ли колонка метаданными, а не временным признаком."""
    normalized = normalize_column_name(column)
    normalized_label = normalize_column_name(label_column)

    if normalized == normalized_label:
        return True

    if normalized in {normalize_column_name(item) for item in META_COLUMNS}:
        return True

    return is_index_like_column(column)


def is_timing_feature_column(column: str) -> bool:
    """Проверить, похожа ли колонка на временной признак клавиатурной динамики."""
    allowed_prefixes = CMU_TIMING_PREFIXES + EXTENDED_TIMING_PREFIXES
    return column.startswith(allowed_prefixes)


def detect_feature_columns(
    dataframe: pd.DataFrame,
    label_column: str,
    expected_feature_count: int,
    allow_feature_count_mismatch: bool,
) -> list[str]:
    """Определить колонки временных признаков.

    Для CMU benchmark приоритетно используются колонки с префиксами
    ``H.``, ``DD.`` и ``UD.``. Это защищает модель от случайного попадания
    служебных числовых колонок, например ``Unnamed: 0``.

    Если таких колонок нет, используется резервный режим: выбираются все
    числовые колонки, кроме метаданных.

    Аргументы:
        dataframe: Таблица признаков.
        label_column: Имя колонки с идентификатором пользователя.
        expected_feature_count: Ожидаемое количество временных признаков.
        allow_feature_count_mismatch: Разрешить обучение при несовпадении
            количества признаков с ожидаемым значением.

    Возвращает:
        Список колонок, используемых как входной вектор модели.

    Исключения:
        ValueError: Возникает, если признаки не найдены или их количество
            отличается от ожидаемого.
    """
    timing_columns = [
        column
        for column in dataframe.columns
        if is_timing_feature_column(column) and pd.api.types.is_numeric_dtype(dataframe[column])
    ]

    if timing_columns:
        feature_columns = timing_columns
    else:
        feature_columns = [
            column
            for column in dataframe.columns
            if (
                not is_metadata_column(column, label_column)
                and pd.api.types.is_numeric_dtype(dataframe[column])
            )
        ]

    if not feature_columns:
        raise ValueError("Не найдены числовые колонки временных признаков.")

    if len(feature_columns) != expected_feature_count and not allow_feature_count_mismatch:
        all_numeric_columns = [
            column
            for column in dataframe.columns
            if pd.api.types.is_numeric_dtype(dataframe[column])
        ]
        excluded_numeric_columns = [
            column for column in all_numeric_columns if column not in feature_columns
        ]

        raise ValueError(
            "Некорректное количество временных признаков. "
            f"Ожидалось: {expected_feature_count}, выбрано: {len(feature_columns)}. "
            f"Выбранные признаки: {feature_columns}. "
            f"Исключённые числовые колонки: {excluded_numeric_columns}. "
            "Если вы намеренно используете другой датасет или расширенный набор признаков, "
            "запустите скрипт с параметром --allow-feature-count-mismatch."
        )

    return feature_columns


def prepare_features_and_labels(
    dataframe: pd.DataFrame,
    expected_feature_count: int,
    allow_feature_count_mismatch: bool,
) -> tuple[np.ndarray, np.ndarray, list[str], str, LabelEncoder]:
    """Подготовить матрицу признаков и закодированные метки пользователей.

    Аргументы:
        dataframe: Таблица признаков клавиатурного почерка.
        expected_feature_count: Ожидаемое количество временных признаков.
        allow_feature_count_mismatch: Разрешить несовпадение числа признаков.

    Возвращает:
        Кортеж из матрицы признаков, меток, списка признаков, имени колонки
        пользователя и обученного label encoder.
    """
    label_column = detect_label_column(dataframe)
    feature_columns = detect_feature_columns(
        dataframe=dataframe,
        label_column=label_column,
        expected_feature_count=expected_feature_count,
        allow_feature_count_mismatch=allow_feature_count_mismatch,
    )

    features = dataframe[feature_columns].replace([np.inf, -np.inf], np.nan)

    if features.isna().any().any():
        raise ValueError(
            "В матрице признаков обнаружены NaN или бесконечные значения. "
            "Перед обучением выполните очистку данных."
        )

    labels_raw = dataframe[label_column].astype(str).to_numpy()

    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(labels_raw)

    x = features.to_numpy(dtype=np.float32)
    y = labels.astype(np.int64)

    return x, y, feature_columns, label_column, label_encoder


def make_stratified_split(
    x: np.ndarray,
    y: np.ndarray,
    config: TrainingConfig,
) -> SplitData:
    """Разделить данные на train, validation и test с сохранением баланса классов.

    Сначала отделяется test split. Затем оставшаяся часть делится на train и
    validation. Test split далее не используется для подбора порогов
    аутентификации.

    Аргументы:
        x: Матрица признаков.
        y: Закодированные метки пользователей.
        config: Параметры обучения.

    Возвращает:
        Объект с разделёнными данными и исходными индексами строк.
    """
    all_indices = np.arange(len(y))

    trainval_indices, test_indices = train_test_split(
        all_indices,
        test_size=config.test_size,
        random_state=config.random_seed,
        stratify=y,
    )

    train_indices, validation_indices = train_test_split(
        trainval_indices,
        test_size=config.validation_size_from_trainval,
        random_state=config.random_seed,
        stratify=y[trainval_indices],
    )

    return SplitData(
        x_train=x[train_indices],
        x_validation=x[validation_indices],
        x_test=x[test_indices],
        y_train=y[train_indices],
        y_validation=y[validation_indices],
        y_test=y[test_indices],
        train_indices=train_indices.astype(int).tolist(),
        validation_indices=validation_indices.astype(int).tolist(),
        test_indices=test_indices.astype(int).tolist(),
    )


def scale_split_data(split_data: SplitData) -> tuple[SplitData, StandardScaler]:
    """Нормализовать признаки через StandardScaler без утечки test split.

    Scaler обучается только на train split. Validation и test преобразуются
    только методом ``transform``.

    Аргументы:
        split_data: Данные после разделения на train, validation и test.

    Возвращает:
        Новый объект SplitData с нормализованными признаками и обученный scaler.
    """
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(split_data.x_train).astype(np.float32)
    x_validation_scaled = scaler.transform(split_data.x_validation).astype(np.float32)
    x_test_scaled = scaler.transform(split_data.x_test).astype(np.float32)

    scaled_split = SplitData(
        x_train=x_train_scaled,
        x_validation=x_validation_scaled,
        x_test=x_test_scaled,
        y_train=split_data.y_train,
        y_validation=split_data.y_validation,
        y_test=split_data.y_test,
        train_indices=split_data.train_indices,
        validation_indices=split_data.validation_indices,
        test_indices=split_data.test_indices,
    )

    return scaled_split, scaler


def train_model(
    split_data: SplitData,
    input_dim: int,
    num_classes: int,
    config: TrainingConfig,
) -> keras.Model:
    """Обучить embedding-классификатор.

    Аргументы:
        split_data: Нормализованные train, validation и test данные.
        input_dim: Количество входных признаков.
        num_classes: Количество пользователей-классов.
        config: Параметры обучения.

    Возвращает:
        Обученную Keras-модель embedding-классификатора.
    """
    classifier = build_embedding_classifier(
        input_dim=input_dim,
        num_classes=num_classes,
        embedding_dim=config.embedding_dim,
        dropout_rate=config.dropout_rate,
        learning_rate=config.learning_rate,
    )

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=config.patience,
            restore_best_weights=True,
        )
    ]

    classifier.fit(
        split_data.x_train,
        split_data.y_train,
        validation_data=(split_data.x_validation, split_data.y_validation),
        epochs=config.epochs,
        batch_size=config.batch_size,
        callbacks=callbacks,
        verbose=2,
    )

    return classifier


def calculate_classifier_metrics(
    classifier: keras.Model,
    split_data: SplitData,
) -> pd.DataFrame:
    """Рассчитать метрики классификации для train, validation и test.

    Эти метрики показывают качество идентификации пользователя. Они не заменяют
    будущие биометрические метрики FAR, FRR и EER, которые будут рассчитаны
    отдельно для проверки по расстоянию.

    Аргументы:
        classifier: Обученный embedding-классификатор.
        split_data: Нормализованные train, validation и test данные.

    Возвращает:
        DataFrame с accuracy, macro F1 и weighted F1.
    """
    rows: list[dict[str, float | str | int]] = []

    for split_name, x_split, y_split in (
        ("train", split_data.x_train, split_data.y_train),
        ("validation", split_data.x_validation, split_data.y_validation),
        ("test", split_data.x_test, split_data.y_test),
    ):
        probabilities = classifier.predict(x_split, verbose=0)
        y_pred = np.argmax(probabilities, axis=1)

        rows.append(
            {
                "split": split_name,
                "samples": int(len(y_split)),
                "accuracy": float(accuracy_score(y_split, y_pred)),
                "macro_f1": float(f1_score(y_split, y_pred, average="macro")),
                "weighted_f1": float(f1_score(y_split, y_pred, average="weighted")),
            }
        )

    return pd.DataFrame(rows)


def save_training_artifacts(
    classifier: keras.Model,
    scaler: StandardScaler,
    label_encoder: LabelEncoder,
    metrics: pd.DataFrame,
    split_data: SplitData,
    feature_columns: list[str],
    label_column: str,
    config: TrainingConfig,
    paths: EmbeddingModelPaths,
) -> None:
    """Сохранить модели, scaler, label encoder, метрики и split-файл.

    Аргументы:
        classifier: Обученный embedding-классификатор.
        scaler: StandardScaler, обученный только на train split.
        label_encoder: Кодировщик идентификаторов пользователей.
        metrics: Таблица метрик классификации.
        split_data: Данные с индексами train, validation и test.
        feature_columns: Список колонок временных признаков.
        label_column: Имя колонки пользователя.
        config: Параметры обучения.
        paths: Пути к артефактам embedding-этапа.
    """
    paths.ensure_directories()

    encoder = build_encoder_from_classifier(classifier)

    classifier.save(paths.classifier_path)
    encoder.save(paths.encoder_path)
    joblib.dump(scaler, paths.scaler_path)
    joblib.dump(label_encoder, paths.label_encoder_path)

    metrics_path = paths.reports_dir / "embedding_classifier_metrics.csv"
    metrics.to_csv(metrics_path, index=False)

    feature_columns_path = paths.reports_dir / "embedding_feature_columns.txt"
    feature_columns_path.write_text(
        "\n".join(feature_columns) + "\n",
        encoding="utf-8",
    )

    split_path = paths.data_processed_dir / "cmu_embedding_split.json"
    split_payload = {
        "label_column": label_column,
        "feature_columns": feature_columns,
        "classes": label_encoder.classes_.tolist(),
        "train_indices": split_data.train_indices,
        "validation_indices": split_data.validation_indices,
        "test_indices": split_data.test_indices,
        "config": {
            **asdict(config),
            "dataset_path": str(config.dataset_path),
        },
    }
    split_path.write_text(
        json.dumps(split_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    """Разобрать аргументы командной строки."""
    default_paths = EmbeddingModelPaths.from_source_dir()
    default_dataset_path = default_paths.project_root / "data" / "processed" / "cmu_features.csv"

    parser = argparse.ArgumentParser(
        description="Обучение embedding-классификатора клавиатурного почерка.",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=default_dataset_path,
        help="Путь к CSV-файлу с подготовленными признаками CMU.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=120,
        help="Максимальное количество эпох обучения.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Размер пакета обучения.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=15,
        help="Терпение EarlyStopping по val_loss.",
    )
    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=DEFAULT_EMBEDDING_DIM,
        help="Размерность embedding-вектора.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=DEFAULT_RANDOM_SEED,
        help="Seed для воспроизводимости эксперимента.",
    )
    parser.add_argument(
        "--expected-feature-count",
        type=int,
        default=DEFAULT_EXPECTED_FEATURE_COUNT,
        help="Ожидаемое количество временных признаков. Для CMU используется 31.",
    )
    parser.add_argument(
        "--allow-feature-count-mismatch",
        action="store_true",
        help=(
            "Разрешить обучение при несовпадении количества выбранных признаков "
            "с ожидаемым значением."
        ),
    )

    return parser.parse_args()


def main() -> None:
    """Выполнить полный цикл обучения embedding-классификатора."""
    args = parse_args()

    config = TrainingConfig(
        dataset_path=args.dataset,
        random_seed=args.random_seed,
        embedding_dim=args.embedding_dim,
        batch_size=args.batch_size,
        epochs=args.epochs,
        patience=args.patience,
        expected_feature_count=args.expected_feature_count,
        allow_feature_count_mismatch=args.allow_feature_count_mismatch,
    )

    set_reproducibility(config.random_seed)

    paths = EmbeddingModelPaths.from_source_dir()
    paths.ensure_directories()

    dataframe = load_feature_table(config.dataset_path)
    x, y, feature_columns, label_column, label_encoder = prepare_features_and_labels(
        dataframe=dataframe,
        expected_feature_count=config.expected_feature_count,
        allow_feature_count_mismatch=config.allow_feature_count_mismatch,
    )

    split_data = make_stratified_split(x, y, config)
    split_data, scaler = scale_split_data(split_data)

    classifier = train_model(
        split_data=split_data,
        input_dim=len(feature_columns),
        num_classes=len(label_encoder.classes_),
        config=config,
    )

    metrics = calculate_classifier_metrics(classifier, split_data)

    save_training_artifacts(
        classifier=classifier,
        scaler=scaler,
        label_encoder=label_encoder,
        metrics=metrics,
        split_data=split_data,
        feature_columns=feature_columns,
        label_column=label_column,
        config=config,
        paths=paths,
    )

    print("Обучение embedding-классификатора завершено.")
    print(f"Количество признаков: {len(feature_columns)}")
    print(f"Колонка пользователя: {label_column}")
    print(f"Количество пользователей: {len(label_encoder.classes_)}")
    print(f"Путь к классификатору: {paths.classifier_path}")
    print(f"Путь к encoder: {paths.encoder_path}")
    print(f"Путь к scaler: {paths.scaler_path}")
    print(f"Путь к label encoder: {paths.label_encoder_path}")
    print(f"Путь к метрикам: {paths.reports_dir / 'embedding_classifier_metrics.csv'}")
    print(f"Путь к списку признаков: {paths.reports_dir / 'embedding_feature_columns.txt'}")
    print(f"Путь к split-файлу: {paths.data_processed_dir / 'cmu_embedding_split.json'}")
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
