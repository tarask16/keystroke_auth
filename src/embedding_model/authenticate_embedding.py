"""CLI-аутентификация через embedding-шаблон пользователя.

Файл размещается в каталоге:

    src/embedding_model/authenticate_embedding.py

Назначение файла:

    1. Загрузить encoder, scaler, embedding-шаблоны и threshold policy.
    2. Проверить один sample против заявленного пользователя.
    3. Выполнить batch-проверку выбранного split-а.
    4. Рассчитать FAR, FRR, число ложных допусков и ложных отказов.

Поддерживаемые режимы:

    Проверка одного sample из исходного CSV:

        python -m src.embedding_model.authenticate_embedding \
            --sample-index 0 \
            --claimed-user-id s011 \
            --distance-metric cosine \
            --policy per_user

    Batch-проверка по заранее извлечённым embedding-векторам:

        python -m src.embedding_model.authenticate_embedding \
            --batch \
            --split test \
            --distance-metric cosine \
            --policy per_user

Методологическое правило:

    Файл не подбирает threshold. Он только применяет уже сохранённую
    threshold policy, полученную на validation split на шаге 6.6.

Терминология:

- ``claimed_user_id`` — заявленный пользователь;
- ``actual_user_id`` — фактический пользователь, которому принадлежит sample;
- ``embedding-шаблон`` — эталонный embedding-вектор пользователя;
- ``threshold`` — порог расстояния для принятия решения;
- ``ACCEPT`` — sample принят как принадлежащий заявленному пользователю;
- ``REJECT`` — sample отклонён;
- ``ложный допуск`` — impostor-попытка ошибочно принята;
- ``ложный отказ`` — genuine-попытка ошибочно отклонена.
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

DEFAULT_DISTANCE_METRIC: Final[str] = "cosine"
DEFAULT_POLICY: Final[str] = "per_user"
DEFAULT_SPLIT: Final[str] = "test"

SUPPORTED_DISTANCE_METRICS: Final[tuple[str, ...]] = (
    "euclidean",
    "cosine",
    "manhattan",
)

SUPPORTED_POLICIES: Final[tuple[str, ...]] = (
    "global",
    "per_user",
    "guarded",
)

SUPPORTED_SPLITS: Final[tuple[str, ...]] = (
    "train",
    "validation",
    "test",
)


@dataclass(frozen=True)
class AuthenticationResources:
    """Загруженные артефакты embedding-аутентификации."""

    encoder: keras.Model
    scaler: object
    templates: dict[str, np.ndarray]
    thresholds: dict[str, float]
    feature_columns: list[str]
    label_column: str


@dataclass(frozen=True)
class AuthenticationResult:
    """Результат одной проверки пользователя."""

    sample_id: str
    source_index: int
    actual_user_id: str
    claimed_user_id: str
    distance_metric: str
    policy: str
    distance: float
    threshold: float
    decision: str
    is_genuine: bool
    is_correct: bool


@dataclass(frozen=True)
class BatchMetrics:
    """Метрики batch-проверки embedding-аутентификации."""

    split: str
    distance_metric: str
    policy: str
    genuine_trials: int
    impostor_trials: int
    genuine_accepts: int
    genuine_rejects: int
    impostor_accepts: int
    impostor_rejects: int
    far: float
    frr: float
    balanced_error: float


def load_json(path: Path) -> dict[str, object]:
    """Загрузить JSON-файл.

    Аргументы:
        path: Путь к JSON-файлу.

    Возвращает:
        JSON-совместимый словарь.

    Исключения:
        FileNotFoundError: Возникает, если файл не найден.
    """
    if not path.exists():
        raise FileNotFoundError(f"JSON-файл не найден: {path}")

    return json.loads(path.read_text(encoding="utf-8"))


def load_split_payload(paths: EmbeddingModelPaths) -> tuple[list[str], str]:
    """Загрузить список признаков и колонку пользователя из split-файла.

    Аргументы:
        paths: Пути к артефактам embedding-этапа.

    Возвращает:
        Кортеж ``(feature_columns, label_column)``.
    """
    split_path = paths.data_processed_dir / "cmu_embedding_split.json"
    payload = load_json(split_path)

    if "feature_columns" not in payload or "label_column" not in payload:
        raise ValueError("Split-файл не содержит feature_columns или label_column.")

    return list(payload["feature_columns"]), str(payload["label_column"])


def load_templates(path: Path) -> dict[str, np.ndarray]:
    """Загрузить embedding-шаблоны пользователей.

    Аргументы:
        path: Путь к ``user_templates_embedding.json``.

    Возвращает:
        Словарь ``user_id -> embedding-шаблон``.
    """
    payload = load_json(path)

    if "templates" not in payload:
        raise ValueError("Файл шаблонов не содержит раздел templates.")

    templates: dict[str, np.ndarray] = {}

    for user_id, template_payload in payload["templates"].items():
        embedding = template_payload.get("embedding")

        if embedding is None:
            raise ValueError(f"У пользователя {user_id} отсутствует embedding.")

        templates[str(user_id)] = np.asarray(embedding, dtype=np.float32)

    return templates


def load_thresholds(
    path: Path,
    distance_metric: str,
    policy: str,
) -> dict[str, float]:
    """Загрузить threshold выбранной metric/policy.

    Аргументы:
        path: Путь к ``user_thresholds_embedding.json``.
        distance_metric: Метрика расстояния.
        policy: Имя threshold policy.

    Возвращает:
        Словарь ``claimed_user_id -> threshold``.
    """
    validate_choice(distance_metric, SUPPORTED_DISTANCE_METRICS, "distance_metric")
    validate_choice(policy, SUPPORTED_POLICIES, "policy")

    payload = load_json(path)

    policies = payload.get("policies")
    if not isinstance(policies, dict):
        raise ValueError("Файл threshold policy не содержит раздел policies.")

    metric_payload = policies.get(distance_metric)
    if not isinstance(metric_payload, dict):
        raise ValueError(f"В threshold policy отсутствует метрика {distance_metric}.")

    if policy == "global":
        global_threshold = float(metric_payload["global_threshold"])
        per_user_thresholds = metric_payload.get("per_user_thresholds")

        if not isinstance(per_user_thresholds, dict):
            raise ValueError("Не найден список пользователей для global policy.")

        return {str(user_id): global_threshold for user_id in per_user_thresholds}

    threshold_key = f"{policy}_thresholds"
    thresholds = metric_payload.get(threshold_key)

    if not isinstance(thresholds, dict):
        raise ValueError(f"В threshold policy отсутствует раздел {threshold_key}.")

    return {str(user_id): float(threshold) for user_id, threshold in thresholds.items()}


def load_resources(
    paths: EmbeddingModelPaths,
    distance_metric: str,
    policy: str,
) -> AuthenticationResources:
    """Загрузить все артефакты для embedding-аутентификации.

    Аргументы:
        paths: Пути к артефактам embedding-этапа.
        distance_metric: Метрика расстояния.
        policy: Threshold policy.

    Возвращает:
        Объект с encoder, scaler, шаблонами, threshold и списком признаков.
    """
    if not paths.encoder_path.exists():
        raise FileNotFoundError(f"Файл encoder не найден: {paths.encoder_path}")

    if not paths.scaler_path.exists():
        raise FileNotFoundError(f"Файл scaler не найден: {paths.scaler_path}")

    encoder = keras.models.load_model(paths.encoder_path)
    scaler = joblib.load(paths.scaler_path)
    templates = load_templates(paths.templates_path)
    thresholds = load_thresholds(
        path=paths.thresholds_path,
        distance_metric=distance_metric,
        policy=policy,
    )
    feature_columns, label_column = load_split_payload(paths)

    return AuthenticationResources(
        encoder=encoder,
        scaler=scaler,
        templates=templates,
        thresholds=thresholds,
        feature_columns=feature_columns,
        label_column=label_column,
    )


def validate_choice(value: str, allowed_values: tuple[str, ...], name: str) -> None:
    """Проверить, что значение входит в допустимый набор."""
    if value not in allowed_values:
        raise ValueError(
            f"Некорректное значение {name}: {value}. "
            f"Допустимые значения: {', '.join(allowed_values)}."
        )


def calculate_distance(
    vector: np.ndarray,
    template: np.ndarray,
    metric: str,
) -> float:
    """Рассчитать расстояние между embedding-вектором и шаблоном.

    Аргументы:
        vector: Embedding-вектор sample.
        template: Embedding-шаблон заявленного пользователя.
        metric: Метрика расстояния.

    Возвращает:
        Значение distance. Чем меньше distance, тем ближе sample к шаблону.
    """
    if metric == "euclidean":
        return float(np.linalg.norm(vector - template, ord=2))

    if metric == "manhattan":
        return float(np.linalg.norm(vector - template, ord=1))

    if metric == "cosine":
        vector_norm = float(np.linalg.norm(vector, ord=2))
        template_norm = float(np.linalg.norm(template, ord=2))

        if vector_norm == 0.0 or template_norm == 0.0:
            return 1.0

        similarity = float(np.dot(vector, template) / (vector_norm * template_norm))
        similarity = float(np.clip(similarity, -1.0, 1.0))

        return 1.0 - similarity

    raise ValueError(f"Неизвестная метрика расстояния: {metric}")


def build_sample_embedding_from_source_row(
    dataframe: pd.DataFrame,
    sample_index: int,
    resources: AuthenticationResources,
) -> tuple[np.ndarray, str, str]:
    """Получить embedding-вектор sample из исходной таблицы признаков.

    Аргументы:
        dataframe: Исходная таблица ``cmu_features.csv``.
        sample_index: Индекс строки исходной таблицы.
        resources: Загруженные артефакты аутентификации.

    Возвращает:
        Кортеж ``(embedding-вектор, actual_user_id, sample_id)``.
    """
    if sample_index < 0 or sample_index >= len(dataframe):
        raise ValueError(
            f"sample_index вне диапазона: {sample_index}. "
            f"Допустимый диапазон: 0..{len(dataframe) - 1}."
        )

    missing_features = set(resources.feature_columns) - set(dataframe.columns)
    if missing_features:
        raise ValueError(
            f"В исходной таблице отсутствуют признаки: {', '.join(sorted(missing_features))}."
        )

    if resources.label_column not in dataframe.columns:
        raise ValueError(f"В исходной таблице отсутствует {resources.label_column}.")

    row = dataframe.iloc[[sample_index]]
    x_raw = row[resources.feature_columns].replace([np.inf, -np.inf], np.nan)

    if x_raw.isna().any().any():
        raise ValueError("В sample обнаружены NaN или бесконечные значения.")

    x_scaled = resources.scaler.transform(x_raw.to_numpy(dtype=np.float32)).astype(np.float32)
    embedding = extract_embeddings(resources.encoder, x_scaled, batch_size=1)[0]

    actual_user_id = str(row[resources.label_column].iloc[0])
    sample_id = str(sample_index)

    return embedding.astype(np.float32), actual_user_id, sample_id


def authenticate_embedding_vector(
    embedding: np.ndarray,
    actual_user_id: str,
    claimed_user_id: str,
    sample_id: str,
    source_index: int,
    distance_metric: str,
    policy: str,
    resources: AuthenticationResources,
) -> AuthenticationResult:
    """Проверить embedding-вектор против заявленного пользователя.

    Аргументы:
        embedding: Embedding-вектор текущего sample.
        actual_user_id: Фактический пользователь.
        claimed_user_id: Заявленный пользователь.
        sample_id: Идентификатор sample для отчёта.
        source_index: Индекс строки исходной таблицы.
        distance_metric: Метрика расстояния.
        policy: Threshold policy.
        resources: Загруженные артефакты аутентификации.

    Возвращает:
        Результат проверки ACCEPT/REJECT.
    """
    if claimed_user_id not in resources.templates:
        raise ValueError(f"Неизвестный claimed_user_id: {claimed_user_id}")

    if claimed_user_id not in resources.thresholds:
        raise ValueError(f"Для claimed_user_id нет threshold: {claimed_user_id}")

    distance = calculate_distance(
        vector=embedding,
        template=resources.templates[claimed_user_id],
        metric=distance_metric,
    )
    threshold = resources.thresholds[claimed_user_id]

    decision = "ACCEPT" if distance <= threshold else "REJECT"
    is_genuine = actual_user_id == claimed_user_id
    is_correct = (decision == "ACCEPT" and is_genuine) or (decision == "REJECT" and not is_genuine)

    return AuthenticationResult(
        sample_id=sample_id,
        source_index=source_index,
        actual_user_id=actual_user_id,
        claimed_user_id=claimed_user_id,
        distance_metric=distance_metric,
        policy=policy,
        distance=distance,
        threshold=threshold,
        decision=decision,
        is_genuine=is_genuine,
        is_correct=is_correct,
    )


def load_source_dataset(path: Path) -> pd.DataFrame:
    """Загрузить исходную таблицу временных признаков."""
    if not path.exists():
        raise FileNotFoundError(f"Файл датасета не найден: {path}")

    dataframe = pd.read_csv(path)

    if dataframe.empty:
        raise ValueError(f"Файл датасета пуст: {path}")

    return dataframe


def load_embeddings_for_split(paths: EmbeddingModelPaths, split: str) -> pd.DataFrame:
    """Загрузить embedding-векторы выбранного split-а.

    Аргументы:
        paths: Пути к артефактам embedding-этапа.
        split: Имя split-а: train, validation или test.

    Возвращает:
        DataFrame с embedding-векторами выбранного split-а.
    """
    validate_choice(split, SUPPORTED_SPLITS, "split")

    path = paths.data_processed_dir / f"embeddings_{split}.csv"

    if not path.exists():
        raise FileNotFoundError(f"Файл embedding-векторов не найден: {path}")

    dataframe = pd.read_csv(path)

    if dataframe.empty:
        raise ValueError(f"Файл embedding-векторов пуст: {path}")

    return dataframe


def detect_embedding_columns(dataframe: pd.DataFrame) -> list[str]:
    """Найти колонки embedding-вектора."""
    columns = [column for column in dataframe.columns if column.startswith("embedding_")]

    if not columns:
        raise ValueError("В таблице не найдены embedding-колонки.")

    return sorted(columns)


def run_batch_authentication(
    embeddings_frame: pd.DataFrame,
    resources: AuthenticationResources,
    split: str,
    distance_metric: str,
    policy: str,
) -> BatchMetrics:
    """Выполнить batch-проверку всех samples против всех claimed users.

    Аргументы:
        embeddings_frame: Таблица embedding-векторов выбранного split-а.
        resources: Загруженные артефакты аутентификации.
        split: Имя split-а.
        distance_metric: Метрика расстояния.
        policy: Threshold policy.

    Возвращает:
        Метрики batch-проверки.
    """
    embedding_columns = detect_embedding_columns(embeddings_frame)

    genuine_trials = 0
    impostor_trials = 0
    genuine_accepts = 0
    genuine_rejects = 0
    impostor_accepts = 0
    impostor_rejects = 0

    claimed_user_ids = sorted(resources.templates)

    for _, sample in embeddings_frame.iterrows():
        actual_user_id = str(sample["user_id"])
        embedding = sample[embedding_columns].to_numpy(dtype=np.float32)

        for claimed_user_id in claimed_user_ids:
            result = authenticate_embedding_vector(
                embedding=embedding,
                actual_user_id=actual_user_id,
                claimed_user_id=claimed_user_id,
                sample_id=str(sample.get("sample_id", "")),
                source_index=int(sample.get("source_index", -1)),
                distance_metric=distance_metric,
                policy=policy,
                resources=resources,
            )

            if result.is_genuine:
                genuine_trials += 1
                if result.decision == "ACCEPT":
                    genuine_accepts += 1
                else:
                    genuine_rejects += 1
            else:
                impostor_trials += 1
                if result.decision == "ACCEPT":
                    impostor_accepts += 1
                else:
                    impostor_rejects += 1

    far = float(impostor_accepts / impostor_trials) if impostor_trials else 0.0
    frr = float(genuine_rejects / genuine_trials) if genuine_trials else 0.0

    return BatchMetrics(
        split=split,
        distance_metric=distance_metric,
        policy=policy,
        genuine_trials=genuine_trials,
        impostor_trials=impostor_trials,
        genuine_accepts=genuine_accepts,
        genuine_rejects=genuine_rejects,
        impostor_accepts=impostor_accepts,
        impostor_rejects=impostor_rejects,
        far=far,
        frr=frr,
        balanced_error=float((far + frr) / 2.0),
    )


def print_authentication_result(result: AuthenticationResult) -> None:
    """Вывести результат одиночной проверки."""
    print("Результат embedding-аутентификации.")
    print(f"Sample ID: {result.sample_id}")
    print(f"Source index: {result.source_index}")
    print(f"Actual user ID: {result.actual_user_id}")
    print(f"Claimed user ID: {result.claimed_user_id}")
    print(f"Метрика расстояния: {result.distance_metric}")
    print(f"Threshold policy: {result.policy}")
    print(f"Distance: {result.distance:.8f}")
    print(f"Threshold: {result.threshold:.8f}")
    print(f"Decision: {result.decision}")
    print(f"Тип попытки: {'genuine' if result.is_genuine else 'impostor'}")
    print(f"Решение корректно: {result.is_correct}")


def print_batch_metrics(metrics: BatchMetrics) -> None:
    """Вывести метрики batch-проверки."""
    print("Batch-проверка embedding-аутентификации завершена.")
    print(f"Split: {metrics.split}")
    print(f"Метрика расстояния: {metrics.distance_metric}")
    print(f"Threshold policy: {metrics.policy}")
    print(f"Genuine trials: {metrics.genuine_trials}")
    print(f"Impostor trials: {metrics.impostor_trials}")
    print(f"Genuine accepts: {metrics.genuine_accepts}")
    print(f"Genuine rejects: {metrics.genuine_rejects}")
    print(f"Impostor accepts: {metrics.impostor_accepts}")
    print(f"Impostor rejects: {metrics.impostor_rejects}")
    print(f"FAR: {metrics.far:.6f}")
    print(f"FRR: {metrics.frr:.6f}")
    print(f"Balanced error: {metrics.balanced_error:.6f}")


def parse_args() -> argparse.Namespace:
    """Разобрать аргументы командной строки."""
    paths = EmbeddingModelPaths.from_source_dir()
    default_dataset = paths.project_root / "data" / "processed" / "cmu_features.csv"

    parser = argparse.ArgumentParser(
        description="CLI-аутентификация по embedding-шаблону пользователя.",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=default_dataset,
        help="Путь к исходной таблице временных признаков.",
    )
    parser.add_argument(
        "--sample-index",
        type=int,
        default=None,
        help="Индекс sample в исходной таблице признаков.",
    )
    parser.add_argument(
        "--claimed-user-id",
        type=str,
        default=None,
        help="Идентификатор заявленного пользователя.",
    )
    parser.add_argument(
        "--distance-metric",
        type=str,
        default=DEFAULT_DISTANCE_METRIC,
        choices=SUPPORTED_DISTANCE_METRICS,
        help="Метрика расстояния.",
    )
    parser.add_argument(
        "--policy",
        type=str,
        default=DEFAULT_POLICY,
        choices=SUPPORTED_POLICIES,
        help="Threshold policy.",
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Запустить batch-проверку выбранного split-а.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default=DEFAULT_SPLIT,
        choices=SUPPORTED_SPLITS,
        help="Split для batch-проверки.",
    )

    return parser.parse_args()


def main() -> None:
    """Выполнить одиночную или batch embedding-аутентификацию."""
    args = parse_args()

    paths = EmbeddingModelPaths.from_source_dir()
    resources = load_resources(
        paths=paths,
        distance_metric=args.distance_metric,
        policy=args.policy,
    )

    if args.batch:
        embeddings_frame = load_embeddings_for_split(paths, args.split)
        metrics = run_batch_authentication(
            embeddings_frame=embeddings_frame,
            resources=resources,
            split=args.split,
            distance_metric=args.distance_metric,
            policy=args.policy,
        )
        print_batch_metrics(metrics)
        return

    if args.sample_index is None or args.claimed_user_id is None:
        raise ValueError(
            "Для одиночной проверки необходимо указать --sample-index "
            "и --claimed-user-id. Для batch-проверки используйте --batch."
        )

    source_dataframe = load_source_dataset(args.dataset)
    embedding, actual_user_id, sample_id = build_sample_embedding_from_source_row(
        dataframe=source_dataframe,
        sample_index=args.sample_index,
        resources=resources,
    )

    result = authenticate_embedding_vector(
        embedding=embedding,
        actual_user_id=actual_user_id,
        claimed_user_id=args.claimed_user_id,
        sample_id=sample_id,
        source_index=args.sample_index,
        distance_metric=args.distance_metric,
        policy=args.policy,
        resources=resources,
    )

    print_authentication_result(result)


if __name__ == "__main__":
    main()
