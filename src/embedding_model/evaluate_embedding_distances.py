"""Диагностика расстояний embedding-аутентификации для этапа 6.

Файл размещается в каталоге:

    src/embedding_model/evaluate_embedding_distances.py

Назначение файла:

    1. Загрузить embedding-шаблоны пользователей, сформированные по train split.
    2. Загрузить embedding-векторы validation и test split.
    3. Для каждого образца рассчитать расстояние до шаблона заявленного пользователя.
    4. Сформировать genuine-попытки и impostor-попытки.
    5. Рассчитать диагностические метрики разделимости:
       FAR, FRR, EER, ROC AUC, средние расстояния и separation margin.
       EER рассчитывается быстрым способом через ROC-кривую, без ручного
       перебора всех уникальных distance-threshold.
    6. Сохранить отчёты для дальнейшего подбора threshold policy.

Входные файлы:

    users/embedding_model/user_templates_embedding.json
    data/processed/embedding_model/embeddings_validation.csv
    data/processed/embedding_model/embeddings_test.csv

Выходные файлы:

    reports/embedding_model/embedding_distance_diagnostics.csv
    reports/embedding_model/embedding_distance_trials_validation.csv
    reports/embedding_model/embedding_distance_trials_test.csv

Методологическое правило:

    Этот файл выполняет диагностику расстояний. Рабочие пороги
    аутентификации не фиксируются здесь как policy. Подбор global,
    per-user и guarded thresholds выполняется на следующем шаге 6.6
    только по validation/calibration split.

Терминология:

- ``genuine-попытка`` — проверка образца против собственного пользователя;
- ``impostor-попытка`` — проверка чужого образца против заявленного пользователя;
- ``FAR`` — доля ложных допусков impostor-попыток;
- ``FRR`` — доля ложных отказов genuine-попыток;
- ``EER`` — точка приблизительного равенства FAR и FRR;
- ``ROC AUC`` — качество разделения genuine и impostor попыток;
- ``separation margin`` — разность между средним impostor-расстоянием
  и средним genuine-расстоянием.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

from src.embedding_model.embedding import EmbeddingModelPaths

# Имена входных файлов с embedding-векторами.
VALIDATION_EMBEDDINGS_FILENAME: Final[str] = "embeddings_validation.csv"
TEST_EMBEDDINGS_FILENAME: Final[str] = "embeddings_test.csv"

# Имена выходных отчётов.
DIAGNOSTICS_FILENAME: Final[str] = "embedding_distance_diagnostics.csv"
VALIDATION_TRIALS_FILENAME: Final[str] = "embedding_distance_trials_validation.csv"
TEST_TRIALS_FILENAME: Final[str] = "embedding_distance_trials_test.csv"

# Поддерживаемые метрики расстояния.
SUPPORTED_DISTANCE_METRICS: Final[tuple[str, ...]] = (
    "euclidean",
    "cosine",
    "manhattan",
)


@dataclass(frozen=True)
class DistanceEvaluationConfig:
    """Параметры диагностики embedding-расстояний."""

    save_trials: bool = True


@dataclass(frozen=True)
class DistanceDiagnostics:
    """Итоговые диагностические метрики для одной метрики расстояния."""

    split: str
    distance_metric: str
    genuine_trials: int
    impostor_trials: int
    genuine_distance_mean: float
    genuine_distance_std: float
    genuine_distance_median: float
    impostor_distance_mean: float
    impostor_distance_std: float
    impostor_distance_median: float
    separation_margin: float
    roc_auc: float
    eer: float
    eer_threshold: float
    far_at_eer_threshold: float
    frr_at_eer_threshold: float


def load_templates(path: Path) -> dict[str, np.ndarray]:
    """Загрузить embedding-шаблоны пользователей.

    Аргументы:
        path: Путь к JSON-файлу ``user_templates_embedding.json``.

    Возвращает:
        Словарь ``user_id -> embedding-шаблон``.

    Исключения:
        FileNotFoundError: Возникает, если файл шаблонов не найден.
        ValueError: Возникает, если структура файла некорректна.
    """
    if not path.exists():
        raise FileNotFoundError(f"Файл embedding-шаблонов не найден: {path}")

    payload = json.loads(path.read_text(encoding="utf-8"))

    if "templates" not in payload:
        raise ValueError("Файл шаблонов не содержит раздел 'templates'.")

    templates: dict[str, np.ndarray] = {}

    for user_id, template_payload in payload["templates"].items():
        if "embedding" not in template_payload:
            raise ValueError(f"У пользователя {user_id} отсутствует embedding-шаблон.")

        template = np.asarray(template_payload["embedding"], dtype=np.float32)

        if template.ndim != 1:
            raise ValueError(f"Embedding-шаблон пользователя {user_id} должен быть одномерным.")

        templates[str(user_id)] = template

    if not templates:
        raise ValueError("Файл шаблонов не содержит ни одного пользователя.")

    return templates


def load_embeddings_table(path: Path) -> pd.DataFrame:
    """Загрузить таблицу embedding-векторов split-а.

    Аргументы:
        path: Путь к CSV-файлу с embedding-векторами.

    Возвращает:
        DataFrame с колонкой ``user_id`` и embedding-координатами.

    Исключения:
        FileNotFoundError: Возникает, если файл не найден.
        ValueError: Возникает, если таблица пуста или не содержит ``user_id``.
    """
    if not path.exists():
        raise FileNotFoundError(f"Файл embedding-векторов не найден: {path}")

    dataframe = pd.read_csv(path)

    if dataframe.empty:
        raise ValueError(f"Файл embedding-векторов пуст: {path}")

    if "user_id" not in dataframe.columns:
        raise ValueError(f"В файле отсутствует колонка user_id: {path}")

    return dataframe


def detect_embedding_columns(dataframe: pd.DataFrame) -> list[str]:
    """Найти колонки embedding-вектора.

    Аргументы:
        dataframe: Таблица embedding-векторов.

    Возвращает:
        Отсортированный список колонок ``embedding_00``, ``embedding_01`` и далее.

    Исключения:
        ValueError: Возникает, если embedding-колонки не найдены.
    """
    embedding_columns = [column for column in dataframe.columns if column.startswith("embedding_")]

    if not embedding_columns:
        raise ValueError("В таблице не найдены embedding-колонки.")

    return sorted(embedding_columns)


def validate_template_dimensions(
    templates: dict[str, np.ndarray],
    embedding_dim: int,
) -> None:
    """Проверить размерность всех embedding-шаблонов.

    Аргументы:
        templates: Словарь пользовательских embedding-шаблонов.
        embedding_dim: Ожидаемая размерность embedding-вектора.

    Исключения:
        ValueError: Возникает, если размерности не совпадают.
    """
    for user_id, template in templates.items():
        if template.shape[0] != embedding_dim:
            raise ValueError(
                f"Размерность шаблона пользователя {user_id} равна "
                f"{template.shape[0]}, ожидалось {embedding_dim}."
            )


def calculate_distance(
    vector: np.ndarray,
    template: np.ndarray,
    metric: str,
) -> float:
    """Рассчитать расстояние между embedding-вектором и шаблоном.

    Аргументы:
        vector: Embedding-вектор текущего образца.
        template: Embedding-шаблон заявленного пользователя.
        metric: Название метрики расстояния.

    Возвращает:
        Значение расстояния. Чем меньше расстояние, тем ближе образец
        к шаблону заявленного пользователя.

    Исключения:
        ValueError: Возникает для неизвестной метрики расстояния.
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

        cosine_similarity = float(np.dot(vector, template) / (vector_norm * template_norm))
        cosine_similarity = float(np.clip(cosine_similarity, -1.0, 1.0))

        return 1.0 - cosine_similarity

    raise ValueError(f"Неизвестная метрика расстояния: {metric}")


def build_distance_trials_for_metric(
    embeddings_frame: pd.DataFrame,
    templates: dict[str, np.ndarray],
    embedding_columns: list[str],
    split_name: str,
    metric: str,
) -> pd.DataFrame:
    """Сформировать таблицу genuine/impostor-попыток для одной метрики.

    Для каждого образца создаётся 51 проверка: одна genuine-попытка против
    собственного пользователя и 50 impostor-попыток против остальных
    пользователей.

    Аргументы:
        embeddings_frame: Таблица embedding-векторов выбранного split-а.
        templates: Словарь пользовательских embedding-шаблонов.
        embedding_columns: Список embedding-координат.
        split_name: Имя split-а: ``validation`` или ``test``.
        metric: Метрика расстояния.

    Возвращает:
        DataFrame с расстояниями и признаком genuine/impostor.
    """
    template_user_ids = sorted(templates)
    rows: list[dict[str, float | int | str | bool]] = []

    for _, sample in embeddings_frame.iterrows():
        actual_user_id = str(sample["user_id"])
        vector = sample[embedding_columns].to_numpy(dtype=np.float32)

        sample_id = sample.get("sample_id", "")
        source_index = sample.get("source_index", "")

        for claimed_user_id in template_user_ids:
            distance = calculate_distance(
                vector=vector,
                template=templates[claimed_user_id],
                metric=metric,
            )
            is_genuine = actual_user_id == claimed_user_id

            rows.append(
                {
                    "split": split_name,
                    "distance_metric": metric,
                    "sample_id": sample_id,
                    "source_index": source_index,
                    "actual_user_id": actual_user_id,
                    "claimed_user_id": claimed_user_id,
                    "is_genuine": bool(is_genuine),
                    "distance": distance,
                }
            )

    return pd.DataFrame(rows)


def calculate_far_frr_at_threshold(
    labels: np.ndarray,
    distances: np.ndarray,
    threshold: float,
) -> tuple[float, float]:
    """Рассчитать FAR и FRR для заданного distance-threshold.

    Решение принимается по правилу:

        distance <= threshold -> ACCEPT
        distance > threshold  -> REJECT

    Аргументы:
        labels: Массив меток, где 1 — genuine-попытка, 0 — impostor-попытка.
        distances: Массив расстояний.
        threshold: Проверяемый порог расстояния.

    Возвращает:
        Кортеж ``(FAR, FRR)``.
    """
    accepted = distances <= threshold

    genuine_mask = labels == 1
    impostor_mask = labels == 0

    false_accepts = np.sum(accepted & impostor_mask)
    false_rejects = np.sum(~accepted & genuine_mask)

    impostor_trials = np.sum(impostor_mask)
    genuine_trials = np.sum(genuine_mask)

    far = float(false_accepts / impostor_trials) if impostor_trials else 0.0
    frr = float(false_rejects / genuine_trials) if genuine_trials else 0.0

    return far, frr


def calculate_eer(
    labels: np.ndarray,
    distances: np.ndarray,
) -> tuple[float, float, float, float]:
    """Быстро рассчитать EER и диагностический distance-threshold.

    Для distance-based verification меньшая дистанция означает большее
    сходство с embedding-шаблоном пользователя. Поэтому для построения
    ROC-кривой используется score = -distance: чем больше score, тем выше
    уверенность, что попытка является genuine.

    Такой расчёт не перебирает все уникальные distance-threshold вручную и
    не пересчитывает FAR/FRR на каждом шаге. Это устраняет зависание на
    больших таблицах genuine/impostor-попыток.

    Аргументы:
        labels: Массив меток, где 1 — genuine-попытка, 0 — impostor-попытка.
        distances: Массив расстояний.

    Возвращает:
        Кортеж ``(EER, threshold, FAR, FRR)`` в точке минимального различия
        между FAR и FRR.

    Исключения:
        ValueError: Возникает, если в данных нет genuine или impostor попыток.
    """
    labels = np.asarray(labels, dtype=np.int8)
    distances = np.asarray(distances, dtype=np.float64)

    if not np.any(labels == 1):
        raise ValueError("Невозможно рассчитать EER: нет genuine-попыток.")

    if not np.any(labels == 0):
        raise ValueError("Невозможно рассчитать EER: нет impostor-попыток.")

    scores = -distances
    far_values, true_accept_rate_values, score_thresholds = roc_curve(
        labels,
        scores,
        pos_label=1,
    )

    frr_values = 1.0 - true_accept_rate_values

    finite_mask = np.isfinite(score_thresholds)
    if not np.any(finite_mask):
        raise ValueError("Невозможно рассчитать EER: нет конечных threshold.")

    far_values = far_values[finite_mask]
    frr_values = frr_values[finite_mask]
    score_thresholds = score_thresholds[finite_mask]

    best_index = int(np.argmin(np.abs(far_values - frr_values)))

    best_far = float(far_values[best_index])
    best_frr = float(frr_values[best_index])
    eer = float((best_far + best_frr) / 2.0)

    # Обратное преобразование score-threshold в distance-threshold.
    # score = -distance, поэтому distance_threshold = -score_threshold.
    best_distance_threshold = float(-score_thresholds[best_index])

    return eer, best_distance_threshold, best_far, best_frr


def calculate_diagnostics(
    trials: pd.DataFrame,
    split_name: str,
    metric: str,
) -> DistanceDiagnostics:
    """Рассчитать диагностические метрики по таблице попыток.

    Аргументы:
        trials: Таблица genuine/impostor-попыток.
        split_name: Имя split-а.
        metric: Метрика расстояния.

    Возвращает:
        Объект с диагностическими метриками.
    """
    labels = trials["is_genuine"].astype(int).to_numpy()
    distances = trials["distance"].to_numpy(dtype=np.float64)

    genuine_distances = distances[labels == 1]
    impostor_distances = distances[labels == 0]

    if len(genuine_distances) == 0 or len(impostor_distances) == 0:
        raise ValueError("Для диагностики нужны и genuine, и impostor попытки.")

    eer, eer_threshold, far_at_eer, frr_at_eer = calculate_eer(
        labels=labels,
        distances=distances,
    )

    # Для ROC AUC большее значение score должно соответствовать genuine-попытке.
    # Поскольку для distance-based verification меньшая дистанция лучше,
    # используем score = -distance.
    roc_auc = float(roc_auc_score(labels, -distances))

    genuine_mean = float(np.mean(genuine_distances))
    impostor_mean = float(np.mean(impostor_distances))

    return DistanceDiagnostics(
        split=split_name,
        distance_metric=metric,
        genuine_trials=int(len(genuine_distances)),
        impostor_trials=int(len(impostor_distances)),
        genuine_distance_mean=genuine_mean,
        genuine_distance_std=float(np.std(genuine_distances)),
        genuine_distance_median=float(np.median(genuine_distances)),
        impostor_distance_mean=impostor_mean,
        impostor_distance_std=float(np.std(impostor_distances)),
        impostor_distance_median=float(np.median(impostor_distances)),
        separation_margin=float(impostor_mean - genuine_mean),
        roc_auc=roc_auc,
        eer=eer,
        eer_threshold=eer_threshold,
        far_at_eer_threshold=far_at_eer,
        frr_at_eer_threshold=frr_at_eer,
    )


def evaluate_split(
    embeddings_frame: pd.DataFrame,
    templates: dict[str, np.ndarray],
    embedding_columns: list[str],
    split_name: str,
    save_trials_path: Path | None,
) -> list[DistanceDiagnostics]:
    """Выполнить диагностику расстояний для одного split-а.

    Аргументы:
        embeddings_frame: Таблица embedding-векторов split-а.
        templates: Словарь embedding-шаблонов пользователей.
        embedding_columns: Список embedding-координат.
        split_name: Имя split-а.
        save_trials_path: Путь для сохранения trial-level CSV или ``None``.

    Возвращает:
        Список диагностик для всех поддерживаемых метрик расстояния.
    """
    diagnostics: list[DistanceDiagnostics] = []
    all_trials: list[pd.DataFrame] = []

    for metric in SUPPORTED_DISTANCE_METRICS:
        trials = build_distance_trials_for_metric(
            embeddings_frame=embeddings_frame,
            templates=templates,
            embedding_columns=embedding_columns,
            split_name=split_name,
            metric=metric,
        )

        diagnostics.append(
            calculate_diagnostics(
                trials=trials,
                split_name=split_name,
                metric=metric,
            )
        )

        all_trials.append(trials)

    if save_trials_path is not None:
        trials_frame = pd.concat(all_trials, ignore_index=True)
        trials_frame.to_csv(save_trials_path, index=False)

    return diagnostics


def save_diagnostics(
    diagnostics: list[DistanceDiagnostics],
    path: Path,
) -> None:
    """Сохранить общий отчёт диагностики расстояний.

    Аргументы:
        diagnostics: Список диагностических метрик.
        path: Путь к CSV-файлу отчёта.
    """
    rows = [diagnostic.__dict__ for diagnostic in diagnostics]
    pd.DataFrame(rows).to_csv(path, index=False)


def parse_args() -> argparse.Namespace:
    """Разобрать аргументы командной строки."""
    parser = argparse.ArgumentParser(
        description="Диагностика genuine/impostor расстояний embedding-модели.",
    )
    parser.add_argument(
        "--no-save-trials",
        action="store_true",
        help="Не сохранять trial-level CSV с расстояниями.",
    )

    return parser.parse_args()


def main() -> None:
    """Выполнить диагностику embedding-расстояний."""
    args = parse_args()
    config = DistanceEvaluationConfig(save_trials=not args.no_save_trials)

    paths = EmbeddingModelPaths.from_source_dir()
    paths.ensure_directories()

    templates = load_templates(paths.templates_path)

    validation_frame = load_embeddings_table(
        paths.data_processed_dir / VALIDATION_EMBEDDINGS_FILENAME
    )
    test_frame = load_embeddings_table(paths.data_processed_dir / TEST_EMBEDDINGS_FILENAME)

    embedding_columns = detect_embedding_columns(validation_frame)
    validate_template_dimensions(
        templates=templates,
        embedding_dim=len(embedding_columns),
    )

    diagnostics: list[DistanceDiagnostics] = []

    validation_trials_path = (
        paths.reports_dir / VALIDATION_TRIALS_FILENAME if config.save_trials else None
    )
    test_trials_path = paths.reports_dir / TEST_TRIALS_FILENAME if config.save_trials else None

    diagnostics.extend(
        evaluate_split(
            embeddings_frame=validation_frame,
            templates=templates,
            embedding_columns=embedding_columns,
            split_name="validation",
            save_trials_path=validation_trials_path,
        )
    )
    diagnostics.extend(
        evaluate_split(
            embeddings_frame=test_frame,
            templates=templates,
            embedding_columns=embedding_columns,
            split_name="test",
            save_trials_path=test_trials_path,
        )
    )

    diagnostics_path = paths.reports_dir / DIAGNOSTICS_FILENAME
    save_diagnostics(diagnostics, diagnostics_path)

    diagnostics_frame = pd.DataFrame([diagnostic.__dict__ for diagnostic in diagnostics])

    print("Диагностика embedding-расстояний завершена.")
    print(f"Количество пользователей-шаблонов: {len(templates)}")
    print(f"Размерность embedding-вектора: {len(embedding_columns)}")
    print(f"Путь к отчёту диагностики: {diagnostics_path}")

    if validation_trials_path is not None:
        print(f"Путь к validation trials: {validation_trials_path}")

    if test_trials_path is not None:
        print(f"Путь к test trials: {test_trials_path}")

    print(diagnostics_frame.to_string(index=False))


if __name__ == "__main__":
    main()
