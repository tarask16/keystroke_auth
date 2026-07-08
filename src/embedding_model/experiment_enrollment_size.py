"""Эксперимент с количеством enrollment samples для embedding-аутентификации.

Файл размещается в каталоге:

    src/embedding_model/experiment_enrollment_size.py

Назначение файла:

    1. Загрузить train, validation и test embedding-векторы.
    2. Для каждого значения N сформировать embedding-шаблоны пользователей
       только по N enrollment samples из train split.
    3. На validation split подобрать global, per-user и guarded thresholds.
    4. На validation и test split оценить FAR, FRR, EER и число ошибок.
    5. Сохранить итоговую таблицу эксперимента.

Проверяемые значения N по умолчанию:

    5, 10, 20, 30, 50

Выходной файл:

    reports/embedding_model/embedding_enrollment_size_experiment.csv

Методологическое правило:

    Enrollment samples берутся только из train split.
    Threshold подбирается только на validation split.
    Test split используется только для финальной оценки выбранных thresholds.

Терминология:

- ``enrollment samples`` — попытки ввода, используемые для регистрации
  пользователя и формирования embedding-шаблона;
- ``embedding-шаблон`` — средний embedding-вектор пользователя;
- ``genuine-попытка`` — проверка sample против собственного пользователя;
- ``impostor-попытка`` — проверка чужого sample против заявленного пользователя;
- ``threshold`` — порог расстояния для ACCEPT/REJECT;
- ``FAR`` — доля ложных допусков;
- ``FRR`` — доля ложных отказов;
- ``EER`` — диагностическая точка приблизительного равенства FAR и FRR.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

from src.embedding_model.embedding import EmbeddingModelPaths

TRAIN_EMBEDDINGS_FILENAME: Final[str] = "embeddings_train.csv"
VALIDATION_EMBEDDINGS_FILENAME: Final[str] = "embeddings_validation.csv"
TEST_EMBEDDINGS_FILENAME: Final[str] = "embeddings_test.csv"
EXPERIMENT_REPORT_FILENAME: Final[str] = "embedding_enrollment_size_experiment.csv"

DEFAULT_ENROLLMENT_SIZES: Final[tuple[int, ...]] = (5, 10, 20, 30, 50)

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


@dataclass(frozen=True)
class EnrollmentExperimentConfig:
    """Параметры эксперимента с числом enrollment samples."""

    enrollment_sizes: tuple[int, ...]
    distance_metrics: tuple[str, ...]
    policies: tuple[str, ...]
    target_far: float = 0.01
    guarded_far_trigger: float = 0.015
    guarded_max_frr_increase: float = 0.02
    random_seed: int = 42


@dataclass(frozen=True)
class SplitEmbeddings:
    """Embedding-векторы одного split-а."""

    split_name: str
    user_ids: np.ndarray
    embeddings: np.ndarray


@dataclass(frozen=True)
class DistanceDiagnostics:
    """Диагностика разделимости для одной distance metric."""

    roc_auc: float
    eer: float
    eer_threshold: float
    far_at_eer_threshold: float
    frr_at_eer_threshold: float
    genuine_distance_mean: float
    impostor_distance_mean: float
    separation_margin: float


@dataclass(frozen=True)
class PolicyMetrics:
    """Итоговые метрики одной threshold policy."""

    enrollment_samples: int
    split: str
    distance_metric: str
    policy: str
    threshold_source: str
    target_far: float
    users_count: int
    samples_count: int
    genuine_trials: int
    impostor_trials: int
    genuine_accepts: int
    genuine_rejects: int
    impostor_accepts: int
    impostor_rejects: int
    far: float
    frr: float
    balanced_error: float
    mean_user_far: float
    mean_user_frr: float
    max_user_far: float
    max_user_frr: float
    guarded_users_count: int
    roc_auc: float
    eer: float
    eer_threshold: float
    far_at_eer_threshold: float
    frr_at_eer_threshold: float
    genuine_distance_mean: float
    impostor_distance_mean: float
    separation_margin: float


def load_embeddings_table(path: Path) -> pd.DataFrame:
    """Загрузить CSV-файл с embedding-векторами.

    Аргументы:
        path: Путь к CSV-файлу split-а.

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
        raise ValueError(f"В таблице отсутствует колонка user_id: {path}")

    return dataframe


def detect_embedding_columns(dataframe: pd.DataFrame) -> list[str]:
    """Найти колонки embedding-вектора."""
    columns = sorted(column for column in dataframe.columns if column.startswith("embedding_"))

    if not columns:
        raise ValueError("В таблице не найдены embedding-колонки.")

    return columns


def dataframe_to_split_embeddings(
    dataframe: pd.DataFrame,
    embedding_columns: list[str],
    split_name: str,
) -> SplitEmbeddings:
    """Преобразовать DataFrame split-а в массивы NumPy."""
    embeddings = dataframe[embedding_columns].to_numpy(dtype=np.float32)
    user_ids = dataframe["user_id"].astype(str).to_numpy()

    return SplitEmbeddings(
        split_name=split_name,
        user_ids=user_ids,
        embeddings=embeddings,
    )


def create_enrollment_templates(
    train_frame: pd.DataFrame,
    embedding_columns: list[str],
    enrollment_samples: int,
    random_seed: int,
) -> tuple[list[str], np.ndarray]:
    """Сформировать embedding-шаблоны по N enrollment samples на пользователя.

    Для каждого пользователя случайно, но воспроизводимо выбирается N samples
    из train split. Шаблон считается как средний embedding-вектор.

    Аргументы:
        train_frame: Таблица train embedding-векторов.
        embedding_columns: Список embedding-координат.
        enrollment_samples: Число enrollment samples на пользователя.
        random_seed: Seed для воспроизводимого выбора samples.

    Возвращает:
        Кортеж ``(user_ids, templates_matrix)``.
    """
    rng = np.random.default_rng(random_seed + enrollment_samples)

    user_ids: list[str] = []
    templates: list[np.ndarray] = []

    for user_id, user_frame in train_frame.groupby("user_id", sort=True):
        if len(user_frame) < enrollment_samples:
            raise ValueError(
                f"У пользователя {user_id} недостаточно train samples: "
                f"{len(user_frame)} < {enrollment_samples}."
            )

        selected_positions = rng.choice(
            len(user_frame),
            size=enrollment_samples,
            replace=False,
        )
        selected_frame = user_frame.iloc[selected_positions]
        template = selected_frame[embedding_columns].to_numpy(dtype=np.float32).mean(axis=0)

        user_ids.append(str(user_id))
        templates.append(template.astype(np.float32))

    return user_ids, np.vstack(templates).astype(np.float32)


def calculate_distance_matrix(
    embeddings: np.ndarray,
    templates: np.ndarray,
    metric: str,
) -> np.ndarray:
    """Рассчитать матрицу расстояний samples x claimed users.

    Аргументы:
        embeddings: Матрица embedding-векторов samples.
        templates: Матрица embedding-шаблонов пользователей.
        metric: Метрика расстояния.

    Возвращает:
        Матрица расстояний формы ``(samples, users)``.
    """
    if metric == "euclidean":
        diff = embeddings[:, np.newaxis, :] - templates[np.newaxis, :, :]
        return np.linalg.norm(diff, ord=2, axis=2).astype(np.float32)

    if metric == "manhattan":
        diff = embeddings[:, np.newaxis, :] - templates[np.newaxis, :, :]
        return np.linalg.norm(diff, ord=1, axis=2).astype(np.float32)

    if metric == "cosine":
        embedding_norms = np.linalg.norm(embeddings, ord=2, axis=1, keepdims=True)
        template_norms = np.linalg.norm(templates, ord=2, axis=1, keepdims=True).T

        denominator = embedding_norms * template_norms
        denominator = np.where(denominator == 0.0, 1.0, denominator)

        similarities = (embeddings @ templates.T) / denominator
        similarities = np.clip(similarities, -1.0, 1.0)

        return (1.0 - similarities).astype(np.float32)

    raise ValueError(f"Неизвестная метрика расстояния: {metric}")


def build_genuine_matrix(
    actual_user_ids: np.ndarray,
    claimed_user_ids: list[str],
) -> np.ndarray:
    """Построить булеву матрицу genuine/impostor."""
    claimed_array = np.asarray(claimed_user_ids, dtype=str)
    return actual_user_ids[:, np.newaxis] == claimed_array[np.newaxis, :]


def choose_max_threshold_for_target_far(
    impostor_distances: np.ndarray,
    target_far: float,
) -> float:
    """Выбрать максимальный threshold, не превышающий целевой FAR.

    Аргументы:
        impostor_distances: Расстояния impostor-попыток.
        target_far: Целевой FAR.

    Возвращает:
        Distance-threshold.
    """
    sorted_distances = np.sort(impostor_distances.astype(np.float64))

    if len(sorted_distances) == 0:
        raise ValueError("Невозможно подобрать threshold: нет impostor-попыток.")

    unique_thresholds = np.unique(sorted_distances)
    accepted_counts = np.searchsorted(
        sorted_distances,
        unique_thresholds,
        side="right",
    )
    far_values = accepted_counts / len(sorted_distances)

    valid_thresholds = unique_thresholds[far_values <= target_far]

    if len(valid_thresholds) == 0:
        return float(np.nextafter(sorted_distances[0], -np.inf))

    return float(valid_thresholds[-1])


def build_global_thresholds(
    validation_distances: np.ndarray,
    validation_genuine: np.ndarray,
    target_far: float,
) -> np.ndarray:
    """Подобрать общий threshold и размножить его на всех пользователей."""
    threshold = choose_max_threshold_for_target_far(
        impostor_distances=validation_distances[~validation_genuine],
        target_far=target_far,
    )

    return np.full(validation_distances.shape[1], threshold, dtype=np.float64)


def build_per_user_thresholds(
    validation_distances: np.ndarray,
    validation_genuine: np.ndarray,
    target_far: float,
) -> np.ndarray:
    """Подобрать индивидуальные threshold для каждого claimed user."""
    thresholds: list[float] = []

    for user_index in range(validation_distances.shape[1]):
        user_impostor_distances = validation_distances[
            ~validation_genuine[:, user_index],
            user_index,
        ]
        thresholds.append(
            choose_max_threshold_for_target_far(
                impostor_distances=user_impostor_distances,
                target_far=target_far,
            )
        )

    return np.asarray(thresholds, dtype=np.float64)


def build_guarded_thresholds(
    validation_distances: np.ndarray,
    validation_genuine: np.ndarray,
    global_thresholds: np.ndarray,
    per_user_thresholds: np.ndarray,
    config: EnrollmentExperimentConfig,
) -> tuple[np.ndarray, int]:
    """Сформировать guarded thresholds по validation split.

    Индивидуальный threshold используется только для пользователей, у которых
    global policy даёт повышенный FAR, а per-user threshold снижает FAR без
    чрезмерного роста FRR.
    """
    global_user_metrics = calculate_per_user_metrics(
        distances=validation_distances,
        genuine_matrix=validation_genuine,
        thresholds=global_thresholds,
    )
    per_user_metrics = calculate_per_user_metrics(
        distances=validation_distances,
        genuine_matrix=validation_genuine,
        thresholds=per_user_thresholds,
    )

    guarded_thresholds = global_thresholds.copy()
    guarded_users_count = 0

    for user_index in range(validation_distances.shape[1]):
        global_far = global_user_metrics.loc[user_index, "far"]
        global_frr = global_user_metrics.loc[user_index, "frr"]
        candidate_far = per_user_metrics.loc[user_index, "far"]
        candidate_frr = per_user_metrics.loc[user_index, "frr"]

        far_is_problematic = global_far > config.guarded_far_trigger
        far_is_reduced = candidate_far < global_far
        frr_growth_is_acceptable = candidate_frr <= global_frr + config.guarded_max_frr_increase

        if far_is_problematic and far_is_reduced and frr_growth_is_acceptable:
            guarded_thresholds[user_index] = per_user_thresholds[user_index]
            guarded_users_count += 1

    return guarded_thresholds, guarded_users_count


def calculate_per_user_metrics(
    distances: np.ndarray,
    genuine_matrix: np.ndarray,
    thresholds: np.ndarray,
) -> pd.DataFrame:
    """Рассчитать per-user FAR и FRR для заданных thresholds."""
    accepted = distances <= thresholds[np.newaxis, :]
    rows: list[dict[str, float | int]] = []

    for user_index in range(distances.shape[1]):
        user_genuine = genuine_matrix[:, user_index]
        user_impostor = ~user_genuine
        user_accepted = accepted[:, user_index]

        genuine_trials = int(np.sum(user_genuine))
        impostor_trials = int(np.sum(user_impostor))
        false_rejects = int(np.sum(~user_accepted & user_genuine))
        false_accepts = int(np.sum(user_accepted & user_impostor))

        rows.append(
            {
                "user_index": user_index,
                "genuine_trials": genuine_trials,
                "impostor_trials": impostor_trials,
                "false_rejects": false_rejects,
                "false_accepts": false_accepts,
                "frr": float(false_rejects / genuine_trials) if genuine_trials else 0.0,
                "far": float(false_accepts / impostor_trials) if impostor_trials else 0.0,
            }
        )

    return pd.DataFrame(rows)


def calculate_policy_metrics(
    enrollment_samples: int,
    split_data: SplitEmbeddings,
    claimed_user_ids: list[str],
    distances: np.ndarray,
    genuine_matrix: np.ndarray,
    thresholds: np.ndarray,
    distance_metric: str,
    policy: str,
    target_far: float,
    guarded_users_count: int,
    diagnostics: DistanceDiagnostics,
) -> PolicyMetrics:
    """Рассчитать итоговые метрики threshold policy."""
    accepted = distances <= thresholds[np.newaxis, :]

    genuine_trials = int(np.sum(genuine_matrix))
    impostor_trials = int(np.sum(~genuine_matrix))

    genuine_accepts = int(np.sum(accepted & genuine_matrix))
    genuine_rejects = int(np.sum(~accepted & genuine_matrix))
    impostor_accepts = int(np.sum(accepted & ~genuine_matrix))
    impostor_rejects = int(np.sum(~accepted & ~genuine_matrix))

    far = float(impostor_accepts / impostor_trials) if impostor_trials else 0.0
    frr = float(genuine_rejects / genuine_trials) if genuine_trials else 0.0

    per_user = calculate_per_user_metrics(
        distances=distances,
        genuine_matrix=genuine_matrix,
        thresholds=thresholds,
    )

    return PolicyMetrics(
        enrollment_samples=enrollment_samples,
        split=split_data.split_name,
        distance_metric=distance_metric,
        policy=policy,
        threshold_source="validation",
        target_far=target_far,
        users_count=len(claimed_user_ids),
        samples_count=len(split_data.user_ids),
        genuine_trials=genuine_trials,
        impostor_trials=impostor_trials,
        genuine_accepts=genuine_accepts,
        genuine_rejects=genuine_rejects,
        impostor_accepts=impostor_accepts,
        impostor_rejects=impostor_rejects,
        far=far,
        frr=frr,
        balanced_error=float((far + frr) / 2.0),
        mean_user_far=float(per_user["far"].mean()),
        mean_user_frr=float(per_user["frr"].mean()),
        max_user_far=float(per_user["far"].max()),
        max_user_frr=float(per_user["frr"].max()),
        guarded_users_count=guarded_users_count,
        roc_auc=diagnostics.roc_auc,
        eer=diagnostics.eer,
        eer_threshold=diagnostics.eer_threshold,
        far_at_eer_threshold=diagnostics.far_at_eer_threshold,
        frr_at_eer_threshold=diagnostics.frr_at_eer_threshold,
        genuine_distance_mean=diagnostics.genuine_distance_mean,
        impostor_distance_mean=diagnostics.impostor_distance_mean,
        separation_margin=diagnostics.separation_margin,
    )


def calculate_distance_diagnostics(
    distances: np.ndarray,
    genuine_matrix: np.ndarray,
) -> DistanceDiagnostics:
    """Рассчитать ROC AUC, EER и средние расстояния."""
    labels = genuine_matrix.astype(np.int8).ravel()
    flat_distances = distances.astype(np.float64).ravel()

    genuine_distances = flat_distances[labels == 1]
    impostor_distances = flat_distances[labels == 0]

    scores = -flat_distances
    far_values, true_accept_rate_values, score_thresholds = roc_curve(
        labels,
        scores,
        pos_label=1,
    )
    frr_values = 1.0 - true_accept_rate_values

    finite_mask = np.isfinite(score_thresholds)
    far_values = far_values[finite_mask]
    frr_values = frr_values[finite_mask]
    score_thresholds = score_thresholds[finite_mask]

    best_index = int(np.argmin(np.abs(far_values - frr_values)))
    far_at_eer = float(far_values[best_index])
    frr_at_eer = float(frr_values[best_index])
    eer = float((far_at_eer + frr_at_eer) / 2.0)
    eer_threshold = float(-score_thresholds[best_index])

    genuine_mean = float(np.mean(genuine_distances))
    impostor_mean = float(np.mean(impostor_distances))

    return DistanceDiagnostics(
        roc_auc=float(roc_auc_score(labels, scores)),
        eer=eer,
        eer_threshold=eer_threshold,
        far_at_eer_threshold=far_at_eer,
        frr_at_eer_threshold=frr_at_eer,
        genuine_distance_mean=genuine_mean,
        impostor_distance_mean=impostor_mean,
        separation_margin=float(impostor_mean - genuine_mean),
    )


def run_experiment_for_size(
    enrollment_samples: int,
    train_frame: pd.DataFrame,
    validation_data: SplitEmbeddings,
    test_data: SplitEmbeddings,
    embedding_columns: list[str],
    config: EnrollmentExperimentConfig,
) -> list[PolicyMetrics]:
    """Выполнить эксперимент для одного значения N."""
    claimed_user_ids, templates = create_enrollment_templates(
        train_frame=train_frame,
        embedding_columns=embedding_columns,
        enrollment_samples=enrollment_samples,
        random_seed=config.random_seed,
    )

    rows: list[PolicyMetrics] = []

    for distance_metric in config.distance_metrics:
        validation_distances = calculate_distance_matrix(
            embeddings=validation_data.embeddings,
            templates=templates,
            metric=distance_metric,
        )
        test_distances = calculate_distance_matrix(
            embeddings=test_data.embeddings,
            templates=templates,
            metric=distance_metric,
        )

        validation_genuine = build_genuine_matrix(
            actual_user_ids=validation_data.user_ids,
            claimed_user_ids=claimed_user_ids,
        )
        test_genuine = build_genuine_matrix(
            actual_user_ids=test_data.user_ids,
            claimed_user_ids=claimed_user_ids,
        )

        global_thresholds = build_global_thresholds(
            validation_distances=validation_distances,
            validation_genuine=validation_genuine,
            target_far=config.target_far,
        )
        per_user_thresholds = build_per_user_thresholds(
            validation_distances=validation_distances,
            validation_genuine=validation_genuine,
            target_far=config.target_far,
        )
        guarded_thresholds, guarded_users_count = build_guarded_thresholds(
            validation_distances=validation_distances,
            validation_genuine=validation_genuine,
            global_thresholds=global_thresholds,
            per_user_thresholds=per_user_thresholds,
            config=config,
        )

        threshold_sets = {
            "global": (global_thresholds, 0),
            "per_user": (per_user_thresholds, 0),
            "guarded": (guarded_thresholds, guarded_users_count),
        }

        validation_diagnostics = calculate_distance_diagnostics(
            distances=validation_distances,
            genuine_matrix=validation_genuine,
        )
        test_diagnostics = calculate_distance_diagnostics(
            distances=test_distances,
            genuine_matrix=test_genuine,
        )

        for policy in config.policies:
            thresholds, guarded_count = threshold_sets[policy]

            rows.append(
                calculate_policy_metrics(
                    enrollment_samples=enrollment_samples,
                    split_data=validation_data,
                    claimed_user_ids=claimed_user_ids,
                    distances=validation_distances,
                    genuine_matrix=validation_genuine,
                    thresholds=thresholds,
                    distance_metric=distance_metric,
                    policy=policy,
                    target_far=config.target_far,
                    guarded_users_count=guarded_count,
                    diagnostics=validation_diagnostics,
                )
            )
            rows.append(
                calculate_policy_metrics(
                    enrollment_samples=enrollment_samples,
                    split_data=test_data,
                    claimed_user_ids=claimed_user_ids,
                    distances=test_distances,
                    genuine_matrix=test_genuine,
                    thresholds=thresholds,
                    distance_metric=distance_metric,
                    policy=policy,
                    target_far=config.target_far,
                    guarded_users_count=guarded_count,
                    diagnostics=test_diagnostics,
                )
            )

    return rows


def parse_int_tuple(value: str) -> tuple[int, ...]:
    """Разобрать строку вида ``5,10,20`` в кортеж целых чисел."""
    result = tuple(int(item.strip()) for item in value.split(",") if item.strip())

    if not result:
        raise ValueError("Список значений N пуст.")

    if any(item <= 0 for item in result):
        raise ValueError("Все значения N должны быть положительными.")

    return result


def parse_str_tuple(value: str, allowed_values: tuple[str, ...], name: str) -> tuple[str, ...]:
    """Разобрать строку со списком значений и проверить допустимые значения."""
    result = tuple(item.strip() for item in value.split(",") if item.strip())

    if not result:
        raise ValueError(f"Список {name} пуст.")

    invalid_values = sorted(set(result) - set(allowed_values))
    if invalid_values:
        raise ValueError(
            f"Недопустимые значения {name}: {', '.join(invalid_values)}. "
            f"Разрешено: {', '.join(allowed_values)}."
        )

    return result


def parse_args() -> argparse.Namespace:
    """Разобрать аргументы командной строки."""
    parser = argparse.ArgumentParser(
        description="Эксперимент с количеством enrollment samples.",
    )
    parser.add_argument(
        "--enrollment-sizes",
        type=str,
        default="5,10,20,30,50",
        help="Список N через запятую.",
    )
    parser.add_argument(
        "--distance-metrics",
        type=str,
        default="euclidean,cosine,manhattan",
        help="Метрики расстояния через запятую.",
    )
    parser.add_argument(
        "--policies",
        type=str,
        default="global,per_user,guarded",
        help="Threshold policies через запятую.",
    )
    parser.add_argument(
        "--target-far",
        type=float,
        default=0.01,
        help="Целевой FAR на validation split.",
    )
    parser.add_argument(
        "--guarded-far-trigger",
        type=float,
        default=0.015,
        help="Validation FAR, выше которого пользователь считается проблемным.",
    )
    parser.add_argument(
        "--guarded-max-frr-increase",
        type=float,
        default=0.02,
        help="Максимально допустимый рост validation FRR для guarded override.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Seed для воспроизводимого выбора enrollment samples.",
    )

    return parser.parse_args()


def main() -> None:
    """Выполнить эксперимент с количеством enrollment samples."""
    args = parse_args()

    config = EnrollmentExperimentConfig(
        enrollment_sizes=parse_int_tuple(args.enrollment_sizes),
        distance_metrics=parse_str_tuple(
            args.distance_metrics,
            SUPPORTED_DISTANCE_METRICS,
            "distance_metrics",
        ),
        policies=parse_str_tuple(args.policies, SUPPORTED_POLICIES, "policies"),
        target_far=args.target_far,
        guarded_far_trigger=args.guarded_far_trigger,
        guarded_max_frr_increase=args.guarded_max_frr_increase,
        random_seed=args.random_seed,
    )

    paths = EmbeddingModelPaths.from_source_dir()
    paths.ensure_directories()

    train_frame = load_embeddings_table(paths.data_processed_dir / TRAIN_EMBEDDINGS_FILENAME)
    validation_frame = load_embeddings_table(
        paths.data_processed_dir / VALIDATION_EMBEDDINGS_FILENAME
    )
    test_frame = load_embeddings_table(paths.data_processed_dir / TEST_EMBEDDINGS_FILENAME)

    embedding_columns = detect_embedding_columns(train_frame)

    validation_data = dataframe_to_split_embeddings(
        dataframe=validation_frame,
        embedding_columns=embedding_columns,
        split_name="validation",
    )
    test_data = dataframe_to_split_embeddings(
        dataframe=test_frame,
        embedding_columns=embedding_columns,
        split_name="test",
    )

    rows: list[PolicyMetrics] = []

    for enrollment_samples in config.enrollment_sizes:
        print(f"Эксперимент для N={enrollment_samples}...")
        rows.extend(
            run_experiment_for_size(
                enrollment_samples=enrollment_samples,
                train_frame=train_frame,
                validation_data=validation_data,
                test_data=test_data,
                embedding_columns=embedding_columns,
                config=config,
            )
        )

    report_frame = pd.DataFrame([asdict(row) for row in rows])
    report_path = paths.reports_dir / EXPERIMENT_REPORT_FILENAME
    report_frame.to_csv(report_path, index=False)

    print("Эксперимент с количеством enrollment samples завершён.")
    print(f"Путь к отчёту: {report_path}")

    test_summary = report_frame[
        (report_frame["split"] == "test")
        & (report_frame["distance_metric"] == "cosine")
        & (report_frame["policy"] == "per_user")
    ][
        [
            "enrollment_samples",
            "far",
            "frr",
            "balanced_error",
            "impostor_accepts",
            "genuine_rejects",
            "eer",
            "roc_auc",
        ]
    ]

    print("Краткая test-сводка для cosine + per_user:")
    print(test_summary.to_string(index=False))


if __name__ == "__main__":
    main()
