"""Подбор threshold policy для embedding-аутентификации этапа 6.

Файл размещается в каталоге:

    src/embedding_model/select_embedding_threshold_policy.py

Назначение файла:

    1. Загрузить validation/test таблицы genuine/impostor-попыток.
    2. Подобрать distance-threshold только на validation split.
    3. Оценить выбранные политики на validation и test split.
    4. Сохранить JSON-файл с порогами аутентификации.
    5. Сохранить CSV-отчёт с FAR, FRR, EER-подобной рабочей точкой,
       числом ложных допусков и ложных отказов.

Поддерживаемые политики:

- ``global`` — один общий distance-threshold для всех пользователей;
- ``per_user`` — отдельный threshold для каждого заявленного пользователя;
- ``guarded`` — общий threshold остаётся базовым, а индивидуальный threshold
  применяется только к пользователям с повышенным validation FAR.

Методологическое правило:

    Все threshold подбираются только по validation split.
    Test split используется только для финальной оценки выбранной policy.

Терминология:

- ``threshold`` — порог расстояния, при котором принимается решение;
- ``ACCEPT`` — distance <= threshold;
- ``REJECT`` — distance > threshold;
- ``FAR`` — доля ложных допусков impostor-попыток;
- ``FRR`` — доля ложных отказов genuine-попыток;
- ``guarded policy`` — осторожная политика индивидуальных порогов,
  ориентированная на снижение ложных допусков без чрезмерного роста FRR.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd

from src.embedding_model.embedding import EmbeddingModelPaths

VALIDATION_TRIALS_FILENAME: Final[str] = "embedding_distance_trials_validation.csv"
TEST_TRIALS_FILENAME: Final[str] = "embedding_distance_trials_test.csv"
THRESHOLD_POLICY_REPORT_FILENAME: Final[str] = "embedding_threshold_policy.csv"

SUPPORTED_DISTANCE_METRICS: Final[tuple[str, ...]] = (
    "euclidean",
    "cosine",
    "manhattan",
)


@dataclass(frozen=True)
class ThresholdSelectionConfig:
    """Параметры подбора threshold policy."""

    target_far: float = 0.01
    guarded_far_trigger: float = 0.015
    guarded_max_frr_increase: float = 0.02


@dataclass(frozen=True)
class PolicyMetrics:
    """Метрики проверки одной threshold policy."""

    split: str
    distance_metric: str
    policy: str
    threshold_source: str
    target_far: float
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


def load_trials(path: Path) -> pd.DataFrame:
    """Загрузить таблицу genuine/impostor-попыток.

    Аргументы:
        path: Путь к CSV-файлу с trial-level расстояниями.

    Возвращает:
        DataFrame с колонками ``distance_metric``, ``claimed_user_id``,
        ``is_genuine`` и ``distance``.

    Исключения:
        FileNotFoundError: Возникает, если файл не найден.
        ValueError: Возникает, если файл пуст или не содержит нужных колонок.
    """
    if not path.exists():
        raise FileNotFoundError(f"Файл попыток не найден: {path}")

    dataframe = pd.read_csv(path)

    if dataframe.empty:
        raise ValueError(f"Файл попыток пуст: {path}")

    required_columns = {
        "distance_metric",
        "claimed_user_id",
        "is_genuine",
        "distance",
    }
    missing_columns = required_columns - set(dataframe.columns)

    if missing_columns:
        raise ValueError(
            f"В trial-файле отсутствуют обязательные колонки: {', '.join(sorted(missing_columns))}."
        )

    dataframe["claimed_user_id"] = dataframe["claimed_user_id"].astype(str)
    dataframe["is_genuine"] = dataframe["is_genuine"].astype(bool)
    dataframe["distance"] = dataframe["distance"].astype(float)

    return dataframe


def choose_max_threshold_for_target_far(
    trials: pd.DataFrame,
    target_far: float,
) -> float:
    """Выбрать максимальный threshold, не превышающий целевой FAR.

    Для distance-based verification принятие выполняется по правилу
    ``distance <= threshold``. Чем выше threshold, тем ниже FRR, но выше FAR.
    Поэтому выбирается максимальный допустимый threshold при ограничении FAR.

    Аргументы:
        trials: Таблица genuine/impostor-попыток.
        target_far: Максимально допустимый FAR на calibration данных.

    Возвращает:
        Distance-threshold.
    """
    impostor_distances = np.sort(
        trials.loc[~trials["is_genuine"], "distance"].to_numpy(dtype=np.float64)
    )

    if len(impostor_distances) == 0:
        raise ValueError("Невозможно подобрать threshold: нет impostor-попыток.")

    unique_thresholds = np.unique(impostor_distances)
    accepted_counts = np.searchsorted(
        impostor_distances,
        unique_thresholds,
        side="right",
    )
    far_values = accepted_counts / len(impostor_distances)

    valid_thresholds = unique_thresholds[far_values <= target_far]

    if len(valid_thresholds) == 0:
        return float(np.nextafter(impostor_distances[0], -np.inf))

    return float(valid_thresholds[-1])


def build_per_user_thresholds(
    validation_trials: pd.DataFrame,
    target_far: float,
) -> dict[str, float]:
    """Подобрать индивидуальные threshold для каждого заявленного пользователя.

    Аргументы:
        validation_trials: Validation-попытки одной метрики расстояния.
        target_far: Целевой FAR для каждого claimed user.

    Возвращает:
        Словарь ``claimed_user_id -> threshold``.
    """
    thresholds: dict[str, float] = {}

    for claimed_user_id, user_trials in validation_trials.groupby("claimed_user_id"):
        thresholds[str(claimed_user_id)] = choose_max_threshold_for_target_far(
            trials=user_trials,
            target_far=target_far,
        )

    return thresholds


def calculate_metrics_for_thresholds(
    trials: pd.DataFrame,
    thresholds: dict[str, float],
    policy: str,
    threshold_source: str,
    target_far: float,
    guarded_users_count: int = 0,
) -> PolicyMetrics:
    """Рассчитать FAR/FRR и per-user диагностику для policy.

    Аргументы:
        trials: Таблица genuine/impostor-попыток.
        thresholds: Словарь threshold для claimed users.
        policy: Имя политики.
        threshold_source: Split, на котором подобраны threshold.
        target_far: Целевой FAR.
        guarded_users_count: Количество пользователей с индивидуальным
            threshold в guarded policy.

    Возвращает:
        Метрики проверки policy.
    """
    thresholds_series = trials["claimed_user_id"].map(thresholds)

    if thresholds_series.isna().any():
        missing_users = sorted(trials.loc[thresholds_series.isna(), "claimed_user_id"].unique())
        raise ValueError(
            f"Для части claimed users отсутствуют threshold: {', '.join(missing_users)}."
        )

    accepted = trials["distance"].to_numpy(dtype=np.float64) <= thresholds_series.to_numpy(
        dtype=np.float64
    )
    is_genuine = trials["is_genuine"].to_numpy(dtype=bool)
    is_impostor = ~is_genuine

    genuine_trials = int(np.sum(is_genuine))
    impostor_trials = int(np.sum(is_impostor))

    genuine_accepts = int(np.sum(accepted & is_genuine))
    genuine_rejects = int(np.sum(~accepted & is_genuine))
    impostor_accepts = int(np.sum(accepted & is_impostor))
    impostor_rejects = int(np.sum(~accepted & is_impostor))

    far = float(impostor_accepts / impostor_trials) if impostor_trials else 0.0
    frr = float(genuine_rejects / genuine_trials) if genuine_trials else 0.0

    per_user = calculate_per_user_metrics(trials, accepted)

    return PolicyMetrics(
        split=str(trials["split"].iloc[0]),
        distance_metric=str(trials["distance_metric"].iloc[0]),
        policy=policy,
        threshold_source=threshold_source,
        target_far=target_far,
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
    )


def calculate_per_user_metrics(
    trials: pd.DataFrame,
    accepted: np.ndarray,
) -> pd.DataFrame:
    """Рассчитать FAR и FRR отдельно по каждому claimed user.

    Аргументы:
        trials: Таблица genuine/impostor-попыток.
        accepted: Булев массив решений ACCEPT/REJECT.

    Возвращает:
        DataFrame с per-user FAR и FRR.
    """
    work_frame = trials.copy()
    work_frame["accepted"] = accepted

    rows: list[dict[str, float | int | str]] = []

    for claimed_user_id, user_trials in work_frame.groupby("claimed_user_id"):
        genuine_mask = user_trials["is_genuine"].to_numpy(dtype=bool)
        impostor_mask = ~genuine_mask
        user_accepted = user_trials["accepted"].to_numpy(dtype=bool)

        genuine_trials = int(np.sum(genuine_mask))
        impostor_trials = int(np.sum(impostor_mask))
        false_rejects = int(np.sum(~user_accepted & genuine_mask))
        false_accepts = int(np.sum(user_accepted & impostor_mask))

        rows.append(
            {
                "claimed_user_id": str(claimed_user_id),
                "genuine_trials": genuine_trials,
                "impostor_trials": impostor_trials,
                "false_rejects": false_rejects,
                "false_accepts": false_accepts,
                "frr": float(false_rejects / genuine_trials) if genuine_trials else 0.0,
                "far": float(false_accepts / impostor_trials) if impostor_trials else 0.0,
            }
        )

    return pd.DataFrame(rows)


def build_global_thresholds(
    validation_trials: pd.DataFrame,
    target_far: float,
) -> dict[str, float]:
    """Построить общий threshold для всех пользователей одной метрики."""
    global_threshold = choose_max_threshold_for_target_far(
        trials=validation_trials,
        target_far=target_far,
    )
    user_ids = sorted(validation_trials["claimed_user_id"].unique())

    return {str(user_id): global_threshold for user_id in user_ids}


def build_guarded_thresholds(
    validation_trials: pd.DataFrame,
    global_thresholds: dict[str, float],
    per_user_thresholds: dict[str, float],
    config: ThresholdSelectionConfig,
) -> tuple[dict[str, float], list[str]]:
    """Сформировать guarded policy на основе validation diagnostics.

    Индивидуальный threshold применяется только если:

    1. У пользователя validation FAR при global policy выше trigger.
    2. Индивидуальный threshold снижает FAR.
    3. Рост validation FRR не превышает допустимую величину.

    Аргументы:
        validation_trials: Validation-попытки одной метрики расстояния.
        global_thresholds: Глобальные threshold для всех claimed users.
        per_user_thresholds: Индивидуальные threshold-кандидаты.
        config: Параметры guarded policy.

    Возвращает:
        Кортеж из итоговых threshold и списка пользователей с override.
    """
    global_metrics = calculate_user_metrics_by_policy(
        validation_trials,
        global_thresholds,
    )
    per_user_metrics = calculate_user_metrics_by_policy(
        validation_trials,
        per_user_thresholds,
    )

    guarded_thresholds = dict(global_thresholds)
    guarded_users: list[str] = []

    for _, global_row in global_metrics.iterrows():
        user_id = str(global_row["claimed_user_id"])
        candidate_row = per_user_metrics.loc[
            per_user_metrics["claimed_user_id"].astype(str) == user_id
        ].iloc[0]

        global_far = float(global_row["far"])
        global_frr = float(global_row["frr"])
        candidate_far = float(candidate_row["far"])
        candidate_frr = float(candidate_row["frr"])

        far_is_problematic = global_far > config.guarded_far_trigger
        far_is_reduced = candidate_far < global_far
        frr_growth_is_acceptable = candidate_frr <= global_frr + config.guarded_max_frr_increase

        if far_is_problematic and far_is_reduced and frr_growth_is_acceptable:
            guarded_thresholds[user_id] = per_user_thresholds[user_id]
            guarded_users.append(user_id)

    return guarded_thresholds, guarded_users


def calculate_user_metrics_by_policy(
    trials: pd.DataFrame,
    thresholds: dict[str, float],
) -> pd.DataFrame:
    """Рассчитать per-user метрики для заданного набора threshold."""
    thresholds_series = trials["claimed_user_id"].map(thresholds)
    accepted = trials["distance"].to_numpy(dtype=np.float64) <= thresholds_series.to_numpy(
        dtype=np.float64
    )
    return calculate_per_user_metrics(trials, accepted)


def evaluate_metric_policy(
    validation_trials: pd.DataFrame,
    test_trials: pd.DataFrame,
    metric: str,
    config: ThresholdSelectionConfig,
) -> tuple[list[PolicyMetrics], dict[str, object]]:
    """Подобрать и оценить threshold policies для одной метрики расстояния.

    Аргументы:
        validation_trials: Validation-попытки одной метрики.
        test_trials: Test-попытки одной метрики.
        metric: Название метрики расстояния.
        config: Параметры подбора policy.

    Возвращает:
        Список строк отчёта и JSON-совместимое описание выбранных threshold.
    """
    global_thresholds = build_global_thresholds(
        validation_trials=validation_trials,
        target_far=config.target_far,
    )
    per_user_thresholds = build_per_user_thresholds(
        validation_trials=validation_trials,
        target_far=config.target_far,
    )
    guarded_thresholds, guarded_users = build_guarded_thresholds(
        validation_trials=validation_trials,
        global_thresholds=global_thresholds,
        per_user_thresholds=per_user_thresholds,
        config=config,
    )

    policies = {
        "global": (global_thresholds, 0),
        "per_user": (per_user_thresholds, 0),
        "guarded": (guarded_thresholds, len(guarded_users)),
    }

    rows: list[PolicyMetrics] = []

    for policy_name, (thresholds, guarded_count) in policies.items():
        rows.append(
            calculate_metrics_for_thresholds(
                trials=validation_trials,
                thresholds=thresholds,
                policy=policy_name,
                threshold_source="validation",
                target_far=config.target_far,
                guarded_users_count=guarded_count,
            )
        )
        rows.append(
            calculate_metrics_for_thresholds(
                trials=test_trials,
                thresholds=thresholds,
                policy=policy_name,
                threshold_source="validation",
                target_far=config.target_far,
                guarded_users_count=guarded_count,
            )
        )

    policy_payload = {
        "distance_metric": metric,
        "target_far": config.target_far,
        "guarded_far_trigger": config.guarded_far_trigger,
        "guarded_max_frr_increase": config.guarded_max_frr_increase,
        "global_threshold": next(iter(global_thresholds.values())),
        "per_user_thresholds": per_user_thresholds,
        "guarded_thresholds": guarded_thresholds,
        "guarded_users": guarded_users,
    }

    return rows, policy_payload


def save_threshold_policy(
    policy_payloads: list[dict[str, object]],
    config: ThresholdSelectionConfig,
    paths: EmbeddingModelPaths,
) -> Path:
    """Сохранить JSON-файл с threshold policy.

    Аргументы:
        policy_payloads: Описание порогов для всех метрик расстояния.
        config: Параметры подбора threshold.
        paths: Пути к артефактам embedding-этапа.

    Возвращает:
        Путь к JSON-файлу.
    """
    payload = {
        "metadata": {
            "created_at_utc": datetime.now(UTC).isoformat(),
            "threshold_source": "validation",
            "decision_rule": "ACCEPT if distance <= threshold else REJECT",
            "target_far": config.target_far,
            "guarded_far_trigger": config.guarded_far_trigger,
            "guarded_max_frr_increase": config.guarded_max_frr_increase,
        },
        "policies": {
            str(policy_payload["distance_metric"]): policy_payload
            for policy_payload in policy_payloads
        },
    }

    paths.thresholds_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return paths.thresholds_path


def parse_args() -> argparse.Namespace:
    """Разобрать аргументы командной строки."""
    parser = argparse.ArgumentParser(
        description="Подбор threshold policy для embedding-аутентификации.",
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
        help="Порог validation FAR, начиная с которого пользователь считается проблемным.",
    )
    parser.add_argument(
        "--guarded-max-frr-increase",
        type=float,
        default=0.02,
        help="Максимально допустимый рост validation FRR для guarded override.",
    )

    return parser.parse_args()


def main() -> None:
    """Выполнить подбор и оценку threshold policy."""
    args = parse_args()

    config = ThresholdSelectionConfig(
        target_far=args.target_far,
        guarded_far_trigger=args.guarded_far_trigger,
        guarded_max_frr_increase=args.guarded_max_frr_increase,
    )

    paths = EmbeddingModelPaths.from_source_dir()
    paths.ensure_directories()

    validation_trials = load_trials(paths.reports_dir / VALIDATION_TRIALS_FILENAME)
    test_trials = load_trials(paths.reports_dir / TEST_TRIALS_FILENAME)

    all_rows: list[PolicyMetrics] = []
    policy_payloads: list[dict[str, object]] = []

    for metric in SUPPORTED_DISTANCE_METRICS:
        validation_metric_trials = validation_trials[
            validation_trials["distance_metric"] == metric
        ].copy()
        test_metric_trials = test_trials[test_trials["distance_metric"] == metric].copy()

        rows, policy_payload = evaluate_metric_policy(
            validation_trials=validation_metric_trials,
            test_trials=test_metric_trials,
            metric=metric,
            config=config,
        )
        all_rows.extend(rows)
        policy_payloads.append(policy_payload)

    report_frame = pd.DataFrame([asdict(row) for row in all_rows])
    report_path = paths.reports_dir / THRESHOLD_POLICY_REPORT_FILENAME
    report_frame.to_csv(report_path, index=False)

    thresholds_path = save_threshold_policy(
        policy_payloads=policy_payloads,
        config=config,
        paths=paths,
    )

    print("Подбор threshold policy завершён.")
    print(f"Целевой FAR: {config.target_far:.4f}")
    print(f"Путь к JSON threshold policy: {thresholds_path}")
    print(f"Путь к CSV-отчёту: {report_path}")
    print(report_frame.to_string(index=False))


if __name__ == "__main__":
    main()
