"""Формирование итогового отчёта по этапу 5.

Отчёт описывает результаты этапа 5: сравнение MLP-архитектур, перевод
baseline v2 в статус основной baseline-модели, анализ per-user FAR/FRR,
проверку naive per-user thresholds и guarded per-user threshold policy.

Выходной файл:

    reports/stage5_biometric_authentication_report.md

Модуль не переобучает модели и не изменяет политики аутентификации.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_V1_POLICY = PROJECT_ROOT / "models" / "auth_policy.json"
DEFAULT_V2_POLICY = PROJECT_ROOT / "models" / "auth_policy_v2_batchnorm.json"
DEFAULT_GUARDED_POLICY = PROJECT_ROOT / "models" / "guarded_auth_policy_v2_batchnorm.json"

DEFAULT_ARCHITECTURE_COMPARISON = PROJECT_ROOT / "reports" / "mlp_architecture_comparison.csv"
DEFAULT_V2_TRAINING_METRICS = PROJECT_ROOT / "reports" / "v2_batchnorm_training_metrics.csv"
DEFAULT_V2_BATCH_REPORT = PROJECT_ROOT / "reports" / "authentication_batch_report_v2_batchnorm.csv"
DEFAULT_V1_V2_COMPARISON = PROJECT_ROOT / "reports" / "baseline_v1_v2_comparison.csv"
DEFAULT_PER_USER_DIAGNOSTICS = (
    PROJECT_ROOT / "reports" / "per_user_auth_diagnostics_v2_batchnorm.csv"
)
DEFAULT_PER_USER_THRESHOLDS = PROJECT_ROOT / "reports" / "per_user_thresholds_v2_batchnorm.csv"
DEFAULT_GUARDED_SUMMARY = (
    PROJECT_ROOT / "reports" / "guarded_per_user_thresholds_v2_batchnorm_summary.csv"
)
DEFAULT_GUARDED_THRESHOLDS = (
    PROJECT_ROOT / "reports" / "guarded_per_user_thresholds_v2_batchnorm.csv"
)
DEFAULT_GUARDED_BATCH_REPORT = (
    PROJECT_ROOT / "reports" / "authentication_batch_report_v2_guarded.csv"
)

DEFAULT_OUTPUT = PROJECT_ROOT / "reports" / "stage5_biometric_authentication_report.md"


COLUMN_NAMES_RU = {
    "architecture": "архитектура",
    "params": "параметры",
    "epochs_run": "эпохи",
    "best_validation_accuracy": "лучшая validation accuracy",
    "test_accuracy": "test accuracy",
    "eer": "EER",
    "actual_far": "фактический FAR",
    "actual_frr": "фактический FRR",
    "threshold_at_target_far": "порог при целевом FAR",
    "model": "модель",
    "auth_threshold": "порог аутентификации",
    "target_far": "целевой FAR",
    "empirical_far": "эмпирический FAR",
    "empirical_frr": "эмпирический FRR",
    "genuine_trials": "genuine-попытки",
    "genuine_accepts": "принятые genuine-попытки",
    "genuine_rejects": "ложные отказы",
    "impostor_trials": "чужие попытки",
    "impostor_accepts": "ложные допуски",
    "impostor_rejects": "корректные отклонения чужих попыток",
    "policy": "политика",
    "false_rejects": "ложные отказы",
    "false_accepts": "ложные допуски",
    "frr": "FRR",
    "far": "FAR",
    "max_user_frr": "максимальный FRR пользователя",
    "max_user_far": "максимальный FAR пользователя",
    "mean_user_frr": "средний FRR по пользователям",
    "mean_user_far": "средний FAR по пользователям",
    "user_id": "пользователь",
    "genuine_score_median": "медиана genuine score",
    "genuine_score_mean": "средний genuine score",
    "impostor_score_p99": "p99 impostor score",
    "impostor_score_max": "максимальный impostor score",
    "global_threshold": "общий порог",
    "candidate_threshold": "кандидатный порог",
    "guarded_threshold": "guarded-порог",
    "guarded_applied": "guarded применён",
    "guarded_reason": "причина решения",
    "threshold_delta": "изменение порога",
    "validation_global_far": "validation FAR при общем пороге",
    "validation_candidate_far": "validation FAR при кандидатном пороге",
    "validation_candidate_frr": "validation FRR при кандидатном пороге",
    "test_global_far": "test FAR при общем пороге",
    "test_guarded_far": "test FAR при guarded-пороге",
    "test_global_frr": "test FRR при общем пороге",
    "test_guarded_frr": "test FRR при guarded-пороге",
}


PERCENT_COLUMNS = {
    "best_validation_accuracy",
    "test_accuracy",
    "eer",
    "actual_far",
    "actual_frr",
    "target_far",
    "empirical_far",
    "empirical_frr",
    "frr",
    "far",
    "max_user_frr",
    "max_user_far",
    "mean_user_frr",
    "mean_user_far",
    "global_far",
    "global_frr",
    "individual_far",
    "individual_frr",
    "test_global_far",
    "test_global_frr",
    "test_naive_far",
    "test_naive_frr",
    "test_guarded_far",
    "test_guarded_frr",
    "validation_global_far",
    "validation_global_frr",
    "validation_candidate_far",
    "validation_candidate_frr",
    "validation_guarded_far",
    "validation_guarded_frr",
}


def read_json(path: Path, *, required: bool = True) -> dict[str, Any]:
    """Прочитать JSON-файл.

    Args:
        path: Путь к JSON-файлу.
        required: Считать ли файл обязательным.

    Returns:
        Содержимое JSON или пустой словарь.
    """
    if not path.exists():
        if required:
            raise FileNotFoundError(f"JSON-файл не найден: {path}")
        return {}

    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path, *, required: bool = True) -> pd.DataFrame:
    """Прочитать CSV-файл.

    Args:
        path: Путь к CSV-файлу.
        required: Считать ли файл обязательным.

    Returns:
        Таблица pandas.
    """
    if not path.exists():
        if required:
            raise FileNotFoundError(f"CSV-файл не найден: {path}")
        return pd.DataFrame()

    return pd.read_csv(path)


def format_float(value: Any, digits: int = 6) -> str:
    """Отформатировать число.

    Args:
        value: Значение.
        digits: Количество знаков после запятой.

    Returns:
        Строковое представление.
    """
    if value is None or pd.isna(value):
        return "нет данных"

    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def format_percent(value: Any, digits: int = 2) -> str:
    """Отформатировать долю как процент.

    Args:
        value: Доля от 0 до 1.
        digits: Количество знаков после запятой.

    Returns:
        Процентное представление.
    """
    if value is None or pd.isna(value):
        return "нет данных"

    try:
        return f"{float(value) * 100:.{digits}f}%"
    except (TypeError, ValueError):
        return str(value)


def format_cell(value: object, column: str) -> str:
    """Отформатировать ячейку Markdown-таблицы.

    Args:
        value: Значение ячейки.
        column: Имя исходной колонки.

    Returns:
        Строковое значение.
    """
    if column in PERCENT_COLUMNS:
        return format_percent(value)

    if isinstance(value, float):
        return f"{value:.6f}"

    return str(value).replace("|", "\\|")


def dataframe_to_markdown(
    df: pd.DataFrame,
    *,
    max_rows: int | None = None,
    columns: list[str] | None = None,
) -> str:
    """Преобразовать DataFrame в Markdown-таблицу без tabulate.

    Args:
        df: Исходная таблица.
        max_rows: Максимальное количество строк.
        columns: Подмножество колонок.

    Returns:
        Markdown-таблица.
    """
    if df.empty:
        return "_Нет данных._"

    if columns is not None:
        existing_columns = [column for column in columns if column in df.columns]
        df = df.loc[:, existing_columns]

    if max_rows is not None:
        df = df.head(max_rows)

    source_columns = list(df.columns)
    display_columns = [COLUMN_NAMES_RU.get(column, column) for column in source_columns]

    header = "| " + " | ".join(display_columns) + " |"
    separator = "| " + " | ".join("---" for _column in display_columns) + " |"

    rows = []
    for _index, row in df.iterrows():
        values = [format_cell(row[column], column) for column in source_columns]
        rows.append("| " + " | ".join(values) + " |")

    return "\n".join([header, separator, *rows])


def one_row(df: pd.DataFrame) -> pd.Series:
    """Получить первую строку таблицы.

    Args:
        df: Таблица.

    Returns:
        Первая строка.

    Raises:
        ValueError: Если таблица пустая.
    """
    if df.empty:
        raise ValueError("Ожидалась непустая таблица.")

    return df.iloc[0]


def get_policy_row(summary_df: pd.DataFrame, policy: str) -> pd.Series:
    """Получить строку политики из summary-таблицы.

    Args:
        summary_df: Таблица с политиками.
        policy: Имя политики.

    Returns:
        Строка политики.

    Raises:
        ValueError: Если политика не найдена.
    """
    if "policy" not in summary_df.columns:
        raise ValueError("В таблице summary отсутствует колонка policy.")

    matches = summary_df[summary_df["policy"] == policy]

    if matches.empty:
        raise ValueError(f"Политика не найдена: {policy}")

    return matches.iloc[0]


def calculate_v1_v2_improvements(comparison_df: pd.DataFrame) -> dict[str, float]:
    """Рассчитать улучшения baseline v2 относительно baseline v1.

    Args:
        comparison_df: Таблица сравнения v1/v2.

    Returns:
        Словарь улучшений.
    """
    v1 = comparison_df[comparison_df["model"] == "baseline_v1_mlp_64"].iloc[0]
    v2 = comparison_df[comparison_df["model"] == "baseline_v2_batchnorm"].iloc[0]

    return {
        "test_accuracy_delta": float(v2["test_accuracy"] - v1["test_accuracy"]),
        "eer_delta": float(v2["eer"] - v1["eer"]),
        "eer_relative_reduction": float((v1["eer"] - v2["eer"]) / v1["eer"]),
        "frr_delta": float(v2["actual_frr"] - v1["actual_frr"]),
        "frr_relative_reduction": float((v1["actual_frr"] - v2["actual_frr"]) / v1["actual_frr"]),
        "false_rejects_delta": float(v2["genuine_rejects"] - v1["genuine_rejects"]),
        "false_accepts_delta": float(v2["impostor_accepts"] - v1["impostor_accepts"]),
    }


def calculate_guarded_delta(summary_df: pd.DataFrame) -> dict[str, float]:
    """Рассчитать отличие guarded policy от global policy.

    Args:
        summary_df: Summary-таблица guarded-эксперимента.

    Returns:
        Словарь изменений.
    """
    global_policy = get_policy_row(summary_df, "global")
    guarded_policy = get_policy_row(summary_df, "guarded")

    return {
        "far_delta": float(guarded_policy["far"] - global_policy["far"]),
        "frr_delta": float(guarded_policy["frr"] - global_policy["frr"]),
        "false_accepts_delta": float(
            guarded_policy["false_accepts"] - global_policy["false_accepts"]
        ),
        "false_rejects_delta": float(
            guarded_policy["false_rejects"] - global_policy["false_rejects"]
        ),
    }


def build_artifact_table(paths: dict[str, Path]) -> pd.DataFrame:
    """Сформировать таблицу артефактов.

    Args:
        paths: Пути к артефактам.

    Returns:
        Таблица артефактов.
    """
    return pd.DataFrame(
        [
            {
                "artifact": name,
                "path": str(path.relative_to(PROJECT_ROOT)),
                "exists": "есть" if path.exists() else "нет",
            }
            for name, path in paths.items()
        ]
    )


def build_report(args: argparse.Namespace) -> str:
    """Сформировать итоговый отчёт этапа 5.

    Args:
        args: Аргументы CLI.

    Returns:
        Markdown-текст отчёта.
    """
    v1_policy = read_json(args.v1_policy, required=False)
    v2_policy = read_json(args.v2_policy)
    guarded_policy = read_json(args.guarded_policy)

    architecture_comparison = read_csv(args.architecture_comparison)
    v2_training_metrics = read_csv(args.v2_training_metrics)
    v2_batch_report = read_csv(args.v2_batch_report)
    v1_v2_comparison = read_csv(args.v1_v2_comparison)
    per_user_diagnostics = read_csv(args.per_user_diagnostics)
    per_user_thresholds = read_csv(args.per_user_thresholds)
    guarded_summary = read_csv(args.guarded_summary)
    guarded_thresholds = read_csv(args.guarded_thresholds)
    guarded_batch_report = read_csv(args.guarded_batch_report)

    v1_v2_improvements = calculate_v1_v2_improvements(v1_v2_comparison)
    guarded_delta = calculate_guarded_delta(guarded_summary)

    worst_frr = per_user_diagnostics.sort_values(
        ["frr", "genuine_rejects"],
        ascending=[False, False],
    )
    worst_far = per_user_diagnostics.sort_values(
        ["far", "impostor_accepts"],
        ascending=[False, False],
    )

    guarded_applied = guarded_thresholds[guarded_thresholds["guarded_applied"].astype(bool)].copy()
    guarded_reasons = guarded_thresholds["guarded_reason"].value_counts().reset_index()
    guarded_reasons.columns = ["reason", "users"]

    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    artifact_table = build_artifact_table(
        {
            "baseline v1 policy": args.v1_policy,
            "baseline v2 policy": args.v2_policy,
            "guarded policy": args.guarded_policy,
            "architecture comparison": args.architecture_comparison,
            "v2 training metrics": args.v2_training_metrics,
            "v2 batch report": args.v2_batch_report,
            "v1/v2 comparison": args.v1_v2_comparison,
            "per-user diagnostics": args.per_user_diagnostics,
            "naive per-user thresholds": args.per_user_thresholds,
            "guarded summary": args.guarded_summary,
            "guarded thresholds": args.guarded_thresholds,
            "guarded batch report": args.guarded_batch_report,
        }
    )

    lines = [
        "# Итоговый отчёт по этапу 5: биометрические метрики и политики аутентификации",
        "",
        f"Дата формирования отчёта: `{generated_at}`",
        "",
        "## 1. Назначение отчёта",
        "",
        (
            "Настоящий отчёт фиксирует результаты этапа 5 проекта Keystroke Auth. "
            "На этом этапе система была переведена от простой классификации "
            "пользователей к оценке пригодности модели для биометрической "
            "аутентификации."
        ),
        "",
        (
            "Основные оцениваемые показатели: `FAR`, `FRR`, `EER`, рабочий порог "
            "аутентификации, распределение ошибок по пользователям и влияние "
            "пользовательских порогов на качество проверки."
        ),
        "",
        "## 2. Термины и обозначения",
        "",
        "| Термин | Значение |",
        "|---|---|",
        "| Genuine-попытка | Проверка, в которой образец принадлежит заявленному пользователю. |",
        "| Impostor-попытка | Проверка, в которой образец принадлежит другому пользователю. |",
        "| FAR | False Acceptance Rate, доля ложных допусков чужих попыток. |",
        "| FRR | False Rejection Rate, доля ложных отказов настоящему пользователю. |",
        "| EER | Equal Error Rate, точка, в которой FAR и FRR приблизительно равны. |",
        (
            "| Authentication score | Оценка принадлежности образца заявленному "
            "пользователю. В текущем baseline используется softmax-вероятность "
            "заявленного пользователя. |"
        ),
        "| Threshold | Порог, выше которого попытка принимается как успешная. |",
        "",
        "## 3. Проверенные MLP-архитектуры",
        "",
        (
            "В рамках этапа 5 были проверены несколько MLP-архитектур. "
            "Цель эксперимента — выбрать более сильную baseline-модель без перехода "
            "к embedding-подходу."
        ),
        "",
        dataframe_to_markdown(
            architecture_comparison,
            columns=[
                "architecture",
                "params",
                "epochs_run",
                "best_validation_accuracy",
                "test_accuracy",
                "eer",
                "actual_far",
                "actual_frr",
                "threshold_at_target_far",
            ],
        ),
        "",
        "### Вывод по архитектурам",
        "",
        (
            "Архитектура `mlp_128_64_batchnorm` выбрана как новая основная "
            "baseline-модель. Она показала лучшую test accuracy и близкий к лучшему "
            "EER при значительно меньшем числе параметров, чем `mlp_256_128`."
        ),
        "",
        "## 4. Baseline v2 BatchNorm",
        "",
        "Новая рабочая модель:",
        "",
        "```text",
        "mlp_v2_batchnorm",
        "Dense(128) → BatchNorm → ReLU → Dropout(0.2)",
        "Dense(64)  → BatchNorm → ReLU → Dropout(0.2)",
        "Dense(num_users, Softmax)",
        "```",
        "",
        "### Метрики обучения и тестирования baseline v2",
        "",
        dataframe_to_markdown(v2_training_metrics),
        "",
        "### Пакетная проверка baseline v2",
        "",
        dataframe_to_markdown(v2_batch_report),
        "",
        "## 5. Сравнение baseline v1 и baseline v2",
        "",
        dataframe_to_markdown(v1_v2_comparison),
        "",
        "### Улучшения baseline v2 относительно baseline v1",
        "",
        "| Показатель | Изменение |",
        "|---|---:|",
        f"| Test accuracy | +{format_percent(v1_v2_improvements['test_accuracy_delta'])} |",
        f"| EER | {format_percent(v1_v2_improvements['eer_delta'])} |",
        (
            "| Относительное снижение EER | "
            f"{format_percent(v1_v2_improvements['eer_relative_reduction'])} |"
        ),
        f"| FRR при FAR около 1% | {format_percent(v1_v2_improvements['frr_delta'])} |",
        (
            "| Относительное снижение FRR | "
            f"{format_percent(v1_v2_improvements['frr_relative_reduction'])} |"
        ),
        f"| Ложные отказы | {int(v1_v2_improvements['false_rejects_delta'])} |",
        f"| Ложные допуски | {int(v1_v2_improvements['false_accepts_delta'])} |",
        "",
        (
            "Baseline v2 существенно улучшает качество: test accuracy выросла, EER "
            "снизился, а количество ложных отказов уменьшилось более чем в два раза. "
            "Поэтому `mlp_v2_batchnorm` фиксируется как основная baseline-модель "
            "для дальнейших экспериментов."
        ),
        "",
        "## 6. Per-user FAR/FRR diagnostics",
        "",
        (
            "После выбора baseline v2 была рассчитана диагностика по каждому "
            "пользователю. Она показывает, что общий порог даёт хорошие глобальные "
            "метрики, но работает неравномерно для разных пользователей."
        ),
        "",
        "### Пользователи с наибольшим FRR",
        "",
        dataframe_to_markdown(
            worst_frr,
            max_rows=10,
            columns=[
                "user_id",
                "genuine_trials",
                "genuine_rejects",
                "frr",
                "genuine_score_median",
                "genuine_score_mean",
            ],
        ),
        "",
        "### Пользователи с наибольшим FAR",
        "",
        dataframe_to_markdown(
            worst_far,
            max_rows=10,
            columns=[
                "user_id",
                "impostor_trials",
                "impostor_accepts",
                "far",
                "impostor_score_p99",
                "impostor_score_max",
            ],
        ),
        "",
        "### Вывод по per-user диагностике",
        "",
        (
            "Пользователи с повышенным FRR требуют более мягкой политики либо "
            "улучшения признакового пространства. Пользователи с повышенным FAR "
            "являются более рискованными с точки зрения безопасности, потому что "
            "для них чаще возникают ложные допуски чужих попыток."
        ),
        "",
        "## 7. Naive per-user thresholds",
        "",
        (
            "Наивная стратегия индивидуальных порогов подбирает отдельный threshold "
            "для каждого пользователя под целевой FAR. Такой подход оказался "
            "неудачным: он не улучшил общую политику и привёл к росту числа ошибок."
        ),
        "",
        dataframe_to_markdown(
            per_user_thresholds,
            max_rows=12,
            columns=[
                "user_id",
                "global_threshold",
                "individual_threshold",
                "global_far",
                "individual_far",
                "global_frr",
                "individual_frr",
                "diagnostic_group",
            ],
        ),
        "",
        "### Вывод по naive per-user thresholds",
        "",
        (
            "Наивные индивидуальные пороги не рекомендуется использовать как "
            "рабочую политику. Они могут ухудшить FAR или FRR, потому что стремятся "
            "механически подогнать каждого пользователя под локальный целевой FAR "
            "без учёта итогового компромисса безопасности и удобства."
        ),
        "",
        "## 8. Guarded per-user threshold policy",
        "",
        (
            "Guarded policy использует более осторожную стратегию. Общий порог "
            "остаётся базовым. Индивидуальный порог применяется только к тем "
            "пользователям, у которых на validation split повышен FAR, а повышение "
            "порога не приводит к чрезмерному росту validation FRR."
        ),
        "",
        "### Сравнение global, naive и guarded policy",
        "",
        dataframe_to_markdown(guarded_summary),
        "",
        "### Пользователи, для которых был применён guarded-порог",
        "",
        dataframe_to_markdown(
            guarded_applied,
            columns=[
                "user_id",
                "global_threshold",
                "candidate_threshold",
                "guarded_threshold",
                "validation_global_far",
                "validation_candidate_far",
                "validation_candidate_frr",
                "test_global_far",
                "test_guarded_far",
                "test_global_frr",
                "test_guarded_frr",
            ],
        ),
        "",
        "### Причины отклонения кандидатных порогов",
        "",
        dataframe_to_markdown(guarded_reasons),
        "",
        "### Проверка guarded policy через CLI-аутентификацию",
        "",
        dataframe_to_markdown(guarded_batch_report),
        "",
        "### Эффект guarded policy относительно global policy",
        "",
        "| Показатель | Изменение |",
        "|---|---:|",
        f"| FAR | {format_percent(guarded_delta['far_delta'])} |",
        f"| FRR | {format_percent(guarded_delta['frr_delta'])} |",
        f"| Ложные допуски | {int(guarded_delta['false_accepts_delta'])} |",
        f"| Ложные отказы | {int(guarded_delta['false_rejects_delta'])} |",
        "",
        (
            "Guarded policy снижает число ложных допусков при практически "
            "неизменном числе ложных отказов. Поэтому её можно рассматривать как "
            "экспериментальную security-oriented политику."
        ),
        "",
        "## 9. Зафиксированные политики аутентификации",
        "",
        "### Основная baseline policy",
        "",
        "| Параметр | Значение |",
        "|---|---:|",
        f"| Файл | `{args.v2_policy.relative_to(PROJECT_ROOT)}` |",
        f"| Модель | `{v2_policy.get('model_name', 'mlp_v2_batchnorm')}` |",
        f"| Score | `{v2_policy.get('score_type', 'нет данных')}` |",
        f"| Порог | {format_float(v2_policy.get('auth_threshold'))} |",
        f"| FAR | {format_percent(v2_policy.get('actual_far'))} |",
        f"| FRR | {format_percent(v2_policy.get('actual_frr'))} |",
        f"| EER | {format_percent(v2_policy.get('eer'))} |",
        "",
        "### Security-oriented guarded policy",
        "",
        "| Параметр | Значение |",
        "|---|---:|",
        f"| Файл | `{args.guarded_policy.relative_to(PROJECT_ROOT)}` |",
        f"| Policy name | `{guarded_policy.get('policy_name', 'нет данных')}` |",
        f"| Базовый общий порог | {format_float(guarded_policy.get('base_global_threshold'))} |",
        f"| Пользователей с guarded-порогом | {guarded_policy.get('guarded_applied_users_count')} |",
        f"| Test FAR | {format_percent(guarded_policy.get('test_far'))} |",
        f"| Test FRR | {format_percent(guarded_policy.get('test_frr'))} |",
        "",
        "## 10. Итоговое решение",
        "",
        "По результатам этапа 5 фиксируются следующие решения:",
        "",
        "1. `mlp_v2_batchnorm` становится основной baseline-моделью.",
        "2. `auth_policy_v2_batchnorm.json` остаётся основной baseline policy.",
        "3. `guarded_auth_policy_v2_batchnorm.json` фиксируется как security-oriented policy.",
        "4. Naive per-user thresholds не используются как рабочая политика.",
        "5. Baseline v1 сохраняется как контрольная точка для сравнения.",
        "",
        "## 11. Ограничения текущего этапа",
        "",
        "- используется fixed-text датасет;",
        "- пользователи в модели являются закрытым набором классов;",
        "- authentication score основан на softmax-вероятности классификатора;",
        "- модель пока не формирует embedding-шаблон нового пользователя;",
        "- отсутствует сценарий полноценной регистрации неизвестного пользователя;",
        "- guarded policy требует дальнейшей проверки на других split-ах и датасетах.",
        "",
        "## 12. Рекомендуемые дальнейшие шаги",
        "",
        "1. Перейти к embedding-модели и сохранить encoder отдельно.",
        "2. Реализовать формирование `user_template` по enrollment samples.",
        "3. Проверить разные объёмы enrollment: 5, 10, 20, 30 и 50 попыток.",
        "4. Подобрать индивидуальные distance-based thresholds.",
        "5. Сравнить softmax-based baseline с embedding-based verification.",
        "6. Подготовить сценарий регистрации нового пользователя.",
        "7. Проверить переносимость подхода на другом fixed-text датасете.",
        "",
        "## 13. Использованные артефакты",
        "",
        dataframe_to_markdown(artifact_table),
        "",
    ]

    if v1_policy:
        lines.extend(
            [
                "## 14. Справочно: baseline v1 policy",
                "",
                "| Параметр | Значение |",
                "|---|---:|",
                f"| Файл | `{args.v1_policy.relative_to(PROJECT_ROOT)}` |",
                f"| Модель | `{v1_policy.get('model_name', 'нет данных')}` |",
                f"| Порог | {format_float(v1_policy.get('auth_threshold'))} |",
                f"| FAR | {format_percent(v1_policy.get('actual_far'))} |",
                f"| FRR | {format_percent(v1_policy.get('actual_frr'))} |",
                f"| EER | {format_percent(v1_policy.get('eer'))} |",
                "",
            ]
        )

    return "\n".join(lines)


def save_report(report_text: str, output_path: Path) -> None:
    """Сохранить Markdown-отчёт.

    Args:
        report_text: Текст отчёта.
        output_path: Путь к выходному файлу.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report_text, encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    """Создать CLI-парсер.

    Returns:
        Настроенный CLI-парсер.
    """
    parser = argparse.ArgumentParser(
        description="Сформировать итоговый отчёт по этапу 5 на русском языке."
    )

    parser.add_argument("--v1-policy", type=Path, default=DEFAULT_V1_POLICY)
    parser.add_argument("--v2-policy", type=Path, default=DEFAULT_V2_POLICY)
    parser.add_argument("--guarded-policy", type=Path, default=DEFAULT_GUARDED_POLICY)
    parser.add_argument(
        "--architecture-comparison",
        type=Path,
        default=DEFAULT_ARCHITECTURE_COMPARISON,
    )
    parser.add_argument(
        "--v2-training-metrics",
        type=Path,
        default=DEFAULT_V2_TRAINING_METRICS,
    )
    parser.add_argument("--v2-batch-report", type=Path, default=DEFAULT_V2_BATCH_REPORT)
    parser.add_argument(
        "--v1-v2-comparison",
        type=Path,
        default=DEFAULT_V1_V2_COMPARISON,
    )
    parser.add_argument(
        "--per-user-diagnostics",
        type=Path,
        default=DEFAULT_PER_USER_DIAGNOSTICS,
    )
    parser.add_argument(
        "--per-user-thresholds",
        type=Path,
        default=DEFAULT_PER_USER_THRESHOLDS,
    )
    parser.add_argument("--guarded-summary", type=Path, default=DEFAULT_GUARDED_SUMMARY)
    parser.add_argument(
        "--guarded-thresholds",
        type=Path,
        default=DEFAULT_GUARDED_THRESHOLDS,
    )
    parser.add_argument(
        "--guarded-batch-report",
        type=Path,
        default=DEFAULT_GUARDED_BATCH_REPORT,
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)

    return parser


def main() -> None:
    """Точка входа CLI."""
    parser = build_arg_parser()
    args = parser.parse_args()

    report_text = build_report(args)
    save_report(report_text=report_text, output_path=args.output)

    print("Stage 5 report saved:")
    print(f"Path: {args.output}")
    print()
    print("Report summary:")
    print("Language: Russian")
    print("Output format: Markdown")
    print("Stage: 5")
    print("Main baseline model: mlp_v2_batchnorm")
    print("Main baseline policy: auth_policy_v2_batchnorm.json")
    print("Security-oriented policy: guarded_auth_policy_v2_batchnorm.json")


if __name__ == "__main__":
    main()
