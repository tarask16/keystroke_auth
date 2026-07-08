"""Генерация итогового отчёта stage 6 по embedding-based verification.

Файл размещается в каталоге:

    src/embedding_model/generate_stage6_embedding_report.py

Назначение файла:

    1. Загрузить все CSV-отчёты этапа 6.
    2. Сформировать итоговый Markdown-отчёт stage6_embedding_report.md.
    3. Зафиксировать архитектуру, артефакты, результаты и выводы.
    4. Подготовить материал для последующего включения в статью.

Входные файлы:

    reports/embedding_model/embedding_classifier_metrics.csv
    reports/embedding_model/embedding_distance_diagnostics.csv
    reports/embedding_model/embedding_threshold_policy.csv
    reports/embedding_model/embedding_enrollment_size_experiment.csv
    reports/embedding_model/embedding_vs_softmax_comparison.csv
    reports/embedding_model/embedding_vs_softmax_summary.md

Выходной файл:

    reports/embedding_model/stage6_embedding_report.md

Методологический вывод этапа:

    Embedding-based verification на базе encoder-а, обученного через softmax
    classification loss, работоспособен, но уступает softmax baseline v2
    по FRR и balanced error при сопоставимом FAR около 1%.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import pandas as pd

from src.embedding_model.embedding import EmbeddingModelPaths

CLASSIFIER_METRICS_FILENAME: Final[str] = "embedding_classifier_metrics.csv"
DISTANCE_DIAGNOSTICS_FILENAME: Final[str] = "embedding_distance_diagnostics.csv"
THRESHOLD_POLICY_FILENAME: Final[str] = "embedding_threshold_policy.csv"
ENROLLMENT_EXPERIMENT_FILENAME: Final[str] = "embedding_enrollment_size_experiment.csv"
COMPARISON_FILENAME: Final[str] = "embedding_vs_softmax_comparison.csv"
COMPARISON_SUMMARY_FILENAME: Final[str] = "embedding_vs_softmax_summary.md"
STAGE6_REPORT_FILENAME: Final[str] = "stage6_embedding_report.md"


@dataclass(frozen=True)
class Stage6Tables:
    """CSV-таблицы, необходимые для формирования итогового отчёта."""

    classifier_metrics: pd.DataFrame
    distance_diagnostics: pd.DataFrame
    threshold_policy: pd.DataFrame
    enrollment_experiment: pd.DataFrame
    comparison: pd.DataFrame
    comparison_summary: str


def load_csv(path: Path) -> pd.DataFrame:
    """Загрузить CSV-файл отчёта.

    Аргументы:
        path: Путь к CSV-файлу.

    Возвращает:
        DataFrame с содержимым отчёта.

    Исключения:
        FileNotFoundError: Возникает, если файл не найден.
        ValueError: Возникает, если таблица пуста.
    """
    if not path.exists():
        raise FileNotFoundError(f"CSV-файл не найден: {path}")

    dataframe = pd.read_csv(path)

    if dataframe.empty:
        raise ValueError(f"CSV-файл пуст: {path}")

    return dataframe


def load_text(path: Path) -> str:
    """Загрузить текстовый файл.

    Аргументы:
        path: Путь к текстовому файлу.

    Возвращает:
        Содержимое файла или пустую строку, если файл отсутствует.
    """
    if not path.exists():
        return ""

    return path.read_text(encoding="utf-8")


def load_stage6_tables(paths: EmbeddingModelPaths) -> Stage6Tables:
    """Загрузить все таблицы этапа 6.

    Аргументы:
        paths: Пути к артефактам embedding-этапа.

    Возвращает:
        Объект со всеми таблицами и Markdown-резюме сравнения.
    """
    reports_dir = paths.reports_dir

    return Stage6Tables(
        classifier_metrics=load_csv(reports_dir / CLASSIFIER_METRICS_FILENAME),
        distance_diagnostics=load_csv(reports_dir / DISTANCE_DIAGNOSTICS_FILENAME),
        threshold_policy=load_csv(reports_dir / THRESHOLD_POLICY_FILENAME),
        enrollment_experiment=load_csv(reports_dir / ENROLLMENT_EXPERIMENT_FILENAME),
        comparison=load_csv(reports_dir / COMPARISON_FILENAME),
        comparison_summary=load_text(reports_dir / COMPARISON_SUMMARY_FILENAME),
    )


def format_percent(value: float) -> str:
    """Отформатировать долю как проценты."""
    return f"{float(value) * 100:.4f}%"


def format_float(value: float) -> str:
    """Отформатировать вещественное число."""
    return f"{float(value):.6f}"


def dataframe_to_markdown(
    dataframe: pd.DataFrame,
    columns: list[str],
    rename: dict[str, str] | None = None,
) -> str:
    """Преобразовать DataFrame в Markdown-таблицу.

    Аргументы:
        dataframe: Исходная таблица.
        columns: Колонки, которые нужно вывести.
        rename: Опциональное переименование колонок.

    Возвращает:
        Markdown-таблица.
    """
    table = dataframe[columns].copy()

    if rename:
        table = table.rename(columns=rename)

    return table.to_markdown(index=False)


def get_classifier_test_row(classifier_metrics: pd.DataFrame) -> pd.Series:
    """Получить test-строку embedding-классификатора."""
    test_rows = classifier_metrics[classifier_metrics["split"] == "test"]

    if test_rows.empty:
        raise ValueError("В embedding_classifier_metrics.csv нет строки test.")

    return test_rows.iloc[0]


def get_best_distance_row(distance_diagnostics: pd.DataFrame) -> pd.Series:
    """Выбрать лучшую метрику расстояния по test EER."""
    test_rows = distance_diagnostics[distance_diagnostics["split"] == "test"].copy()

    if test_rows.empty:
        raise ValueError("В embedding_distance_diagnostics.csv нет test-строк.")

    return test_rows.loc[test_rows["eer"].astype(float).idxmin()]


def get_best_threshold_policy_row(threshold_policy: pd.DataFrame) -> pd.Series:
    """Выбрать лучшую threshold policy по test balanced error."""
    test_rows = threshold_policy[threshold_policy["split"] == "test"].copy()

    if test_rows.empty:
        raise ValueError("В embedding_threshold_policy.csv нет test-строк.")

    return test_rows.loc[test_rows["balanced_error"].astype(float).idxmin()]


def get_best_enrollment_row(enrollment_experiment: pd.DataFrame) -> pd.Series:
    """Выбрать лучший enrollment-вариант по test balanced error."""
    test_rows = enrollment_experiment[enrollment_experiment["split"] == "test"].copy()

    if test_rows.empty:
        raise ValueError("В embedding_enrollment_size_experiment.csv нет test-строк.")

    return test_rows.loc[test_rows["balanced_error"].astype(float).idxmin()]


def get_best_embedding_comparison_row(comparison: pd.DataFrame) -> pd.Series:
    """Выбрать лучший embedding-вариант из общей таблицы сравнения."""
    embedding_rows = comparison[comparison["approach"] == "embedding_based_verification"].copy()

    if embedding_rows.empty:
        raise ValueError("В comparison-таблице нет embedding-вариантов.")

    return embedding_rows.loc[embedding_rows["balanced_error"].astype(float).idxmin()]


def get_softmax_baseline_row(comparison: pd.DataFrame) -> pd.Series:
    """Получить global softmax baseline v2."""
    baseline_rows = comparison[
        (comparison["approach"] == "softmax_based_verification")
        & (comparison["policy"] == "global")
    ]

    if baseline_rows.empty:
        raise ValueError("В comparison-таблице нет softmax baseline global.")

    return baseline_rows.iloc[0]


def build_classifier_section(classifier_metrics: pd.DataFrame) -> str:
    """Сформировать раздел с качеством embedding-классификатора."""
    table = classifier_metrics.copy()
    for column in ("accuracy", "macro_f1", "weighted_f1"):
        table[column] = table[column].map(format_percent)

    return "\n".join(
        [
            "## 2. Качество embedding-классификатора",
            "",
            dataframe_to_markdown(
                table,
                columns=["split", "samples", "accuracy", "macro_f1", "weighted_f1"],
                rename={
                    "split": "Split",
                    "samples": "Samples",
                    "accuracy": "Accuracy",
                    "macro_f1": "Macro F1",
                    "weighted_f1": "Weighted F1",
                },
            ),
            "",
            (
                "Классификационное качество encoder-а достаточно высокое, "
                "но эта метрика не эквивалентна качеству biometric verification. "
                "Для аутентификации основными являются FAR, FRR и EER."
            ),
            "",
        ]
    )


def build_distance_section(distance_diagnostics: pd.DataFrame) -> str:
    """Сформировать раздел диагностики расстояний."""
    table = distance_diagnostics[distance_diagnostics["split"].isin(["validation", "test"])].copy()

    percent_columns = ["roc_auc", "eer", "far_at_eer_threshold", "frr_at_eer_threshold"]
    for column in percent_columns:
        if column == "roc_auc":
            table[column] = table[column].map(format_float)
        else:
            table[column] = table[column].map(format_percent)

    for column in (
        "genuine_distance_mean",
        "impostor_distance_mean",
        "separation_margin",
    ):
        table[column] = table[column].map(format_float)

    best_row = get_best_distance_row(distance_diagnostics)

    return "\n".join(
        [
            "## 3. Диагностика embedding-расстояний",
            "",
            dataframe_to_markdown(
                table,
                columns=[
                    "split",
                    "distance_metric",
                    "roc_auc",
                    "eer",
                    "genuine_distance_mean",
                    "impostor_distance_mean",
                    "separation_margin",
                ],
                rename={
                    "split": "Split",
                    "distance_metric": "Distance",
                    "roc_auc": "ROC AUC",
                    "eer": "EER",
                    "genuine_distance_mean": "Genuine mean",
                    "impostor_distance_mean": "Impostor mean",
                    "separation_margin": "Margin",
                },
            ),
            "",
            (
                "Лучшая метрика по test EER: "
                f"`{best_row['distance_metric']}` с EER "
                f"{format_percent(best_row['eer'])}."
            ),
            "",
        ]
    )


def build_threshold_policy_section(threshold_policy: pd.DataFrame) -> str:
    """Сформировать раздел подбора threshold policy."""
    test_rows = threshold_policy[threshold_policy["split"] == "test"].copy()

    for column in ("far", "frr", "balanced_error", "max_user_far", "max_user_frr"):
        test_rows[column] = test_rows[column].map(format_percent)

    best_row = get_best_threshold_policy_row(threshold_policy)

    return "\n".join(
        [
            "## 4. Threshold policy",
            "",
            "Пороги подбирались только по validation split. Test split "
            "использовался только для финальной оценки.",
            "",
            dataframe_to_markdown(
                test_rows,
                columns=[
                    "distance_metric",
                    "policy",
                    "far",
                    "frr",
                    "balanced_error",
                    "impostor_accepts",
                    "genuine_rejects",
                    "max_user_far",
                    "max_user_frr",
                ],
                rename={
                    "distance_metric": "Distance",
                    "policy": "Policy",
                    "far": "FAR",
                    "frr": "FRR",
                    "balanced_error": "Balanced error",
                    "impostor_accepts": "False accepts",
                    "genuine_rejects": "False rejects",
                    "max_user_far": "Max user FAR",
                    "max_user_frr": "Max user FRR",
                },
            ),
            "",
            (
                "Лучшая test-policy: "
                f"`{best_row['distance_metric']} + {best_row['policy']}`. "
                f"FAR = {format_percent(best_row['far'])}, "
                f"FRR = {format_percent(best_row['frr'])}."
            ),
            "",
        ]
    )


def build_enrollment_section(enrollment_experiment: pd.DataFrame) -> str:
    """Сформировать раздел эксперимента enrollment size."""
    table = enrollment_experiment[
        (enrollment_experiment["split"] == "test")
        & (enrollment_experiment["distance_metric"] == "cosine")
        & (enrollment_experiment["policy"] == "per_user")
    ].copy()

    for column in ("far", "frr", "balanced_error", "eer"):
        table[column] = table[column].map(format_percent)

    table["roc_auc"] = table["roc_auc"].map(format_float)

    best_row = get_best_enrollment_row(enrollment_experiment)

    return "\n".join(
        [
            "## 5. Эксперимент с количеством enrollment samples",
            "",
            "Ниже показан основной вариант `cosine + per_user`.",
            "",
            dataframe_to_markdown(
                table,
                columns=[
                    "enrollment_samples",
                    "far",
                    "frr",
                    "balanced_error",
                    "impostor_accepts",
                    "genuine_rejects",
                    "eer",
                    "roc_auc",
                ],
                rename={
                    "enrollment_samples": "N",
                    "far": "FAR",
                    "frr": "FRR",
                    "balanced_error": "Balanced error",
                    "impostor_accepts": "False accepts",
                    "genuine_rejects": "False rejects",
                    "eer": "EER",
                    "roc_auc": "ROC AUC",
                },
            ),
            "",
            (
                "Лучший enrollment-вариант по test balanced error: "
                f"N = `{best_row['enrollment_samples']}`, "
                f"distance = `{best_row['distance_metric']}`, "
                f"policy = `{best_row['policy']}`."
            ),
            "",
            (
                "Увеличение числа enrollment samples снижает FRR, но даже при "
                "N=50 FRR остаётся существенно выше softmax baseline v2."
            ),
            "",
        ]
    )


def build_comparison_section(comparison: pd.DataFrame) -> str:
    """Сформировать раздел сравнения с softmax baseline v2."""
    best_embedding = get_best_embedding_comparison_row(comparison)
    baseline = get_softmax_baseline_row(comparison)

    rows = pd.DataFrame(
        [
            {
                "Approach": "Softmax baseline v2",
                "Variant": baseline["variant"],
                "Metric": baseline["distance_metric"],
                "Policy": baseline["policy"],
                "Enrollment": "-",
                "FAR": format_percent(baseline["far"]),
                "FRR": format_percent(baseline["frr"]),
                "Balanced error": format_percent(baseline["balanced_error"]),
                "EER": format_percent(baseline["eer"]),
                "False accepts": int(baseline["false_accepts"]),
                "False rejects": int(baseline["false_rejects"]),
            },
            {
                "Approach": "Best embedding",
                "Variant": best_embedding["variant"],
                "Metric": best_embedding["distance_metric"],
                "Policy": best_embedding["policy"],
                "Enrollment": best_embedding["enrollment_samples"],
                "FAR": format_percent(best_embedding["far"]),
                "FRR": format_percent(best_embedding["frr"]),
                "Balanced error": format_percent(best_embedding["balanced_error"]),
                "EER": (
                    "-" if pd.isna(best_embedding["eer"]) else format_percent(best_embedding["eer"])
                ),
                "False accepts": int(best_embedding["false_accepts"]),
                "False rejects": int(best_embedding["false_rejects"]),
            },
        ]
    )

    far_delta = float(best_embedding["far"] - baseline["far"])
    frr_delta = float(best_embedding["frr"] - baseline["frr"])
    balanced_delta = float(best_embedding["balanced_error"] - baseline["balanced_error"])

    return "\n".join(
        [
            "## 6. Сравнение с softmax baseline v2",
            "",
            rows.to_markdown(index=False),
            "",
            "Разница лучшего embedding-варианта относительно baseline v2:",
            "",
            f"- FAR delta: {far_delta * 100:+.4f} п.п.",
            f"- FRR delta: {frr_delta * 100:+.4f} п.п.",
            f"- Balanced error delta: {balanced_delta * 100:+.4f} п.п.",
            (
                f"- False accepts delta: "
                f"{int(best_embedding['false_accepts'] - baseline['false_accepts']):+d}"
            ),
            (
                f"- False rejects delta: "
                f"{int(best_embedding['false_rejects'] - baseline['false_rejects']):+d}"
            ),
            "",
        ]
    )


def build_artifacts_section(paths: EmbeddingModelPaths) -> str:
    """Сформировать раздел с перечнем артефактов этапа 6."""
    artifacts = [
        paths.classifier_path,
        paths.encoder_path,
        paths.scaler_path,
        paths.label_encoder_path,
        paths.templates_path,
        paths.thresholds_path,
        paths.reports_dir / CLASSIFIER_METRICS_FILENAME,
        paths.reports_dir / DISTANCE_DIAGNOSTICS_FILENAME,
        paths.reports_dir / THRESHOLD_POLICY_FILENAME,
        paths.reports_dir / ENROLLMENT_EXPERIMENT_FILENAME,
        paths.reports_dir / COMPARISON_FILENAME,
    ]

    lines = ["## 7. Сформированные артефакты", ""]

    for artifact in artifacts:
        lines.append(f"- `{artifact}`")

    lines.append("")

    return "\n".join(lines)


def build_final_conclusion() -> str:
    """Сформировать итоговый вывод этапа 6."""
    return "\n".join(
        [
            "## 8. Итоговый вывод",
            "",
            (
                "Embedding-based verification реализована и проверена на полном "
                "цикле: encoder, user templates, distance diagnostics, threshold "
                "policy, CLI-аутентификация и enrollment-size experiment."
            ),
            "",
            (
                "Лучший embedding-вариант использует cosine distance и per-user "
                "threshold policy. Однако при FAR около 1% он даёт существенно "
                "более высокий FRR, чем softmax baseline v2."
            ),
            "",
            (
                "Основная причина: encoder обучался как softmax-классификатор "
                "через cross-entropy loss. Такое обучение хорошо решает задачу "
                "закрытой классификации пользователей, но не оптимизирует "
                "embedding-пространство напрямую под distance-based verification."
            ),
            "",
            "Практический вывод:",
            "",
            ("- softmax baseline v2 остаётся основной рабочей моделью текущего этапа;"),
            (
                "- embedding-подход следует развивать через Siamese / contrastive / "
                "triplet loss или metric-learning fine-tuning;"
            ),
            (
                "- результаты stage 6 можно использовать как отрицательный, но "
                "методологически значимый эксперимент для статьи."
            ),
            "",
        ]
    )


def build_stage6_report(tables: Stage6Tables, paths: EmbeddingModelPaths) -> str:
    """Сформировать полный Markdown-отчёт stage 6."""
    test_row = get_classifier_test_row(tables.classifier_metrics)

    sections = [
        "# Stage 6. Embedding-based verification",
        "",
        "## 1. Назначение этапа",
        "",
        (
            "Цель этапа — проверить, можно ли перейти от softmax-based "
            "аутентификации к embedding-based verification, где решение "
            "принимается по расстоянию между текущим embedding-вектором и "
            "шаблоном заявленного пользователя."
        ),
        "",
        "Проверенная схема:",
        "",
        "```text",
        (
            "sample -> encoder -> embedding vector -> distance to user_template "
            "-> threshold_user -> ACCEPT/REJECT"
        ),
        "```",
        "",
        "Базовая embedding-модель:",
        "",
        "```text",
        "Input(31 timing features)",
        "Dense(128, ReLU)",
        "BatchNormalization",
        "Dropout(0.2)",
        "Dense(32, ReLU, name='embedding')",
        "BatchNormalization",
        "Dense(51, Softmax)",
        "```",
        "",
        (f"Test accuracy embedding-классификатора: {format_percent(test_row['accuracy'])}."),
        "",
        build_classifier_section(tables.classifier_metrics),
        build_distance_section(tables.distance_diagnostics),
        build_threshold_policy_section(tables.threshold_policy),
        build_enrollment_section(tables.enrollment_experiment),
        build_comparison_section(tables.comparison),
        build_artifacts_section(paths),
        build_final_conclusion(),
    ]

    return "\n".join(sections)


def parse_args() -> argparse.Namespace:
    """Разобрать аргументы командной строки."""
    parser = argparse.ArgumentParser(
        description="Генерация stage6_embedding_report.md.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Опциональный путь к выходному Markdown-файлу.",
    )

    return parser.parse_args()


def main() -> None:
    """Сформировать итоговый Markdown-отчёт stage 6."""
    args = parse_args()

    paths = EmbeddingModelPaths.from_source_dir()
    paths.ensure_directories()

    tables = load_stage6_tables(paths)
    report_text = build_stage6_report(tables, paths)

    output_path = args.output or paths.reports_dir / STAGE6_REPORT_FILENAME
    output_path.write_text(report_text, encoding="utf-8")

    best_embedding = get_best_embedding_comparison_row(tables.comparison)
    baseline = get_softmax_baseline_row(tables.comparison)

    print("Итоговый отчёт stage 6 сформирован.")
    print(f"Путь к отчёту: {output_path}")
    print()
    print("Ключевой вывод:")
    print(
        "Embedding-based verification уступает softmax baseline v2: "
        f"FRR {format_percent(best_embedding['frr'])} против "
        f"{format_percent(baseline['frr'])} при сопоставимом FAR."
    )


if __name__ == "__main__":
    main()
