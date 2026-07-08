"""Сравнение embedding-based verification с softmax baseline v2.

Файл размещается в каталоге:

    src/embedding_model/compare_embedding_vs_softmax.py

Назначение файла:

    1. Загрузить результаты embedding threshold policy.
    2. Загрузить результаты эксперимента enrollment size.
    3. Сформировать единую таблицу сравнения с softmax baseline v2.
    4. Зафиксировать вывод: embedding лучше, хуже или сопоставим с baseline v2.
    5. Сохранить CSV-отчёт для будущей статьи.

Выходные файлы:

    reports/embedding_model/embedding_vs_softmax_comparison.csv
    reports/embedding_model/embedding_vs_softmax_summary.md

Методологическое правило:

    Сравниваются только test-результаты.
    Значения softmax baseline v2 берутся из зафиксированного baseline-отчёта
    проекта или задаются через CLI-аргументы. По умолчанию используются
    значения baseline v2 BatchNorm, уже полученные в предыдущем этапе:

        test accuracy = 91.79%
        EER = 1.71%
        FAR ≈ 1.00%
        FRR = 2.52%
        false accepts = 2041
        false rejects = 103

Терминология:

- ``softmax baseline v2`` — baseline-модель mlp_128_64_batchnorm,
  использующая softmax score заявленного пользователя;
- ``embedding-based verification`` — проверка по расстоянию между текущим
  embedding-вектором и embedding-шаблоном заявленного пользователя;
- ``global threshold`` — общий порог для всех пользователей;
- ``per-user threshold`` — индивидуальный порог заявленного пользователя;
- ``guarded threshold policy`` — осторожная индивидуализация порогов;
- ``FAR`` — доля ложных допусков;
- ``FRR`` — доля ложных отказов;
- ``EER`` — точка приблизительного равенства FAR и FRR.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd

from src.embedding_model.embedding import EmbeddingModelPaths

EMBEDDING_THRESHOLD_POLICY_FILENAME: Final[str] = "embedding_threshold_policy.csv"
ENROLLMENT_EXPERIMENT_FILENAME: Final[str] = "embedding_enrollment_size_experiment.csv"
COMPARISON_FILENAME: Final[str] = "embedding_vs_softmax_comparison.csv"
SUMMARY_FILENAME: Final[str] = "embedding_vs_softmax_summary.md"

# Зафиксированные результаты baseline v2 BatchNorm из предыдущего этапа проекта.
DEFAULT_BASELINE_V2_TEST_ACCURACY: Final[float] = 0.9179
DEFAULT_BASELINE_V2_EER: Final[float] = 0.0171
DEFAULT_BASELINE_V2_FAR: Final[float] = 0.010005
DEFAULT_BASELINE_V2_FRR: Final[float] = 0.025245
DEFAULT_BASELINE_V2_FALSE_ACCEPTS: Final[int] = 2041
DEFAULT_BASELINE_V2_FALSE_REJECTS: Final[int] = 103
DEFAULT_BASELINE_V2_GENUINE_TRIALS: Final[int] = 4080
DEFAULT_BASELINE_V2_IMPOSTOR_TRIALS: Final[int] = 204000
DEFAULT_BASELINE_V2_MAX_USER_FAR: Final[float] = 0.029
DEFAULT_BASELINE_V2_MAX_USER_FRR: Final[float] = 0.075

# Зафиксированные результаты guarded softmax policy из черновика статьи.
DEFAULT_GUARDED_SOFTMAX_FAR: Final[float] = 0.009613
DEFAULT_GUARDED_SOFTMAX_FRR: Final[float] = 0.025490
DEFAULT_GUARDED_SOFTMAX_FALSE_ACCEPTS: Final[int] = 1961
DEFAULT_GUARDED_SOFTMAX_FALSE_REJECTS: Final[int] = 104
DEFAULT_GUARDED_SOFTMAX_MAX_USER_FAR: Final[float] = 0.029
DEFAULT_GUARDED_SOFTMAX_MAX_USER_FRR: Final[float] = 0.075


@dataclass(frozen=True)
class BaselineSoftmaxMetrics:
    """Метрики softmax baseline v2."""

    approach: str
    variant: str
    distance_metric: str
    policy: str
    enrollment_samples: int | None
    test_accuracy: float | None
    far: float
    frr: float
    eer: float | None
    false_accepts: int
    false_rejects: int
    genuine_trials: int
    impostor_trials: int
    max_user_far: float | None
    max_user_frr: float | None
    mean_user_far: float | None
    mean_user_frr: float | None
    balanced_error: float


@dataclass(frozen=True)
class ComparisonConclusion:
    """Итог сравнения embedding-подхода с baseline v2."""

    best_embedding_variant: str
    baseline_variant: str
    far_delta: float
    frr_delta: float
    eer_delta: float | None
    false_accepts_delta: int
    false_rejects_delta: int
    conclusion: str


def load_csv(path: Path) -> pd.DataFrame:
    """Загрузить CSV-файл.

    Аргументы:
        path: Путь к CSV-файлу.

    Возвращает:
        DataFrame с данными отчёта.

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


def build_baseline_rows(args: argparse.Namespace) -> list[BaselineSoftmaxMetrics]:
    """Сформировать строки сравнения для softmax baseline v2.

    Аргументы:
        args: Аргументы CLI с baseline-метриками.

    Возвращает:
        Список строк baseline-сравнения.
    """
    global_row = BaselineSoftmaxMetrics(
        approach="softmax_based_verification",
        variant="baseline_v2_mlp_128_64_batchnorm",
        distance_metric="softmax_score",
        policy="global",
        enrollment_samples=None,
        test_accuracy=args.baseline_test_accuracy,
        far=args.baseline_far,
        frr=args.baseline_frr,
        eer=args.baseline_eer,
        false_accepts=args.baseline_false_accepts,
        false_rejects=args.baseline_false_rejects,
        genuine_trials=args.baseline_genuine_trials,
        impostor_trials=args.baseline_impostor_trials,
        max_user_far=args.baseline_max_user_far,
        max_user_frr=args.baseline_max_user_frr,
        mean_user_far=None,
        mean_user_frr=None,
        balanced_error=float((args.baseline_far + args.baseline_frr) / 2.0),
    )

    guarded_row = BaselineSoftmaxMetrics(
        approach="softmax_based_verification",
        variant="baseline_v2_mlp_128_64_batchnorm",
        distance_metric="softmax_score",
        policy="guarded",
        enrollment_samples=None,
        test_accuracy=args.baseline_test_accuracy,
        far=args.guarded_softmax_far,
        frr=args.guarded_softmax_frr,
        eer=args.baseline_eer,
        false_accepts=args.guarded_softmax_false_accepts,
        false_rejects=args.guarded_softmax_false_rejects,
        genuine_trials=args.baseline_genuine_trials,
        impostor_trials=args.baseline_impostor_trials,
        max_user_far=args.guarded_softmax_max_user_far,
        max_user_frr=args.guarded_softmax_max_user_frr,
        mean_user_far=None,
        mean_user_frr=None,
        balanced_error=float((args.guarded_softmax_far + args.guarded_softmax_frr) / 2.0),
    )

    return [global_row, guarded_row]


def select_embedding_rows(threshold_policy: pd.DataFrame) -> pd.DataFrame:
    """Выбрать test-строки embedding threshold policy.

    Аргументы:
        threshold_policy: Таблица ``embedding_threshold_policy.csv``.

    Возвращает:
        DataFrame со строками test split.
    """
    required_columns = {
        "split",
        "distance_metric",
        "policy",
        "far",
        "frr",
        "balanced_error",
        "impostor_accepts",
        "genuine_rejects",
        "genuine_trials",
        "impostor_trials",
        "mean_user_far",
        "mean_user_frr",
        "max_user_far",
        "max_user_frr",
    }
    missing_columns = required_columns - set(threshold_policy.columns)

    if missing_columns:
        raise ValueError(
            "В embedding_threshold_policy.csv отсутствуют колонки: "
            f"{', '.join(sorted(missing_columns))}."
        )

    return threshold_policy[threshold_policy["split"] == "test"].copy()


def select_enrollment_rows(enrollment_experiment: pd.DataFrame) -> pd.DataFrame:
    """Выбрать test-строки эксперимента enrollment size.

    Аргументы:
        enrollment_experiment: Таблица ``embedding_enrollment_size_experiment.csv``.

    Возвращает:
        DataFrame со строками test split.
    """
    required_columns = {
        "enrollment_samples",
        "split",
        "distance_metric",
        "policy",
        "far",
        "frr",
        "balanced_error",
        "impostor_accepts",
        "genuine_rejects",
        "genuine_trials",
        "impostor_trials",
        "mean_user_far",
        "mean_user_frr",
        "max_user_far",
        "max_user_frr",
        "eer",
        "roc_auc",
    }
    missing_columns = required_columns - set(enrollment_experiment.columns)

    if missing_columns:
        raise ValueError(
            "В embedding_enrollment_size_experiment.csv отсутствуют колонки: "
            f"{', '.join(sorted(missing_columns))}."
        )

    return enrollment_experiment[enrollment_experiment["split"] == "test"].copy()


def convert_embedding_policy_rows(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Преобразовать строки embedding threshold policy к общей схеме."""
    rows: list[dict[str, object]] = []

    for _, row in dataframe.iterrows():
        rows.append(
            {
                "approach": "embedding_based_verification",
                "variant": "mean_template_full_train_256_samples",
                "distance_metric": row["distance_metric"],
                "policy": row["policy"],
                "enrollment_samples": 256,
                "test_accuracy": None,
                "far": float(row["far"]),
                "frr": float(row["frr"]),
                "eer": None,
                "false_accepts": int(row["impostor_accepts"]),
                "false_rejects": int(row["genuine_rejects"]),
                "genuine_trials": int(row["genuine_trials"]),
                "impostor_trials": int(row["impostor_trials"]),
                "max_user_far": float(row["max_user_far"]),
                "max_user_frr": float(row["max_user_frr"]),
                "mean_user_far": float(row["mean_user_far"]),
                "mean_user_frr": float(row["mean_user_frr"]),
                "balanced_error": float(row["balanced_error"]),
                "roc_auc": None,
                "notes": "Шаблон построен по всем 256 train samples пользователя.",
            }
        )

    return pd.DataFrame(rows)


def convert_enrollment_rows(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Преобразовать строки enrollment experiment к общей схеме."""
    rows: list[dict[str, object]] = []

    for _, row in dataframe.iterrows():
        enrollment_samples = int(row["enrollment_samples"])
        rows.append(
            {
                "approach": "embedding_based_verification",
                "variant": f"mean_template_{enrollment_samples}_enrollment_samples",
                "distance_metric": row["distance_metric"],
                "policy": row["policy"],
                "enrollment_samples": enrollment_samples,
                "test_accuracy": None,
                "far": float(row["far"]),
                "frr": float(row["frr"]),
                "eer": float(row["eer"]),
                "false_accepts": int(row["impostor_accepts"]),
                "false_rejects": int(row["genuine_rejects"]),
                "genuine_trials": int(row["genuine_trials"]),
                "impostor_trials": int(row["impostor_trials"]),
                "max_user_far": float(row["max_user_far"]),
                "max_user_frr": float(row["max_user_frr"]),
                "mean_user_far": float(row["mean_user_far"]),
                "mean_user_frr": float(row["mean_user_frr"]),
                "balanced_error": float(row["balanced_error"]),
                "roc_auc": float(row["roc_auc"]),
                "notes": "Шаблон построен по ограниченному числу enrollment samples.",
            }
        )

    return pd.DataFrame(rows)


def convert_baseline_rows(rows: list[BaselineSoftmaxMetrics]) -> pd.DataFrame:
    """Преобразовать baseline-строки к общей схеме."""
    dataframe = pd.DataFrame([asdict(row) for row in rows])
    dataframe["roc_auc"] = None
    dataframe["notes"] = "Зафиксированный softmax baseline v2 из предыдущего этапа."

    return dataframe


def build_comparison_table(
    baseline_rows: list[BaselineSoftmaxMetrics],
    embedding_policy_rows: pd.DataFrame,
    enrollment_rows: pd.DataFrame,
) -> pd.DataFrame:
    """Сформировать единую таблицу сравнения подходов.

    Аргументы:
        baseline_rows: Строки softmax baseline v2.
        embedding_policy_rows: Test-строки полного embedding-шаблона.
        enrollment_rows: Test-строки enrollment-size эксперимента.

    Возвращает:
        Общая таблица сравнения.
    """
    baseline_frame = convert_baseline_rows(baseline_rows)
    embedding_policy_frame = convert_embedding_policy_rows(embedding_policy_rows)
    enrollment_frame = convert_enrollment_rows(enrollment_rows)

    comparison = pd.concat(
        [baseline_frame, embedding_policy_frame, enrollment_frame],
        ignore_index=True,
    )

    comparison["far_percent"] = comparison["far"] * 100.0
    comparison["frr_percent"] = comparison["frr"] * 100.0
    comparison["balanced_error_percent"] = comparison["balanced_error"] * 100.0
    comparison["eer_percent"] = comparison["eer"] * 100.0

    sort_columns = [
        "approach",
        "distance_metric",
        "policy",
        "enrollment_samples",
        "balanced_error",
    ]

    return comparison.sort_values(sort_columns, na_position="first").reset_index(drop=True)


def select_best_embedding_variant(comparison: pd.DataFrame) -> pd.Series:
    """Выбрать лучший embedding-вариант по test balanced error.

    Аргументы:
        comparison: Общая таблица сравнения.

    Возвращает:
        Строка лучшего embedding-варианта.
    """
    embedding_rows = comparison[comparison["approach"] == "embedding_based_verification"].copy()

    if embedding_rows.empty:
        raise ValueError("В таблице сравнения отсутствуют embedding-варианты.")

    best_index = embedding_rows["balanced_error"].astype(float).idxmin()

    return comparison.loc[best_index]


def select_baseline_global(comparison: pd.DataFrame) -> pd.Series:
    """Выбрать global softmax baseline v2 из общей таблицы."""
    baseline_rows = comparison[
        (comparison["approach"] == "softmax_based_verification")
        & (comparison["policy"] == "global")
    ]

    if baseline_rows.empty:
        raise ValueError("В таблице сравнения отсутствует softmax baseline global.")

    return baseline_rows.iloc[0]


def build_conclusion(
    best_embedding: pd.Series,
    baseline: pd.Series,
) -> ComparisonConclusion:
    """Сформировать текстовый вывод по сравнению.

    Аргументы:
        best_embedding: Лучший embedding-вариант.
        baseline: Global softmax baseline v2.

    Возвращает:
        Структура с разницами метрик и выводом.
    """
    far_delta = float(best_embedding["far"] - baseline["far"])
    frr_delta = float(best_embedding["frr"] - baseline["frr"])
    false_accepts_delta = int(best_embedding["false_accepts"] - baseline["false_accepts"])
    false_rejects_delta = int(best_embedding["false_rejects"] - baseline["false_rejects"])

    embedding_eer = best_embedding.get("eer")
    baseline_eer = baseline.get("eer")

    if pd.notna(embedding_eer) and pd.notna(baseline_eer):
        eer_delta: float | None = float(embedding_eer - baseline_eer)
    else:
        eer_delta = None

    baseline_balanced = float(baseline["balanced_error"])
    embedding_balanced = float(best_embedding["balanced_error"])

    if embedding_balanced < baseline_balanced * 0.95:
        conclusion = "embedding лучше baseline v2"
    elif embedding_balanced <= baseline_balanced * 1.10:
        conclusion = "embedding сопоставим с baseline v2"
    else:
        conclusion = "embedding хуже baseline v2"

    return ComparisonConclusion(
        best_embedding_variant=str(best_embedding["variant"]),
        baseline_variant=str(baseline["variant"]),
        far_delta=far_delta,
        frr_delta=frr_delta,
        eer_delta=eer_delta,
        false_accepts_delta=false_accepts_delta,
        false_rejects_delta=false_rejects_delta,
        conclusion=conclusion,
    )


def build_summary_markdown(
    comparison: pd.DataFrame,
    conclusion: ComparisonConclusion,
) -> str:
    """Сформировать Markdown-резюме сравнения для отчёта stage 6."""
    best_embedding = select_best_embedding_variant(comparison)
    baseline = select_baseline_global(comparison)

    lines = [
        "# Сравнение embedding-based verification с softmax baseline v2",
        "",
        "## Лучший embedding-вариант",
        "",
        f"- Вариант: `{best_embedding['variant']}`",
        f"- Метрика расстояния: `{best_embedding['distance_metric']}`",
        f"- Threshold policy: `{best_embedding['policy']}`",
        f"- Enrollment samples: `{best_embedding['enrollment_samples']}`",
        f"- FAR: {float(best_embedding['far']) * 100:.4f}%",
        f"- FRR: {float(best_embedding['frr']) * 100:.4f}%",
        f"- Balanced error: {float(best_embedding['balanced_error']) * 100:.4f}%",
        f"- False accepts: {int(best_embedding['false_accepts'])}",
        f"- False rejects: {int(best_embedding['false_rejects'])}",
        "",
        "## Softmax baseline v2",
        "",
        f"- Вариант: `{baseline['variant']}`",
        f"- FAR: {float(baseline['far']) * 100:.4f}%",
        f"- FRR: {float(baseline['frr']) * 100:.4f}%",
        f"- Balanced error: {float(baseline['balanced_error']) * 100:.4f}%",
        f"- EER: {float(baseline['eer']) * 100:.4f}%",
        f"- False accepts: {int(baseline['false_accepts'])}",
        f"- False rejects: {int(baseline['false_rejects'])}",
        "",
        "## Вывод",
        "",
        f"Итог: **{conclusion.conclusion}**.",
        "",
        "Разница относительно softmax baseline v2:",
        "",
        f"- FAR delta: {conclusion.far_delta * 100:+.4f} п.п.",
        f"- FRR delta: {conclusion.frr_delta * 100:+.4f} п.п.",
        f"- False accepts delta: {conclusion.false_accepts_delta:+d}",
        f"- False rejects delta: {conclusion.false_rejects_delta:+d}",
    ]

    if conclusion.eer_delta is not None:
        lines.append(f"- EER delta: {conclusion.eer_delta * 100:+.4f} п.п.")

    lines.extend(
        [
            "",
            "Методологическая интерпретация:",
            "",
            (
                "Текущий encoder обучался через cross-entropy классификацию, "
                "поэтому его embedding-пространство не оптимизировано напрямую "
                "для distance-based verification. Это объясняет рост FRR при "
                "целевом FAR около 1%."
            ),
            "",
        ]
    )

    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    """Разобрать аргументы командной строки."""
    parser = argparse.ArgumentParser(
        description="Сравнение embedding-based verification с softmax baseline v2.",
    )
    parser.add_argument(
        "--baseline-test-accuracy",
        type=float,
        default=DEFAULT_BASELINE_V2_TEST_ACCURACY,
        help="Test accuracy softmax baseline v2.",
    )
    parser.add_argument(
        "--baseline-eer",
        type=float,
        default=DEFAULT_BASELINE_V2_EER,
        help="EER softmax baseline v2.",
    )
    parser.add_argument(
        "--baseline-far",
        type=float,
        default=DEFAULT_BASELINE_V2_FAR,
        help="FAR softmax baseline v2 global policy.",
    )
    parser.add_argument(
        "--baseline-frr",
        type=float,
        default=DEFAULT_BASELINE_V2_FRR,
        help="FRR softmax baseline v2 global policy.",
    )
    parser.add_argument(
        "--baseline-false-accepts",
        type=int,
        default=DEFAULT_BASELINE_V2_FALSE_ACCEPTS,
        help="Число ложных допусков softmax baseline v2 global policy.",
    )
    parser.add_argument(
        "--baseline-false-rejects",
        type=int,
        default=DEFAULT_BASELINE_V2_FALSE_REJECTS,
        help="Число ложных отказов softmax baseline v2 global policy.",
    )
    parser.add_argument(
        "--baseline-genuine-trials",
        type=int,
        default=DEFAULT_BASELINE_V2_GENUINE_TRIALS,
        help="Число genuine-попыток baseline v2.",
    )
    parser.add_argument(
        "--baseline-impostor-trials",
        type=int,
        default=DEFAULT_BASELINE_V2_IMPOSTOR_TRIALS,
        help="Число impostor-попыток baseline v2.",
    )
    parser.add_argument(
        "--baseline-max-user-far",
        type=float,
        default=DEFAULT_BASELINE_V2_MAX_USER_FAR,
        help="Максимальный per-user FAR baseline v2.",
    )
    parser.add_argument(
        "--baseline-max-user-frr",
        type=float,
        default=DEFAULT_BASELINE_V2_MAX_USER_FRR,
        help="Максимальный per-user FRR baseline v2.",
    )
    parser.add_argument(
        "--guarded-softmax-far",
        type=float,
        default=DEFAULT_GUARDED_SOFTMAX_FAR,
        help="FAR guarded softmax policy.",
    )
    parser.add_argument(
        "--guarded-softmax-frr",
        type=float,
        default=DEFAULT_GUARDED_SOFTMAX_FRR,
        help="FRR guarded softmax policy.",
    )
    parser.add_argument(
        "--guarded-softmax-false-accepts",
        type=int,
        default=DEFAULT_GUARDED_SOFTMAX_FALSE_ACCEPTS,
        help="Число ложных допусков guarded softmax policy.",
    )
    parser.add_argument(
        "--guarded-softmax-false-rejects",
        type=int,
        default=DEFAULT_GUARDED_SOFTMAX_FALSE_REJECTS,
        help="Число ложных отказов guarded softmax policy.",
    )
    parser.add_argument(
        "--guarded-softmax-max-user-far",
        type=float,
        default=DEFAULT_GUARDED_SOFTMAX_MAX_USER_FAR,
        help="Максимальный per-user FAR guarded softmax policy.",
    )
    parser.add_argument(
        "--guarded-softmax-max-user-frr",
        type=float,
        default=DEFAULT_GUARDED_SOFTMAX_MAX_USER_FRR,
        help="Максимальный per-user FRR guarded softmax policy.",
    )

    return parser.parse_args()


def main() -> None:
    """Выполнить сравнение embedding-подхода с softmax baseline v2."""
    args = parse_args()

    paths = EmbeddingModelPaths.from_source_dir()
    paths.ensure_directories()

    threshold_policy = load_csv(paths.reports_dir / EMBEDDING_THRESHOLD_POLICY_FILENAME)
    enrollment_experiment = load_csv(paths.reports_dir / ENROLLMENT_EXPERIMENT_FILENAME)

    baseline_rows = build_baseline_rows(args)
    embedding_policy_rows = select_embedding_rows(threshold_policy)
    enrollment_rows = select_enrollment_rows(enrollment_experiment)

    comparison = build_comparison_table(
        baseline_rows=baseline_rows,
        embedding_policy_rows=embedding_policy_rows,
        enrollment_rows=enrollment_rows,
    )

    best_embedding = select_best_embedding_variant(comparison)
    baseline = select_baseline_global(comparison)
    conclusion = build_conclusion(
        best_embedding=best_embedding,
        baseline=baseline,
    )

    comparison_path = paths.reports_dir / COMPARISON_FILENAME
    summary_path = paths.reports_dir / SUMMARY_FILENAME

    comparison.to_csv(comparison_path, index=False)
    summary_path.write_text(
        build_summary_markdown(comparison, conclusion),
        encoding="utf-8",
    )

    print("Сравнение embedding-based verification с softmax baseline v2 завершено.")
    print(f"Путь к CSV-сравнению: {comparison_path}")
    print(f"Путь к Markdown-резюме: {summary_path}")
    print(f"Итог: {conclusion.conclusion}")
    print()
    print("Лучший embedding-вариант:")
    print(
        best_embedding[
            [
                "variant",
                "distance_metric",
                "policy",
                "enrollment_samples",
                "far",
                "frr",
                "balanced_error",
                "false_accepts",
                "false_rejects",
                "eer",
                "roc_auc",
            ]
        ].to_string()
    )
    print()
    print("Softmax baseline v2:")
    print(
        baseline[
            [
                "variant",
                "policy",
                "far",
                "frr",
                "balanced_error",
                "false_accepts",
                "false_rejects",
                "eer",
            ]
        ].to_string()
    )


if __name__ == "__main__":
    main()
