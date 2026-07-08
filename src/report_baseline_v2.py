"""Формирование итогового отчёта по baseline v2 BatchNorm.

Модуль собирает сохранённые артефакты и отчёты по baseline v2, затем
формирует русскоязычный Markdown-отчёт:

    reports/baseline_v2_batchnorm_report.md

Модуль не переобучает модель и не изменяет артефакты.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import tensorflow as tf
from tensorflow import keras


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "mlp_v2_batchnorm.keras"
DEFAULT_SCALER_PATH = PROJECT_ROOT / "models" / "scaler_v2_batchnorm.pkl"
DEFAULT_LABEL_ENCODER_PATH = PROJECT_ROOT / "models" / "label_encoder_v2_batchnorm.pkl"
DEFAULT_AUTH_POLICY_PATH = PROJECT_ROOT / "models" / "auth_policy_v2_batchnorm.json"
DEFAULT_TRAINING_METRICS_PATH = PROJECT_ROOT / "reports" / "v2_batchnorm_training_metrics.csv"
DEFAULT_BATCH_REPORT_PATH = (
    PROJECT_ROOT / "reports" / "authentication_batch_report_v2_batchnorm.csv"
)
DEFAULT_COMPARISON_PATH = PROJECT_ROOT / "reports" / "baseline_v1_v2_comparison.csv"
DEFAULT_ARCHITECTURE_COMPARISON_PATH = PROJECT_ROOT / "reports" / "mlp_architecture_comparison.csv"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "reports" / "baseline_v2_batchnorm_report.md"


COLUMN_NAMES_RU = {
    "model": "модель",
    "architecture": "архитектура",
    "params": "параметры",
    "test_accuracy": "test accuracy",
    "eer": "EER",
    "auth_threshold": "порог аутентификации",
    "target_far": "целевой FAR",
    "actual_far": "фактический FAR",
    "actual_frr": "фактический FRR",
    "empirical_far": "эмпирический FAR",
    "empirical_frr": "эмпирический FRR",
    "genuine_rejects": "ложные отказы",
    "impostor_accepts": "ложные допуски",
    "genuine_trials": "genuine-попытки",
    "genuine_accepts": "принятые genuine-попытки",
    "impostor_trials": "чужие попытки",
    "impostor_rejects": "корректные отклонения чужих попыток",
    "policy_actual_far": "FAR из policy",
    "policy_actual_frr": "FRR из policy",
    "policy_eer": "EER из policy",
    "predicted_classes": "предсказанные классы",
    "model_name": "имя модели",
    "epochs_run": "эпохи",
    "best_validation_accuracy": "лучшая validation accuracy",
    "final_train_accuracy": "финальная train accuracy",
    "final_validation_accuracy": "финальная validation accuracy",
    "test_loss": "test loss",
    "eer_threshold": "порог EER",
    "layer": "слой",
    "type": "тип",
    "output_shape": "размерность выхода",
    "artifact": "артефакт",
    "path": "путь",
    "exists": "наличие",
}


PERCENT_COLUMNS = {
    "test_accuracy",
    "eer",
    "target_far",
    "actual_far",
    "actual_frr",
    "empirical_far",
    "empirical_frr",
    "policy_actual_far",
    "policy_actual_frr",
    "policy_eer",
    "best_validation_accuracy",
    "final_train_accuracy",
    "final_validation_accuracy",
}


ARTIFACT_NAMES_RU = {
    "model": "модель baseline v2",
    "scaler": "параметры нормализации baseline v2",
    "label_encoder": "кодировщик меток baseline v2",
    "auth_policy": "политика аутентификации baseline v2",
    "training_metrics": "метрики обучения baseline v2",
    "batch_report": "пакетная проверка baseline v2",
    "v1_v2_comparison": "сравнение baseline v1/v2",
    "architecture_comparison": "сравнение MLP-архитектур",
}


def read_json(path: Path) -> dict[str, Any]:
    """Прочитать JSON-файл.

    Args:
        path: Путь к JSON-файлу.

    Returns:
        Содержимое файла.

    Raises:
        FileNotFoundError: Если файл отсутствует.
    """
    if not path.exists():
        raise FileNotFoundError(f"JSON-файл не найден: {path}")

    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> pd.DataFrame:
    """Прочитать CSV-файл.

    Args:
        path: Путь к CSV-файлу.

    Returns:
        Загруженная таблица.

    Raises:
        FileNotFoundError: Если файл отсутствует.
    """
    if not path.exists():
        raise FileNotFoundError(f"CSV-файл не найден: {path}")

    return pd.read_csv(path)


def format_float(value: Any, digits: int = 6) -> str:
    """Отформатировать число.

    Args:
        value: Значение.
        digits: Количество знаков после запятой.

    Returns:
        Строковое представление.
    """
    if value is None:
        return "нет данных"

    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def format_percent(value: Any, digits: int = 2) -> str:
    """Отформатировать долю как процент.

    Args:
        value: Доля.
        digits: Количество знаков после запятой.

    Returns:
        Строка с процентом.
    """
    if value is None:
        return "нет данных"

    try:
        return f"{float(value) * 100:.{digits}f}%"
    except (TypeError, ValueError):
        return str(value)


def format_cell(value: object, column: str) -> str:
    """Отформатировать значение для Markdown-таблицы.

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
    max_rows: int | None = None,
    rename_columns: bool = True,
) -> str:
    """Преобразовать DataFrame в Markdown-таблицу без tabulate.

    Args:
        df: Исходная таблица.
        max_rows: Максимальное количество строк.
        rename_columns: Нужно ли переводить названия колонок.

    Returns:
        Markdown-таблица.
    """
    if max_rows is not None:
        df = df.head(max_rows)

    if df.empty:
        return "_Нет данных._"

    source_columns = list(df.columns)
    display_columns = [
        COLUMN_NAMES_RU.get(column, column) if rename_columns else column
        for column in source_columns
    ]

    header = "| " + " | ".join(display_columns) + " |"
    separator = "| " + " | ".join("---" for _column in display_columns) + " |"

    rows = []
    for _index, row in df.iterrows():
        values = [format_cell(row[column], column) for column in source_columns]
        rows.append("| " + " | ".join(values) + " |")

    return "\n".join([header, separator, *rows])


def extract_model_layers(model_path: Path) -> tuple[pd.DataFrame, int]:
    """Извлечь описание слоёв Keras-модели.

    Args:
        model_path: Путь к модели.

    Returns:
        Таблица слоёв и общее количество параметров.
    """
    if not model_path.exists():
        raise FileNotFoundError(f"Модель не найдена: {model_path}")

    model = keras.models.load_model(model_path)

    rows = []
    for layer in model.layers:
        output_shape = getattr(layer, "output_shape", "нет данных")
        rows.append(
            {
                "layer": layer.name,
                "type": layer.__class__.__name__,
                "output_shape": str(output_shape),
                "params": int(layer.count_params()),
            }
        )

    return pd.DataFrame(rows), int(model.count_params())


def render_artifact_table(paths: dict[str, Path]) -> str:
    """Сформировать таблицу артефактов.

    Args:
        paths: Словарь с путями артефактов.

    Returns:
        Markdown-таблица.
    """
    rows = []
    for name, path in paths.items():
        rows.append(
            {
                "artifact": ARTIFACT_NAMES_RU.get(name, name),
                "path": str(path.relative_to(PROJECT_ROOT)),
                "exists": "есть" if path.exists() else "нет",
            }
        )

    return dataframe_to_markdown(pd.DataFrame(rows))


def get_v2_comparison_row(comparison_df: pd.DataFrame) -> pd.DataFrame:
    """Получить строку baseline v2 из таблицы сравнения.

    Args:
        comparison_df: Таблица baseline_v1_v2_comparison.csv.

    Returns:
        Одна строка с baseline v2.
    """
    if "model" not in comparison_df.columns:
        return pd.DataFrame()

    result = comparison_df[comparison_df["model"] == "baseline_v2_batchnorm"]

    if result.empty:
        return pd.DataFrame()

    return result.reset_index(drop=True)


def calculate_improvements(comparison_df: pd.DataFrame) -> dict[str, float]:
    """Рассчитать улучшения v2 относительно v1.

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
        "far_delta": float(v2["actual_far"] - v1["actual_far"]),
        "genuine_rejects_delta": float(v2["genuine_rejects"] - v1["genuine_rejects"]),
        "impostor_accepts_delta": float(v2["impostor_accepts"] - v1["impostor_accepts"]),
    }


def build_report(args: argparse.Namespace) -> str:
    """Сформировать русскоязычный отчёт по baseline v2.

    Args:
        args: Аргументы командной строки.

    Returns:
        Markdown-текст.
    """
    auth_policy = read_json(args.auth_policy_input)
    training_metrics = read_csv(args.training_metrics_input)
    batch_report = read_csv(args.batch_report_input)
    comparison = read_csv(args.comparison_input)
    architecture_comparison = read_csv(args.architecture_comparison_input)

    model_layers, total_params = extract_model_layers(args.model_input)
    v2_comparison_row = get_v2_comparison_row(comparison)
    improvements = calculate_improvements(comparison)

    artifact_paths = {
        "model": args.model_input,
        "scaler": args.scaler_input,
        "label_encoder": args.label_encoder_input,
        "auth_policy": args.auth_policy_input,
        "training_metrics": args.training_metrics_input,
        "batch_report": args.batch_report_input,
        "v1_v2_comparison": args.comparison_input,
        "architecture_comparison": args.architecture_comparison_input,
    }

    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines = [
        "# Итоговый отчёт по baseline v2 BatchNorm",
        "",
        f"Дата формирования: `{generated_at}`",
        "",
        "## 1. Назначение отчёта",
        "",
        (
            "Отчёт фиксирует результаты модели `mlp_v2_batchnorm`, выбранной "
            "в качестве основной baseline-модели после сравнения нескольких "
            "MLP-архитектур."
        ),
        "",
        (
            "Целевая задача системы — аутентификация пользователя по признакам "
            "клавиатурного почерка. Поэтому основное внимание уделяется не только "
            "точности классификации, но и биометрическим метрикам FAR, FRR и EER."
        ),
        "",
        "## 2. Краткая сводка baseline v2",
        "",
        "| Показатель | Значение |",
        "|---|---:|",
        f"| Имя модели | `{auth_policy.get('model_name', 'mlp_v2_batchnorm')}` |",
        f"| Тип authentication score | `{auth_policy.get('score_type', 'нет данных')}` |",
        f"| Количество параметров | {total_params} |",
        f"| Test accuracy | {format_percent(auth_policy.get('test_accuracy'))} |",
        f"| EER | {format_percent(auth_policy.get('eer'))} |",
        f"| Порог EER | {format_float(auth_policy.get('eer_threshold'))} |",
        f"| Рабочий порог аутентификации | {format_float(auth_policy.get('auth_threshold'))} |",
        f"| Целевой FAR | {format_percent(auth_policy.get('target_far'))} |",
        f"| Фактический FAR | {format_percent(auth_policy.get('actual_far'))} |",
        f"| Фактический FRR | {format_percent(auth_policy.get('actual_frr'))} |",
        "",
        "## 3. Архитектура модели",
        "",
        (
            "Baseline v2 использует два скрытых полносвязных слоя, BatchNorm, "
            "ReLU-активации и Dropout. BatchNorm стабилизирует распределение "
            "активаций между слоями, а Dropout снижает риск переобучения."
        ),
        "",
        dataframe_to_markdown(model_layers),
        "",
        "## 4. Метрики обучения и тестирования",
        "",
        dataframe_to_markdown(training_metrics),
        "",
        "## 5. Пакетная проверка аутентификации",
        "",
        (
            "Пакетная проверка использует всю тестовую выборку. Для genuine-попыток "
            "заявленный пользователь совпадает с истинным пользователем sample. "
            "Для чужих попыток sample проверяется против всех остальных пользователей."
        ),
        "",
        dataframe_to_markdown(batch_report),
        "",
        "## 6. Сравнение с baseline v1",
        "",
        dataframe_to_markdown(comparison),
        "",
        "## 7. Улучшения baseline v2 относительно baseline v1",
        "",
        "| Показатель | Изменение |",
        "|---|---:|",
        f"| Test accuracy | +{format_percent(improvements['test_accuracy_delta'])} |",
        f"| EER | {format_percent(improvements['eer_delta'])} |",
        (
            "| Относительное снижение EER | "
            f"{format_percent(improvements['eer_relative_reduction'])} |"
        ),
        f"| FRR при FAR≈1% | {format_percent(improvements['frr_delta'])} |",
        (
            "| Относительное снижение FRR | "
            f"{format_percent(improvements['frr_relative_reduction'])} |"
        ),
        f"| FAR | {format_percent(improvements['far_delta'])} |",
        f"| Ложные отказы | {int(improvements['genuine_rejects_delta'])} |",
        f"| Ложные допуски | {int(improvements['impostor_accepts_delta'])} |",
        "",
        "## 8. Место v2 среди проверенных MLP-архитектур",
        "",
        dataframe_to_markdown(architecture_comparison),
        "",
        "## 9. Рабочая политика аутентификации",
        "",
        (
            "Для baseline v2 зафиксирована политика с целевым FAR около 1%. "
            "Рабочий порог применяется к softmax-вероятности заявленного пользователя."
        ),
        "",
        "| Параметр | Значение |",
        "|---|---:|",
        f"| Score | `{auth_policy.get('score_type', 'нет данных')}` |",
        f"| Порог | {format_float(auth_policy.get('auth_threshold'))} |",
        f"| Целевой FAR | {format_percent(auth_policy.get('target_far'))} |",
        f"| Фактический FAR | {format_percent(auth_policy.get('actual_far'))} |",
        f"| Фактический FRR | {format_percent(auth_policy.get('actual_frr'))} |",
        f"| EER | {format_percent(auth_policy.get('eer'))} |",
        "",
        "## 10. Артефакты baseline v2",
        "",
        render_artifact_table(artifact_paths),
        "",
        "## 11. Решение по статусу модели",
        "",
        (
            "Модель `mlp_v2_batchnorm` рекомендуется зафиксировать как основную "
            "baseline-модель для дальнейших экспериментов. Она существенно "
            "снижает FRR при сохранении FAR около 1% и улучшает test accuracy "
            "по сравнению с исходной моделью `mlp_64`."
        ),
        "",
        "Исходную модель `mlp_64` следует оставить как контрольную точку.",
        "",
        "## 12. Ограничения и дальнейшие шаги",
        "",
        "Ограничения текущего baseline v2:",
        "",
        "- используется закрытый набор пользователей;",
        "- отсутствует явная модель неизвестного пользователя;",
        "- применяется единый порог для всех пользователей;",
        "- authentication score основан на softmax-вероятности классификатора;",
        "- качество проверено на фиксированной парольной фразе датасета CMU.",
        "",
        "Рекомендуемые следующие шаги:",
        "",
        "- рассчитать FAR/FRR отдельно по каждому пользователю;",
        "- добавить индивидуальные пороги пользователей;",
        "- сравнить v2 с embedding-based verification;",
        "- проверить устойчивость между сессиями;",
        "- добавить сценарий регистрации нового пользователя;",
        "- подготовить `authenticate_v2_batchnorm.py` к работе с реальными samples.",
        "",
    ]

    if not v2_comparison_row.empty:
        lines.extend(
            [
                "## 13. Компактная строка v2 из сравнения",
                "",
                dataframe_to_markdown(v2_comparison_row),
                "",
            ]
        )

    return "\n".join(lines)


def save_report(report_text: str, output_path: Path) -> None:
    """Сохранить отчёт.

    Args:
        report_text: Текст отчёта.
        output_path: Путь сохранения.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report_text, encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    """Создать CLI-парсер.

    Returns:
        Настроенный парсер.
    """
    parser = argparse.ArgumentParser(description="Сформировать отчёт по baseline v2 BatchNorm.")

    parser.add_argument("--model-input", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--scaler-input", type=Path, default=DEFAULT_SCALER_PATH)
    parser.add_argument(
        "--label-encoder-input",
        type=Path,
        default=DEFAULT_LABEL_ENCODER_PATH,
    )
    parser.add_argument(
        "--auth-policy-input",
        type=Path,
        default=DEFAULT_AUTH_POLICY_PATH,
    )
    parser.add_argument(
        "--training-metrics-input",
        type=Path,
        default=DEFAULT_TRAINING_METRICS_PATH,
    )
    parser.add_argument(
        "--batch-report-input",
        type=Path,
        default=DEFAULT_BATCH_REPORT_PATH,
    )
    parser.add_argument("--comparison-input", type=Path, default=DEFAULT_COMPARISON_PATH)
    parser.add_argument(
        "--architecture-comparison-input",
        type=Path,
        default=DEFAULT_ARCHITECTURE_COMPARISON_PATH,
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)

    return parser


def main() -> None:
    """Точка входа CLI."""
    parser = build_arg_parser()
    args = parser.parse_args()

    tf.get_logger().setLevel("ERROR")

    report_text = build_report(args)
    save_report(report_text=report_text, output_path=args.output)

    print("Baseline v2 report saved:")
    print(f"Path: {args.output}")
    print()
    print("Report summary:")
    print("Model: mlp_v2_batchnorm")
    print("Language: Russian")
    print("Output format: Markdown")
    print("Status: main baseline candidate")


if __name__ == "__main__":
    main()
