"""Формирование итогового отчёта по baseline MLP для Keystroke Auth.

Модуль собирает уже созданные артефакты модели и диагностические отчёты,
после чего формирует Markdown-отчёт на русском языке:

    reports/baseline_mlp_report.md

Модуль не переобучает модель и не пересчитывает метрики. Он только читает
готовые CSV/JSON/модельные артефакты и оформляет их в итоговый отчёт.
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

DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "mlp_baseline.keras"
DEFAULT_SCALER_PATH = PROJECT_ROOT / "models" / "scaler.pkl"
DEFAULT_LABEL_ENCODER_PATH = PROJECT_ROOT / "models" / "label_encoder.pkl"
DEFAULT_AUTH_POLICY_PATH = PROJECT_ROOT / "models" / "auth_policy.json"
DEFAULT_SPLIT_PATH = PROJECT_ROOT / "data" / "processed" / "cmu_split.json"

DEFAULT_CLASSIFICATION_REPORT_PATH = PROJECT_ROOT / "reports" / "classification_report.csv"
DEFAULT_CONFUSION_MATRIX_PATH = PROJECT_ROOT / "reports" / "confusion_matrix.csv"
DEFAULT_AUTHENTICATION_METRICS_PATH = PROJECT_ROOT / "reports" / "authentication_metrics.csv"
DEFAULT_AUTHENTICATION_SCORE_SUMMARY_PATH = (
    PROJECT_ROOT / "reports" / "authentication_score_summary.csv"
)
DEFAULT_AUTHENTICATION_THRESHOLD_POLICY_PATH = (
    PROJECT_ROOT / "reports" / "authentication_threshold_policy.csv"
)
DEFAULT_AUTHENTICATION_BATCH_REPORT_PATH = (
    PROJECT_ROOT / "reports" / "authentication_batch_report.csv"
)
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "reports" / "baseline_mlp_report.md"


RUSSIAN_COLUMN_NAMES = {
    "label": "метка",
    "user_id": "пользователь",
    "precision": "точность",
    "recall": "полнота",
    "f1-score": "F1-мера",
    "support": "количество samples",
    "true_user_id": "истинный пользователь",
    "predicted_user_id": "предсказанный пользователь",
    "count": "количество ошибок",
    "score_type": "тип score",
    "min": "минимум",
    "p01": "перцентиль 1%",
    "p05": "перцентиль 5%",
    "median": "медиана",
    "mean": "среднее",
    "p95": "перцентиль 95%",
    "p99": "перцентиль 99%",
    "max": "максимум",
    "target_far": "целевой FAR",
    "threshold": "порог",
    "actual_far": "фактический FAR",
    "actual_frr": "фактический FRR",
    "far_per_10000_impostor_trials": "ложных допусков на 10 000 чужих попыток",
    "frr_per_10000_genuine_trials": "ложных отказов на 10 000 genuine-попыток",
    "genuine_trials": "genuine-попытки",
    "genuine_accepts": "принятые genuine-попытки",
    "genuine_rejects": "отклонённые genuine-попытки",
    "empirical_frr": "эмпирический FRR",
    "impostor_trials": "чужие попытки",
    "impostor_accepts": "ложные допуски",
    "impostor_rejects": "корректные отклонения чужих попыток",
    "empirical_far": "эмпирический FAR",
    "auth_threshold": "порог аутентификации",
    "policy_actual_far": "FAR из auth_policy",
    "policy_actual_frr": "FRR из auth_policy",
    "predicted_classes": "предсказанные классы",
    "eer": "EER",
    "eer_threshold": "порог EER",
    "far_at_eer_threshold": "FAR на пороге EER",
    "frr_at_eer_threshold": "FRR на пороге EER",
    "far_at_threshold_0_50": "FAR при пороге 0.50",
    "frr_at_threshold_0_50": "FRR при пороге 0.50",
    "layer": "слой",
    "type": "тип",
    "output_shape": "размерность выхода",
    "params": "параметры",
    "artifact": "артефакт",
    "path": "путь",
    "exists": "наличие",
}


RUSSIAN_LABEL_NAMES = {
    "macro avg": "macro average",
    "weighted avg": "weighted average",
    "genuine": "genuine-попытки",
    "impostor": "чужие попытки",
}


ARTIFACT_RUSSIAN_NAMES = {
    "model": "обученная модель",
    "scaler": "параметры нормализации",
    "label_encoder": "кодировщик меток пользователей",
    "auth_policy": "политика аутентификации",
    "split_metadata": "метаданные разбиения выборки",
    "classification_report": "отчёт классификации",
    "confusion_matrix": "матрица ошибок",
    "authentication_metrics": "метрики аутентификации",
    "authentication_score_summary": "сводка authentication score",
    "authentication_threshold_policy": "таблица выбора порога",
    "authentication_batch_report": "пакетная проверка аутентификации",
}


def read_json(path: Path) -> dict[str, Any]:
    """Прочитать JSON-файл.

    Args:
        path: Путь к JSON-файлу.

    Returns:
        Содержимое JSON-файла.

    Raises:
        FileNotFoundError: Если файл не найден.
    """
    if not path.exists():
        raise FileNotFoundError(f"JSON-файл не найден: {path}")

    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path, index_col: int | None = None) -> pd.DataFrame:
    """Прочитать CSV-файл.

    Args:
        path: Путь к CSV-файлу.
        index_col: Индексная колонка, если требуется.

    Returns:
        Загруженная таблица.

    Raises:
        FileNotFoundError: Если файл не найден.
    """
    if not path.exists():
        raise FileNotFoundError(f"CSV-файл не найден: {path}")

    return pd.read_csv(path, index_col=index_col)


def format_float(value: Any, digits: int = 6) -> str:
    """Отформатировать числовое значение.

    Args:
        value: Значение.
        digits: Количество знаков после запятой.

    Returns:
        Строковое представление значения.
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
        Процентное представление значения.
    """
    if value is None:
        return "нет данных"

    try:
        return f"{float(value) * 100:.{digits} %}"
    except (TypeError, ValueError):
        return str(value)


def dataframe_to_markdown(
    df: pd.DataFrame,
    max_rows: int | None = None,
    column_names: dict[str, str] | None = None,
) -> str:
    """Преобразовать DataFrame в Markdown-таблицу без внешних зависимостей.

    Args:
        df: Исходная таблица.
        max_rows: Максимальное количество строк.
        column_names: Словарь переименования колонок.

    Returns:
        Markdown-таблица.
    """
    if max_rows is not None:
        df = df.head(max_rows)

    if df.empty:
        return "_Нет данных._"

    prepared = df.copy()
    prepared = localize_label_values(prepared)

    if column_names:
        prepared = prepared.rename(columns=column_names)

    columns = list(prepared.columns)
    header = "| " + " | ".join(str(column) for column in columns) + " |"
    separator = "| " + " | ".join("---" for _column in columns) + " |"

    rows = []
    for _index, row in prepared.iterrows():
        values = [format_markdown_cell(row[column]) for column in columns]
        rows.append("| " + " | ".join(values) + " |")

    return "\n".join([header, separator, *rows])


def format_markdown_cell(value: object) -> str:
    """Отформатировать одну ячейку Markdown-таблицы.

    Args:
        value: Значение ячейки.

    Returns:
        Строковое значение.
    """
    if isinstance(value, float):
        return f"{value:.6f}"

    return str(value).replace("|", "\\|")


def localize_label_values(df: pd.DataFrame) -> pd.DataFrame:
    """Локализовать отдельные значения в таблицах.

    Args:
        df: Исходная таблица.

    Returns:
        Таблица с локализованными значениями.
    """
    result = df.copy()

    for column in ("label", "user_id", "score_type"):
        if column in result.columns:
            result[column] = result[column].replace(RUSSIAN_LABEL_NAMES)

    return result


def extract_split_counts(split_data: dict[str, Any]) -> dict[str, int | str]:
    """Извлечь размеры выборок из cmu_split.json.

    Функция допускает небольшие отличия схемы JSON, чтобы отчёт не ломался
    при последующих изменениях структуры файла.

    Args:
        split_data: Содержимое cmu_split.json.

    Returns:
        Сводка по размерам выборок, пользователям и признакам.
    """
    counts = split_data.get("counts", {})
    splits = split_data.get("splits", {})

    train_count = counts.get("train") or counts.get("train_samples")
    validation_count = counts.get("validation") or counts.get("validation_samples")
    test_count = counts.get("test") or counts.get("test_samples")

    if train_count is None and isinstance(splits.get("train"), list):
        train_count = len(splits["train"])
    if validation_count is None and isinstance(splits.get("validation"), list):
        validation_count = len(splits["validation"])
    if test_count is None and isinstance(splits.get("test"), list):
        test_count = len(splits["test"])

    users = counts.get("users")
    if users is None:
        users_value = split_data.get("users")
        users = len(users_value) if isinstance(users_value, list) else users_value

    features = counts.get("features")
    feature_columns = split_data.get("feature_columns")
    if features is None and isinstance(feature_columns, list):
        features = len(feature_columns)

    total = counts.get("total")
    if total is None:
        numeric_counts = [
            value for value in (train_count, validation_count, test_count) if isinstance(value, int)
        ]
        total = sum(numeric_counts) if numeric_counts else "нет данных"

    return {
        "total": total or "нет данных",
        "train": train_count or "нет данных",
        "validation": validation_count or "нет данных",
        "test": test_count or "нет данных",
        "users": users or "нет данных",
        "features": features or "нет данных",
    }


def extract_main_classification_metrics(report_df: pd.DataFrame) -> pd.DataFrame:
    """Извлечь macro/weighted-метрики классификации.

    Args:
        report_df: Таблица classification_report.csv.

    Returns:
        Компактная таблица основных метрик.
    """
    df = report_df.copy()

    if "Unnamed: 0" in df.columns:
        df = df.rename(columns={"Unnamed: 0": "label"})

    if "label" not in df.columns:
        df = df.reset_index().rename(columns={"index": "label"})

    labels = {"macro avg", "weighted avg"}
    result = df[df["label"].isin(labels)].copy()

    columns = ["label", "precision", "recall", "f1-score", "support"]
    existing_columns = [column for column in columns if column in result.columns]
    return result.loc[:, existing_columns]


def extract_worst_users_by_recall(
    report_df: pd.DataFrame,
    count: int = 10,
) -> pd.DataFrame:
    """Извлечь пользователей с минимальной полнотой распознавания.

    Args:
        report_df: Таблица classification_report.csv.
        count: Количество пользователей в итоговой таблице.

    Returns:
        Таблица пользователей с наихудшей recall-метрикой.
    """
    df = report_df.copy()

    if "Unnamed: 0" in df.columns:
        df = df.rename(columns={"Unnamed: 0": "user_id"})

    if "user_id" not in df.columns:
        df = df.reset_index().rename(columns={"index": "user_id"})

    user_rows = df[df["user_id"].astype(str).str.startswith("s")].copy()

    if user_rows.empty or "recall" not in user_rows.columns:
        return pd.DataFrame()

    columns = ["user_id", "precision", "recall", "f1-score", "support"]
    existing_columns = [column for column in columns if column in user_rows.columns]

    return user_rows.loc[:, existing_columns].sort_values("recall", ascending=True).head(count)


def extract_top_confusions(confusion_matrix_path: Path, count: int = 10) -> pd.DataFrame:
    """Извлечь наиболее частые ошибки классификации.

    Args:
        confusion_matrix_path: Путь к confusion_matrix.csv.
        count: Количество ошибок в итоговой таблице.

    Returns:
        Таблица наиболее частых ошибок вида true_user -> predicted_user.
    """
    confusion_df = read_csv(confusion_matrix_path, index_col=0)

    rows: list[dict[str, Any]] = []
    for true_user_id in confusion_df.index:
        for predicted_user_id in confusion_df.columns:
            if str(true_user_id) == str(predicted_user_id):
                continue

            value = int(confusion_df.loc[true_user_id, predicted_user_id])
            if value > 0:
                rows.append(
                    {
                        "true_user_id": true_user_id,
                        "predicted_user_id": predicted_user_id,
                        "count": value,
                    }
                )

    if not rows:
        return pd.DataFrame(columns=["true_user_id", "predicted_user_id", "count"])

    return (
        pd.DataFrame(rows).sort_values("count", ascending=False).head(count).reset_index(drop=True)
    )


def extract_authentication_metrics(metrics_df: pd.DataFrame) -> pd.DataFrame:
    """Подготовить таблицу метрик аутентификации.

    Args:
        metrics_df: Таблица authentication_metrics.csv.

    Returns:
        Нормализованная таблица метрик.
    """
    if metrics_df.empty:
        return metrics_df

    df = metrics_df.copy()

    expected_columns = [
        "eer",
        "eer_threshold",
        "far_at_eer_threshold",
        "frr_at_eer_threshold",
        "far_at_threshold_0_50",
        "frr_at_threshold_0_50",
    ]

    existing_columns = [column for column in expected_columns if column in df.columns]

    if existing_columns:
        return df.loc[:, existing_columns]

    return df


def extract_model_summary(model_path: Path) -> tuple[pd.DataFrame, int]:
    """Загрузить Keras-модель и извлечь краткое описание слоёв.

    Args:
        model_path: Путь к сохранённой модели.

    Returns:
        Таблица слоёв и общее количество параметров.
    """
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
    """Сформировать таблицу наличия артефактов.

    Args:
        paths: Словарь имя артефакта -> путь.

    Returns:
        Markdown-таблица.
    """
    rows = [
        {
            "artifact": ARTIFACT_RUSSIAN_NAMES.get(name, name),
            "path": str(path.relative_to(PROJECT_ROOT)),
            "exists": "есть" if path.exists() else "нет",
        }
        for name, path in paths.items()
    ]

    return dataframe_to_markdown(
        pd.DataFrame(rows),
        column_names=RUSSIAN_COLUMN_NAMES,
    )


def build_report(args: argparse.Namespace) -> str:
    """Сформировать Markdown-отчёт.

    Args:
        args: Аргументы командной строки.

    Returns:
        Текст Markdown-отчёта.
    """
    split_data = read_json(args.split_input)
    auth_policy = read_json(args.auth_policy_input)

    classification_report_df = read_csv(args.classification_report_input)
    authentication_metrics_df = read_csv(args.authentication_metrics_input)
    authentication_score_summary_df = read_csv(args.authentication_score_summary_input)
    threshold_policy_df = read_csv(args.authentication_threshold_policy_input)
    batch_report_df = read_csv(args.authentication_batch_report_input)

    split_counts = extract_split_counts(split_data)
    main_classification_metrics = extract_main_classification_metrics(classification_report_df)
    worst_users = extract_worst_users_by_recall(classification_report_df)
    top_confusions = extract_top_confusions(args.confusion_matrix_input)
    authentication_metrics = extract_authentication_metrics(authentication_metrics_df)
    model_layers, total_params = extract_model_summary(args.model_input)

    artifact_paths = {
        "model": args.model_input,
        "scaler": args.scaler_input,
        "label_encoder": args.label_encoder_input,
        "auth_policy": args.auth_policy_input,
        "split_metadata": args.split_input,
        "classification_report": args.classification_report_input,
        "confusion_matrix": args.confusion_matrix_input,
        "authentication_metrics": args.authentication_metrics_input,
        "authentication_score_summary": args.authentication_score_summary_input,
        "authentication_threshold_policy": args.authentication_threshold_policy_input,
        "authentication_batch_report": args.authentication_batch_report_input,
    }

    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines = [
        "# Итоговый отчёт по baseline MLP",
        "",
        f"Дата формирования: `{generated_at}`",
        "",
        "## 1. Краткая сводка",
        "",
        "| Показатель | Значение |",
        "|---|---:|",
        f"| Модель | `{auth_policy.get('model_name', 'mlp_baseline')}` |",
        f"| Тип authentication score | `{auth_policy.get('score_type', 'нет данных')}` |",
        f"| Всего samples | {split_counts['total']} |",
        f"| Количество пользователей | {split_counts['users']} |",
        f"| Количество признаков | {split_counts['features']} |",
        f"| Обучающая выборка | {split_counts['train']} |",
        f"| Валидационная выборка | {split_counts['validation']} |",
        f"| Тестовая выборка | {split_counts['test']} |",
        f"| Количество параметров модели | {total_params} |",
        f"| Рабочий порог аутентификации | {format_float(auth_policy.get('auth_threshold'))} |",
        f"| Целевой FAR | {format_percent(auth_policy.get('target_far'))} |",
        f"| Фактический FAR | {format_percent(auth_policy.get('actual_far'))} |",
        f"| Фактический FRR | {format_percent(auth_policy.get('actual_frr'))} |",
        "",
        "## 2. Архитектура модели",
        "",
        (
            "Baseline-модель представляет собой полносвязную нейронную сеть "
            "для закрытой классификации пользователей по признакам клавиатурной динамики."
        ),
        "",
        dataframe_to_markdown(model_layers, column_names=RUSSIAN_COLUMN_NAMES),
        "",
        "## 3. Разбиение датасета",
        "",
        "| Выборка | Количество samples | Назначение |",
        "|---|---:|---|",
        f"| Обучающая | {split_counts['train']} | Обучение модели и StandardScaler |",
        f"| Валидационная | {split_counts['validation']} | Контроль качества во время обучения |",
        f"| Тестовая | {split_counts['test']} | Финальная независимая оценка |",
        "",
        "## 4. Метрики классификации",
        "",
        (
            "Эти метрики оценивают задачу закрытой классификации: "
            "к какому из известных пользователей относится sample."
        ),
        "",
        dataframe_to_markdown(
            main_classification_metrics,
            column_names=RUSSIAN_COLUMN_NAMES,
        ),
        "",
        "## 5. Пользователи с наименьшей полнотой распознавания",
        "",
        (
            "Полнота показывает, какая доля samples конкретного пользователя "
            "была правильно отнесена к этому же пользователю."
        ),
        "",
        dataframe_to_markdown(worst_users, column_names=RUSSIAN_COLUMN_NAMES),
        "",
        "## 6. Наиболее частые ошибки классификации",
        "",
        ("Таблица показывает пары пользователей, между которыми модель чаще всего ошибается."),
        "",
        dataframe_to_markdown(top_confusions, column_names=RUSSIAN_COLUMN_NAMES),
        "",
        "## 7. Метрики аутентификации",
        "",
        (
            "Для целевой задачи важнее не только точность классификации, "
            "а вероятность ложного допуска и ложного отказа."
        ),
        "",
        "- **FAR** — доля чужих попыток, ошибочно принятых системой.",
        "- **FRR** — доля genuine-попыток, ошибочно отклонённых системой.",
        "- **EER** — точка, в которой FAR и FRR примерно равны.",
        "",
        dataframe_to_markdown(
            authentication_metrics,
            column_names=RUSSIAN_COLUMN_NAMES,
        ),
        "",
        "## 8. Распределение authentication score",
        "",
        (
            "Authentication score — это softmax-вероятность, которую модель "
            "назначила заявленному пользователю."
        ),
        "",
        dataframe_to_markdown(
            authentication_score_summary_df,
            column_names=RUSSIAN_COLUMN_NAMES,
        ),
        "",
        "## 9. Политика выбора порога",
        "",
        (
            "Порог аутентификации выбирается исходя из допустимого уровня FAR. "
            "Снижение FAR обычно увеличивает FRR."
        ),
        "",
        dataframe_to_markdown(threshold_policy_df, column_names=RUSSIAN_COLUMN_NAMES),
        "",
        "## 10. Пакетная проверка аутентификации",
        "",
        (
            "Пакетная проверка имитирует genuine- и чужие попытки на всей "
            "тестовой выборке и подтверждает значения FAR/FRR из auth_policy.json."
        ),
        "",
        dataframe_to_markdown(batch_report_df, column_names=RUSSIAN_COLUMN_NAMES),
        "",
        "## 11. Артефакты эксперимента",
        "",
        render_artifact_table(artifact_paths),
        "",
        "## 12. Вывод по baseline-модели",
        "",
        (
            "Текущий baseline подтверждает, что простая MLP-модель способна "
            "использовать признаки клавиатурной динамики для распознавания "
            "пользователей и первичной аутентификации. Качество достаточно для "
            "исследовательского MVP, но недостаточно для промышленного применения "
            "без дополнительных механизмов защиты и настройки порогов."
        ),
        "",
        "Зафиксированная рабочая точка:",
        "",
        f"- порог аутентификации: `{format_float(auth_policy.get('auth_threshold'))}`;",
        f"- целевой FAR: `{format_percent(auth_policy.get('target_far'))}`;",
        f"- фактический FAR: `{format_percent(auth_policy.get('actual_far'))}`;",
        f"- фактический FRR: `{format_percent(auth_policy.get('actual_frr'))}`.",
        "",
        "Основные ограничения текущего baseline:",
        "",
        "- authentication score основан на softmax-вероятности классификатора;",
        "- используются единый общий порог и закрытый набор пользователей;",
        "- пока отсутствуют индивидуальные пороги для разных пользователей;",
        "- не реализована явная модель неизвестного пользователя;",
        "- часть genuine-попыток отклоняется, что отражается в FRR;",
        "- часть чужих попыток проходит порог, что отражается в FAR.",
        "",
        "Рекомендуемые следующие шаги:",
        "",
        "- сравнить несколько архитектур MLP;",
        "- добавить индивидуальные пороги пользователей;",
        "- проверить embedding-based verification;",
        "- добавить per-user FAR/FRR диагностику;",
        "- оценить устойчивость по сессиям и условиям ввода;",
        "- рассмотреть объединение клавиатурной биометрии с дополнительным фактором.",
        "",
    ]

    return "\n".join(lines)


def save_report(report_text: str, output_path: Path) -> None:
    """Сохранить Markdown-отчёт.

    Args:
        report_text: Текст отчёта.
        output_path: Путь сохранения.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report_text, encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    """Создать парсер аргументов командной строки.

    Returns:
        Настроенный парсер.
    """
    parser = argparse.ArgumentParser(description="Сформировать итоговый отчёт по baseline MLP.")

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
    parser.add_argument("--split-input", type=Path, default=DEFAULT_SPLIT_PATH)
    parser.add_argument(
        "--classification-report-input",
        type=Path,
        default=DEFAULT_CLASSIFICATION_REPORT_PATH,
    )
    parser.add_argument(
        "--confusion-matrix-input",
        type=Path,
        default=DEFAULT_CONFUSION_MATRIX_PATH,
    )
    parser.add_argument(
        "--authentication-metrics-input",
        type=Path,
        default=DEFAULT_AUTHENTICATION_METRICS_PATH,
    )
    parser.add_argument(
        "--authentication-score-summary-input",
        type=Path,
        default=DEFAULT_AUTHENTICATION_SCORE_SUMMARY_PATH,
    )
    parser.add_argument(
        "--authentication-threshold-policy-input",
        type=Path,
        default=DEFAULT_AUTHENTICATION_THRESHOLD_POLICY_PATH,
    )
    parser.add_argument(
        "--authentication-batch-report-input",
        type=Path,
        default=DEFAULT_AUTHENTICATION_BATCH_REPORT_PATH,
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)

    return parser


def main() -> None:
    """Точка входа CLI."""
    parser = build_arg_parser()
    args = parser.parse_args()

    report_text = build_report(args)
    save_report(report_text=report_text, output_path=args.output)

    print("Итоговый отчёт по baseline MLP сохранён:")
    print(f"Путь: {args.output}")
    print()
    print("Сводка отчёта:")
    print("Модель: mlp_baseline")
    print("Формат: Markdown")
    print("Язык: русский")
    print("Разделов: 12")


if __name__ == "__main__":
    tf.get_logger().setLevel("ERROR")
    main()
