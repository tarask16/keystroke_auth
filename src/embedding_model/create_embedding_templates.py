"""Формирование embedding-шаблонов пользователей для этапа 6 проекта Keystroke Auth.

Файл размещается в каталоге:

    src/embedding_model/create_embedding_templates.py

Назначение файла:

    1. Загрузить embedding-векторы train split.
    2. Найти все координаты embedding-вектора.
    3. Для каждого пользователя сформировать embedding-шаблон.
    4. Сохранить шаблоны в JSON-файл.
    5. Сохранить краткий CSV-отчёт по числу samples и нормам шаблонов.

Выходные файлы:

    users/embedding_model/user_templates_embedding.json
    reports/embedding_model/embedding_template_summary.csv

Методологическое правило:

    На этом шаге шаблоны строятся только по train split.
    Validation split будет использован позднее для подбора порогов.
    Test split не используется ни для формирования шаблонов, ни для подбора
    порогов аутентификации.

Терминология:

- ``embedding-вектор`` — компактное числовое представление одной попытки ввода;
- ``embedding-шаблон`` — эталонный вектор пользователя;
- ``mean embedding`` — шаблон, рассчитанный как среднее значение
  embedding-векторов пользователя;
- ``genuine-попытка`` — попытка, принадлежащая заявленному пользователю;
- ``impostor-попытка`` — чужая попытка, проверяемая против заявленного
  пользователя.
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

# Имя файла с embedding-векторами train split.
DEFAULT_TRAIN_EMBEDDINGS_FILENAME: Final[str] = "embeddings_train.csv"

# Имя выходного отчёта по сформированным шаблонам.
TEMPLATE_SUMMARY_FILENAME: Final[str] = "embedding_template_summary.csv"

# Метод формирования шаблона пользователя на первом шаге этапа 6.
TEMPLATE_METHOD_MEAN: Final[str] = "mean_embedding"


@dataclass(frozen=True)
class TemplateCreationConfig:
    """Параметры формирования embedding-шаблонов."""

    train_embeddings_path: Path
    min_samples_per_user: int = 1
    template_method: str = TEMPLATE_METHOD_MEAN


@dataclass(frozen=True)
class UserTemplate:
    """Embedding-шаблон одного пользователя."""

    embedding: list[float]
    samples_count: int
    embedding_dim: int
    template_method: str
    source_split: str


def load_embeddings_table(path: Path) -> pd.DataFrame:
    """Загрузить таблицу embedding-векторов.

    Аргументы:
        path: Путь к CSV-файлу с embedding-векторами.

    Возвращает:
        DataFrame с колонкой ``user_id`` и embedding-координатами.

    Исключения:
        FileNotFoundError: Возникает, если файл не найден.
        ValueError: Возникает, если таблица пуста.
    """
    if not path.exists():
        raise FileNotFoundError(f"Файл embedding-векторов не найден: {path}")

    dataframe = pd.read_csv(path)

    if dataframe.empty:
        raise ValueError(f"Файл embedding-векторов пуст: {path}")

    return dataframe


def detect_embedding_columns(dataframe: pd.DataFrame) -> list[str]:
    """Найти колонки embedding-вектора.

    Аргументы:
        dataframe: Таблица embedding-векторов.

    Возвращает:
        Список колонок вида ``embedding_00``, ``embedding_01`` и далее.

    Исключения:
        ValueError: Возникает, если embedding-колонки не найдены.
    """
    embedding_columns = [column for column in dataframe.columns if column.startswith("embedding_")]

    if not embedding_columns:
        raise ValueError("В таблице не найдены колонки embedding-вектора.")

    return sorted(embedding_columns)


def validate_embeddings_table(
    dataframe: pd.DataFrame,
    embedding_columns: list[str],
) -> None:
    """Проверить корректность таблицы embedding-векторов.

    Аргументы:
        dataframe: Таблица embedding-векторов.
        embedding_columns: Список колонок embedding-вектора.

    Исключения:
        ValueError: Возникает, если отсутствует ``user_id`` или есть
            некорректные числовые значения.
    """
    if "user_id" not in dataframe.columns:
        raise ValueError("В таблице embedding-векторов отсутствует колонка user_id.")

    embedding_values = dataframe[embedding_columns].replace([np.inf, -np.inf], np.nan)

    if embedding_values.isna().any().any():
        raise ValueError("В embedding-векторах обнаружены NaN или бесконечные значения.")


def create_mean_template(
    user_frame: pd.DataFrame,
    embedding_columns: list[str],
) -> np.ndarray:
    """Сформировать средний embedding-шаблон пользователя.

    Аргументы:
        user_frame: Таблица embedding-векторов одного пользователя.
        embedding_columns: Список колонок embedding-вектора.

    Возвращает:
        Средний embedding-вектор пользователя.
    """
    embedding_matrix = user_frame[embedding_columns].to_numpy(dtype=np.float32)
    return embedding_matrix.mean(axis=0).astype(np.float32)


def create_user_templates(
    dataframe: pd.DataFrame,
    embedding_columns: list[str],
    config: TemplateCreationConfig,
) -> dict[str, UserTemplate]:
    """Сформировать embedding-шаблоны для всех пользователей.

    Аргументы:
        dataframe: Таблица embedding-векторов train split.
        embedding_columns: Список колонок embedding-вектора.
        config: Параметры формирования шаблонов.

    Возвращает:
        Словарь ``user_id -> UserTemplate``.

    Исключения:
        ValueError: Возникает, если у пользователя меньше samples, чем
            требуется параметром ``min_samples_per_user``.
        NotImplementedError: Возникает, если выбран неизвестный метод шаблона.
    """
    if config.template_method != TEMPLATE_METHOD_MEAN:
        raise NotImplementedError(f"Метод шаблона не поддерживается: {config.template_method}")

    templates: dict[str, UserTemplate] = {}

    for user_id, user_frame in dataframe.groupby("user_id", sort=True):
        samples_count = len(user_frame)

        if samples_count < config.min_samples_per_user:
            raise ValueError(
                f"У пользователя {user_id} недостаточно samples: "
                f"{samples_count} < {config.min_samples_per_user}."
            )

        template_vector = create_mean_template(user_frame, embedding_columns)

        templates[str(user_id)] = UserTemplate(
            embedding=[float(value) for value in template_vector],
            samples_count=int(samples_count),
            embedding_dim=int(template_vector.shape[0]),
            template_method=config.template_method,
            source_split="train",
        )

    return templates


def build_templates_payload(
    templates: dict[str, UserTemplate],
    embedding_columns: list[str],
    config: TemplateCreationConfig,
) -> dict[str, object]:
    """Сформировать JSON-структуру с embedding-шаблонами.

    Аргументы:
        templates: Словарь пользовательских шаблонов.
        embedding_columns: Список embedding-координат.
        config: Параметры формирования шаблонов.

    Возвращает:
        JSON-совместимый словарь с метаданными и шаблонами пользователей.
    """
    return {
        "metadata": {
            "created_at_utc": datetime.now(UTC).isoformat(),
            "template_method": config.template_method,
            "source_split": "train",
            "users_count": len(templates),
            "embedding_dim": len(embedding_columns),
            "embedding_columns": embedding_columns,
            "train_embeddings_path": str(config.train_embeddings_path),
            "min_samples_per_user": config.min_samples_per_user,
        },
        "templates": {user_id: asdict(template) for user_id, template in templates.items()},
    }


def build_template_summary(
    templates: dict[str, UserTemplate],
) -> pd.DataFrame:
    """Сформировать CSV-отчёт по embedding-шаблонам.

    Аргументы:
        templates: Словарь пользовательских шаблонов.

    Возвращает:
        DataFrame с числом samples и нормой каждого шаблона.
    """
    rows: list[dict[str, float | int | str]] = []

    for user_id, template in templates.items():
        vector = np.asarray(template.embedding, dtype=np.float32)
        rows.append(
            {
                "user_id": user_id,
                "samples_count": template.samples_count,
                "embedding_dim": template.embedding_dim,
                "template_method": template.template_method,
                "source_split": template.source_split,
                "template_l2_norm": float(np.linalg.norm(vector, ord=2)),
                "template_l1_norm": float(np.linalg.norm(vector, ord=1)),
                "template_mean": float(np.mean(vector)),
                "template_std": float(np.std(vector)),
            }
        )

    return pd.DataFrame(rows).sort_values("user_id").reset_index(drop=True)


def save_templates_and_summary(
    templates: dict[str, UserTemplate],
    embedding_columns: list[str],
    config: TemplateCreationConfig,
    paths: EmbeddingModelPaths,
) -> tuple[Path, Path]:
    """Сохранить embedding-шаблоны и краткий отчёт.

    Аргументы:
        templates: Словарь пользовательских шаблонов.
        embedding_columns: Список embedding-координат.
        config: Параметры формирования шаблонов.
        paths: Пути к артефактам embedding-этапа.

    Возвращает:
        Кортеж путей к JSON-файлу шаблонов и CSV-отчёту.
    """
    paths.ensure_directories()

    payload = build_templates_payload(
        templates=templates,
        embedding_columns=embedding_columns,
        config=config,
    )

    templates_path = paths.templates_path
    templates_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    summary = build_template_summary(templates)
    summary_path = paths.reports_dir / TEMPLATE_SUMMARY_FILENAME
    summary.to_csv(summary_path, index=False)

    return templates_path, summary_path


def parse_args() -> argparse.Namespace:
    """Разобрать аргументы командной строки."""
    paths = EmbeddingModelPaths.from_source_dir()
    default_train_embeddings_path = paths.data_processed_dir / DEFAULT_TRAIN_EMBEDDINGS_FILENAME

    parser = argparse.ArgumentParser(
        description="Формирование embedding-шаблонов пользователей.",
    )
    parser.add_argument(
        "--train-embeddings",
        type=Path,
        default=default_train_embeddings_path,
        help="Путь к CSV-файлу embedding-векторов train split.",
    )
    parser.add_argument(
        "--min-samples-per-user",
        type=int,
        default=1,
        help="Минимально допустимое число train samples на пользователя.",
    )

    return parser.parse_args()


def main() -> None:
    """Выполнить формирование embedding-шаблонов пользователей."""
    args = parse_args()

    config = TemplateCreationConfig(
        train_embeddings_path=args.train_embeddings,
        min_samples_per_user=args.min_samples_per_user,
    )

    paths = EmbeddingModelPaths.from_source_dir()
    paths.ensure_directories()

    dataframe = load_embeddings_table(config.train_embeddings_path)
    embedding_columns = detect_embedding_columns(dataframe)
    validate_embeddings_table(dataframe, embedding_columns)

    templates = create_user_templates(
        dataframe=dataframe,
        embedding_columns=embedding_columns,
        config=config,
    )

    templates_path, summary_path = save_templates_and_summary(
        templates=templates,
        embedding_columns=embedding_columns,
        config=config,
        paths=paths,
    )

    samples_counts = [template.samples_count for template in templates.values()]

    print("Формирование embedding-шаблонов завершено.")
    print(f"Количество пользователей: {len(templates)}")
    print(f"Размерность embedding-вектора: {len(embedding_columns)}")
    print(f"Минимум samples на пользователя: {min(samples_counts)}")
    print(f"Максимум samples на пользователя: {max(samples_counts)}")
    print(f"Метод формирования шаблона: {config.template_method}")
    print(f"Путь к шаблонам: {templates_path}")
    print(f"Путь к отчёту: {summary_path}")


if __name__ == "__main__":
    main()
