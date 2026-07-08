"""Построение графика сравнения MLP-архитектур.

Скрипт читает файл:

    reports/mlp_architecture_comparison.csv

и формирует одно изображение с двумя столбчатыми диаграммами:

    слева  — точность на тестовой выборке, test accuracy;
    справа — равная вероятность ошибок, EER.

Выходные файлы:

    reports/figures/mlp_architecture_accuracy_eer_ru.png
    reports/figures/mlp_architecture_accuracy_eer_ru.svg
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_INPUT = PROJECT_ROOT / "reports" / "mlp_architecture_comparison.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "reports" / "figures"

DEFAULT_OUTPUT_PNG = DEFAULT_OUTPUT_DIR / "mlp_architecture_accuracy_eer_ru.png"
DEFAULT_OUTPUT_SVG = DEFAULT_OUTPUT_DIR / "mlp_architecture_accuracy_eer_ru.svg"

REQUIRED_COLUMNS = ["architecture", "test_accuracy", "eer"]


def read_architecture_comparison(input_path: Path) -> pd.DataFrame:
    """Прочитать CSV-отчёт со сравнением MLP-архитектур.

    Args:
        input_path: Путь к CSV-файлу `mlp_architecture_comparison.csv`.

    Returns:
        Таблица со сравнением архитектур.

    Raises:
        FileNotFoundError: Если входной CSV-файл отсутствует.
        ValueError: Если в таблице отсутствуют обязательные колонки.
    """
    if not input_path.exists():
        raise FileNotFoundError(f"CSV-файл со сравнением архитектур не найден: {input_path}")

    df = pd.read_csv(input_path)
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in df.columns]

    if missing_columns:
        raise ValueError(
            "В CSV-файле отсутствуют обязательные колонки: "
            f"{missing_columns}. Требуются колонки: {REQUIRED_COLUMNS}"
        )

    return df


def prepare_plot_data(df: pd.DataFrame) -> pd.DataFrame:
    """Подготовить данные для построения графика.

    Args:
        df: Исходная таблица со сравнением архитектур.

    Returns:
        Таблица с процентными значениями test accuracy и EER.
    """
    plot_df = df.loc[:, REQUIRED_COLUMNS].copy()
    plot_df["test_accuracy_percent"] = plot_df["test_accuracy"] * 100.0
    plot_df["eer_percent"] = plot_df["eer"] * 100.0

    return plot_df


def build_accuracy_eer_chart(
    plot_df: pd.DataFrame,
    output_png: Path,
    output_svg: Path,
) -> None:
    """Построить и сохранить график сравнения test accuracy и EER.

    Args:
        plot_df: Подготовленная таблица для построения графика.
        output_png: Путь сохранения PNG-файла.
        output_svg: Путь сохранения SVG-файла.
    """
    output_png.parent.mkdir(parents=True, exist_ok=True)
    output_svg.parent.mkdir(parents=True, exist_ok=True)

    plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["axes.unicode_minus"] = False

    figure, axes = plt.subplots(nrows=1, ncols=2, figsize=(14, 6), constrained_layout=True)

    architecture_labels = plot_df["architecture"].tolist()
    x_positions = list(range(len(architecture_labels)))

    build_test_accuracy_subplot(
        axis=axes[0],
        x_positions=x_positions,
        architecture_labels=architecture_labels,
        values=plot_df["test_accuracy_percent"],
    )
    build_eer_subplot(
        axis=axes[1],
        x_positions=x_positions,
        architecture_labels=architecture_labels,
        values=plot_df["eer_percent"],
    )

    figure.suptitle(
        "Сравнение MLP-архитектур по test accuracy и EER",
        fontsize=14,
    )

    figure.savefig(output_png, dpi=300, bbox_inches="tight")
    figure.savefig(output_svg, bbox_inches="tight")
    plt.close(figure)


def build_test_accuracy_subplot(
    axis: plt.Axes,
    x_positions: list[int],
    architecture_labels: list[str],
    values: pd.Series,
) -> None:
    """Построить столбчатую диаграмму test accuracy.

    Args:
        axis: Область построения matplotlib.
        x_positions: Позиции столбцов по оси X.
        architecture_labels: Подписи MLP-архитектур.
        values: Значения test accuracy в процентах.
    """
    axis.bar(x_positions, values)
    axis.set_title("Точность на тестовой выборке")
    axis.set_xlabel("MLP-архитектура")
    axis.set_ylabel("Test accuracy, %")
    axis.set_xticks(x_positions)
    axis.set_xticklabels(architecture_labels, rotation=35, ha="right")
    axis.set_ylim(0, 100)
    axis.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.7)

    for index, value in enumerate(values):
        axis.text(
            index,
            value + 1.0,
            f"{value:.2f}%",
            ha="center",
            va="bottom",
            fontsize=9,
        )


def build_eer_subplot(
    axis: plt.Axes,
    x_positions: list[int],
    architecture_labels: list[str],
    values: pd.Series,
) -> None:
    """Построить столбчатую диаграмму EER.

    Args:
        axis: Область построения matplotlib.
        x_positions: Позиции столбцов по оси X.
        architecture_labels: Подписи MLP-архитектур.
        values: Значения EER в процентах.
    """
    axis.bar(x_positions, values)
    axis.set_title("Равная вероятность ошибок")
    axis.set_xlabel("MLP-архитектура")
    axis.set_ylabel("EER, %")
    axis.set_xticks(x_positions)
    axis.set_xticklabels(architecture_labels, rotation=35, ha="right")
    axis.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.7)

    eer_max = float(values.max())
    axis.set_ylim(0, eer_max * 1.25)

    for index, value in enumerate(values):
        axis.text(
            index,
            value + eer_max * 0.03,
            f"{value:.2f}%",
            ha="center",
            va="bottom",
            fontsize=9,
        )


def build_arg_parser() -> argparse.ArgumentParser:
    """Создать CLI-парсер.

    Returns:
        Настроенный парсер аргументов командной строки.
    """
    parser = argparse.ArgumentParser(
        description=("Построить столбчатый график сравнения MLP-архитектур по test accuracy и EER.")
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Путь к файлу reports/mlp_architecture_comparison.csv.",
    )
    parser.add_argument(
        "--output-png",
        type=Path,
        default=DEFAULT_OUTPUT_PNG,
        help="Путь сохранения PNG-файла.",
    )
    parser.add_argument(
        "--output-svg",
        type=Path,
        default=DEFAULT_OUTPUT_SVG,
        help="Путь сохранения SVG-файла.",
    )

    return parser


def main() -> None:
    """Точка входа CLI."""
    parser = build_arg_parser()
    args = parser.parse_args()

    df = read_architecture_comparison(args.input)
    plot_df = prepare_plot_data(df)

    build_accuracy_eer_chart(
        plot_df=plot_df,
        output_png=args.output_png,
        output_svg=args.output_svg,
    )

    print("График сравнения MLP-архитектур сохранён:")
    print(f"Входной CSV: {args.input}")
    print(f"PNG: {args.output_png}")
    print(f"SVG: {args.output_svg}")
    print(f"Количество архитектур: {len(plot_df)}")


if __name__ == "__main__":
    main()
