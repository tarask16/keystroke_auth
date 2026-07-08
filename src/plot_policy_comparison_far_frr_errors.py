"""Построение графиков сравнения global, naive и guarded policies.

Скрипт читает summary-файл:

    reports/guarded_per_user_thresholds_v2_batchnorm_summary.csv

и строит два отдельных bar chart:

    1. Сравнение FAR и FRR.
    2. Сравнение числа ошибок: false accepts и false rejects.

Выходные файлы:

    reports/figures/policy_far_frr_comparison.png
    reports/figures/policy_far_frr_comparison.svg
    reports/figures/policy_errors_comparison.png
    reports/figures/policy_errors_comparison.svg
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_INPUT = PROJECT_ROOT / "reports" / "guarded_per_user_thresholds_v2_batchnorm_summary.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "reports" / "figures"

DEFAULT_FAR_FRR_OUTPUT_PNG = DEFAULT_OUTPUT_DIR / "policy_far_frr_comparison.png"
DEFAULT_FAR_FRR_OUTPUT_SVG = DEFAULT_OUTPUT_DIR / "policy_far_frr_comparison.svg"
DEFAULT_ERRORS_OUTPUT_PNG = DEFAULT_OUTPUT_DIR / "policy_errors_comparison.png"
DEFAULT_ERRORS_OUTPUT_SVG = DEFAULT_OUTPUT_DIR / "policy_errors_comparison.svg"

REQUIRED_COLUMNS = ["policy", "far", "frr", "false_accepts", "false_rejects"]
POLICY_ORDER = ["global", "naive", "guarded"]

POLICY_LABELS_RU = {
    "global": "Global policy",
    "naive": "Naive per-user policy",
    "guarded": "Guarded per-user policy",
}


def read_policy_summary(input_path: Path) -> pd.DataFrame:
    """Прочитать summary-таблицу сравнения политик.

    Args:
        input_path: Путь к CSV-файлу summary.

    Returns:
        Таблица сравнения политик.

    Raises:
        FileNotFoundError: Если входной CSV-файл отсутствует.
        ValueError: Если отсутствуют обязательные колонки.
    """
    if not input_path.exists():
        raise FileNotFoundError(f"CSV-файл summary не найден: {input_path}")

    df = pd.read_csv(input_path)
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in df.columns]

    if missing_columns:
        raise ValueError(
            "В CSV-файле отсутствуют обязательные колонки: "
            f"{missing_columns}. Требуются колонки: {REQUIRED_COLUMNS}"
        )

    return df


def prepare_policy_data(df: pd.DataFrame) -> pd.DataFrame:
    """Подготовить данные для построения графиков.

    Args:
        df: Исходная таблица сравнения политик.

    Returns:
        Подготовленная таблица.
    """
    plot_df = df.loc[:, REQUIRED_COLUMNS].copy()

    plot_df["policy_order"] = plot_df["policy"].map(
        {policy: index for index, policy in enumerate(POLICY_ORDER)}
    )
    plot_df = plot_df.sort_values("policy_order").reset_index(drop=True)

    plot_df["policy_label"] = plot_df["policy"].map(POLICY_LABELS_RU)
    plot_df["far_percent"] = plot_df["far"] * 100.0
    plot_df["frr_percent"] = plot_df["frr"] * 100.0

    return plot_df


def build_far_frr_chart(
    plot_df: pd.DataFrame,
    output_png: Path,
    output_svg: Path,
) -> None:
    """Построить bar chart сравнения FAR и FRR.

    Args:
        plot_df: Подготовленная таблица.
        output_png: Путь сохранения PNG-файла.
        output_svg: Путь сохранения SVG-файла.
    """
    output_png.parent.mkdir(parents=True, exist_ok=True)
    output_svg.parent.mkdir(parents=True, exist_ok=True)

    plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["axes.unicode_minus"] = False

    figure, axis = plt.subplots(figsize=(9, 6), constrained_layout=True)

    x_positions = np.arange(len(plot_df))
    bar_width = 0.36

    far_values = plot_df["far_percent"]
    frr_values = plot_df["frr_percent"]

    axis.bar(
        x_positions - bar_width / 2,
        far_values,
        width=bar_width,
        label="FAR, ложный допуск",
    )
    axis.bar(
        x_positions + bar_width / 2,
        frr_values,
        width=bar_width,
        label="FRR, ложный отказ",
    )

    axis.set_title("Сравнение global, naive и guarded policies по FAR и FRR")
    axis.set_xlabel("Политика аутентификации")
    axis.set_ylabel("Доля ошибок, %")
    axis.set_xticks(x_positions)
    axis.set_xticklabels(plot_df["policy_label"])
    axis.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.7)
    axis.legend(loc="best")

    y_max = max(float(far_values.max()), float(frr_values.max()))
    axis.set_ylim(0, y_max * 1.25)

    add_bar_labels(axis, x_positions - bar_width / 2, far_values)
    add_bar_labels(axis, x_positions + bar_width / 2, frr_values)

    figure.savefig(output_png, dpi=300, bbox_inches="tight")
    figure.savefig(output_svg, bbox_inches="tight")
    plt.close(figure)


def build_errors_chart(
    plot_df: pd.DataFrame,
    output_png: Path,
    output_svg: Path,
) -> None:
    """Построить bar chart сравнения числа ошибок.

    Args:
        plot_df: Подготовленная таблица.
        output_png: Путь сохранения PNG-файла.
        output_svg: Путь сохранения SVG-файла.
    """
    output_png.parent.mkdir(parents=True, exist_ok=True)
    output_svg.parent.mkdir(parents=True, exist_ok=True)

    plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["axes.unicode_minus"] = False

    figure, axis = plt.subplots(figsize=(9, 6), constrained_layout=True)

    x_positions = np.arange(len(plot_df))
    bar_width = 0.36

    false_accepts = plot_df["false_accepts"].astype(int)
    false_rejects = plot_df["false_rejects"].astype(int)

    axis.bar(
        x_positions - bar_width / 2,
        false_accepts,
        width=bar_width,
        label="False accepts, ложные допуски",
    )
    axis.bar(
        x_positions + bar_width / 2,
        false_rejects,
        width=bar_width,
        label="False rejects, ложные отказы",
    )

    axis.set_title("Сравнение global, naive и guarded policies по числу ошибок")
    axis.set_xlabel("Политика аутентификации")
    axis.set_ylabel("Количество ошибок")
    axis.set_xticks(x_positions)
    axis.set_xticklabels(plot_df["policy_label"])
    axis.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.7)
    axis.legend(loc="best")

    y_max = max(int(false_accepts.max()), int(false_rejects.max()))
    axis.set_ylim(0, y_max * 1.18)

    add_bar_labels(axis, x_positions - bar_width / 2, false_accepts, integer=True)
    add_bar_labels(axis, x_positions + bar_width / 2, false_rejects, integer=True)

    figure.savefig(output_png, dpi=300, bbox_inches="tight")
    figure.savefig(output_svg, bbox_inches="tight")
    plt.close(figure)


def add_bar_labels(
    axis: plt.Axes,
    x_positions: np.ndarray,
    values: pd.Series,
    *,
    integer: bool = False,
) -> None:
    """Добавить числовые подписи над столбцами.

    Args:
        axis: Область построения matplotlib.
        x_positions: Позиции столбцов по оси X.
        values: Значения столбцов.
        integer: Форматировать значения как целые числа.
    """
    y_max = float(max(values))
    offset = y_max * 0.025 if y_max > 0 else 0.01

    for x_position, value in zip(x_positions, values, strict=True):
        label = f"{int(value)}" if integer else f"{float(value):.2f}%"

        axis.text(
            x_position,
            float(value) + offset,
            label,
            ha="center",
            va="bottom",
            fontsize=9,
        )


def build_arg_parser() -> argparse.ArgumentParser:
    """Создать CLI-парсер.

    Returns:
        Настроенный parser.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Построить графики сравнения global, naive и guarded policies "
            "по FAR/FRR и числу ошибок."
        )
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Путь к guarded_per_user_thresholds_v2_batchnorm_summary.csv.",
    )
    parser.add_argument(
        "--far-frr-output-png",
        type=Path,
        default=DEFAULT_FAR_FRR_OUTPUT_PNG,
        help="Путь сохранения PNG-графика FAR/FRR.",
    )
    parser.add_argument(
        "--far-frr-output-svg",
        type=Path,
        default=DEFAULT_FAR_FRR_OUTPUT_SVG,
        help="Путь сохранения SVG-графика FAR/FRR.",
    )
    parser.add_argument(
        "--errors-output-png",
        type=Path,
        default=DEFAULT_ERRORS_OUTPUT_PNG,
        help="Путь сохранения PNG-графика числа ошибок.",
    )
    parser.add_argument(
        "--errors-output-svg",
        type=Path,
        default=DEFAULT_ERRORS_OUTPUT_SVG,
        help="Путь сохранения SVG-графика числа ошибок.",
    )

    return parser


def main() -> None:
    """Точка входа CLI."""
    parser = build_arg_parser()
    args = parser.parse_args()

    df = read_policy_summary(args.input)
    plot_df = prepare_policy_data(df)

    build_far_frr_chart(
        plot_df=plot_df,
        output_png=args.far_frr_output_png,
        output_svg=args.far_frr_output_svg,
    )
    build_errors_chart(
        plot_df=plot_df,
        output_png=args.errors_output_png,
        output_svg=args.errors_output_svg,
    )

    print("Policy comparison charts saved:")
    print(f"Input CSV: {args.input}")
    print(f"Policies: {', '.join(plot_df['policy'].tolist())}")
    print(f"FAR/FRR PNG: {args.far_frr_output_png}")
    print(f"FAR/FRR SVG: {args.far_frr_output_svg}")
    print(f"Errors PNG: {args.errors_output_png}")
    print(f"Errors SVG: {args.errors_output_svg}")


if __name__ == "__main__":
    main()
