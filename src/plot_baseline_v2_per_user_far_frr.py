"""Построение per-user FAR/FRR-графика для baseline v2.

Скрипт читает файл:

    reports/per_user_auth_diagnostics_v2_batchnorm.csv

и формирует горизонтальный bar chart для 51 пользователя:

    слева  — FAR, доля ложных допусков;
    справа — FRR, доля ложных отказов.

Пользователи s007, s037, s002, s032 и s054 выделяются отдельным цветом
и дополнительной подписью.

Выходные файлы:

    reports/figures/baseline_v2_per_user_far_frr.png
    reports/figures/baseline_v2_per_user_far_frr.svg
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_INPUT = PROJECT_ROOT / "reports" / "per_user_auth_diagnostics_v2_batchnorm.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "reports" / "figures"

DEFAULT_OUTPUT_PNG = DEFAULT_OUTPUT_DIR / "baseline_v2_per_user_far_frr.png"
DEFAULT_OUTPUT_SVG = DEFAULT_OUTPUT_DIR / "baseline_v2_per_user_far_frr.svg"

REQUIRED_COLUMNS = ["user_id", "far", "frr"]
DEFAULT_HIGHLIGHT_USERS = ["s007", "s037", "s002", "s032", "s054"]

BASE_COLOR = "#9AA6B2"
HIGHLIGHT_COLOR = "#C0392B"
GRID_COLOR = "#D0D7DE"


def read_per_user_diagnostics(input_path: Path) -> pd.DataFrame:
    """Прочитать CSV-отчёт с per-user FAR/FRR diagnostics.

    Args:
        input_path: Путь к `per_user_auth_diagnostics_v2_batchnorm.csv`.

    Returns:
        Таблица per-user diagnostics.

    Raises:
        FileNotFoundError: Если входной CSV-файл отсутствует.
        ValueError: Если отсутствуют обязательные колонки.
    """
    if not input_path.exists():
        raise FileNotFoundError(f"CSV-файл per-user diagnostics не найден: {input_path}")

    df = pd.read_csv(input_path)
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in df.columns]

    if missing_columns:
        raise ValueError(
            "В CSV-файле отсутствуют обязательные колонки: "
            f"{missing_columns}. Требуются колонки: {REQUIRED_COLUMNS}"
        )

    return df


def prepare_plot_data(
    df: pd.DataFrame,
    highlight_users: list[str],
    sort_by: str,
) -> pd.DataFrame:
    """Подготовить данные для построения графика.

    Args:
        df: Исходная таблица per-user diagnostics.
        highlight_users: Список пользователей, которых нужно выделить.
        sort_by: Колонка сортировки: `far`, `frr`, `user_id` или `max_error`.

    Returns:
        Подготовленная таблица.
    """
    plot_df = df.loc[:, REQUIRED_COLUMNS].copy()
    plot_df["far_percent"] = plot_df["far"] * 100.0
    plot_df["frr_percent"] = plot_df["frr"] * 100.0
    plot_df["max_error_percent"] = plot_df[["far_percent", "frr_percent"]].max(axis=1)
    plot_df["is_highlighted"] = plot_df["user_id"].isin(highlight_users)

    if sort_by == "user_id":
        plot_df = plot_df.sort_values("user_id", ascending=True)
    elif sort_by == "frr":
        plot_df = plot_df.sort_values(["frr_percent", "far_percent"], ascending=False)
    elif sort_by == "max_error":
        plot_df = plot_df.sort_values(["max_error_percent", "far_percent"], ascending=False)
    else:
        plot_df = plot_df.sort_values(["far_percent", "frr_percent"], ascending=False)

    return plot_df.reset_index(drop=True)


def build_per_user_far_frr_chart(
    plot_df: pd.DataFrame,
    highlight_users: list[str],
    output_png: Path,
    output_svg: Path,
) -> None:
    """Построить и сохранить per-user FAR/FRR-график.

    Args:
        plot_df: Подготовленная таблица.
        highlight_users: Список выделяемых пользователей.
        output_png: Путь сохранения PNG-файла.
        output_svg: Путь сохранения SVG-файла.
    """
    output_png.parent.mkdir(parents=True, exist_ok=True)
    output_svg.parent.mkdir(parents=True, exist_ok=True)

    plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["axes.unicode_minus"] = False

    figure_height = max(12.0, len(plot_df) * 0.26)
    figure, axes = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(15, figure_height),
        sharey=True,
        constrained_layout=True,
    )

    y_positions = list(range(len(plot_df)))
    user_labels = build_user_labels(plot_df=plot_df, highlight_users=highlight_users)
    bar_colors = [
        HIGHLIGHT_COLOR if is_highlighted else BASE_COLOR
        for is_highlighted in plot_df["is_highlighted"]
    ]

    build_metric_subplot(
        axis=axes[0],
        y_positions=y_positions,
        user_labels=user_labels,
        values=plot_df["far_percent"],
        bar_colors=bar_colors,
        title="Per-user FAR",
        xlabel="FAR, ложный допуск, %",
    )
    build_metric_subplot(
        axis=axes[1],
        y_positions=y_positions,
        user_labels=user_labels,
        values=plot_df["frr_percent"],
        bar_colors=bar_colors,
        title="Per-user FRR",
        xlabel="FRR, ложный отказ, %",
    )

    figure.suptitle(
        "Per-user FAR и FRR для baseline v2",
        fontsize=15,
    )

    add_legend(figure)

    figure.savefig(output_png, dpi=300, bbox_inches="tight")
    figure.savefig(output_svg, bbox_inches="tight")
    plt.close(figure)


def build_user_labels(plot_df: pd.DataFrame, highlight_users: list[str]) -> list[str]:
    """Сформировать подписи пользователей.

    Args:
        plot_df: Подготовленная таблица.
        highlight_users: Список выделяемых пользователей.

    Returns:
        Подписи для оси Y.
    """
    highlight_set = set(highlight_users)

    return [
        f"{user_id}  ★" if user_id in highlight_set else str(user_id)
        for user_id in plot_df["user_id"]
    ]


def build_metric_subplot(
    axis: plt.Axes,
    y_positions: list[int],
    user_labels: list[str],
    values: pd.Series,
    bar_colors: list[str],
    title: str,
    xlabel: str,
) -> None:
    """Построить один horizontal bar chart для метрики.

    Args:
        axis: Область построения matplotlib.
        y_positions: Позиции пользователей по оси Y.
        user_labels: Подписи пользователей.
        values: Значения метрики в процентах.
        bar_colors: Цвета столбцов.
        title: Заголовок subplot.
        xlabel: Подпись оси X.
    """
    axis.barh(y_positions, values, color=bar_colors)
    axis.set_title(title)
    axis.set_xlabel(xlabel)
    axis.set_yticks(y_positions)
    axis.set_yticklabels(user_labels)
    axis.invert_yaxis()
    axis.grid(axis="x", linestyle="--", linewidth=0.5, color=GRID_COLOR, alpha=0.9)

    x_max = max(float(values.max()) * 1.18, 0.1)
    axis.set_xlim(0, x_max)

    for y_position, value in zip(y_positions, values, strict=True):
        if value <= 0:
            continue

        axis.text(
            value + x_max * 0.01,
            y_position,
            f"{value:.2f}%",
            va="center",
            fontsize=8,
        )


def add_legend(figure: plt.Figure) -> None:
    """Добавить легенду к графику.

    Args:
        figure: Объект matplotlib Figure.
    """
    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, color=BASE_COLOR, label="Остальные пользователи"),
        plt.Rectangle(
            (0, 0),
            1,
            1,
            color=HIGHLIGHT_COLOR,
            label="Выделенные пользователи",
        ),
    ]

    figure.legend(
        handles=legend_handles,
        loc="lower center",
        ncols=2,
        frameon=False,
        fontsize=10,
    )


def parse_highlight_users(value: str) -> list[str]:
    """Разобрать список пользователей для выделения.

    Args:
        value: Строка вида `s007,s037,s002`.

    Returns:
        Список user_id.
    """
    return [user_id.strip() for user_id in value.split(",") if user_id.strip()]


def build_arg_parser() -> argparse.ArgumentParser:
    """Создать CLI-парсер.

    Returns:
        Настроенный parser.
    """
    parser = argparse.ArgumentParser(
        description=("Построить горизонтальный bar chart per-user FAR и FRR для baseline v2.")
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Путь к reports/per_user_auth_diagnostics_v2_batchnorm.csv.",
    )
    parser.add_argument(
        "--highlight-users",
        type=parse_highlight_users,
        default=DEFAULT_HIGHLIGHT_USERS,
        help="Список user_id через запятую: s007,s037,s002,s032,s054.",
    )
    parser.add_argument(
        "--sort-by",
        choices=["far", "frr", "max_error", "user_id"],
        default="far",
        help="Порядок пользователей на графике.",
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

    df = read_per_user_diagnostics(args.input)
    plot_df = prepare_plot_data(
        df=df,
        highlight_users=args.highlight_users,
        sort_by=args.sort_by,
    )

    build_per_user_far_frr_chart(
        plot_df=plot_df,
        highlight_users=args.highlight_users,
        output_png=args.output_png,
        output_svg=args.output_svg,
    )

    highlighted_existing = sorted(set(args.highlight_users).intersection(set(plot_df["user_id"])))

    print("Per-user FAR/FRR chart for baseline v2 saved:")
    print(f"Input CSV: {args.input}")
    print(f"Users: {len(plot_df)}")
    print(f"Sort by: {args.sort_by}")
    print(f"Highlighted users: {', '.join(highlighted_existing)}")
    print(f"PNG path: {args.output_png}")
    print(f"SVG path: {args.output_svg}")


if __name__ == "__main__":
    main()
