"""Скрипт генерации общей схемы сравниваемых подходов.

Схема иллюстрирует три сравниваемых подхода для аутентификации
по клавиатурному почерку:
1. MLP softmax score.
2. Encoder -> embedding -> distance to user template.
3. Metric-learning encoder -> distance -> threshold.

На схеме отдельно показаны:
- этап калибровки на validation split;
- этап финальной оценки на test split.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle


FIG_WIDTH = 18
FIG_HEIGHT = 11
DEFAULT_OUTPUT = Path("reports/figures/compared_approaches_overview.png")
DEFAULT_SVG_OUTPUT = Path("reports/figures/compared_approaches_overview.svg")


class DiagramDrawer:
    """Вспомогательный класс для отрисовки блок-схемы."""

    def __init__(self, ax: plt.Axes) -> None:
        """Сохранить ссылку на matplotlib axes."""
        self.ax = ax

    def draw_box(
        self,
        x: float,
        y: float,
        w: float,
        h: float,
        text: str,
        *,
        fontsize: int = 10,
        linewidth: float = 1.5,
        facecolor: str = "#f8f9fa",
        edgecolor: str = "#495057",
        boxstyle: str = "round,pad=0.02,rounding_size=0.08",
        fontweight: str = "normal",
        zorder: int = 3,
    ) -> None:
        """Нарисовать прямоугольный блок с подписью."""
        patch = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle=boxstyle,
            linewidth=linewidth,
            facecolor=facecolor,
            edgecolor=edgecolor,
            zorder=zorder,
        )
        self.ax.add_patch(patch)
        self.ax.text(
            x + w / 2,
            y + h / 2,
            text,
            ha="center",
            va="center",
            fontsize=fontsize,
            fontweight=fontweight,
            wrap=True,
            zorder=zorder + 1,
        )

    def draw_arrow(
        self,
        start: tuple[float, float],
        end: tuple[float, float],
        *,
        linewidth: float = 1.6,
        mutation_scale: float = 14,
        color: str = "#343a40",
        style: str = "-|>",
        connectionstyle: str = "arc3,rad=0.0",
        zorder: int = 2,
    ) -> None:
        """Нарисовать стрелку между двумя точками."""
        arrow = FancyArrowPatch(
            start,
            end,
            arrowstyle=style,
            mutation_scale=mutation_scale,
            linewidth=linewidth,
            color=color,
            connectionstyle=connectionstyle,
            zorder=zorder,
        )
        self.ax.add_patch(arrow)

    def draw_group_band(
        self,
        x: float,
        y: float,
        w: float,
        h: float,
        text: str,
        *,
        facecolor: str,
        edgecolor: str = "#adb5bd",
        fontsize: int = 12,
    ) -> None:
        """Нарисовать полосу-группу для логического этапа."""
        patch = Rectangle(
            (x, y),
            w,
            h,
            linewidth=1.2,
            edgecolor=edgecolor,
            facecolor=facecolor,
            zorder=0,
        )
        self.ax.add_patch(patch)
        self.ax.text(
            x + 0.5,
            y + h - 0.45,
            text,
            ha="left",
            va="top",
            fontsize=fontsize,
            fontweight="bold",
            zorder=1,
        )

    def draw_note(
        self,
        x: float,
        y: float,
        text: str,
        *,
        fontsize: int = 9,
        color: str = "#495057",
    ) -> None:
        """Нарисовать текстовую заметку без рамки."""
        self.ax.text(
            x,
            y,
            text,
            ha="left",
            va="center",
            fontsize=fontsize,
            color=color,
            zorder=4,
        )


def build_diagram(output_path: Path, svg_output_path: Path | None = None) -> None:
    """Построить и сохранить итоговую блок-схему."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if svg_output_path is not None:
        svg_output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 13)
    ax.axis("off")

    drawer = DiagramDrawer(ax)

    # Заголовок.
    ax.text(
        9,
        12.5,
        "Общая схема сравниваемых подходов",
        ha="center",
        va="center",
        fontsize=18,
        fontweight="bold",
    )
    ax.text(
        9,
        12.05,
        ("Сравнение baseline softmax, embedding-based verification и metric-learning verification"),
        ha="center",
        va="center",
        fontsize=11,
    )

    # Группы этапов.
    drawer.draw_group_band(
        0.4,
        7.1,
        17.0,
        4.15,
        "Основные конвейеры сравниваемых подходов",
        facecolor="#f1f3f5",
    )
    drawer.draw_group_band(
        0.4,
        3.35,
        17.0,
        2.8,
        "Calibration на validation split",
        facecolor="#fff3bf",
    )
    drawer.draw_group_band(
        0.4,
        0.45,
        17.0,
        2.35,
        "Финальная оценка на test split",
        facecolor="#d3f9d8",
    )

    # Общий вход.
    common_y = 10.0
    box_h = 0.8
    drawer.draw_box(
        0.8,
        common_y,
        2.3,
        box_h,
        "Входной вектор\nвременных признаков",
        fontsize=11,
        facecolor="#e7f5ff",
        edgecolor="#1c7ed6",
        fontweight="bold",
    )
    drawer.draw_box(
        3.6,
        common_y,
        1.7,
        box_h,
        "Scaler",
        fontsize=11,
        facecolor="#e7f5ff",
        edgecolor="#1c7ed6",
        fontweight="bold",
    )
    drawer.draw_arrow((3.1, common_y + box_h / 2), (3.6, common_y + box_h / 2))

    # Разветвление.
    branch_x = 5.8
    branch_center_y = common_y + box_h / 2
    drawer.draw_arrow((5.3, branch_center_y), (6.2, branch_center_y))
    drawer.draw_arrow((6.2, branch_center_y), (6.2, 9.1), connectionstyle="arc3,rad=0.0")
    drawer.draw_arrow((6.2, branch_center_y), (6.2, 7.9), connectionstyle="arc3,rad=0.0")
    drawer.draw_arrow((6.2, branch_center_y), (6.2, 6.7), connectionstyle="arc3,rad=0.0")

    # Координаты строк подходов.
    row_y = {
        "a": 8.7,
        "b": 7.5,
        "c": 6.3,
    }

    # Метки подходов.
    drawer.draw_note(0.85, row_y["a"] + 0.4, "(а) Baseline softmax", fontsize=11)
    drawer.draw_note(0.85, row_y["b"] + 0.4, "(б) Embedding-based verification", fontsize=11)
    drawer.draw_note(0.85, row_y["c"] + 0.4, "(в) Metric-learning verification", fontsize=11)

    # Подход (а): MLP softmax.
    drawer.draw_box(6.6, row_y["a"], 2.2, 0.8, "MLP", fontsize=11)
    drawer.draw_box(
        9.2, row_y["a"], 2.5, 0.8, "Softmax score\nзаявленного пользователя", fontsize=10
    )
    drawer.draw_box(12.1, row_y["a"], 1.7, 0.8, "Threshold", fontsize=11)
    drawer.draw_arrow((6.2, 9.1), (6.6, 9.1))
    drawer.draw_arrow((8.8, 9.1), (9.2, 9.1))
    drawer.draw_arrow((11.7, 9.1), (12.1, 9.1))

    # Подход (б): encoder -> embedding -> distance to user template.
    drawer.draw_box(6.6, row_y["b"], 2.2, 0.8, "Encoder", fontsize=11)
    drawer.draw_box(9.2, row_y["b"], 1.9, 0.8, "Embedding", fontsize=11)
    drawer.draw_box(11.5, row_y["b"], 2.8, 0.8, "Distance до\nuser template", fontsize=10)
    drawer.draw_box(14.7, row_y["b"], 1.7, 0.8, "Threshold", fontsize=11)
    drawer.draw_arrow((6.2, 7.9), (6.6, 7.9))
    drawer.draw_arrow((8.8, 7.9), (9.2, 7.9))
    drawer.draw_arrow((11.1, 7.9), (11.5, 7.9))
    drawer.draw_arrow((14.3, 7.9), (14.7, 7.9))
    drawer.draw_note(
        11.55, 7.15, "user template формируется по train / enrollment samples", fontsize=8
    )

    # Подход (в): metric-learning encoder -> distance -> threshold.
    drawer.draw_box(6.6, row_y["c"], 2.8, 0.8, "Metric-learning\nencoder", fontsize=10)
    drawer.draw_box(9.8, row_y["c"], 2.7, 0.8, "Distance до\nclaimed-user template", fontsize=10)
    drawer.draw_box(12.9, row_y["c"], 1.7, 0.8, "Threshold", fontsize=11)
    drawer.draw_arrow((6.2, 6.7), (6.6, 6.7))
    drawer.draw_arrow((9.4, 6.7), (9.8, 6.7))
    drawer.draw_arrow((12.5, 6.7), (12.9, 6.7))

    # Блоки калибровки на validation split.
    calib_y = 4.3
    drawer.draw_box(
        6.5,
        calib_y,
        3.2,
        1.0,
        "Validation split:\nкалибровка threshold\nпо softmax score",
        fontsize=10,
        facecolor="#fff9db",
        edgecolor="#f08c00",
    )
    drawer.draw_box(
        10.15,
        calib_y,
        3.2,
        1.0,
        "Validation split:\nкалибровка distance\nthreshold(s)",
        fontsize=10,
        facecolor="#fff9db",
        edgecolor="#f08c00",
    )
    drawer.draw_box(
        13.8,
        calib_y,
        3.0,
        1.0,
        "Политики threshold:\n global / per-user / guarded",
        fontsize=9,
        facecolor="#fff9db",
        edgecolor="#f08c00",
    )

    # Связи из основных конвейеров к калибровке.
    drawer.draw_arrow((12.95, 8.7), (8.1, 5.3), connectionstyle="arc3,rad=0.0")
    drawer.draw_arrow((15.55, 7.5), (11.75, 5.3), connectionstyle="arc3,rad=0.0")
    drawer.draw_arrow((13.75, 6.3), (11.75, 5.3), connectionstyle="arc3,rad=0.0")
    drawer.draw_arrow((13.35, 4.8), (13.8, 4.8))

    # Блоки финальной оценки на test split.
    test_y = 1.15
    drawer.draw_box(
        6.5,
        test_y,
        3.7,
        1.0,
        "Test split:\nприменение откалиброванных\nthreshold(s)",
        fontsize=10,
        facecolor="#ebfbee",
        edgecolor="#2b8a3e",
    )
    drawer.draw_box(
        10.7,
        test_y,
        2.7,
        1.0,
        "Решение:\nACCEPT / REJECT",
        fontsize=11,
        facecolor="#ebfbee",
        edgecolor="#2b8a3e",
    )
    drawer.draw_box(
        13.9,
        test_y,
        2.8,
        1.0,
        "Метрики:\nFAR / FRR / EER",
        fontsize=11,
        facecolor="#ebfbee",
        edgecolor="#2b8a3e",
    )

    drawer.draw_arrow((8.1, 4.3), (8.1, 2.15))
    drawer.draw_arrow((11.75, 4.3), (8.85, 2.15), connectionstyle="arc3,rad=0.0")
    drawer.draw_arrow((15.3, 4.3), (8.85, 2.15), connectionstyle="arc3,rad=0.0")
    drawer.draw_arrow((10.2, 1.65), (10.7, 1.65))
    drawer.draw_arrow((13.4, 1.65), (13.9, 1.65))

    # Легенда / пояснения.
    drawer.draw_note(
        0.9,
        2.45,
        (
            "Пояснение: calibration выполняется только на validation split; "
            "финальная report-оценка проводится только на test split."
        ),
        fontsize=9,
    )
    drawer.draw_note(
        0.9,
        1.95,
        (
            "Вариант (б) соответствует Stage 6, вариант (в) соответствует "
            "Stage 7 (Siamese / Triplet и их модификациям)."
        ),
        fontsize=9,
    )

    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    if svg_output_path is not None:
        fig.savefig(svg_output_path, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    """Разобрать аргументы командной строки."""
    parser = argparse.ArgumentParser(description="Построить общую схему сравниваемых подходов.")
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Путь к PNG-файлу схемы.",
    )
    parser.add_argument(
        "--svg-output",
        type=Path,
        default=DEFAULT_SVG_OUTPUT,
        help="Путь к SVG-файлу схемы.",
    )
    return parser.parse_args()


def main() -> None:
    """Точка входа в CLI."""
    args = parse_args()
    build_diagram(args.output, args.svg_output)
    print("Схема успешно построена.")
    print(f"PNG: {args.output.resolve()}")
    print(f"SVG: {args.svg_output.resolve()}")


if __name__ == "__main__":
    main()
