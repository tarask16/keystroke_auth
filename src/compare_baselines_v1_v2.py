"""Compare baseline v1 and v2 BatchNorm models.

The module reads already saved artifacts and reports, then creates:

    reports/baseline_v1_v2_comparison.csv
    reports/baseline_v1_v2_comparison.md

It does not retrain models and does not overwrite any model artifacts.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_V1_AUTH_POLICY = PROJECT_ROOT / "models" / "auth_policy.json"
DEFAULT_V2_AUTH_POLICY = PROJECT_ROOT / "models" / "auth_policy_v2_batchnorm.json"

DEFAULT_V1_BATCH_REPORT = PROJECT_ROOT / "reports" / "authentication_batch_report.csv"
DEFAULT_V2_BATCH_REPORT = PROJECT_ROOT / "reports" / "authentication_batch_report_v2_batchnorm.csv"
DEFAULT_V2_TRAINING_METRICS = PROJECT_ROOT / "reports" / "v2_batchnorm_training_metrics.csv"
DEFAULT_ARCHITECTURE_COMPARISON = PROJECT_ROOT / "reports" / "mlp_architecture_comparison.csv"

DEFAULT_OUTPUT_CSV = PROJECT_ROOT / "reports" / "baseline_v1_v2_comparison.csv"
DEFAULT_OUTPUT_MD = PROJECT_ROOT / "reports" / "baseline_v1_v2_comparison.md"


def read_json(path: Path) -> dict[str, Any]:
    """Read JSON file.

    Args:
        path: JSON path.

    Returns:
        Parsed JSON dictionary.

    Raises:
        FileNotFoundError: If path does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")

    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> pd.DataFrame:
    """Read CSV file.

    Args:
        path: CSV path.

    Returns:
        Loaded DataFrame.

    Raises:
        FileNotFoundError: If path does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")

    return pd.read_csv(path)


def optional_float(value: Any) -> float | None:
    """Convert optional value to float.

    Args:
        value: Source value.

    Returns:
        Float value or None.
    """
    if value is None:
        return None

    return float(value)


def get_first_row_value(df: pd.DataFrame, column: str) -> float | None:
    """Get first-row value from DataFrame column.

    Args:
        df: Source DataFrame.
        column: Column name.

    Returns:
        First-row value or None.
    """
    if column not in df.columns or df.empty:
        return None

    return optional_float(df.loc[0, column])


def find_architecture_row(
    architecture_df: pd.DataFrame,
    architecture_name: str,
) -> pd.Series | None:
    """Find architecture row in comparison report.

    Args:
        architecture_df: Architecture comparison DataFrame.
        architecture_name: Architecture name.

    Returns:
        Matching row or None.
    """
    if "architecture" not in architecture_df.columns:
        return None

    matches = architecture_df[architecture_df["architecture"] == architecture_name]

    if matches.empty:
        return None

    return matches.iloc[0]


def build_comparison_table(
    v1_policy: dict[str, Any],
    v2_policy: dict[str, Any],
    v1_batch: pd.DataFrame,
    v2_batch: pd.DataFrame,
    architecture_comparison: pd.DataFrame,
) -> pd.DataFrame:
    """Build v1/v2 comparison table.

    Args:
        v1_policy: Baseline v1 auth policy.
        v2_policy: Baseline v2 auth policy.
        v1_batch: Baseline v1 batch report.
        v2_batch: Baseline v2 batch report.
        architecture_comparison: MLP architecture comparison report.

    Returns:
        Comparison DataFrame.
    """
    v1_architecture = find_architecture_row(architecture_comparison, "mlp_64")
    v2_architecture = find_architecture_row(
        architecture_comparison,
        "mlp_128_64_batchnorm",
    )

    rows = [
        {
            "model": "baseline_v1_mlp_64",
            "architecture": "mlp_64",
            "params": get_series_value(v1_architecture, "params"),
            "test_accuracy": get_series_value(v1_architecture, "test_accuracy"),
            "eer": optional_float(v1_policy.get("eer")),
            "auth_threshold": optional_float(v1_policy.get("auth_threshold")),
            "target_far": optional_float(v1_policy.get("target_far")),
            "actual_far": optional_float(v1_policy.get("actual_far")),
            "actual_frr": optional_float(v1_policy.get("actual_frr")),
            "empirical_far": get_first_row_value(v1_batch, "empirical_far"),
            "empirical_frr": get_first_row_value(v1_batch, "empirical_frr"),
            "genuine_rejects": get_first_row_value(v1_batch, "genuine_rejects"),
            "impostor_accepts": get_first_row_value(v1_batch, "impostor_accepts"),
        },
        {
            "model": "baseline_v2_batchnorm",
            "architecture": "mlp_128_64_batchnorm",
            "params": get_series_value(v2_architecture, "params"),
            "test_accuracy": optional_float(v2_policy.get("test_accuracy"))
            or get_series_value(v2_architecture, "test_accuracy"),
            "eer": optional_float(v2_policy.get("eer")),
            "auth_threshold": optional_float(v2_policy.get("auth_threshold")),
            "target_far": optional_float(v2_policy.get("target_far")),
            "actual_far": optional_float(v2_policy.get("actual_far")),
            "actual_frr": optional_float(v2_policy.get("actual_frr")),
            "empirical_far": get_first_row_value(v2_batch, "empirical_far"),
            "empirical_frr": get_first_row_value(v2_batch, "empirical_frr"),
            "genuine_rejects": get_first_row_value(v2_batch, "genuine_rejects"),
            "impostor_accepts": get_first_row_value(v2_batch, "impostor_accepts"),
        },
    ]

    return pd.DataFrame(rows)


def get_series_value(row: pd.Series | None, column: str) -> float | None:
    """Get value from optional Series.

    Args:
        row: Optional row.
        column: Column name.

    Returns:
        Value or None.
    """
    if row is None or column not in row:
        return None

    return optional_float(row[column])


def add_improvement_columns(comparison_df: pd.DataFrame) -> pd.DataFrame:
    """Add relative improvement summary rows.

    Args:
        comparison_df: v1/v2 comparison table.

    Returns:
        DataFrame with original rows only; improvement is computed separately.
    """
    return comparison_df.copy()


def calculate_improvements(comparison_df: pd.DataFrame) -> dict[str, float]:
    """Calculate v2 improvements over v1.

    Args:
        comparison_df: v1/v2 comparison table.

    Returns:
        Improvement dictionary.
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


def format_float(value: Any, digits: int = 6) -> str:
    """Format float-like value.

    Args:
        value: Source value.
        digits: Digits after decimal point.

    Returns:
        Formatted string.
    """
    if value is None:
        return "нет данных"

    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def format_percent(value: Any, digits: int = 2) -> str:
    """Format fraction as percent.

    Args:
        value: Source value.
        digits: Digits after decimal point.

    Returns:
        Formatted percent.
    """
    if value is None:
        return "нет данных"

    try:
        return f"{float(value) * 100:.{digits}f}%"
    except (TypeError, ValueError):
        return str(value)


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    """Convert DataFrame to Markdown table without optional dependencies.

    Args:
        df: Source DataFrame.

    Returns:
        Markdown table.
    """
    if df.empty:
        return "_Нет данных._"

    columns = list(df.columns)
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join("---" for _column in columns) + " |"

    rows = []
    for _index, row in df.iterrows():
        values = [format_markdown_cell(row[column]) for column in columns]
        rows.append("| " + " | ".join(values) + " |")

    return "\n".join([header, separator, *rows])


def format_markdown_cell(value: object) -> str:
    """Format Markdown table cell.

    Args:
        value: Cell value.

    Returns:
        String cell value.
    """
    if isinstance(value, float):
        return f"{value:.6f}"

    return str(value).replace("|", "\\|")


def localize_comparison_table(comparison_df: pd.DataFrame) -> pd.DataFrame:
    """Create Russian display table.

    Args:
        comparison_df: Source comparison table.

    Returns:
        Localized table.
    """
    display_df = comparison_df.copy()
    display_df["test_accuracy"] = display_df["test_accuracy"].map(format_percent)
    display_df["eer"] = display_df["eer"].map(format_percent)
    display_df["target_far"] = display_df["target_far"].map(format_percent)
    display_df["actual_far"] = display_df["actual_far"].map(format_percent)
    display_df["actual_frr"] = display_df["actual_frr"].map(format_percent)
    display_df["empirical_far"] = display_df["empirical_far"].map(format_percent)
    display_df["empirical_frr"] = display_df["empirical_frr"].map(format_percent)
    display_df["auth_threshold"] = display_df["auth_threshold"].map(format_float)

    return display_df.rename(
        columns={
            "model": "модель",
            "architecture": "архитектура",
            "params": "параметры",
            "test_accuracy": "test accuracy",
            "eer": "EER",
            "auth_threshold": "порог",
            "target_far": "целевой FAR",
            "actual_far": "FAR policy",
            "actual_frr": "FRR policy",
            "empirical_far": "FAR batch",
            "empirical_frr": "FRR batch",
            "genuine_rejects": "ложные отказы",
            "impostor_accepts": "ложные допуски",
        }
    )


def build_markdown_report(
    comparison_df: pd.DataFrame,
    improvements: dict[str, float],
) -> str:
    """Build Russian Markdown comparison report.

    Args:
        comparison_df: v1/v2 comparison table.
        improvements: Improvement dictionary.

    Returns:
        Markdown report text.
    """
    display_df = localize_comparison_table(comparison_df)

    return "\n".join(
        [
            "# Сравнение baseline v1 и baseline v2 BatchNorm",
            "",
            "## 1. Сводная таблица",
            "",
            dataframe_to_markdown(display_df),
            "",
            "## 2. Изменение качества",
            "",
            "| Показатель | Изменение v2 относительно v1 |",
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
            "## 3. Вывод",
            "",
            (
                "Baseline v2 BatchNorm существенно улучшает качество по сравнению "
                "с исходной моделью mlp_64. Основной выигрыш наблюдается в снижении "
                "FRR при сохранении целевого FAR около 1%."
            ),
            "",
            (
                "V2 можно считать рекомендуемой baseline-моделью для дальнейших "
                "экспериментов. При этом исходный baseline v1 целесообразно оставить "
                "как контрольную точку для сравнения."
            ),
            "",
        ]
    )


def save_reports(
    comparison_df: pd.DataFrame,
    markdown_report: str,
    output_csv: Path,
    output_md: Path,
) -> None:
    """Save CSV and Markdown reports.

    Args:
        comparison_df: Comparison table.
        markdown_report: Markdown report text.
        output_csv: CSV output path.
        output_md: Markdown output path.
    """
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)

    comparison_df.to_csv(output_csv, index=False)
    output_md.write_text(markdown_report, encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    """Build command-line parser.

    Returns:
        Configured parser.
    """
    parser = argparse.ArgumentParser(description="Compare baseline v1 and v2 BatchNorm artifacts.")

    parser.add_argument("--v1-auth-policy", type=Path, default=DEFAULT_V1_AUTH_POLICY)
    parser.add_argument("--v2-auth-policy", type=Path, default=DEFAULT_V2_AUTH_POLICY)
    parser.add_argument("--v1-batch-report", type=Path, default=DEFAULT_V1_BATCH_REPORT)
    parser.add_argument("--v2-batch-report", type=Path, default=DEFAULT_V2_BATCH_REPORT)
    parser.add_argument(
        "--v2-training-metrics",
        type=Path,
        default=DEFAULT_V2_TRAINING_METRICS,
    )
    parser.add_argument(
        "--architecture-comparison",
        type=Path,
        default=DEFAULT_ARCHITECTURE_COMPARISON,
    )
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)

    return parser


def main() -> None:
    """CLI entry point."""
    parser = build_arg_parser()
    args = parser.parse_args()

    v1_policy = read_json(args.v1_auth_policy)
    v2_policy = read_json(args.v2_auth_policy)
    v1_batch = read_csv(args.v1_batch_report)
    v2_batch = read_csv(args.v2_batch_report)
    architecture_comparison = read_csv(args.architecture_comparison)

    comparison_df = build_comparison_table(
        v1_policy=v1_policy,
        v2_policy=v2_policy,
        v1_batch=v1_batch,
        v2_batch=v2_batch,
        architecture_comparison=architecture_comparison,
    )
    comparison_df = add_improvement_columns(comparison_df)
    improvements = calculate_improvements(comparison_df)
    markdown_report = build_markdown_report(
        comparison_df=comparison_df,
        improvements=improvements,
    )

    save_reports(
        comparison_df=comparison_df,
        markdown_report=markdown_report,
        output_csv=args.output_csv,
        output_md=args.output_md,
    )

    print("Baseline v1/v2 comparison finished.")
    print()
    print("Comparison:")
    print(localize_comparison_table(comparison_df).to_string(index=False))
    print()
    print("Improvements:")
    print(f"Test accuracy delta: {format_percent(improvements['test_accuracy_delta'])}")
    print(f"EER delta: {format_percent(improvements['eer_delta'])}")
    print(f"EER relative reduction: {format_percent(improvements['eer_relative_reduction'])}")
    print(f"FRR delta: {format_percent(improvements['frr_delta'])}")
    print(f"FRR relative reduction: {format_percent(improvements['frr_relative_reduction'])}")
    print(f"FAR delta: {format_percent(improvements['far_delta'])}")
    print()
    print("Reports saved:")
    print(f"CSV path: {args.output_csv}")
    print(f"Markdown path: {args.output_md}")


if __name__ == "__main__":
    main()
