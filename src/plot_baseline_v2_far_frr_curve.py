"""Построение FAR/FRR-кривой для baseline v2.

Скрипт рассчитывает genuine/impostor scores на test split для модели
`mlp_v2_batchnorm` и строит график trade-off между FAR и FRR.

На графике отмечаются:
- рабочий порог baseline v2: 0.058608;
- точка EER, где FAR и FRR приблизительно равны.

Выходные файлы:

    reports/figures/baseline_v2_far_frr_curve.png
    reports/figures/baseline_v2_far_frr_curve.svg
    reports/baseline_v2_far_frr_curve.csv
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve
from sklearn.preprocessing import LabelEncoder, StandardScaler

from src.config import CMU_FEATURES_FILE
from src.preprocessing import (
    TrainValidationTestSplit,
    clean_prepared_dataset,
    create_train_validation_test_split,
    load_processed_dataset,
    prepare_features_and_labels,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_MODEL_INPUT = PROJECT_ROOT / "models" / "mlp_v2_batchnorm.keras"
DEFAULT_SCALER_INPUT = PROJECT_ROOT / "models" / "scaler_v2_batchnorm.pkl"
DEFAULT_LABEL_ENCODER_INPUT = PROJECT_ROOT / "models" / "label_encoder_v2_batchnorm.pkl"
DEFAULT_AUTH_POLICY_INPUT = PROJECT_ROOT / "models" / "auth_policy_v2_batchnorm.json"

DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "reports" / "figures"
DEFAULT_OUTPUT_PNG = DEFAULT_OUTPUT_DIR / "baseline_v2_far_frr_curve.png"
DEFAULT_OUTPUT_SVG = DEFAULT_OUTPUT_DIR / "baseline_v2_far_frr_curve.svg"
DEFAULT_OUTPUT_CSV = PROJECT_ROOT / "reports" / "baseline_v2_far_frr_curve.csv"

DEFAULT_WORKING_THRESHOLD = 0.058608


@dataclass(frozen=True)
class BaselineV2Artifacts:
    """Артефакты baseline v2."""

    model: Any
    scaler: StandardScaler
    label_encoder: LabelEncoder
    auth_policy: dict[str, Any]
    working_threshold: float


@dataclass(frozen=True)
class ScoreSet:
    """Genuine/impostor scores для проверки аутентификации."""

    genuine_scores: np.ndarray
    impostor_scores: np.ndarray


@dataclass(frozen=True)
class OperatingPoint:
    """Рабочая точка на FAR/FRR-кривой."""

    threshold: float
    far: float
    frr: float


def load_unscaled_split(input_path: Path) -> TrainValidationTestSplit:
    """Загрузить, очистить и разделить обработанный CMU dataset.

    Args:
        input_path: Путь к CSV-файлу с обработанными признаками.

    Returns:
        Ненормализованный train/validation/test split.
    """
    df = load_processed_dataset(input_path)
    prepared = prepare_features_and_labels(df)
    cleaned, _cleaning_report = clean_prepared_dataset(prepared)

    return create_train_validation_test_split(cleaned)


def load_baseline_v2_artifacts(
    model_path: Path,
    scaler_path: Path,
    label_encoder_path: Path,
    auth_policy_path: Path,
    fallback_threshold: float,
) -> BaselineV2Artifacts:
    """Загрузить модель baseline v2, scaler, label encoder и auth policy.

    Args:
        model_path: Путь к Keras-модели baseline v2.
        scaler_path: Путь к StandardScaler.
        label_encoder_path: Путь к LabelEncoder.
        auth_policy_path: Путь к JSON-файлу политики аутентификации.
        fallback_threshold: Порог, используемый при отсутствии значения в policy.

    Returns:
        Загруженные артефакты baseline v2.

    Raises:
        FileNotFoundError: Если один из обязательных артефактов отсутствует.
        TypeError: Если scaler или label encoder имеет неожиданный тип.
    """
    for path in (model_path, scaler_path, label_encoder_path, auth_policy_path):
        if not path.exists():
            raise FileNotFoundError(f"Обязательный артефакт не найден: {path}")

    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

    from tensorflow import keras  # noqa: PLC0415

    model = keras.models.load_model(model_path)
    scaler = joblib.load(scaler_path)
    label_encoder = joblib.load(label_encoder_path)
    auth_policy = json.loads(auth_policy_path.read_text(encoding="utf-8"))

    if not isinstance(scaler, StandardScaler):
        raise TypeError(f"Ожидался StandardScaler, получен: {type(scaler)!r}")

    if not isinstance(label_encoder, LabelEncoder):
        raise TypeError(f"Ожидался LabelEncoder, получен: {type(label_encoder)!r}")

    working_threshold = float(auth_policy.get("auth_threshold", fallback_threshold))

    return BaselineV2Artifacts(
        model=model,
        scaler=scaler,
        label_encoder=label_encoder,
        auth_policy=auth_policy,
        working_threshold=working_threshold,
    )


def calculate_scores(
    artifacts: BaselineV2Artifacts,
    split: TrainValidationTestSplit,
) -> ScoreSet:
    """Рассчитать genuine и impostor scores для test split.

    Для текущего baseline v2 authentication score — это softmax-вероятность
    заявленного пользователя.

    Args:
        artifacts: Артефакты baseline v2.
        split: Ненормализованный train/validation/test split.

    Returns:
        Набор genuine/impostor scores.
    """
    X_test_scaled = artifacts.scaler.transform(split.X_test)
    probabilities = artifacts.model.predict(
        X_test_scaled.astype(np.float32),
        verbose=0,
    )

    true_class_indices = artifacts.label_encoder.transform(split.y_test)
    row_indices = np.arange(len(split.y_test))

    genuine_scores = probabilities[row_indices, true_class_indices]

    impostor_mask = np.ones_like(probabilities, dtype=bool)
    impostor_mask[row_indices, true_class_indices] = False
    impostor_scores = probabilities[impostor_mask]

    return ScoreSet(
        genuine_scores=genuine_scores.astype(np.float64),
        impostor_scores=impostor_scores.astype(np.float64),
    )


def build_far_frr_curve(scores: ScoreSet) -> pd.DataFrame:
    """Построить таблицу FAR/FRR-кривой.

    Args:
        scores: Genuine/impostor scores.

    Returns:
        Таблица с threshold, FAR и FRR.
    """
    y_true = np.concatenate(
        [
            np.ones_like(scores.genuine_scores, dtype=np.int8),
            np.zeros_like(scores.impostor_scores, dtype=np.int8),
        ]
    )
    y_score = np.concatenate([scores.genuine_scores, scores.impostor_scores])

    far, true_accept_rate, thresholds = roc_curve(y_true=y_true, y_score=y_score)
    frr = 1.0 - true_accept_rate

    curve_df = pd.DataFrame(
        {
            "threshold": thresholds,
            "far": far,
            "frr": frr,
        }
    )

    curve_df = curve_df.replace([np.inf, -np.inf], np.nan).dropna()
    curve_df = curve_df.sort_values("far").reset_index(drop=True)

    return curve_df


def calculate_operating_point(scores: ScoreSet, threshold: float) -> OperatingPoint:
    """Рассчитать FAR и FRR для заданного threshold.

    Args:
        scores: Genuine/impostor scores.
        threshold: Рабочий порог аутентификации.

    Returns:
        Рабочая точка FAR/FRR.
    """
    false_accepts = int(np.sum(scores.impostor_scores >= threshold))
    false_rejects = int(np.sum(scores.genuine_scores < threshold))

    far = false_accepts / len(scores.impostor_scores)
    frr = false_rejects / len(scores.genuine_scores)

    return OperatingPoint(threshold=threshold, far=far, frr=frr)


def calculate_eer_point(curve_df: pd.DataFrame) -> OperatingPoint:
    """Найти точку EER на FAR/FRR-кривой.

    Args:
        curve_df: Таблица FAR/FRR-кривой.

    Returns:
        Точка EER.
    """
    eer_index = (curve_df["far"] - curve_df["frr"]).abs().idxmin()
    row = curve_df.loc[eer_index]

    return OperatingPoint(
        threshold=float(row["threshold"]),
        far=float(row["far"]),
        frr=float(row["frr"]),
    )


def save_curve_csv(curve_df: pd.DataFrame, output_csv: Path) -> None:
    """Сохранить FAR/FRR-кривую в CSV.

    Args:
        curve_df: Таблица FAR/FRR-кривой.
        output_csv: Путь сохранения CSV-файла.
    """
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    curve_df.to_csv(output_csv, index=False)


def build_far_frr_plot(
    curve_df: pd.DataFrame,
    working_point: OperatingPoint,
    eer_point: OperatingPoint,
    output_png: Path,
    output_svg: Path,
) -> None:
    """Построить и сохранить график FAR/FRR.

    Args:
        curve_df: Таблица FAR/FRR-кривой.
        working_point: Рабочая точка baseline v2.
        eer_point: Точка EER.
        output_png: Путь сохранения PNG-файла.
        output_svg: Путь сохранения SVG-файла.
    """
    output_png.parent.mkdir(parents=True, exist_ok=True)
    output_svg.parent.mkdir(parents=True, exist_ok=True)

    plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["axes.unicode_minus"] = False

    figure, axis = plt.subplots(figsize=(8, 6), constrained_layout=True)

    axis.plot(
        curve_df["far"] * 100.0,
        curve_df["frr"] * 100.0,
        linewidth=2,
        label="Кривая trade-off FAR/FRR",
    )

    axis.scatter(
        [working_point.far * 100.0],
        [working_point.frr * 100.0],
        s=70,
        marker="o",
        label=(
            "Рабочий порог "
            f"θ = {format_threshold_ru(working_point.threshold)} "
            f"(FAR = {working_point.far * 100.0:.2f}%, "
            f"FRR = {working_point.frr * 100.0:.2f}%)"
        ),
        zorder=3,
    )

    axis.scatter(
        [eer_point.far * 100.0],
        [eer_point.frr * 100.0],
        s=80,
        marker="x",
        label=(
            "Точка EER "
            f"(θ = {format_threshold_ru(eer_point.threshold)}, "
            f"EER ≈ {(eer_point.far + eer_point.frr) * 50.0:.2f}%)"
        ),
        zorder=3,
    )

    axis.plot(
        [0, 100],
        [0, 100],
        linestyle="--",
        linewidth=1,
        label="Линия FAR = FRR",
    )

    axis.set_title("Trade-off между FAR и FRR для baseline v2")
    axis.set_xlabel("FAR, ложный допуск, %")
    axis.set_ylabel("FRR, ложный отказ, %")
    axis.grid(linestyle="--", linewidth=0.5, alpha=0.7)

    max_axis_value = max(
        float(curve_df["far"].max() * 100.0),
        float(curve_df["frr"].max() * 100.0),
        working_point.far * 100.0,
        working_point.frr * 100.0,
        eer_point.far * 100.0,
        eer_point.frr * 100.0,
    )
    axis.set_xlim(0, min(100.0, max_axis_value * 1.05))
    axis.set_ylim(0, min(100.0, max_axis_value * 1.05))
    axis.legend(loc="best", fontsize=9)

    figure.savefig(output_png, dpi=300, bbox_inches="tight")
    figure.savefig(output_svg, bbox_inches="tight")
    plt.close(figure)


def format_threshold_ru(value: float) -> str:
    """Отформатировать threshold с запятой в русской подписи.

    Args:
        value: Значение threshold.

    Returns:
        Строка threshold.
    """
    return f"{value:.6f}".replace(".", ",")


def build_arg_parser() -> argparse.ArgumentParser:
    """Создать CLI-парсер.

    Returns:
        Настроенный parser.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Построить FAR/FRR-кривую для baseline v2 с отметкой рабочего порога и точки EER."
        )
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=CMU_FEATURES_FILE,
        help="Путь к обработанному CMU dataset.",
    )
    parser.add_argument(
        "--model-input",
        type=Path,
        default=DEFAULT_MODEL_INPUT,
        help="Путь к модели mlp_v2_batchnorm.keras.",
    )
    parser.add_argument(
        "--scaler-input",
        type=Path,
        default=DEFAULT_SCALER_INPUT,
        help="Путь к scaler_v2_batchnorm.pkl.",
    )
    parser.add_argument(
        "--label-encoder-input",
        type=Path,
        default=DEFAULT_LABEL_ENCODER_INPUT,
        help="Путь к label_encoder_v2_batchnorm.pkl.",
    )
    parser.add_argument(
        "--auth-policy-input",
        type=Path,
        default=DEFAULT_AUTH_POLICY_INPUT,
        help="Путь к auth_policy_v2_batchnorm.json.",
    )
    parser.add_argument(
        "--working-threshold",
        type=float,
        default=DEFAULT_WORKING_THRESHOLD,
        help="Fallback-значение рабочего threshold, если его нет в auth policy.",
    )
    parser.add_argument(
        "--output-png",
        type=Path,
        default=DEFAULT_OUTPUT_PNG,
        help="Путь сохранения PNG-графика.",
    )
    parser.add_argument(
        "--output-svg",
        type=Path,
        default=DEFAULT_OUTPUT_SVG,
        help="Путь сохранения SVG-графика.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=DEFAULT_OUTPUT_CSV,
        help="Путь сохранения CSV-файла с FAR/FRR-кривой.",
    )

    return parser


def main() -> None:
    """Точка входа CLI."""
    parser = build_arg_parser()
    args = parser.parse_args()

    artifacts = load_baseline_v2_artifacts(
        model_path=args.model_input,
        scaler_path=args.scaler_input,
        label_encoder_path=args.label_encoder_input,
        auth_policy_path=args.auth_policy_input,
        fallback_threshold=args.working_threshold,
    )
    split = load_unscaled_split(args.input)
    scores = calculate_scores(artifacts=artifacts, split=split)

    curve_df = build_far_frr_curve(scores=scores)
    working_point = calculate_operating_point(
        scores=scores,
        threshold=artifacts.working_threshold,
    )
    eer_point = calculate_eer_point(curve_df=curve_df)

    save_curve_csv(curve_df=curve_df, output_csv=args.output_csv)
    build_far_frr_plot(
        curve_df=curve_df,
        working_point=working_point,
        eer_point=eer_point,
        output_png=args.output_png,
        output_svg=args.output_svg,
    )

    print("FAR/FRR curve for baseline v2 saved:")
    print(f"Input dataset: {args.input}")
    print(f"Model input: {args.model_input}")
    print(f"Auth policy input: {args.auth_policy_input}")
    print(f"Working threshold: {artifacts.working_threshold:.6f}")
    print(f"Working FAR: {working_point.far:.6f}")
    print(f"Working FRR: {working_point.frr:.6f}")
    print(f"EER threshold: {eer_point.threshold:.6f}")
    print(f"EER FAR: {eer_point.far:.6f}")
    print(f"EER FRR: {eer_point.frr:.6f}")
    print(f"PNG path: {args.output_png}")
    print(f"SVG path: {args.output_svg}")
    print(f"CSV path: {args.output_csv}")


if __name__ == "__main__":
    main()
