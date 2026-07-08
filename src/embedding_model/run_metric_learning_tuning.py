"""Запуск tuning-экспериментов для Stage 7 metric-learning.

Скрипт автоматизирует серию обучений Siamese/Triplet encoder-ов с разными
значениями margin и сразу прогоняет единый evaluation protocol из задачи 7.4.
Основная цель — проверить, можно ли снизить FRR при целевом FAR около 1%.

Результаты сохраняются раздельно по каждому кандидату, чтобы не перезаписывать
рабочие артефакты `siamese_encoder.keras` и `triplet_encoder.keras`.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from src.embedding_model.evaluate_metric_learning_encoders import (
    DEFAULT_BATCH_SIZE as DEFAULT_EVALUATION_BATCH_SIZE,
)
from src.embedding_model.evaluate_metric_learning_encoders import (
    DEFAULT_MODELS_DIR,
    DEFAULT_REPORTS_DIR,
    DEFAULT_TARGET_FAR,
    DEFAULT_TRAIN_FRACTION,
    DEFAULT_VALIDATION_FRACTION,
    SUPPORTED_DISTANCES,
    EncoderConfig,
    evaluate_one_encoder,
    resolve_distances,
)
from src.embedding_model.generate_metric_learning_pairs import DEFAULT_FEATURES_FILE

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "processed" / "embedding_model"
DEFAULT_TUNING_MODELS_DIR = DEFAULT_MODELS_DIR / "tuning"
DEFAULT_TUNING_REPORTS_DIR = DEFAULT_REPORTS_DIR / "tuning"
DEFAULT_METRIC_SCALER_FILE = DEFAULT_MODELS_DIR / "metric_learning_scaler.pkl"
DEFAULT_TRAIN_PAIRS_FILE = DEFAULT_DATA_DIR / "metric_pairs_train.npz"
DEFAULT_VALIDATION_PAIRS_FILE = DEFAULT_DATA_DIR / "metric_pairs_validation.npz"
DEFAULT_TRAIN_TRIPLETS_FILE = DEFAULT_DATA_DIR / "metric_triplets_train.npz"
DEFAULT_VALIDATION_TRIPLETS_FILE = DEFAULT_DATA_DIR / "metric_triplets_validation.npz"

DEFAULT_SIAMESE_MARGINS = (0.2, 0.5, 1.0, 1.5)
DEFAULT_TRIPLET_MARGINS = (0.1, 0.2, 0.5, 1.0)
SUPPORTED_OBJECTIVES = ("siamese", "triplet")
TEST_SPLIT = "test"


@dataclass(frozen=True)
class TuningCandidate:
    """Один вариант обучения metric-learning encoder-а."""

    objective: str
    margin: float
    embedding_dim: int
    hidden_units: int
    dropout_rate: float
    learning_rate: float
    epochs: int
    batch_size: int
    patience: int
    random_state: int
    embedding_activation: str | None
    l2_normalize: bool

    @property
    def name(self) -> str:
        """Вернуть стабильное имя кандидата для каталогов и таблиц."""
        margin_label = format_margin_for_name(self.margin)
        norm_label = "l2" if self.l2_normalize else "raw"
        return f"{self.objective}_m{margin_label}_{norm_label}"


@dataclass(frozen=True)
class CandidatePaths:
    """Пути к артефактам одного tuning-кандидата."""

    model_path: Path
    encoder_path: Path
    training_metrics_path: Path
    evaluation_output_root: Path
    evaluation_dir: Path


@dataclass(frozen=True)
class CandidateSummary:
    """Сводные результаты одного tuning-кандидата."""

    candidate: str
    objective: str
    margin: float
    l2_normalize: bool
    best_distance: str
    best_policy: str
    best_far: float
    best_frr: float
    best_balanced_error: float
    best_eer: float
    best_roc_auc: float
    training_metrics_path: str
    evaluation_dir: str


def format_margin_for_name(margin: float) -> str:
    """Преобразовать margin в безопасную часть имени файла/каталога."""
    return f"{margin:g}".replace(".", "p")


def parse_objectives(objective: str) -> list[str]:
    """Развернуть CLI-значение objective в список вариантов обучения."""
    if objective == "all":
        return list(SUPPORTED_OBJECTIVES)
    if objective not in SUPPORTED_OBJECTIVES:
        raise ValueError(f"Unsupported objective: {objective}")
    return [objective]


def resolve_margins(
    objective: str,
    siamese_margins: list[float] | None,
    triplet_margins: list[float] | None,
) -> list[float]:
    """Вернуть список margin-ов для выбранного objective."""
    if objective == "siamese":
        return siamese_margins or list(DEFAULT_SIAMESE_MARGINS)
    if objective == "triplet":
        return triplet_margins or list(DEFAULT_TRIPLET_MARGINS)
    raise ValueError(f"Unsupported objective: {objective}")


def build_candidates(
    objectives: list[str],
    siamese_margins: list[float] | None,
    triplet_margins: list[float] | None,
    embedding_dim: int,
    hidden_units: int,
    dropout_rate: float,
    learning_rate: float,
    epochs: int,
    batch_size: int,
    patience: int,
    random_state: int,
    embedding_activation: str | None,
    l2_normalize: bool,
    limit_candidates: int | None,
) -> list[TuningCandidate]:
    """Сформировать план tuning-эксперимента."""
    candidates: list[TuningCandidate] = []

    for objective in objectives:
        margins = resolve_margins(
            objective=objective,
            siamese_margins=siamese_margins,
            triplet_margins=triplet_margins,
        )
        for margin in margins:
            candidates.append(
                TuningCandidate(
                    objective=objective,
                    margin=margin,
                    embedding_dim=embedding_dim,
                    hidden_units=hidden_units,
                    dropout_rate=dropout_rate,
                    learning_rate=learning_rate,
                    epochs=epochs,
                    batch_size=batch_size,
                    patience=patience,
                    random_state=random_state,
                    embedding_activation=embedding_activation,
                    l2_normalize=l2_normalize,
                )
            )

    if limit_candidates is not None:
        if limit_candidates < 1:
            raise ValueError("limit_candidates must be positive")
        candidates = candidates[:limit_candidates]

    return candidates


def build_candidate_paths(
    candidate: TuningCandidate,
    models_root: Path,
    reports_root: Path,
) -> CandidatePaths:
    """Построить пути к артефактам tuning-кандидата."""
    candidate_models_dir = models_root / candidate.name
    evaluation_output_root = reports_root
    evaluation_dir = evaluation_output_root / f"{candidate.name}_encoder"

    return CandidatePaths(
        model_path=candidate_models_dir / "model.keras",
        encoder_path=candidate_models_dir / "encoder.keras",
        training_metrics_path=reports_root / candidate.name / "training_metrics.csv",
        evaluation_output_root=evaluation_output_root,
        evaluation_dir=evaluation_dir,
    )


def normalize_embedding_activation(value: str) -> str | None:
    """Преобразовать CLI-значение activation в значение для Keras Dense."""
    if value == "linear":
        return None
    return value


def train_candidate(
    candidate: TuningCandidate,
    paths: CandidatePaths,
    train_pairs_path: Path,
    validation_pairs_path: Path,
    train_triplets_path: Path,
    validation_triplets_path: Path,
) -> None:
    """Обучить один Siamese или Triplet tuning-кандидат."""
    if candidate.objective == "siamese":
        from src.embedding_model.train_siamese_embedding import run_training as run_siamese

        run_siamese(
            train_pairs_path=train_pairs_path,
            validation_pairs_path=validation_pairs_path,
            model_output_path=paths.model_path,
            encoder_output_path=paths.encoder_path,
            metrics_output_path=paths.training_metrics_path,
            embedding_dim=candidate.embedding_dim,
            hidden_units=candidate.hidden_units,
            dropout_rate=candidate.dropout_rate,
            margin=candidate.margin,
            learning_rate=candidate.learning_rate,
            epochs=candidate.epochs,
            batch_size=candidate.batch_size,
            patience=candidate.patience,
            random_state=candidate.random_state,
            embedding_activation=candidate.embedding_activation,
            l2_normalize=candidate.l2_normalize,
        )
        return

    if candidate.objective == "triplet":
        from src.embedding_model.train_triplet_embedding import run_training as run_triplet

        run_triplet(
            train_triplets_path=train_triplets_path,
            validation_triplets_path=validation_triplets_path,
            model_output_path=paths.model_path,
            encoder_output_path=paths.encoder_path,
            metrics_output_path=paths.training_metrics_path,
            embedding_dim=candidate.embedding_dim,
            hidden_units=candidate.hidden_units,
            dropout_rate=candidate.dropout_rate,
            margin=candidate.margin,
            learning_rate=candidate.learning_rate,
            epochs=candidate.epochs,
            batch_size=candidate.batch_size,
            patience=candidate.patience,
            random_state=candidate.random_state,
            embedding_activation=candidate.embedding_activation,
            l2_normalize=candidate.l2_normalize,
        )
        return

    raise ValueError(f"Unsupported objective: {candidate.objective}")


def evaluate_candidate(
    candidate: TuningCandidate,
    paths: CandidatePaths,
    features_file: Path,
    scaler_path: Path,
    distances: list[str],
    target_far: float,
    evaluation_batch_size: int,
    train_fraction: float,
    validation_fraction: float,
    max_validation_frr_increase: float,
) -> None:
    """Оценить encoder tuning-кандидата через общий Stage 7 protocol."""
    config = EncoderConfig(
        name=candidate.name,
        encoder_path=paths.encoder_path,
        scaler_path=scaler_path,
    )

    evaluate_one_encoder(
        config=config,
        features_file=features_file,
        output_root=paths.evaluation_output_root,
        distances=distances,
        target_far=target_far,
        batch_size=evaluation_batch_size,
        train_fraction=train_fraction,
        validation_fraction=validation_fraction,
        max_validation_frr_increase=max_validation_frr_increase,
    )


def select_best_policy_row(policy_path: Path) -> pd.Series:
    """Выбрать лучшую строку test policy по balanced error."""
    policy_df = pd.read_csv(policy_path)
    test_df = policy_df[policy_df["split"] == TEST_SPLIT].copy()

    if test_df.empty:
        raise ValueError(f"No test rows in policy file: {policy_path}")

    test_df = test_df.sort_values(
        by=["balanced_error", "far", "frr"],
        ascending=[True, True, True],
    )
    return test_df.iloc[0]


def select_best_diagnostics_row(diagnostics_path: Path) -> pd.Series:
    """Выбрать лучшую строку test diagnostics по EER."""
    diagnostics_df = pd.read_csv(diagnostics_path)
    test_df = diagnostics_df[diagnostics_df["split"] == TEST_SPLIT].copy()

    if test_df.empty:
        raise ValueError(f"No test rows in diagnostics file: {diagnostics_path}")

    test_df = test_df.sort_values(by=["eer", "distance"], ascending=[True, True])
    return test_df.iloc[0]


def summarize_candidate(candidate: TuningCandidate, paths: CandidatePaths) -> CandidateSummary:
    """Собрать итоговые метрики tuning-кандидата."""
    policy_path = paths.evaluation_dir / "embedding_threshold_policy.csv"
    diagnostics_path = paths.evaluation_dir / "embedding_distance_diagnostics.csv"

    best_policy = select_best_policy_row(policy_path)
    best_diagnostics = select_best_diagnostics_row(diagnostics_path)

    return CandidateSummary(
        candidate=candidate.name,
        objective=candidate.objective,
        margin=candidate.margin,
        l2_normalize=candidate.l2_normalize,
        best_distance=str(best_policy["distance"]),
        best_policy=str(best_policy["policy"]),
        best_far=float(best_policy["far"]),
        best_frr=float(best_policy["frr"]),
        best_balanced_error=float(best_policy["balanced_error"]),
        best_eer=float(best_diagnostics["eer"]),
        best_roc_auc=float(best_diagnostics["roc_auc"]),
        training_metrics_path=str(paths.training_metrics_path),
        evaluation_dir=str(paths.evaluation_dir),
    )


def save_candidate_plan(candidates: list[TuningCandidate], output_path: Path) -> None:
    """Сохранить план tuning-эксперимента в CSV."""
    rows = []
    for candidate in candidates:
        row = asdict(candidate)
        row["candidate"] = candidate.name
        rows.append(row)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False)


def save_tuning_summary(summaries: list[CandidateSummary], output_path: Path) -> None:
    """Сохранить сводную таблицу tuning-результатов."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    summary_df = pd.DataFrame(asdict(summary) for summary in summaries)
    summary_df = summary_df.sort_values(
        by=["best_balanced_error", "best_far", "best_frr"],
        ascending=[True, True, True],
    )
    summary_df.to_csv(output_path, index=False)


def run_tuning(args: argparse.Namespace) -> None:
    """Выполнить полный tuning experiment по CLI-аргументам."""
    objectives = parse_objectives(args.objective)
    distances = resolve_distances(args.distance or ["all"])
    embedding_activation = normalize_embedding_activation(args.embedding_activation)

    candidates = build_candidates(
        objectives=objectives,
        siamese_margins=args.siamese_margin,
        triplet_margins=args.triplet_margin,
        embedding_dim=args.embedding_dim,
        hidden_units=args.hidden_units,
        dropout_rate=args.dropout_rate,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        batch_size=args.batch_size,
        patience=args.patience,
        random_state=args.random_state,
        embedding_activation=embedding_activation,
        l2_normalize=not args.no_l2_normalize,
        limit_candidates=args.limit_candidates,
    )

    plan_path = args.reports_root / "metric_learning_tuning_plan.csv"
    summary_path = args.reports_root / "metric_learning_tuning_summary.csv"
    save_candidate_plan(candidates, plan_path)

    print(f"Tuning candidates: {len(candidates)}")
    print(f"Plan: {plan_path}")

    if args.dry_run:
        print("Dry run enabled. Training and evaluation were not executed.")
        return

    summaries: list[CandidateSummary] = []
    for index, candidate in enumerate(candidates, start=1):
        print(f"[{index}/{len(candidates)}] Run candidate: {candidate.name}")
        paths = build_candidate_paths(candidate, args.models_root, args.reports_root)

        train_candidate(
            candidate=candidate,
            paths=paths,
            train_pairs_path=args.train_pairs,
            validation_pairs_path=args.validation_pairs,
            train_triplets_path=args.train_triplets,
            validation_triplets_path=args.validation_triplets,
        )
        evaluate_candidate(
            candidate=candidate,
            paths=paths,
            features_file=args.features_file,
            scaler_path=args.scaler,
            distances=distances,
            target_far=args.target_far,
            evaluation_batch_size=args.evaluation_batch_size,
            train_fraction=args.train_fraction,
            validation_fraction=args.validation_fraction,
            max_validation_frr_increase=args.max_validation_frr_increase,
        )
        summaries.append(summarize_candidate(candidate, paths))
        save_tuning_summary(summaries, summary_path)

    print("Metric-learning tuning completed successfully.")
    print(f"Summary: {summary_path}")


def build_arg_parser() -> argparse.ArgumentParser:
    """Создать CLI parser для tuning-экспериментов Stage 7."""
    parser = argparse.ArgumentParser(description="Run Stage 7 metric-learning tuning.")
    parser.add_argument("--objective", choices=["all", *SUPPORTED_OBJECTIVES], default="all")
    parser.add_argument("--siamese-margin", action="append", type=float, default=None)
    parser.add_argument("--triplet-margin", action="append", type=float, default=None)
    parser.add_argument("--train-pairs", type=Path, default=DEFAULT_TRAIN_PAIRS_FILE)
    parser.add_argument("--validation-pairs", type=Path, default=DEFAULT_VALIDATION_PAIRS_FILE)
    parser.add_argument("--train-triplets", type=Path, default=DEFAULT_TRAIN_TRIPLETS_FILE)
    parser.add_argument(
        "--validation-triplets",
        type=Path,
        default=DEFAULT_VALIDATION_TRIPLETS_FILE,
    )
    parser.add_argument("--features-file", type=Path, default=DEFAULT_FEATURES_FILE)
    parser.add_argument("--scaler", type=Path, default=DEFAULT_METRIC_SCALER_FILE)
    parser.add_argument("--models-root", type=Path, default=DEFAULT_TUNING_MODELS_DIR)
    parser.add_argument("--reports-root", type=Path, default=DEFAULT_TUNING_REPORTS_DIR)
    parser.add_argument("--embedding-dim", type=int, default=32)
    parser.add_argument("--hidden-units", type=int, default=128)
    parser.add_argument("--dropout-rate", type=float, default=0.2)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--embedding-activation", choices=["linear", "relu"], default="linear")
    parser.add_argument("--no-l2-normalize", action="store_true")
    parser.add_argument(
        "--distance",
        action="append",
        choices=["all", *SUPPORTED_DISTANCES],
        default=None,
    )
    parser.add_argument("--target-far", type=float, default=DEFAULT_TARGET_FAR)
    parser.add_argument("--evaluation-batch-size", type=int, default=DEFAULT_EVALUATION_BATCH_SIZE)
    parser.add_argument("--train-fraction", type=float, default=DEFAULT_TRAIN_FRACTION)
    parser.add_argument("--validation-fraction", type=float, default=DEFAULT_VALIDATION_FRACTION)
    parser.add_argument("--max-validation-frr-increase", type=float, default=0.02)
    parser.add_argument("--limit-candidates", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    """CLI entry point."""
    parser = build_arg_parser()
    args = parser.parse_args()
    run_tuning(args)


if __name__ == "__main__":
    main()
