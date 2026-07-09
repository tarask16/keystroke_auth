"""Эксперимент 7.6: Triplet fine-tuning с hard/semi-hard negative mining.

Скрипт объединяет три шага:
1. генерация hard-mined triplets через seed encoder;
2. обучение Triplet encoder-а на новых triplets;
3. evaluation через единый Stage 7 protocol.

Все артефакты сохраняются в отдельных каталогах, чтобы не перезаписывать
основные модели Stage 7 и результаты tuning 7.5.
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
from src.embedding_model.generate_hard_metric_learning_triplets import (
    DEFAULT_MARGIN,
    DEFAULT_TOP_K,
    SUPPORTED_MINING_STRATEGIES,
    normalize_margin_label,
    resolve_default_seed_encoder,
)
from src.embedding_model.generate_hard_metric_learning_triplets import (
    run_generation as run_hard_triplet_generation,
)
from src.embedding_model.generate_metric_learning_pairs import (
    DEFAULT_FEATURES_FILE,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_SCALER_FILE,
)
from src.embedding_model.run_metric_learning_tuning import (
    CandidateSummary,
    save_tuning_summary,
    select_best_diagnostics_row,
    select_best_policy_row,
)

DEFAULT_HARD_DATA_ROOT = DEFAULT_OUTPUT_DIR / "hard_negative_mining"
DEFAULT_HARD_MODELS_ROOT = DEFAULT_MODELS_DIR / "hard_negative_mining"
DEFAULT_HARD_REPORTS_ROOT = DEFAULT_REPORTS_DIR / "hard_negative_mining"
DEFAULT_STRATEGIES = ("semi_hard", "hard")
DEFAULT_MARGINS = (0.1, 0.2)
TEST_SPLIT = "test"


@dataclass(frozen=True)
class HardMiningCandidate:
    """Один кандидат hard/semi-hard negative mining эксперимента."""

    strategy: str
    margin: float
    top_k: int
    mining_distance: str
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
        """Вернуть стабильное имя кандидата для каталогов."""
        margin_label = normalize_margin_label(self.margin)
        strategy_label = self.strategy.replace("_", "")
        norm_label = "l2" if self.l2_normalize else "raw"
        return (
            f"triplet_{strategy_label}_m{margin_label}_"
            f"top{self.top_k}_{self.mining_distance}_{norm_label}"
        )


@dataclass(frozen=True)
class HardMiningPaths:
    """Пути к данным, моделям и отчётам одного кандидата."""

    train_triplets_path: Path
    validation_triplets_path: Path
    manifest_path: Path
    model_path: Path
    encoder_path: Path
    training_metrics_path: Path
    evaluation_output_root: Path
    evaluation_dir: Path


def normalize_embedding_activation(value: str) -> str | None:
    """Преобразовать CLI activation в значение для Keras Dense."""
    if value == "linear":
        return None
    return value


def build_hard_mining_candidates(
    strategies: list[str],
    margins: list[float],
    top_k_values: list[int],
    mining_distance: str,
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
) -> list[HardMiningCandidate]:
    """Сформировать план hard negative mining эксперимента."""
    candidates: list[HardMiningCandidate] = []
    for strategy in strategies:
        if strategy not in SUPPORTED_MINING_STRATEGIES:
            raise ValueError(f"Unsupported strategy: {strategy}")
        for margin in margins:
            for top_k in top_k_values:
                candidates.append(
                    HardMiningCandidate(
                        strategy=strategy,
                        margin=margin,
                        top_k=top_k,
                        mining_distance=mining_distance,
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


def build_hard_mining_paths(
    candidate: HardMiningCandidate,
    data_root: Path,
    models_root: Path,
    reports_root: Path,
) -> HardMiningPaths:
    """Построить изолированные пути артефактов кандидата."""
    candidate_data_dir = data_root / candidate.name
    candidate_model_dir = models_root / candidate.name
    candidate_report_dir = reports_root / candidate.name
    evaluation_dir = reports_root / f"{candidate.name}_encoder"

    return HardMiningPaths(
        train_triplets_path=candidate_data_dir / "metric_hard_triplets_train.npz",
        validation_triplets_path=candidate_data_dir / "metric_hard_triplets_validation.npz",
        manifest_path=candidate_data_dir / "metric_hard_triplets_manifest.csv",
        model_path=candidate_model_dir / "model.keras",
        encoder_path=candidate_model_dir / "encoder.keras",
        training_metrics_path=candidate_report_dir / "training_metrics.csv",
        evaluation_output_root=reports_root,
        evaluation_dir=evaluation_dir,
    )


def summarize_hard_mining_candidate(
    candidate: HardMiningCandidate,
    paths: HardMiningPaths,
) -> CandidateSummary:
    """Собрать итоговые метрики одного hard-mining кандидата."""
    best_policy = select_best_policy_row(
        paths.evaluation_dir / "embedding_threshold_policy.csv"
    )
    best_diagnostics = select_best_diagnostics_row(
        paths.evaluation_dir / "embedding_distance_diagnostics.csv"
    )

    return CandidateSummary(
        candidate=candidate.name,
        objective="triplet_hard_negative_mining",
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


def save_hard_mining_plan(
    candidates: list[HardMiningCandidate],
    output_path: Path,
) -> None:
    """Сохранить план hard negative mining эксперимента."""
    rows = []
    for candidate in candidates:
        row = asdict(candidate)
        row["candidate"] = candidate.name
        rows.append(row)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False)


def run_one_candidate(
    candidate: HardMiningCandidate,
    paths: HardMiningPaths,
    args: argparse.Namespace,
    distances: list[str],
) -> CandidateSummary:
    """Сгенерировать данные, обучить модель и выполнить evaluation."""
    run_hard_triplet_generation(
        input_path=args.features_file,
        scaler_path=args.scaler,
        seed_encoder_path=args.seed_encoder,
        train_output_path=paths.train_triplets_path,
        validation_output_path=paths.validation_triplets_path,
        manifest_output_path=paths.manifest_path,
        train_triplets_per_user=args.train_triplets_per_user,
        validation_triplets_per_user=args.validation_triplets_per_user,
        strategy=candidate.strategy,
        distance=candidate.mining_distance,
        margin=candidate.margin,
        top_k=candidate.top_k,
        batch_size=args.mining_batch_size,
        random_state=candidate.random_state,
        train_fraction=args.train_fraction,
        validation_fraction=args.validation_fraction,
    )

    from src.embedding_model.train_triplet_embedding import run_training as run_triplet_training

    run_triplet_training(
        train_triplets_path=paths.train_triplets_path,
        validation_triplets_path=paths.validation_triplets_path,
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

    evaluate_one_encoder(
        config=EncoderConfig(
            name=candidate.name,
            encoder_path=paths.encoder_path,
            scaler_path=args.scaler,
        ),
        features_file=args.features_file,
        output_root=paths.evaluation_output_root,
        distances=distances,
        target_far=args.target_far,
        batch_size=args.evaluation_batch_size,
        train_fraction=args.train_fraction,
        validation_fraction=args.validation_fraction,
        max_validation_frr_increase=args.max_validation_frr_increase,
    )

    return summarize_hard_mining_candidate(candidate, paths)


def run_experiment(args: argparse.Namespace) -> None:
    """Выполнить hard/semi-hard negative mining experiment."""
    embedding_activation = normalize_embedding_activation(args.embedding_activation)
    distances = resolve_distances(args.distance or ["all"])
    candidates = build_hard_mining_candidates(
        strategies=args.strategy,
        margins=args.margin,
        top_k_values=args.top_k,
        mining_distance=args.mining_distance,
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

    plan_path = args.reports_root / "hard_negative_mining_plan.csv"
    summary_path = args.reports_root / "hard_negative_mining_summary.csv"
    save_hard_mining_plan(candidates, plan_path)

    print(f"Hard-mining candidates: {len(candidates)}")
    print(f"Plan: {plan_path}")

    if args.dry_run:
        print("Dry run enabled. Generation, training and evaluation were not executed.")
        return

    summaries: list[CandidateSummary] = []
    for index, candidate in enumerate(candidates, start=1):
        print(f"[{index}/{len(candidates)}] Run candidate: {candidate.name}")
        paths = build_hard_mining_paths(
            candidate=candidate,
            data_root=args.data_root,
            models_root=args.models_root,
            reports_root=args.reports_root,
        )
        summaries.append(run_one_candidate(candidate, paths, args, distances))
        save_tuning_summary(summaries, summary_path)

    print("Hard negative mining experiment completed successfully.")
    print(f"Summary: {summary_path}")


def build_arg_parser() -> argparse.ArgumentParser:
    """Создать CLI parser для эксперимента 7.6."""
    parser = argparse.ArgumentParser(description="Run Stage 7.6 hard negative mining.")
    parser.add_argument("--features-file", type=Path, default=DEFAULT_FEATURES_FILE)
    parser.add_argument("--scaler", type=Path, default=DEFAULT_SCALER_FILE)
    parser.add_argument("--seed-encoder", type=Path, default=resolve_default_seed_encoder())
    parser.add_argument("--data-root", type=Path, default=DEFAULT_HARD_DATA_ROOT)
    parser.add_argument("--models-root", type=Path, default=DEFAULT_HARD_MODELS_ROOT)
    parser.add_argument("--reports-root", type=Path, default=DEFAULT_HARD_REPORTS_ROOT)
    parser.add_argument("--strategy", action="append", choices=SUPPORTED_MINING_STRATEGIES)
    parser.add_argument("--margin", action="append", type=float)
    parser.add_argument("--top-k", action="append", type=int)
    parser.add_argument("--mining-distance", choices=SUPPORTED_DISTANCES, default="cosine")
    parser.add_argument("--train-triplets-per-user", type=int, default=1000)
    parser.add_argument("--validation-triplets-per-user", type=int, default=250)
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
    parser.add_argument("--distance", action="append", choices=["all", *SUPPORTED_DISTANCES])
    parser.add_argument("--target-far", type=float, default=DEFAULT_TARGET_FAR)
    parser.add_argument("--mining-batch-size", type=int, default=DEFAULT_EVALUATION_BATCH_SIZE)
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
    if args.strategy is None:
        args.strategy = list(DEFAULT_STRATEGIES)
    if args.margin is None:
        args.margin = list(DEFAULT_MARGINS)
    if args.top_k is None:
        args.top_k = [DEFAULT_TOP_K]
    if args.margin == []:
        args.margin = [DEFAULT_MARGIN]
    run_experiment(args)


if __name__ == "__main__":
    main()
