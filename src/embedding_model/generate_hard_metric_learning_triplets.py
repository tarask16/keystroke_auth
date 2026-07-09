"""Генерация triplets с hard/semi-hard negative mining для Stage 7.

Базовый генератор Stage 7 создаёт triplets со случайным negative sample.
Такие negative-примеры часто оказываются слишком простыми: encoder быстро
отделяет их, но это не снижает FRR при template-based verification.

Данный модуль использует уже обученный seed encoder и выбирает negative samples,
которые находятся близко к anchor в embedding-пространстве. Это формирует более
сложный обучающий набор для последующего Triplet fine-tuning.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.embedding_model.evaluate_metric_learning_encoders import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_MODELS_DIR,
    DEFAULT_TRAIN_FRACTION,
    DEFAULT_VALIDATION_FRACTION,
    SUPPORTED_DISTANCES,
    extract_embedding_splits,
    load_encoder_model,
)
from src.embedding_model.evaluate_metric_learning_encoders import (
    SplitFeatures as EvaluationSplitFeatures,
)
from src.embedding_model.generate_metric_learning_pairs import (
    DEFAULT_FEATURES_FILE,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_SCALER_FILE,
    TRAIN_SPLIT,
    VALIDATION_SPLIT,
    add_deterministic_split,
    get_feature_columns,
    load_feature_table,
)
from src.embedding_model.generate_metric_learning_triplets import (
    TripletDataset,
    append_triplet,
    save_triplet_dataset,
    shuffle_triplet_dataset,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SEED_ENCODER_FILE = DEFAULT_MODELS_DIR / "tuning" / "triplet_m0p1_l2" / "encoder.keras"
DEFAULT_TRAIN_OUTPUT = DEFAULT_OUTPUT_DIR / "metric_hard_triplets_train.npz"
DEFAULT_VALIDATION_OUTPUT = DEFAULT_OUTPUT_DIR / "metric_hard_triplets_validation.npz"
DEFAULT_MANIFEST_OUTPUT = DEFAULT_OUTPUT_DIR / "metric_hard_triplets_manifest.csv"

SUPPORTED_MINING_STRATEGIES = ("hard", "semi_hard", "mixed")
DEFAULT_MARGIN = 0.1
DEFAULT_TOP_K = 32
DEFAULT_TRAIN_TRIPLETS_PER_USER = 1000
DEFAULT_VALIDATION_TRIPLETS_PER_USER = 250


@dataclass(frozen=True)
class ScaledSplitTables:
    """Train/validation split-ы после transform существующим scaler-ом."""

    train: pd.DataFrame
    validation: pd.DataFrame
    feature_columns: list[str]


@dataclass(frozen=True)
class NegativeCandidateIndex:
    """Top-k negative candidates для каждого sample внутри одного split-а."""

    candidates: np.ndarray
    distances: np.ndarray


@dataclass(frozen=True)
class MinedTripletResult:
    """Triplet-набор и диагностическая таблица mining-решений."""

    dataset: TripletDataset
    manifest: pd.DataFrame


def normalize_margin_label(margin: float) -> str:
    """Преобразовать margin в безопасную часть имени файла."""
    return f"{margin:g}".replace(".", "p")


def resolve_default_seed_encoder() -> Path:
    """Вернуть лучший доступный seed encoder для hard negative mining."""
    if DEFAULT_SEED_ENCODER_FILE.exists():
        return DEFAULT_SEED_ENCODER_FILE
    return DEFAULT_MODELS_DIR / "triplet_encoder.keras"


def load_existing_scaler(scaler_path: Path) -> Any:
    """Загрузить scaler, обученный только на train split."""
    if not scaler_path.exists():
        raise FileNotFoundError(
            "Scaler not found. Run metric-learning pair/triplet generation first: "
            f"{scaler_path}"
        )
    return joblib.load(scaler_path)


def scale_train_validation_with_existing_scaler(
    df: pd.DataFrame,
    feature_columns: list[str],
    scaler_path: Path,
) -> ScaledSplitTables:
    """Нормализовать train/validation через уже сохранённый scaler.

    Scaler не обучается повторно. Это важно, потому что seed encoder был обучен
    в той же feature space, что и сохранённый `metric_learning_scaler.pkl`.
    """
    scaler = load_existing_scaler(scaler_path)
    train_df = df[df["split"] == TRAIN_SPLIT].copy().reset_index(drop=True)
    validation_df = df[df["split"] == VALIDATION_SPLIT].copy().reset_index(drop=True)

    if train_df.empty:
        raise ValueError("Train split is empty.")
    if validation_df.empty:
        raise ValueError("Validation split is empty.")

    train_df.loc[:, feature_columns] = scaler.transform(
        train_df.loc[:, feature_columns].to_numpy(dtype=np.float32)
    )
    validation_df.loc[:, feature_columns] = scaler.transform(
        validation_df.loc[:, feature_columns].to_numpy(dtype=np.float32)
    )

    return ScaledSplitTables(
        train=train_df,
        validation=validation_df,
        feature_columns=feature_columns,
    )


def calculate_pairwise_distances(
    left_embeddings: np.ndarray,
    right_embeddings: np.ndarray,
    distance: str,
) -> np.ndarray:
    """Рассчитать матрицу расстояний между двумя наборами embeddings."""
    if distance == "euclidean":
        delta = left_embeddings[:, None, :] - right_embeddings[None, :, :]
        return np.sqrt(np.sum(delta * delta, axis=2)).astype(np.float32)

    if distance == "manhattan":
        delta = np.abs(left_embeddings[:, None, :] - right_embeddings[None, :, :])
        return np.sum(delta, axis=2).astype(np.float32)

    if distance == "cosine":
        left_norm = normalize_rows(left_embeddings)
        right_norm = normalize_rows(right_embeddings)
        similarity = left_norm @ right_norm.T
        return (1.0 - similarity).astype(np.float32)

    raise ValueError(f"Unsupported distance: {distance}")


def normalize_rows(values: np.ndarray, epsilon: float = 1e-12) -> np.ndarray:
    """L2-нормализовать строки матрицы с защитой от деления на ноль."""
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norms, epsilon)


def build_negative_candidate_index(
    df: pd.DataFrame,
    embeddings: np.ndarray,
    distance: str,
    top_k: int,
) -> NegativeCandidateIndex:
    """Построить top-k ближайших negative samples для каждого anchor sample."""
    if top_k <= 0:
        raise ValueError("top_k must be positive.")

    row_count = len(df)
    candidate_count = min(top_k, row_count - 1)
    if candidate_count <= 0:
        raise ValueError("At least two samples are required for negative mining.")

    candidate_indices = np.full((row_count, candidate_count), -1, dtype=np.int64)
    candidate_distances = np.full((row_count, candidate_count), np.inf, dtype=np.float32)
    user_ids = df["user_id"].astype(str).to_numpy()

    for user_id in sorted(df["user_id"].astype(str).unique().tolist()):
        anchor_positions = np.where(user_ids == user_id)[0]
        negative_positions = np.where(user_ids != user_id)[0]

        if negative_positions.size == 0:
            raise ValueError("At least two users are required for negative mining.")

        distances = calculate_pairwise_distances(
            left_embeddings=embeddings[anchor_positions],
            right_embeddings=embeddings[negative_positions],
            distance=distance,
        )
        local_top_k = min(candidate_count, negative_positions.size)
        local_top_indices = np.argpartition(distances, kth=local_top_k - 1, axis=1)[
            :, :local_top_k
        ]
        local_top_distances = np.take_along_axis(distances, local_top_indices, axis=1)
        order = np.argsort(local_top_distances, axis=1)
        local_top_indices = np.take_along_axis(local_top_indices, order, axis=1)
        local_top_distances = np.take_along_axis(local_top_distances, order, axis=1)

        global_top_indices = negative_positions[local_top_indices]
        candidate_indices[anchor_positions, :local_top_k] = global_top_indices
        candidate_distances[anchor_positions, :local_top_k] = local_top_distances

    return NegativeCandidateIndex(
        candidates=candidate_indices,
        distances=candidate_distances,
    )


def select_negative_position(
    candidate_positions: np.ndarray,
    candidate_distances: np.ndarray,
    positive_distance: float,
    strategy: str,
    margin: float,
    rng: np.random.Generator,
) -> tuple[int, str, float]:
    """Выбрать negative sample из top-k кандидатов.

    Returns:
        Кортеж: индекс строки negative, фактически применённая стратегия,
        расстояние anchor-negative.
    """
    valid_mask = candidate_positions >= 0
    positions = candidate_positions[valid_mask]
    distances = candidate_distances[valid_mask]

    if positions.size == 0:
        raise ValueError("No negative candidates available.")

    if strategy == "hard":
        selected_index = int(rng.integers(0, positions.size))
        return int(positions[selected_index]), "hard", float(distances[selected_index])

    if strategy == "mixed":
        effective_strategy = "semi_hard" if rng.random() < 0.5 else "hard"
        return select_negative_position(
            candidate_positions=positions,
            candidate_distances=distances,
            positive_distance=positive_distance,
            strategy=effective_strategy,
            margin=margin,
            rng=rng,
        )

    if strategy == "semi_hard":
        semi_hard_mask = (distances > positive_distance) & (
            distances < positive_distance + margin
        )
        if np.any(semi_hard_mask):
            semi_hard_positions = positions[semi_hard_mask]
            semi_hard_distances = distances[semi_hard_mask]
            selected_index = int(rng.integers(0, semi_hard_positions.size))
            return (
                int(semi_hard_positions[selected_index]),
                "semi_hard",
                float(semi_hard_distances[selected_index]),
            )

        farther_mask = distances > positive_distance
        if np.any(farther_mask):
            farther_positions = positions[farther_mask]
            farther_distances = distances[farther_mask]
            selected_index = int(np.argmin(farther_distances))
            return (
                int(farther_positions[selected_index]),
                "semi_hard_fallback_farther",
                float(farther_distances[selected_index]),
            )

        return int(positions[0]), "semi_hard_fallback_hard", float(distances[0])

    raise ValueError(f"Unsupported mining strategy: {strategy}")


def calculate_positive_distance(
    anchor_embedding: np.ndarray,
    positive_embedding: np.ndarray,
    distance: str,
) -> float:
    """Рассчитать расстояние anchor-positive для одного triplet-а."""
    distances = calculate_pairwise_distances(
        left_embeddings=anchor_embedding.reshape(1, -1),
        right_embeddings=positive_embedding.reshape(1, -1),
        distance=distance,
    )
    return float(distances[0, 0])


def create_mined_triplet_dataset(
    df: pd.DataFrame,
    feature_columns: list[str],
    embeddings: np.ndarray,
    triplets_per_user: int,
    candidate_index: NegativeCandidateIndex,
    strategy: str,
    distance: str,
    margin: float,
    rng: np.random.Generator,
) -> MinedTripletResult:
    """Сформировать triplets с hard/semi-hard negative mining."""
    if strategy not in SUPPORTED_MINING_STRATEGIES:
        raise ValueError(f"Unsupported mining strategy: {strategy}")
    if triplets_per_user <= 0:
        raise ValueError("triplets_per_user must be positive.")
    if margin <= 0:
        raise ValueError("margin must be positive.")

    users = sorted(df["user_id"].astype(str).unique().tolist())
    if len(users) < 2:
        raise ValueError("At least two users are required for triplet generation.")

    grouped_indices = {
        user_id: df.index[df["user_id"].astype(str) == user_id].to_numpy(dtype=np.int64)
        for user_id in users
    }
    for user_id, indices in grouped_indices.items():
        if len(indices) < 2:
            raise ValueError(f"User {user_id} has less than two samples.")

    anchor_rows: list[np.ndarray] = []
    positive_rows: list[np.ndarray] = []
    negative_rows: list[np.ndarray] = []
    anchor_user_ids: list[str] = []
    positive_user_ids: list[str] = []
    negative_user_ids: list[str] = []
    anchor_sample_ids: list[str] = []
    positive_sample_ids: list[str] = []
    negative_sample_ids: list[str] = []
    manifest_rows: list[dict[str, Any]] = []

    for anchor_user_id in users:
        user_indices = grouped_indices[anchor_user_id]

        for _ in range(triplets_per_user):
            anchor_index, positive_index = rng.choice(user_indices, size=2, replace=False)
            anchor_index = int(anchor_index)
            positive_index = int(positive_index)
            positive_distance = calculate_positive_distance(
                anchor_embedding=embeddings[anchor_index],
                positive_embedding=embeddings[positive_index],
                distance=distance,
            )
            negative_index, effective_strategy, negative_distance = select_negative_position(
                candidate_positions=candidate_index.candidates[anchor_index],
                candidate_distances=candidate_index.distances[anchor_index],
                positive_distance=positive_distance,
                strategy=strategy,
                margin=margin,
                rng=rng,
            )

            append_triplet(
                df=df,
                feature_columns=feature_columns,
                anchor_index=anchor_index,
                positive_index=positive_index,
                negative_index=negative_index,
                anchor_rows=anchor_rows,
                positive_rows=positive_rows,
                negative_rows=negative_rows,
                anchor_user_ids=anchor_user_ids,
                positive_user_ids=positive_user_ids,
                negative_user_ids=negative_user_ids,
                anchor_sample_ids=anchor_sample_ids,
                positive_sample_ids=positive_sample_ids,
                negative_sample_ids=negative_sample_ids,
            )
            negative_row = df.loc[negative_index]
            manifest_rows.append(
                {
                    "anchor_user_id": anchor_user_id,
                    "negative_user_id": str(negative_row["user_id"]),
                    "anchor_sample_id": str(df.loc[anchor_index, "sample_id"]),
                    "positive_sample_id": str(df.loc[positive_index, "sample_id"]),
                    "negative_sample_id": str(negative_row["sample_id"]),
                    "positive_distance": positive_distance,
                    "negative_distance": negative_distance,
                    "strategy": strategy,
                    "effective_strategy": effective_strategy,
                    "margin": margin,
                    "distance": distance,
                }
            )

    dataset = TripletDataset(
        anchor=np.vstack(anchor_rows).astype(np.float32),
        positive=np.vstack(positive_rows).astype(np.float32),
        negative=np.vstack(negative_rows).astype(np.float32),
        anchor_user_id=np.asarray(anchor_user_ids, dtype=str),
        positive_user_id=np.asarray(positive_user_ids, dtype=str),
        negative_user_id=np.asarray(negative_user_ids, dtype=str),
        anchor_sample_id=np.asarray(anchor_sample_ids, dtype=str),
        positive_sample_id=np.asarray(positive_sample_ids, dtype=str),
        negative_sample_id=np.asarray(negative_sample_ids, dtype=str),
    )
    shuffled_dataset = shuffle_triplet_dataset(dataset, rng)
    manifest = pd.DataFrame(manifest_rows)

    return MinedTripletResult(dataset=shuffled_dataset, manifest=manifest)


def build_split_features(
    split_name: str,
    df: pd.DataFrame,
    feature_columns: list[str],
) -> EvaluationSplitFeatures:
    """Преобразовать DataFrame split-а в контейнер evaluation pipeline."""
    return EvaluationSplitFeatures(
        split_name=split_name,
        x=df.loc[:, feature_columns].to_numpy(dtype=np.float32),
        user_ids=df["user_id"].astype(str).to_numpy(),
        sample_ids=df["sample_id"].astype(str).to_numpy(),
    )


def run_generation(
    input_path: Path,
    scaler_path: Path,
    seed_encoder_path: Path,
    train_output_path: Path,
    validation_output_path: Path,
    manifest_output_path: Path,
    train_triplets_per_user: int,
    validation_triplets_per_user: int,
    strategy: str,
    distance: str,
    margin: float,
    top_k: int,
    batch_size: int,
    random_state: int,
    train_fraction: float,
    validation_fraction: float,
) -> None:
    """Выполнить полный цикл генерации hard-mined train/validation triplets."""
    rng = np.random.default_rng(random_state)
    df = load_feature_table(input_path)
    df = add_deterministic_split(
        df=df,
        train_fraction=train_fraction,
        validation_fraction=validation_fraction,
    )
    feature_columns = get_feature_columns(df)
    split_tables = scale_train_validation_with_existing_scaler(
        df=df,
        feature_columns=feature_columns,
        scaler_path=scaler_path,
    )

    encoder = load_encoder_model(seed_encoder_path)
    embedding_splits = extract_embedding_splits(
        encoder=encoder,
        split_features=[
            build_split_features(TRAIN_SPLIT, split_tables.train, feature_columns),
            build_split_features(VALIDATION_SPLIT, split_tables.validation, feature_columns),
        ],
        batch_size=batch_size,
    )
    embeddings_by_split = {split.split_name: split.embeddings for split in embedding_splits}

    train_candidate_index = build_negative_candidate_index(
        df=split_tables.train,
        embeddings=embeddings_by_split[TRAIN_SPLIT],
        distance=distance,
        top_k=top_k,
    )
    validation_candidate_index = build_negative_candidate_index(
        df=split_tables.validation,
        embeddings=embeddings_by_split[VALIDATION_SPLIT],
        distance=distance,
        top_k=top_k,
    )

    train_result = create_mined_triplet_dataset(
        df=split_tables.train,
        feature_columns=feature_columns,
        embeddings=embeddings_by_split[TRAIN_SPLIT],
        triplets_per_user=train_triplets_per_user,
        candidate_index=train_candidate_index,
        strategy=strategy,
        distance=distance,
        margin=margin,
        rng=rng,
    )
    validation_result = create_mined_triplet_dataset(
        df=split_tables.validation,
        feature_columns=feature_columns,
        embeddings=embeddings_by_split[VALIDATION_SPLIT],
        triplets_per_user=validation_triplets_per_user,
        candidate_index=validation_candidate_index,
        strategy=strategy,
        distance=distance,
        margin=margin,
        rng=rng,
    )

    save_triplet_dataset(train_result.dataset, train_output_path)
    save_triplet_dataset(validation_result.dataset, validation_output_path)
    save_manifest(
        train_manifest=train_result.manifest,
        validation_manifest=validation_result.manifest,
        output_path=manifest_output_path,
    )

    print("Hard-mined metric-learning triplets generated successfully.")
    print(f"Strategy: {strategy}")
    print(f"Distance: {distance}")
    print(f"Margin: {margin}")
    print(f"Top-k candidates: {top_k}")
    print(f"Train triplets: {train_result.dataset.anchor.shape[0]}")
    print(f"Validation triplets: {validation_result.dataset.anchor.shape[0]}")
    print(f"Seed encoder: {seed_encoder_path}")
    print(f"Train output: {train_output_path}")
    print(f"Validation output: {validation_output_path}")
    print(f"Manifest: {manifest_output_path}")


def save_manifest(
    train_manifest: pd.DataFrame,
    validation_manifest: pd.DataFrame,
    output_path: Path,
) -> None:
    """Сохранить CSV-manifest с диагностикой mining-выбора."""
    train_manifest = train_manifest.copy()
    validation_manifest = validation_manifest.copy()
    train_manifest.insert(0, "split", TRAIN_SPLIT)
    validation_manifest.insert(0, "split", VALIDATION_SPLIT)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.concat([train_manifest, validation_manifest], ignore_index=True).to_csv(
        output_path,
        index=False,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    """Создать CLI parser для hard negative mining генератора."""
    parser = argparse.ArgumentParser(
        description="Generate hard/semi-hard triplets for Stage 7 metric learning."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_FEATURES_FILE)
    parser.add_argument("--scaler", type=Path, default=DEFAULT_SCALER_FILE)
    parser.add_argument("--seed-encoder", type=Path, default=resolve_default_seed_encoder())
    parser.add_argument("--train-output", type=Path, default=DEFAULT_TRAIN_OUTPUT)
    parser.add_argument("--validation-output", type=Path, default=DEFAULT_VALIDATION_OUTPUT)
    parser.add_argument("--manifest-output", type=Path, default=DEFAULT_MANIFEST_OUTPUT)
    parser.add_argument(
        "--train-triplets-per-user",
        type=int,
        default=DEFAULT_TRAIN_TRIPLETS_PER_USER,
    )
    parser.add_argument(
        "--validation-triplets-per-user",
        type=int,
        default=DEFAULT_VALIDATION_TRIPLETS_PER_USER,
    )
    parser.add_argument("--strategy", choices=SUPPORTED_MINING_STRATEGIES, default="semi_hard")
    parser.add_argument("--distance", choices=SUPPORTED_DISTANCES, default="cosine")
    parser.add_argument("--margin", type=float, default=DEFAULT_MARGIN)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--train-fraction", type=float, default=DEFAULT_TRAIN_FRACTION)
    parser.add_argument("--validation-fraction", type=float, default=DEFAULT_VALIDATION_FRACTION)
    return parser


def main() -> None:
    """CLI entry point."""
    parser = build_arg_parser()
    args = parser.parse_args()
    run_generation(
        input_path=args.input,
        scaler_path=args.scaler,
        seed_encoder_path=args.seed_encoder,
        train_output_path=args.train_output,
        validation_output_path=args.validation_output,
        manifest_output_path=args.manifest_output,
        train_triplets_per_user=args.train_triplets_per_user,
        validation_triplets_per_user=args.validation_triplets_per_user,
        strategy=args.strategy,
        distance=args.distance,
        margin=args.margin,
        top_k=args.top_k,
        batch_size=args.batch_size,
        random_state=args.random_state,
        train_fraction=args.train_fraction,
        validation_fraction=args.validation_fraction,
    )


if __name__ == "__main__":
    main()
