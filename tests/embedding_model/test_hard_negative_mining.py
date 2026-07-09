"""Тесты hard/semi-hard negative mining для Stage 7.6."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.embedding_model.generate_hard_metric_learning_triplets import (
    NegativeCandidateIndex,
    build_negative_candidate_index,
    calculate_pairwise_distances,
    create_mined_triplet_dataset,
    normalize_margin_label,
    select_negative_position,
)
from src.embedding_model.run_hard_negative_mining_experiment import (
    build_hard_mining_candidates,
    build_hard_mining_paths,
)


def test_calculate_pairwise_distances_supports_euclidean_and_cosine() -> None:
    """Расстояния должны считаться предсказуемо на простых embeddings."""
    left = np.asarray([[1.0, 0.0]], dtype=np.float32)
    right = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)

    euclidean = calculate_pairwise_distances(left, right, "euclidean")
    cosine = calculate_pairwise_distances(left, right, "cosine")

    assert euclidean.shape == (1, 2)
    assert euclidean[0, 0] == pytest.approx(0.0)
    assert euclidean[0, 1] == pytest.approx(np.sqrt(2.0))
    assert cosine[0, 0] == pytest.approx(0.0)
    assert cosine[0, 1] == pytest.approx(1.0)


def test_build_negative_candidate_index_excludes_same_user() -> None:
    """Top-k negative candidates не должны включать samples того же пользователя."""
    df = pd.DataFrame(
        {
            "user_id": ["s001", "s001", "s002", "s002"],
            "sample_id": ["a1", "a2", "b1", "b2"],
        }
    )
    embeddings = np.asarray(
        [
            [0.0, 0.0],
            [0.1, 0.0],
            [1.0, 0.0],
            [1.1, 0.0],
        ],
        dtype=np.float32,
    )

    candidate_index = build_negative_candidate_index(
        df=df,
        embeddings=embeddings,
        distance="euclidean",
        top_k=2,
    )

    first_user_candidates = candidate_index.candidates[0]
    assert set(first_user_candidates.tolist()) == {2, 3}


def test_select_negative_position_uses_semi_hard_when_available() -> None:
    """Semi-hard mining должен выбирать negative между pos_dist и pos_dist+margin."""
    rng = np.random.default_rng(42)
    position, effective_strategy, distance = select_negative_position(
        candidate_positions=np.asarray([10, 11, 12]),
        candidate_distances=np.asarray([0.2, 0.35, 0.8], dtype=np.float32),
        positive_distance=0.3,
        strategy="semi_hard",
        margin=0.2,
        rng=rng,
    )

    assert position == 11
    assert effective_strategy == "semi_hard"
    assert distance == pytest.approx(0.35)


def test_select_negative_position_falls_back_to_hard_when_needed() -> None:
    """При отсутствии semi-hard candidates должен быть явный fallback."""
    rng = np.random.default_rng(42)
    position, effective_strategy, distance = select_negative_position(
        candidate_positions=np.asarray([10, 11, 12]),
        candidate_distances=np.asarray([0.1, 0.2, 0.25], dtype=np.float32),
        positive_distance=0.3,
        strategy="semi_hard",
        margin=0.2,
        rng=rng,
    )

    assert position == 10
    assert effective_strategy == "semi_hard_fallback_hard"
    assert distance == pytest.approx(0.1)


def test_create_mined_triplet_dataset_preserves_user_constraints() -> None:
    """Anchor/positive должны быть genuine, negative должен быть impostor."""
    df = pd.DataFrame(
        {
            "user_id": ["s001", "s001", "s002", "s002"],
            "sample_id": ["a1", "a2", "b1", "b2"],
            "f1": [0.0, 0.1, 1.0, 1.1],
            "f2": [0.0, 0.0, 0.0, 0.0],
        }
    )
    embeddings = df[["f1", "f2"]].to_numpy(dtype=np.float32)
    candidate_index = NegativeCandidateIndex(
        candidates=np.asarray([[2, 3], [2, 3], [0, 1], [0, 1]], dtype=np.int64),
        distances=np.asarray(
            [[1.0, 1.1], [0.9, 1.0], [1.0, 0.9], [1.1, 1.0]],
            dtype=np.float32,
        ),
    )

    result = create_mined_triplet_dataset(
        df=df,
        feature_columns=["f1", "f2"],
        embeddings=embeddings,
        triplets_per_user=2,
        candidate_index=candidate_index,
        strategy="hard",
        distance="euclidean",
        margin=0.1,
        rng=np.random.default_rng(42),
    )

    assert result.dataset.anchor.shape == (4, 2)
    assert np.all(result.dataset.anchor_user_id == result.dataset.positive_user_id)
    assert np.all(result.dataset.anchor_user_id != result.dataset.negative_user_id)
    assert set(result.manifest["strategy"].unique()) == {"hard"}


def test_build_hard_mining_candidates_and_paths_are_isolated(tmp_path: Path) -> None:
    """Hard-mining кандидаты должны иметь отдельные каталоги данных и моделей."""
    candidates = build_hard_mining_candidates(
        strategies=["semi_hard"],
        margins=[0.1],
        top_k_values=[16],
        mining_distance="cosine",
        embedding_dim=32,
        hidden_units=128,
        dropout_rate=0.2,
        learning_rate=1e-3,
        epochs=2,
        batch_size=512,
        patience=2,
        random_state=42,
        embedding_activation=None,
        l2_normalize=True,
        limit_candidates=None,
    )
    paths = build_hard_mining_paths(
        candidate=candidates[0],
        data_root=tmp_path / "data",
        models_root=tmp_path / "models",
        reports_root=tmp_path / "reports",
    )

    assert candidates[0].name == "triplet_semihard_m0p1_top16_cosine_l2"
    assert candidates[0].name in str(paths.train_triplets_path)
    assert candidates[0].name in str(paths.encoder_path)
    assert paths.encoder_path.name == "encoder.keras"


def test_normalize_margin_label() -> None:
    """Margin label должен быть пригоден для имени каталога."""
    assert normalize_margin_label(0.1) == "0p1"
    assert normalize_margin_label(1.0) == "1"
