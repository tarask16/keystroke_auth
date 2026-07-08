"""Тесты generalized evaluation pipeline для Stage 7."""

from __future__ import annotations

import numpy as np

from src.embedding_model.evaluate_metric_learning_encoders import (
    EmbeddingSplit,
    TrialDistanceSet,
    build_mean_templates,
    calculate_distance_matrix,
    evaluate_threshold_policy,
    select_distance_threshold_at_far,
    select_global_policy,
    split_genuine_impostor_distances,
)


def test_calculate_distance_matrix_euclidean() -> None:
    """Euclidean distance matrix должна иметь размер samples x templates."""
    embeddings = np.array([[0.0, 0.0], [3.0, 4.0]], dtype=np.float32)
    templates = np.array([[0.0, 0.0], [6.0, 8.0]], dtype=np.float32)

    distances = calculate_distance_matrix(embeddings, templates, "euclidean")

    assert distances.shape == (2, 2)
    np.testing.assert_allclose(distances[0], np.array([0.0, 10.0]), atol=1e-6)
    np.testing.assert_allclose(distances[1], np.array([5.0, 5.0]), atol=1e-6)


def test_calculate_distance_matrix_cosine() -> None:
    """Cosine distance должен быть близок к нулю для одинаковых направлений."""
    embeddings = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    templates = np.array([[1.0, 0.0], [1.0, 1.0]], dtype=np.float32)

    distances = calculate_distance_matrix(embeddings, templates, "cosine")

    assert distances.shape == (2, 2)
    assert distances[0, 0] < 1e-6
    assert distances[1, 0] > 0.99


def test_build_mean_templates() -> None:
    """User template должен быть средним embedding-вектором пользователя."""
    split = EmbeddingSplit(
        split_name="train",
        embeddings=np.array(
            [
                [1.0, 1.0],
                [3.0, 3.0],
                [10.0, 10.0],
            ],
            dtype=np.float32,
        ),
        user_ids=np.array(["s001", "s001", "s002"]),
        sample_ids=np.array(["a", "b", "c"]),
    )

    templates = build_mean_templates(split)

    assert templates.user_ids == ["s001", "s002"]
    assert templates.samples_count == {"s001": 2, "s002": 1}
    np.testing.assert_allclose(templates.template_matrix[0], np.array([2.0, 2.0]))
    np.testing.assert_allclose(templates.template_matrix[1], np.array([10.0, 10.0]))


def test_split_genuine_impostor_distances() -> None:
    """Genuine distance берётся из колонки фактического пользователя."""
    trials = TrialDistanceSet(
        split_name="test",
        distance_name="euclidean",
        distance_matrix=np.array([[0.1, 0.9], [0.8, 0.2]], dtype=np.float32),
        sample_user_ids=np.array(["s001", "s002"]),
        sample_ids=np.array(["a", "b"]),
        template_user_ids=["s001", "s002"],
    )

    genuine, impostor = split_genuine_impostor_distances(trials)

    np.testing.assert_allclose(genuine, np.array([0.1, 0.2], dtype=np.float32))
    np.testing.assert_allclose(impostor, np.array([0.9, 0.8], dtype=np.float32))


def test_select_distance_threshold_at_far() -> None:
    """Threshold selection должен ограничивать число принятых impostor-попыток."""
    impostor_distances = np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float32)

    threshold = select_distance_threshold_at_far(impostor_distances, target_far=0.4)

    assert abs(threshold - 0.2) < 1e-6
    assert np.mean(impostor_distances <= threshold) <= 0.4


def test_evaluate_threshold_policy() -> None:
    """FAR/FRR должны считаться относительно claimed_user_id."""
    trials = TrialDistanceSet(
        split_name="test",
        distance_name="euclidean",
        distance_matrix=np.array([[0.1, 0.3], [0.7, 0.2]], dtype=np.float32),
        sample_user_ids=np.array(["s001", "s002"]),
        sample_ids=np.array(["a", "b"]),
        template_user_ids=["s001", "s002"],
    )
    thresholds = {"s001": 0.5, "s002": 0.5}

    metrics = evaluate_threshold_policy(trials, thresholds)

    assert metrics["genuine_trials"] == 2
    assert metrics["impostor_trials"] == 2
    assert metrics["false_rejects"] == 0
    assert metrics["false_accepts"] == 1
    assert metrics["far"] == 0.5
    assert metrics["frr"] == 0.0


def test_select_global_policy() -> None:
    """Global policy должна назначить один и тот же threshold всем пользователям."""
    trials = TrialDistanceSet(
        split_name="validation",
        distance_name="euclidean",
        distance_matrix=np.array([[0.1, 0.8], [0.7, 0.2]], dtype=np.float32),
        sample_user_ids=np.array(["s001", "s002"]),
        sample_ids=np.array(["a", "b"]),
        template_user_ids=["s001", "s002"],
    )

    policy = select_global_policy(trials, target_far=0.5)

    assert policy.policy_name == "global"
    assert set(policy.thresholds) == {"s001", "s002"}
    assert len(set(policy.thresholds.values())) == 1
