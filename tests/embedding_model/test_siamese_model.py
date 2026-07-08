"""Тесты Siamese-модели Stage 7."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("tensorflow")

from src.embedding_model.siamese_model import (  # noqa: E402
    ContrastiveAccuracy,
    ContrastiveLoss,
    build_siamese_encoder,
    build_siamese_model,
)
from src.embedding_model.train_siamese_embedding import load_pair_dataset  # noqa: E402


def test_contrastive_loss_penalizes_wrong_distances() -> None:
    """Проверить, что loss ниже для корректных genuine/impostor расстояний."""
    import tensorflow as tf

    loss = ContrastiveLoss(margin=1.0)
    labels = tf.constant([[1.0], [0.0]], dtype=tf.float32)
    good_distances = tf.constant([[0.1], [1.2]], dtype=tf.float32)
    bad_distances = tf.constant([[1.2], [0.1]], dtype=tf.float32)

    good_loss = float(loss(labels, good_distances).numpy())
    bad_loss = float(loss(labels, bad_distances).numpy())

    assert good_loss < bad_loss


def test_contrastive_accuracy_uses_distance_threshold() -> None:
    """Проверить threshold-логику метрики accuracy."""
    import tensorflow as tf

    metric = ContrastiveAccuracy(threshold=0.5)
    labels = tf.constant([1.0, 0.0, 1.0, 0.0], dtype=tf.float32)
    distances = tf.constant([0.1, 0.7, 0.9, 0.2], dtype=tf.float32)

    metric.update_state(labels, distances)

    assert float(metric.result().numpy()) == pytest.approx(0.5)


def test_build_siamese_encoder_output_shape() -> None:
    """Проверить форму embedding-вектора encoder-а."""
    encoder = build_siamese_encoder(input_dim=31, embedding_dim=8, hidden_units=16)
    samples = np.zeros((4, 31), dtype=np.float32)
    embeddings = encoder.predict(samples, verbose=0)

    assert embeddings.shape == (4, 8)


def test_build_siamese_model_output_shape() -> None:
    """Проверить, что Siamese-модель возвращает одно расстояние на пару."""
    model, encoder = build_siamese_model(
        input_dim=31,
        embedding_dim=8,
        hidden_units=16,
        margin=1.0,
    )
    x_a = np.zeros((5, 31), dtype=np.float32)
    x_b = np.ones((5, 31), dtype=np.float32)
    distances = model.predict([x_a, x_b], verbose=0)

    assert encoder.output_shape == (None, 8)
    assert distances.shape == (5, 1)


def test_load_pair_dataset_validates_npz_structure(tmp_path: Path) -> None:
    """Проверить загрузку pair dataset из `.npz`."""
    pair_path = tmp_path / "pairs.npz"
    np.savez_compressed(
        pair_path,
        x_a=np.zeros((6, 3), dtype=np.float32),
        x_b=np.ones((6, 3), dtype=np.float32),
        y=np.asarray([1.0, 0.0, 1.0, 0.0, 1.0, 0.0], dtype=np.float32),
    )

    pairs = load_pair_dataset(pair_path)

    assert pairs.x_a.shape == (6, 3)
    assert pairs.x_b.shape == (6, 3)
    assert pairs.y.shape == (6,)
    assert pairs.input_dim == 3
