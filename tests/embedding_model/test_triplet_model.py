"""Тесты Triplet-модели Stage 7."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("tensorflow")

from src.embedding_model.train_triplet_embedding import load_triplet_dataset  # noqa: E402
from src.embedding_model.triplet_model import (  # noqa: E402
    TripletDistanceLayer,
    TripletLoss,
    TripletMarginAccuracy,
    build_triplet_encoder,
    build_triplet_model,
)


def test_triplet_loss_penalizes_wrong_distances() -> None:
    """Проверить, что loss ниже для корректных triplet расстояний."""
    import tensorflow as tf

    loss = TripletLoss(margin=0.2)
    dummy_labels = tf.zeros((2,), dtype=tf.float32)
    good_distances = tf.constant([[0.1, 0.8], [0.2, 1.0]], dtype=tf.float32)
    bad_distances = tf.constant([[0.8, 0.1], [1.0, 0.2]], dtype=tf.float32)

    good_loss = float(loss(dummy_labels, good_distances).numpy())
    bad_loss = float(loss(dummy_labels, bad_distances).numpy())

    assert good_loss < bad_loss


def test_triplet_margin_accuracy_uses_margin_condition() -> None:
    """Проверить margin-логику TripletMarginAccuracy."""
    import tensorflow as tf

    metric = TripletMarginAccuracy(margin=0.5)
    dummy_labels = tf.zeros((4,), dtype=tf.float32)
    distances = tf.constant(
        [
            [0.1, 0.8],
            [0.2, 0.5],
            [0.3, 0.9],
            [0.6, 0.7],
        ],
        dtype=tf.float32,
    )

    metric.update_state(dummy_labels, distances)

    assert float(metric.result().numpy()) == pytest.approx(0.5)


def test_triplet_distance_layer_output_shape() -> None:
    """Проверить форму выхода слоя расстояний."""
    import tensorflow as tf

    layer = TripletDistanceLayer()
    anchor = tf.zeros((5, 8), dtype=tf.float32)
    positive = tf.ones((5, 8), dtype=tf.float32)
    negative = tf.ones((5, 8), dtype=tf.float32) * 2.0
    distances = layer([anchor, positive, negative])

    assert tuple(distances.shape) == (5, 2)


def test_build_triplet_encoder_output_shape() -> None:
    """Проверить форму embedding-вектора Triplet encoder-а."""
    encoder = build_triplet_encoder(input_dim=31, embedding_dim=8, hidden_units=16)
    samples = np.zeros((4, 31), dtype=np.float32)
    embeddings = encoder.predict(samples, verbose=0)

    assert embeddings.shape == (4, 8)


def test_build_triplet_model_output_shape() -> None:
    """Проверить, что Triplet-модель возвращает AP/AN расстояния."""
    model, encoder = build_triplet_model(
        input_dim=31,
        embedding_dim=8,
        hidden_units=16,
        margin=0.2,
    )
    anchor = np.zeros((5, 31), dtype=np.float32)
    positive = np.ones((5, 31), dtype=np.float32)
    negative = np.ones((5, 31), dtype=np.float32) * 2.0
    distances = model.predict([anchor, positive, negative], verbose=0)

    assert encoder.output_shape == (None, 8)
    assert distances.shape == (5, 2)


def test_load_triplet_dataset_validates_npz_structure(tmp_path: Path) -> None:
    """Проверить загрузку triplet dataset из `.npz`."""
    triplet_path = tmp_path / "triplets.npz"
    np.savez_compressed(
        triplet_path,
        anchor=np.zeros((6, 3), dtype=np.float32),
        positive=np.ones((6, 3), dtype=np.float32),
        negative=np.ones((6, 3), dtype=np.float32) * 2.0,
    )

    triplets = load_triplet_dataset(triplet_path)

    assert triplets.anchor.shape == (6, 3)
    assert triplets.positive.shape == (6, 3)
    assert triplets.negative.shape == (6, 3)
    assert triplets.input_dim == 3
    assert triplets.dummy_labels.shape == (6,)
