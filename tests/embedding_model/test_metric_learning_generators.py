"""Тесты генераторов пар и triplets для Stage 7."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.embedding_model.generate_metric_learning_pairs import (
    add_deterministic_split,
    create_pair_dataset,
    get_feature_columns,
)
from src.embedding_model.generate_metric_learning_triplets import create_triplet_dataset


def build_synthetic_feature_table() -> pd.DataFrame:
    """Создать компактную таблицу для unit-тестов."""
    rows = []

    for user_index, user_id in enumerate(["s001", "s002", "s003"]):
        for rep in range(10):
            rows.append(
                {
                    "user_id": user_id,
                    "session_index": 1,
                    "rep": rep,
                    "sample_id": f"{user_id}_r{rep:02d}",
                    "H.key": float(user_index + rep),
                    "DD.key.key": float(user_index + rep + 0.1),
                    "UD.key.key": float(user_index + rep + 0.2),
                }
            )

    return pd.DataFrame(rows)


def test_add_deterministic_split_keeps_all_users_in_train_validation() -> None:
    """Проверить per-user split."""
    df = build_synthetic_feature_table()
    split_df = add_deterministic_split(df, train_fraction=0.6, validation_fraction=0.2)

    assert set(split_df["split"].unique()) == {"train", "validation", "test"}

    train_users = set(split_df[split_df["split"] == "train"]["user_id"].unique())
    validation_users = set(split_df[split_df["split"] == "validation"]["user_id"].unique())

    assert train_users == {"s001", "s002", "s003"}
    assert validation_users == {"s001", "s002", "s003"}


def test_create_pair_dataset_is_balanced_and_has_valid_labels() -> None:
    """Проверить баланс genuine/impostor пар и форму массивов."""
    rng = np.random.default_rng(42)
    df = build_synthetic_feature_table()
    feature_columns = get_feature_columns(df)

    dataset = create_pair_dataset(
        df=df,
        feature_columns=feature_columns,
        pairs_per_user=3,
        rng=rng,
    )

    assert dataset.x_a.shape == (18, 3)
    assert dataset.x_b.shape == (18, 3)
    assert dataset.y.shape == (18,)
    assert int(dataset.y.sum()) == 9
    assert int((dataset.y == 0).sum()) == 9

    genuine_mask = dataset.y == 1
    impostor_mask = dataset.y == 0

    assert np.all(dataset.user_id_a[genuine_mask] == dataset.user_id_b[genuine_mask])
    assert np.all(dataset.user_id_a[impostor_mask] != dataset.user_id_b[impostor_mask])


def test_create_triplet_dataset_has_valid_user_relations() -> None:
    """Проверить корректность user_id внутри triplet-записи."""
    rng = np.random.default_rng(42)
    df = build_synthetic_feature_table()
    feature_columns = get_feature_columns(df)

    dataset = create_triplet_dataset(
        df=df,
        feature_columns=feature_columns,
        triplets_per_user=4,
        rng=rng,
    )

    assert dataset.anchor.shape == (12, 3)
    assert dataset.positive.shape == (12, 3)
    assert dataset.negative.shape == (12, 3)
    assert np.all(dataset.anchor_user_id == dataset.positive_user_id)
    assert np.all(dataset.anchor_user_id != dataset.negative_user_id)
