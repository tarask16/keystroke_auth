"""Triplet-модель для metric-learning аутентификации.

Модуль реализует encoder и triplet loss для обучения embedding-пространства.
Каждая training-запись содержит три образца:
- anchor — базовый образец пользователя;
- positive — другой образец того же пользователя;
- negative — образец другого пользователя.

Цель обучения: сделать расстояние anchor-positive меньше расстояния
anchor-negative минимум на заданный margin.
"""

from __future__ import annotations

from typing import Any

import tensorflow as tf

from src.embedding_model.siamese_model import build_siamese_encoder


@tf.keras.utils.register_keras_serializable(package="keystroke_auth")
class TripletLoss(tf.keras.losses.Loss):
    """Triplet loss для metric-learning encoder-а.

    Loss использует выход модели вида `[ap_distance, an_distance]`, где:
    - `ap_distance` — расстояние anchor-positive;
    - `an_distance` — расстояние anchor-negative.

    Формула штрафа:
    `max(ap_distance - an_distance + margin, 0)`.

    Args:
        margin: Минимальный зазор между positive и negative расстояниями.
        name: Имя loss-функции для Keras.
        kwargs: Дополнительные параметры базового класса Keras Loss.
    """

    def __init__(self, margin: float = 0.2, name: str = "triplet_loss", **kwargs: Any):
        super().__init__(name=name, **kwargs)
        if margin <= 0:
            raise ValueError("margin must be positive.")
        self.margin = float(margin)

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """Рассчитать triplet loss для batch-а расстояний."""
        del y_true
        distances = tf.cast(y_pred, tf.float32)
        anchor_positive_distance = distances[:, 0]
        anchor_negative_distance = distances[:, 1]
        losses = tf.maximum(
            anchor_positive_distance - anchor_negative_distance + self.margin,
            0.0,
        )
        return tf.reduce_mean(losses)

    def get_config(self) -> dict[str, Any]:
        """Вернуть параметры для сериализации Keras."""
        config = super().get_config()
        config.update({"margin": self.margin})
        return config


@tf.keras.utils.register_keras_serializable(package="keystroke_auth")
class TripletMarginAccuracy(tf.keras.metrics.Metric):
    """Доля triplets, удовлетворяющих margin-условию.

    Triplet считается корректным, если:
    `distance(anchor, positive) + margin < distance(anchor, negative)`.
    """

    def __init__(
        self,
        margin: float = 0.2,
        name: str = "triplet_margin_accuracy",
        **kwargs: Any,
    ):
        super().__init__(name=name, **kwargs)
        if margin <= 0:
            raise ValueError("margin must be positive.")
        self.margin = float(margin)
        self.total = self.add_weight(name="total", initializer="zeros")
        self.count = self.add_weight(name="count", initializer="zeros")

    def update_state(
        self,
        y_true: tf.Tensor,
        y_pred: tf.Tensor,
        sample_weight: tf.Tensor | None = None,
    ) -> None:
        """Обновить накопленную метрику для текущего batch-а."""
        del y_true
        distances = tf.cast(y_pred, tf.float32)
        anchor_positive_distance = distances[:, 0]
        anchor_negative_distance = distances[:, 1]
        valid_triplets = tf.cast(
            anchor_positive_distance + self.margin < anchor_negative_distance,
            tf.float32,
        )

        if sample_weight is not None:
            weights = tf.cast(tf.reshape(sample_weight, (-1,)), tf.float32)
            valid_triplets *= weights
            batch_count = tf.reduce_sum(weights)
        else:
            batch_count = tf.cast(tf.size(valid_triplets), tf.float32)

        self.total.assign_add(tf.reduce_sum(valid_triplets))
        self.count.assign_add(batch_count)

    def result(self) -> tf.Tensor:
        """Вернуть среднюю долю корректных triplets."""
        return tf.math.divide_no_nan(self.total, self.count)

    def reset_state(self) -> None:
        """Сбросить накопленные значения метрики."""
        self.total.assign(0.0)
        self.count.assign(0.0)

    def get_config(self) -> dict[str, Any]:
        """Вернуть параметры для сериализации Keras."""
        config = super().get_config()
        config.update({"margin": self.margin})
        return config


@tf.keras.utils.register_keras_serializable(package="keystroke_auth")
class TripletDistanceLayer(tf.keras.layers.Layer):
    """Слой расчёта anchor-positive и anchor-negative расстояний."""

    def call(self, inputs: list[tf.Tensor]) -> tf.Tensor:
        """Вернуть tensor `[ap_distance, an_distance]` для каждого triplet."""
        if len(inputs) != 3:
            raise ValueError("TripletDistanceLayer expects exactly three tensors.")

        anchor_embedding, positive_embedding, negative_embedding = inputs
        anchor_positive_distance = self._euclidean_distance(anchor_embedding, positive_embedding)
        anchor_negative_distance = self._euclidean_distance(anchor_embedding, negative_embedding)
        return tf.concat([anchor_positive_distance, anchor_negative_distance], axis=1)

    @staticmethod
    def _euclidean_distance(embedding_a: tf.Tensor, embedding_b: tf.Tensor) -> tf.Tensor:
        """Рассчитать Euclidean distance между двумя batch-ами embedding-векторов."""
        squared_difference = tf.square(embedding_a - embedding_b)
        squared_distance = tf.reduce_sum(squared_difference, axis=1, keepdims=True)
        return tf.sqrt(tf.maximum(squared_distance, tf.keras.backend.epsilon()))


def build_triplet_encoder(
    input_dim: int,
    embedding_dim: int = 32,
    hidden_units: int = 128,
    dropout_rate: float = 0.2,
    embedding_activation: str | None = None,
    l2_normalize: bool = True,
) -> tf.keras.Model:
    """Создать encoder для Triplet-модели.

    Архитектура encoder-а синхронизирована с Siamese baseline, чтобы сравнение
    Stage 7 зависело от objective, а не от разных сетевых архитектур.
    """
    encoder = build_siamese_encoder(
        input_dim=input_dim,
        embedding_dim=embedding_dim,
        hidden_units=hidden_units,
        dropout_rate=dropout_rate,
        embedding_activation=embedding_activation,
        l2_normalize=l2_normalize,
    )
    encoder._name = "triplet_encoder"
    return encoder


def build_triplet_model(
    input_dim: int,
    embedding_dim: int = 32,
    hidden_units: int = 128,
    dropout_rate: float = 0.2,
    margin: float = 0.2,
    learning_rate: float = 1e-3,
    embedding_activation: str | None = None,
    l2_normalize: bool = True,
) -> tuple[tf.keras.Model, tf.keras.Model]:
    """Создать и скомпилировать Triplet-модель.

    Args:
        input_dim: Размерность входных признаков.
        embedding_dim: Размерность embedding-вектора.
        hidden_units: Число нейронов скрытого слоя encoder-а.
        dropout_rate: Dropout encoder-а.
        margin: Margin для triplet loss.
        learning_rate: Learning rate Adam optimizer-а.
        embedding_activation: Активация embedding-слоя.
        l2_normalize: Нормализовать ли embedding-векторы.

    Returns:
        Кортеж `(triplet_model, encoder)`.
    """
    if margin <= 0:
        raise ValueError("margin must be positive.")

    encoder = build_triplet_encoder(
        input_dim=input_dim,
        embedding_dim=embedding_dim,
        hidden_units=hidden_units,
        dropout_rate=dropout_rate,
        embedding_activation=embedding_activation,
        l2_normalize=l2_normalize,
    )

    anchor_input = tf.keras.Input(shape=(input_dim,), name="anchor")
    positive_input = tf.keras.Input(shape=(input_dim,), name="positive")
    negative_input = tf.keras.Input(shape=(input_dim,), name="negative")

    anchor_embedding = encoder(anchor_input)
    positive_embedding = encoder(positive_input)
    negative_embedding = encoder(negative_input)

    distances = TripletDistanceLayer(name="triplet_distances")(
        [anchor_embedding, positive_embedding, negative_embedding]
    )

    model = tf.keras.Model(
        inputs=[anchor_input, positive_input, negative_input],
        outputs=distances,
        name="triplet_model",
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=TripletLoss(margin=margin),
        metrics=[TripletMarginAccuracy(margin=margin)],
    )

    return model, encoder
