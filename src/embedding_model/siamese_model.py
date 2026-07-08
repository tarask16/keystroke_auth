"""Siamese-модель для metric-learning аутентификации.

Модуль содержит общий encoder, distance layer и contrastive loss.
Метка пары интерпретируется так:
- 1.0 — genuine-пара, образцы принадлежат одному пользователю;
- 0.0 — impostor-пара, образцы принадлежат разным пользователям.

Для genuine-пар loss уменьшает расстояние между embedding-векторами.
Для impostor-пар loss штрафует расстояния меньше заданного margin.
"""

from __future__ import annotations

from typing import Any

import tensorflow as tf


@tf.keras.utils.register_keras_serializable(package="keystroke_auth")
class L2NormalizeLayer(tf.keras.layers.Layer):
    """Сериализуемый слой L2-нормализации embedding-векторов.

    Слой заменяет Keras `Lambda`, потому что сохранённые Lambda-слои
    без явного `output_shape` могут не загружаться в Keras 3.
    """

    def __init__(self, axis: int = 1, **kwargs: Any):
        super().__init__(**kwargs)
        self.axis = int(axis)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """Вернуть L2-нормализованный batch embedding-векторов."""
        return tf.math.l2_normalize(inputs, axis=self.axis)

    def compute_output_shape(self, input_shape: tuple[int | None, ...]) -> tuple[int | None, ...]:
        """Форма выхода совпадает с формой входа."""
        return input_shape

    def get_config(self) -> dict[str, Any]:
        """Вернуть параметры слоя для сериализации Keras."""
        config = super().get_config()
        config.update({"axis": self.axis})
        return config


@tf.keras.utils.register_keras_serializable(package="keystroke_auth")
class EuclideanDistanceLayer(tf.keras.layers.Layer):
    """Сериализуемый слой Euclidean distance для Siamese-модели."""

    def call(self, inputs: list[tf.Tensor]) -> tf.Tensor:
        """Рассчитать расстояние между двумя batch-ами embedding-векторов."""
        if len(inputs) != 2:
            raise ValueError("EuclideanDistanceLayer expects exactly two tensors.")

        embedding_a, embedding_b = inputs
        squared_difference = tf.square(embedding_a - embedding_b)
        squared_distance = tf.reduce_sum(squared_difference, axis=1, keepdims=True)
        return tf.sqrt(tf.maximum(squared_distance, tf.keras.backend.epsilon()))

    def compute_output_shape(
        self,
        input_shape: list[tuple[int | None, ...]],
    ) -> tuple[int | None, int]:
        """Вернуть форму расстояний `(batch, 1)`."""
        if len(input_shape) != 2:
            raise ValueError("EuclideanDistanceLayer expects exactly two input shapes.")
        return (input_shape[0][0], 1)


@tf.keras.utils.register_keras_serializable(package="keystroke_auth")
class ContrastiveLoss(tf.keras.losses.Loss):
    """Contrastive loss для Siamese metric-learning.

    Формула использует соглашение проекта:
    `y_true = 1` для genuine-пар и `y_true = 0` для impostor-пар.

    Args:
        margin: Минимально желательное расстояние между impostor-парами.
        name: Имя loss-функции для Keras.
        kwargs: Дополнительные параметры базового класса Keras Loss.
    """

    def __init__(self, margin: float = 1.0, name: str = "contrastive_loss", **kwargs: Any):
        super().__init__(name=name, **kwargs)
        if margin <= 0:
            raise ValueError("margin must be positive.")
        self.margin = float(margin)

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """Рассчитать contrastive loss для batch-а расстояний."""
        labels = tf.cast(tf.reshape(y_true, (-1, 1)), y_pred.dtype)
        distances = tf.cast(tf.reshape(y_pred, (-1, 1)), y_pred.dtype)

        genuine_loss = labels * tf.square(distances)
        impostor_margin = tf.maximum(self.margin - distances, 0.0)
        impostor_loss = (1.0 - labels) * tf.square(impostor_margin)

        return tf.reduce_mean(genuine_loss + impostor_loss)

    def get_config(self) -> dict[str, Any]:
        """Вернуть параметры для сериализации Keras."""
        config = super().get_config()
        config.update({"margin": self.margin})
        return config


@tf.keras.utils.register_keras_serializable(package="keystroke_auth")
class ContrastiveAccuracy(tf.keras.metrics.Metric):
    """Accuracy для расстояний Siamese-модели.

    Если расстояние не превышает threshold, пара считается genuine.
    Иначе пара считается impostor.
    """

    def __init__(
        self,
        threshold: float = 0.5,
        name: str = "contrastive_accuracy",
        **kwargs: Any,
    ):
        super().__init__(name=name, **kwargs)
        if threshold <= 0:
            raise ValueError("threshold must be positive.")
        self.threshold = float(threshold)
        self.total = self.add_weight(name="total", initializer="zeros")
        self.count = self.add_weight(name="count", initializer="zeros")

    def update_state(
        self,
        y_true: tf.Tensor,
        y_pred: tf.Tensor,
        sample_weight: tf.Tensor | None = None,
    ) -> None:
        """Обновить накопленную метрику для текущего batch-а."""
        labels = tf.cast(tf.reshape(y_true, (-1,)), tf.float32)
        distances = tf.cast(tf.reshape(y_pred, (-1,)), tf.float32)
        predicted_genuine = tf.cast(distances <= self.threshold, tf.float32)
        matches = tf.cast(tf.equal(predicted_genuine, labels), tf.float32)

        if sample_weight is not None:
            weights = tf.cast(tf.reshape(sample_weight, (-1,)), tf.float32)
            matches *= weights
            batch_count = tf.reduce_sum(weights)
        else:
            batch_count = tf.cast(tf.size(matches), tf.float32)

        self.total.assign_add(tf.reduce_sum(matches))
        self.count.assign_add(batch_count)

    def result(self) -> tf.Tensor:
        """Вернуть среднюю accuracy по накопленным batch-ам."""
        return tf.math.divide_no_nan(self.total, self.count)

    def reset_state(self) -> None:
        """Сбросить накопленные значения метрики."""
        self.total.assign(0.0)
        self.count.assign(0.0)

    def get_config(self) -> dict[str, Any]:
        """Вернуть параметры для сериализации Keras."""
        config = super().get_config()
        config.update({"threshold": self.threshold})
        return config


def build_siamese_encoder(
    input_dim: int,
    embedding_dim: int = 32,
    hidden_units: int = 128,
    dropout_rate: float = 0.2,
    embedding_activation: str | None = None,
    l2_normalize: bool = True,
) -> tf.keras.Model:
    """Создать shared encoder для Siamese-модели.

    Args:
        input_dim: Размерность входного вектора признаков.
        embedding_dim: Размерность итогового embedding-вектора.
        hidden_units: Число нейронов скрытого Dense-слоя.
        dropout_rate: Доля dropout после первого BatchNormalization.
        embedding_activation: Активация embedding-слоя. По умолчанию linear.
        l2_normalize: Нормализовать ли embedding к единичной L2-норме.

    Returns:
        Keras encoder-модель `feature_vector -> embedding_vector`.

    Raises:
        ValueError: Если параметры размерности некорректны.
    """
    if input_dim <= 0:
        raise ValueError("input_dim must be positive.")
    if embedding_dim <= 0:
        raise ValueError("embedding_dim must be positive.")
    if hidden_units <= 0:
        raise ValueError("hidden_units must be positive.")
    if not 0.0 <= dropout_rate < 1.0:
        raise ValueError("dropout_rate must be in [0.0, 1.0).")

    inputs = tf.keras.Input(shape=(input_dim,), name="feature_vector")
    x = tf.keras.layers.Dense(hidden_units, activation="relu", name="dense_128")(inputs)
    x = tf.keras.layers.BatchNormalization(name="bn_dense_128")(x)
    x = tf.keras.layers.Dropout(dropout_rate, name="dropout_128")(x)
    x = tf.keras.layers.Dense(
        embedding_dim,
        activation=embedding_activation,
        name="embedding",
    )(x)
    x = tf.keras.layers.BatchNormalization(name="bn_embedding")(x)

    if l2_normalize:
        outputs = L2NormalizeLayer(name="l2_normalized_embedding")(x)
    else:
        outputs = tf.keras.layers.Activation("linear", name="embedding_output")(x)

    return tf.keras.Model(inputs=inputs, outputs=outputs, name="siamese_encoder")


def euclidean_distance(tensors: list[tf.Tensor]) -> tf.Tensor:
    """Рассчитать Euclidean distance между двумя batch-ами embedding-векторов.

    Функция оставлена для обратной совместимости с ранними тестами и импортами.
    Для сериализуемой модели используется `EuclideanDistanceLayer`.
    """
    if len(tensors) != 2:
        raise ValueError("euclidean_distance expects exactly two tensors.")

    embedding_a, embedding_b = tensors
    squared_difference = tf.square(embedding_a - embedding_b)
    squared_distance = tf.reduce_sum(squared_difference, axis=1, keepdims=True)
    return tf.sqrt(tf.maximum(squared_distance, tf.keras.backend.epsilon()))


def build_siamese_model(
    input_dim: int,
    embedding_dim: int = 32,
    hidden_units: int = 128,
    dropout_rate: float = 0.2,
    margin: float = 1.0,
    learning_rate: float = 1e-3,
    embedding_activation: str | None = None,
    l2_normalize: bool = True,
) -> tuple[tf.keras.Model, tf.keras.Model]:
    """Создать и скомпилировать Siamese-модель.

    Args:
        input_dim: Размерность входных признаков.
        embedding_dim: Размерность embedding-вектора.
        hidden_units: Число нейронов скрытого слоя encoder-а.
        dropout_rate: Dropout encoder-а.
        margin: Margin для contrastive loss.
        learning_rate: Learning rate Adam optimizer-а.
        embedding_activation: Активация embedding-слоя.
        l2_normalize: Нормализовать ли embedding-векторы.

    Returns:
        Кортеж `(siamese_model, encoder)`.
    """
    encoder = build_siamese_encoder(
        input_dim=input_dim,
        embedding_dim=embedding_dim,
        hidden_units=hidden_units,
        dropout_rate=dropout_rate,
        embedding_activation=embedding_activation,
        l2_normalize=l2_normalize,
    )

    input_a = tf.keras.Input(shape=(input_dim,), name="sample_a")
    input_b = tf.keras.Input(shape=(input_dim,), name="sample_b")

    embedding_a = encoder(input_a)
    embedding_b = encoder(input_b)
    distance = EuclideanDistanceLayer(name="euclidean_distance")([embedding_a, embedding_b])

    model = tf.keras.Model(inputs=[input_a, input_b], outputs=distance, name="siamese_model")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=ContrastiveLoss(margin=margin),
        metrics=[ContrastiveAccuracy(threshold=margin / 2.0)],
    )

    return model, encoder
