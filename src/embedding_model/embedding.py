"""Модуль embedding-модели для этапа 6 проекта Keystroke Auth.

Файл размещается в каталоге исходного кода:

    src/embedding_model/embedding.py

Артефакты этапа 6 сохраняются не рядом с исходным кодом, а в уже
существующих каталогах проекта, внутри отдельного подкаталога
``embedding_model``:

    models/embedding_model/
    users/embedding_model/
    reports/embedding_model/
    data/processed/embedding_model/

Модель обучается как многоклассовый классификатор пользователей.
Промежуточный слой с именем ``embedding`` используется как компактное
векторное представление одной попытки ввода фиксированной фразы.

Терминология:

- embedding-вектор — компактное числовое представление образца ввода;
- encoder — часть модели от входного слоя до embedding-слоя включительно;
- embedding-шаблон — эталонный вектор заявленного пользователя;
- проверка по расстоянию — сравнение текущего embedding-вектора с шаблоном;
- genuine-попытка — проверка настоящего образца заявленного пользователя;
- impostor-попытка — проверка чужого образца против заявленного пользователя.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
import tensorflow as tf
from numpy.typing import NDArray
from tensorflow import keras
from tensorflow.keras import layers

# Размерность входного вектора для CMU Keystroke Dynamics Benchmark: 31 признак.
DEFAULT_INPUT_DIM: Final[int] = 31

# Количество пользователей в CMU benchmark: 51 пользователь.
DEFAULT_NUM_CLASSES: Final[int] = 51

# Базовая размерность embedding-вектора для первого варианта этапа 6.
DEFAULT_EMBEDDING_DIM: Final[int] = 32

# Dropout используется для снижения переобучения классификатора.
DEFAULT_DROPOUT_RATE: Final[float] = 0.2

# Базовая скорость обучения Adam для стартового эксперимента.
DEFAULT_LEARNING_RATE: Final[float] = 1e-3

# Фиксированный seed нужен для воспроизводимости smoke-проверки.
DEFAULT_RANDOM_SEED: Final[int] = 42

# Типовая аннотация для матриц признаков и embedding-векторов.
FloatArray = NDArray[np.floating]


@dataclass(frozen=True)
class EmbeddingModelPaths:
    """Описание путей артефактов embedding-этапа.

    Исходный код находится в ``src/embedding_model``.

    Артефакты сохраняются в корневых каталогах проекта:

    - ``models/embedding_model`` — Keras-модели, scaler и label encoder;
    - ``users/embedding_model`` — embedding-шаблоны и пороги пользователей;
    - ``reports/embedding_model`` — отчёты, метрики и диагностические таблицы;
    - ``data/processed/embedding_model`` — извлечённые embedding-векторы.
    """

    project_root: Path
    source_dir: Path
    models_dir: Path
    users_dir: Path
    reports_dir: Path
    data_processed_dir: Path
    classifier_path: Path
    encoder_path: Path
    scaler_path: Path
    label_encoder_path: Path
    templates_path: Path
    thresholds_path: Path

    @classmethod
    def from_source_dir(cls, source_dir: Path | None = None) -> EmbeddingModelPaths:
        """Сформировать стандартные пути к артефактам этапа 6.

        Аргументы:
            source_dir: Каталог с файлом ``embedding.py``. Если путь не передан,
                он определяется автоматически через ``__file__``.

        Возвращает:
            Объект с путями к исходному коду и артефактам embedding-этапа.

        Исключения:
            RuntimeError: Возникает при нарушении ожидаемой структуры проекта.
        """
        resolved_source_dir = source_dir or Path(__file__).resolve().parent

        # Проверяем, что файл действительно расположен в src/embedding_model.
        if resolved_source_dir.name != "embedding_model":
            raise RuntimeError(
                "Ожидалось имя каталога исходного кода 'embedding_model', "
                f"получено: '{resolved_source_dir.name}'."
            )

        src_dir = resolved_source_dir.parent
        if src_dir.name != "src":
            raise RuntimeError(
                "Ожидалось размещение каталога 'embedding_model' внутри 'src', "
                f"получен родительский каталог: '{src_dir.name}'."
            )

        # Корень проекта находится на один уровень выше каталога src.
        project_root = src_dir.parent
        artifact_subdir = "embedding_model"

        # Артефакты размещаются в стандартных каталогах проекта.
        models_dir = project_root / "models" / artifact_subdir
        users_dir = project_root / "users" / artifact_subdir
        reports_dir = project_root / "reports" / artifact_subdir
        data_processed_dir = project_root / "data" / "processed" / artifact_subdir

        return cls(
            project_root=project_root,
            source_dir=resolved_source_dir,
            models_dir=models_dir,
            users_dir=users_dir,
            reports_dir=reports_dir,
            data_processed_dir=data_processed_dir,
            classifier_path=models_dir / "embedding_classifier.keras",
            encoder_path=models_dir / "encoder.keras",
            scaler_path=models_dir / "embedding_scaler.pkl",
            label_encoder_path=models_dir / "embedding_label_encoder.pkl",
            templates_path=users_dir / "user_templates_embedding.json",
            thresholds_path=users_dir / "user_thresholds_embedding.json",
        )

    def ensure_directories(self) -> None:
        """Создать каталоги артефактов, если они ещё не существуют."""
        for directory in (
            self.models_dir,
            self.users_dir,
            self.reports_dir,
            self.data_processed_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)


def build_embedding_classifier(
    input_dim: int,
    num_classes: int,
    embedding_dim: int = DEFAULT_EMBEDDING_DIM,
    dropout_rate: float = DEFAULT_DROPOUT_RATE,
    learning_rate: float = DEFAULT_LEARNING_RATE,
) -> keras.Model:
    """Создать и скомпилировать embedding-классификатор.

    Архитектура:

    1. Входной слой принимает вектор временных признаков.
    2. Первый полносвязный слой формирует скрытое представление.
    3. BatchNormalization стабилизирует обучение.
    4. Dropout снижает риск переобучения.
    5. Слой ``embedding`` формирует компактный embedding-вектор.
    6. Выходной softmax-слой классифицирует известных пользователей.

    Аргументы:
        input_dim: Количество входных временных признаков.
        num_classes: Количество известных пользователей-классов.
        embedding_dim: Размерность embedding-вектора.
        dropout_rate: Вероятность dropout после первого скрытого блока.
        learning_rate: Скорость обучения оптимизатора Adam.

    Возвращает:
        Скомпилированную Keras-модель классификатора.

    Исключения:
        ValueError: Возникает при некорректных размерностях или гиперпараметрах.
    """
    _validate_positive_int(input_dim, "input_dim")
    _validate_positive_int(num_classes, "num_classes")
    _validate_positive_int(embedding_dim, "embedding_dim")

    if not 0.0 <= dropout_rate < 1.0:
        raise ValueError("dropout_rate должен находиться в диапазоне [0.0, 1.0).")

    if learning_rate <= 0.0:
        raise ValueError("learning_rate должен быть положительным числом.")

    # Вход модели — нормализованный вектор временных признаков набора.
    inputs = keras.Input(shape=(input_dim,), name="features")

    # Первый скрытый блок наследует идею baseline v2 с нормализацией пакета.
    x = layers.Dense(128, activation="relu", name="dense_128")(inputs)
    x = layers.BatchNormalization(name="bn_128")(x)
    x = layers.Dropout(dropout_rate, name="dropout_128")(x)

    # Этот слой является целевым компактным представлением образца ввода.
    embedding = layers.Dense(
        embedding_dim,
        activation="relu",
        name="embedding",
    )(x)

    # Нормализация embedding-вектора перед softmax-классификацией.
    x = layers.BatchNormalization(name="embedding_bn")(embedding)

    # Выходной слой используется только на этапе обучения классификатора.
    outputs = layers.Dense(
        num_classes,
        activation="softmax",
        name="user_softmax",
    )(x)

    model = keras.Model(
        inputs=inputs,
        outputs=outputs,
        name="embedding_classifier",
    )

    # На этапе 6 модель сначала обучается как обычный классификатор.
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def build_encoder_from_classifier(
    classifier_model: keras.Model,
    embedding_layer_name: str = "embedding",
) -> keras.Model:
    """Извлечь encoder из embedding-классификатора.

    Encoder преобразует вектор временных признаков в embedding-вектор.
    Далее этот вектор используется для проверки по расстоянию до
    embedding-шаблона заявленного пользователя.

    Аргументы:
        classifier_model: Keras-модель с выделенным embedding-слоем.
        embedding_layer_name: Имя слоя, выход которого считается
            embedding-вектором.

    Возвращает:
        Keras-модель encoder: вектор признаков -> embedding-вектор.

    Исключения:
        ValueError: Возникает, если слой с указанным именем отсутствует.
    """
    try:
        embedding_layer = classifier_model.get_layer(embedding_layer_name)
    except ValueError as exc:
        raise ValueError(f"В классификаторе отсутствует слой '{embedding_layer_name}'.") from exc

    # Отбрасываем softmax-голову и оставляем только преобразование в embedding.
    return keras.Model(
        inputs=classifier_model.input,
        outputs=embedding_layer.output,
        name="encoder",
    )


def extract_embeddings(
    encoder: keras.Model,
    x: FloatArray,
    batch_size: int = 256,
) -> FloatArray:
    """Получить embedding-векторы для матрицы признаков.

    Аргументы:
        encoder: Keras-модель encoder.
        x: Матрица входных признаков формы
            ``(число_образцов, число_признаков)``.
        batch_size: Размер пакета при вычислении embedding-векторов.

    Возвращает:
        Матрицу embedding-векторов формы
        ``(число_образцов, размерность_embedding_вектора)``.

    Исключения:
        ValueError: Возникает, если форма матрицы признаков некорректна.
    """
    x = np.asarray(x, dtype=np.float32)

    if x.ndim != 2:
        raise ValueError(f"x должен быть двумерной матрицей, получена форма {x.shape}.")

    expected_input_dim = encoder.input_shape[-1]
    if x.shape[1] != expected_input_dim:
        raise ValueError(
            "Некорректное количество признаков: "
            f"ожидалось {expected_input_dim}, получено {x.shape[1]}."
        )

    # Encoder обрабатывает признаки пакетами и возвращает embedding-матрицу.
    embeddings = encoder.predict(x, batch_size=batch_size, verbose=0)
    embeddings = np.asarray(embeddings, dtype=np.float32)

    if embeddings.ndim != 2:
        raise ValueError(
            f"Выход encoder должен быть двумерной матрицей, получена форма {embeddings.shape}."
        )

    return embeddings


def save_classifier_and_encoder(
    classifier_model: keras.Model,
    paths: EmbeddingModelPaths | None = None,
) -> None:
    """Сохранить классификатор и извлечённый encoder.

    Файлы сохраняются в каталоге ``models/embedding_model`` относительно
    корня проекта.

    Аргументы:
        classifier_model: Обученная или необученная embedding-модель.
        paths: Необязательная конфигурация путей. Если не передана,
            используются стандартные пути этапа 6.
    """
    resolved_paths = paths or EmbeddingModelPaths.from_source_dir()
    resolved_paths.ensure_directories()

    # Encoder извлекается из того же классификатора, чтобы веса совпадали.
    encoder = build_encoder_from_classifier(classifier_model)

    classifier_model.save(resolved_paths.classifier_path)
    encoder.save(resolved_paths.encoder_path)


def load_encoder(
    encoder_path: Path | None = None,
) -> keras.Model:
    """Загрузить сохранённую Keras-модель encoder.

    Аргументы:
        encoder_path: Путь к файлу ``encoder.keras``. Если не передан,
            используется стандартный путь ``models/embedding_model/encoder.keras``.

    Возвращает:
        Загруженную Keras-модель encoder.

    Исключения:
        FileNotFoundError: Возникает, если файл encoder не найден.
    """
    path = encoder_path or EmbeddingModelPaths.from_source_dir().encoder_path

    if not path.exists():
        raise FileNotFoundError(f"Файл encoder не найден: {path}")

    return keras.models.load_model(path)


def _validate_positive_int(value: int, name: str) -> None:
    """Проверить, что значение является положительным целым числом."""
    if not isinstance(value, int):
        raise ValueError(f"{name} должен иметь тип int.")

    if value <= 0:
        raise ValueError(f"{name} должен быть положительным целым числом.")


def _demo() -> None:
    """Выполнить быструю проверку работоспособности для задачи 6.1."""
    tf.keras.utils.set_random_seed(DEFAULT_RANDOM_SEED)

    paths = EmbeddingModelPaths.from_source_dir()
    paths.ensure_directories()

    classifier = build_embedding_classifier(
        input_dim=DEFAULT_INPUT_DIM,
        num_classes=DEFAULT_NUM_CLASSES,
        embedding_dim=DEFAULT_EMBEDDING_DIM,
    )
    encoder = build_encoder_from_classifier(classifier)

    # Тестовый пакет имитирует четыре нормализованных образца ввода.
    sample_batch = np.zeros((4, DEFAULT_INPUT_DIM), dtype=np.float32)
    embeddings = extract_embeddings(encoder, sample_batch)

    print(f"Корень проекта: {paths.project_root}")
    print(f"Каталог исходного кода: {paths.source_dir}")
    print(f"Каталог моделей: {paths.models_dir}")
    print(f"Каталог пользовательских шаблонов: {paths.users_dir}")
    print(f"Каталог отчётов: {paths.reports_dir}")
    print(f"Каталог обработанных данных: {paths.data_processed_dir}")
    print(f"Путь к embedding-классификатору: {paths.classifier_path}")
    print(f"Путь к encoder: {paths.encoder_path}")
    print(f"Имя классификатора: {classifier.name}")
    print(f"Имя encoder: {encoder.name}")
    print(f"Форма входного пакета: {sample_batch.shape}")
    print(f"Форма embedding-матрицы: {embeddings.shape}")
    print(f"Размерность embedding-вектора: {embeddings.shape[1]}")


if __name__ == "__main__":
    _demo()
