"""Model definitions for Keystroke Auth.

This module contains the baseline MLP classifier architecture.

Current step:
- build a simple MLP baseline for user classification;
- compile the model;
- print model summary from CLI.
"""

from __future__ import annotations

import argparse

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


DEFAULT_INPUT_DIM = 31
DEFAULT_NUM_CLASSES = 51
DEFAULT_HIDDEN_UNITS = 64
DEFAULT_DROPOUT_RATE = 0.2
DEFAULT_LEARNING_RATE = 0.001


def build_mlp_baseline(
    input_dim: int,
    num_classes: int,
    hidden_units: int = DEFAULT_HIDDEN_UNITS,
    dropout_rate: float = DEFAULT_DROPOUT_RATE,
) -> keras.Model:
    """Build baseline MLP classifier.

    Architecture:
    - input vector with timing features;
    - Dense(hidden_units, ReLU);
    - Dropout(dropout_rate);
    - Dense(num_classes, Softmax).

    Args:
        input_dim: Number of input features.
        num_classes: Number of user classes.
        hidden_units: Number of neurons in hidden dense layer.
        dropout_rate: Dropout probability.

    Returns:
        Uncompiled Keras model.

    Raises:
        ValueError: If model parameters are invalid.
    """
    validate_model_parameters(
        input_dim=input_dim,
        num_classes=num_classes,
        hidden_units=hidden_units,
        dropout_rate=dropout_rate,
    )

    model = keras.Sequential(
        [
            keras.Input(shape=(input_dim,), name="keystroke_features"),
            layers.Dense(hidden_units, activation="relu", name="dense_1"),
            layers.Dropout(dropout_rate, name="dropout_1"),
            layers.Dense(num_classes, activation="softmax", name="user_classifier"),
        ],
        name="mlp_baseline",
    )

    return model


def compile_mlp_model(
    model: keras.Model,
    learning_rate: float = DEFAULT_LEARNING_RATE,
) -> keras.Model:
    """Compile MLP classifier.

    Args:
        model: Keras model to compile.
        learning_rate: Adam optimizer learning rate.

    Returns:
        Compiled Keras model.

    Raises:
        ValueError: If learning rate is invalid.
    """
    if learning_rate <= 0:
        raise ValueError(f"learning_rate must be positive, got: {learning_rate}")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def build_compiled_mlp_baseline(
    input_dim: int,
    num_classes: int,
    hidden_units: int = DEFAULT_HIDDEN_UNITS,
    dropout_rate: float = DEFAULT_DROPOUT_RATE,
    learning_rate: float = DEFAULT_LEARNING_RATE,
) -> keras.Model:
    """Build and compile baseline MLP classifier.

    Args:
        input_dim: Number of input features.
        num_classes: Number of user classes.
        hidden_units: Number of neurons in hidden dense layer.
        dropout_rate: Dropout probability.
        learning_rate: Adam optimizer learning rate.

    Returns:
        Compiled Keras model.
    """
    model = build_mlp_baseline(
        input_dim=input_dim,
        num_classes=num_classes,
        hidden_units=hidden_units,
        dropout_rate=dropout_rate,
    )
    return compile_mlp_model(model=model, learning_rate=learning_rate)


def validate_model_parameters(
    input_dim: int,
    num_classes: int,
    hidden_units: int,
    dropout_rate: float,
) -> None:
    """Validate MLP model parameters.

    Args:
        input_dim: Number of input features.
        num_classes: Number of user classes.
        hidden_units: Number of neurons in hidden dense layer.
        dropout_rate: Dropout probability.

    Raises:
        ValueError: If any parameter is invalid.
    """
    if input_dim <= 0:
        raise ValueError(f"input_dim must be positive, got: {input_dim}")

    if num_classes < 2:
        raise ValueError(f"num_classes must be at least 2, got: {num_classes}")

    if hidden_units <= 0:
        raise ValueError(f"hidden_units must be positive, got: {hidden_units}")

    if not 0 <= dropout_rate < 1:
        raise ValueError(f"dropout_rate must be in [0, 1), got: {dropout_rate}")


def get_model_summary_lines(model: keras.Model) -> list[str]:
    """Collect Keras model summary as a list of strings.

    Args:
        model: Keras model.

    Returns:
        Model summary lines.
    """
    lines: list[str] = []
    model.summary(print_fn=lines.append)
    return lines


def build_arg_parser() -> argparse.ArgumentParser:
    """Build command-line argument parser.

    Returns:
        Configured ArgumentParser.
    """
    parser = argparse.ArgumentParser(description="Build Keystroke Auth MLP baseline.")

    parser.add_argument(
        "--input-dim",
        type=int,
        default=DEFAULT_INPUT_DIM,
        help="Number of input features.",
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=DEFAULT_NUM_CLASSES,
        help="Number of user classes.",
    )
    parser.add_argument(
        "--hidden-units",
        type=int,
        default=DEFAULT_HIDDEN_UNITS,
        help="Number of neurons in the hidden dense layer.",
    )
    parser.add_argument(
        "--dropout-rate",
        type=float,
        default=DEFAULT_DROPOUT_RATE,
        help="Dropout probability.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=DEFAULT_LEARNING_RATE,
        help="Adam optimizer learning rate.",
    )

    return parser


def main() -> None:
    """CLI entry point."""
    parser = build_arg_parser()
    args = parser.parse_args()

    tf.keras.utils.set_random_seed(42)

    model = build_compiled_mlp_baseline(
        input_dim=args.input_dim,
        num_classes=args.num_classes,
        hidden_units=args.hidden_units,
        dropout_rate=args.dropout_rate,
        learning_rate=args.learning_rate,
    )

    print("MLP baseline built successfully.")
    print(f"Input features: {args.input_dim}")
    print(f"User classes: {args.num_classes}")
    print(f"Hidden units: {args.hidden_units}")
    print(f"Dropout rate: {args.dropout_rate}")
    print(f"Learning rate: {args.learning_rate}")
    print()
    print("Model summary:")
    for line in get_model_summary_lines(model):
        print(line)


if __name__ == "__main__":
    main()
