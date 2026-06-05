"""CNN model architecture."""

from __future__ import annotations

from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, Input, MaxPooling2D
from tensorflow.keras.models import Sequential


def build_cnn_model(
    num_classes: int, input_shape: tuple[int, int, int] = (48, 48, 1)
) -> Sequential:
    """Build a compact CNN suitable for grayscale facial-expression classification."""
    model = Sequential(
        [
            Input(shape=input_shape),
            Conv2D(32, (3, 3), activation="relu"),
            MaxPooling2D((2, 2)),
            Conv2D(64, (3, 3), activation="relu"),
            MaxPooling2D((2, 2)),
            Conv2D(128, (3, 3), activation="relu"),
            MaxPooling2D((2, 2)),
            Flatten(),
            Dense(128, activation="relu"),
            Dropout(0.3),
            Dense(num_classes, activation="softmax"),
        ]
    )
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model
