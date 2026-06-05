"""Dataset loading utilities for Cohn-Kanade-style CSV data."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from tensorflow.keras.utils import to_categorical


def load_emotion_csv(
    csv_path: str | Path,
    image_size: tuple[int, int] = (48, 48),
) -> tuple[np.ndarray, np.ndarray, int]:
    """Load pixel/emotion CSV into normalized tensors and one-hot labels."""
    data = pd.read_csv(csv_path)
    required_columns = {"pixels", "emotion"}
    missing = required_columns.difference(data.columns)
    if missing:
        raise ValueError(f"Dataset missing required columns: {sorted(missing)}")

    pixels = data["pixels"].apply(lambda value: np.array(str(value).split(), dtype="float32"))
    expected_pixels = image_size[0] * image_size[1]
    if any(pixel.size != expected_pixels for pixel in pixels):
        raise ValueError(f"Each image must contain {expected_pixels} pixels.")

    x = np.array([pixel.reshape(image_size[0], image_size[1], 1) for pixel in pixels]) / 255.0
    emotions = data["emotion"].astype(int).values
    num_classes = int(max(emotions)) + 1
    y = to_categorical(emotions, num_classes=num_classes)
    return x, y, num_classes
