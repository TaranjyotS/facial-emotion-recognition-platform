"""Model loading and prediction service."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from app.config import get_settings
from app.preprocessing import preprocess_image_file


@dataclass
class PredictionResult:
    """Structured prediction response."""

    emotion: str
    confidence: float
    probabilities: dict[str, float]


class EmotionRecognitionService:
    """Lazy-loaded TensorFlow/Keras inference service."""

    def __init__(self, model_path: Path | None = None) -> None:
        self.settings = get_settings()
        self.model_path = model_path or self.settings.model_path
        self._model: Any | None = None

    @property
    def model(self) -> Any:
        """Load model once and reuse it across requests."""
        if self._model is None:
            if not self.model_path.exists():
                raise FileNotFoundError(
                    f"Model not found at {self.model_path}. Train the model or place the .keras file there."
                )
            from tensorflow.keras.models import load_model

            self._model = load_model(self.model_path)
        return self._model

    def predict_file(self, image_path: str | Path) -> PredictionResult:
        """Predict emotion from an image file path."""
        valid_path = Path(image_path)
        image_batch = preprocess_image_file(valid_path, target_size=self.settings.image_size)
        raw_prediction = self.model.predict(image_batch, verbose=0)[0]
        probabilities = self._format_probabilities(raw_prediction)
        emotion = max(probabilities, key=probabilities.get)
        return PredictionResult(
            emotion=emotion,
            confidence=probabilities[emotion],
            probabilities=probabilities,
        )

    def _format_probabilities(self, prediction: np.ndarray) -> dict[str, float]:
        """Map model output probabilities to emotion labels."""
        labels = self.settings.emotion_labels
        if len(prediction) > len(labels):
            labels = [*labels, *[f"Class_{idx}" for idx in range(len(labels), len(prediction))]]
        return {
            label: round(float(score), 6)
            for label, score in zip(labels[: len(prediction)], prediction, strict=False)
        }
