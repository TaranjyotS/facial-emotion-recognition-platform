"""Application configuration."""

from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel, Field


class Settings(BaseModel):
    """Runtime settings for model inference."""

    app_name: str = "Facial Emotion Recognition Platform"
    model_path: Path = Field(default=Path("artifacts/models/emotion_detection_model.keras"))
    image_size: tuple[int, int] = (48, 48)
    allowed_extensions: set[str] = {".jpg", ".jpeg", ".png"}
    emotion_labels: list[str] = [
        "Anger",
        "Contempt",
        "Disgust",
        "Fear",
        "Happiness",
        "Sadness",
        "Surprise",
        "Neutral",
    ]


@lru_cache
def get_settings() -> Settings:
    """Return cached settings so the model service uses one consistent config."""
    return Settings()
