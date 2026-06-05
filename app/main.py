"""FastAPI application exposing facial emotion recognition inference."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, HTTPException, UploadFile

from app.config import get_settings
from app.inference import EmotionRecognitionService
from app.preprocessing import validate_image_path

settings = get_settings()
app = FastAPI(title=settings.app_name, version="1.0.0")
service = EmotionRecognitionService()


@app.get("/health")
def health() -> dict[str, str]:
    """Health endpoint used by CI/CD, Docker, and monitoring checks."""
    return {"status": "ok", "service": settings.app_name}


@app.post("/predict")
def predict(
    file: Annotated[UploadFile, File(description="PNG, JPG, or JPEG image")],
) -> dict[str, object]:
    """Predict facial emotion from an uploaded image."""
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in settings.allowed_extensions:
        allowed = ", ".join(sorted(settings.allowed_extensions))
        raise HTTPException(status_code=400, detail=f"Only these files are supported: {allowed}.")

    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            shutil.copyfileobj(file.file, tmp)
            temp_path = Path(tmp.name)

        validate_image_path(temp_path, settings.allowed_extensions)
        result = service.predict_file(temp_path)
        return {
            "emotion": result.emotion,
            "confidence": result.confidence,
            "probabilities": result.probabilities,
        }
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
