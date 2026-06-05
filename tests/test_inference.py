from pathlib import Path

import pytest

from app.inference import EmotionRecognitionService


def test_missing_model_raises_file_not_found(tmp_path):
    service = EmotionRecognitionService(model_path=Path(tmp_path / "missing.keras"))
    with pytest.raises(FileNotFoundError):
        _ = service.model
