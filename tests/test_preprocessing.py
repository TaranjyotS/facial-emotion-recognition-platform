import numpy as np
import pytest

from app.preprocessing import preprocess_image_array


def test_preprocess_image_array_returns_model_shape():
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    result = preprocess_image_array(image)
    assert result.shape == (1, 48, 48, 1)
    assert result.dtype == np.float32


def test_preprocess_image_array_rejects_empty_input():
    with pytest.raises(ValueError):
        preprocess_image_array(np.array([]))
