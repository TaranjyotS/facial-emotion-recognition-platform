"""Image preprocessing utilities used by training and inference."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def validate_image_path(image_path: str | Path, allowed_extensions: set[str]) -> Path:
    """Validate that an image path exists and uses a supported extension."""
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image file not found: {path}")
    if path.suffix.lower() not in allowed_extensions:
        raise ValueError(
            f"Unsupported file type '{path.suffix}'. Allowed: {sorted(allowed_extensions)}"
        )
    return path


def crop_largest_face(image: np.ndarray) -> np.ndarray:
    """Crop the largest detected face; return the original image if no face is found.

    The original academic model expects a centered facial-expression crop. Real uploaded
    images often contain background or large margins, so this lightweight OpenCV step
    makes API inference closer to the training data without changing the original CNN aim.
    """
    if image is None or image.size == 0:
        raise ValueError("Input image array is empty.")

    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(cascade_path)
    faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) == 0:
        return gray

    x, y, w, h = max(faces, key=lambda box: box[2] * box[3])
    pad_x = int(w * 0.10)
    pad_y = int(h * 0.15)
    y1 = max(y - pad_y, 0)
    y2 = min(y + h + pad_y, gray.shape[0])
    x1 = max(x - pad_x, 0)
    x2 = min(x + w + pad_x, gray.shape[1])
    return gray[y1:y2, x1:x2]


def preprocess_image_array(
    image_array: np.ndarray,
    target_size: tuple[int, int] = (48, 48),
    detect_face: bool = True,
) -> np.ndarray:
    """Convert an image array into normalized model-ready shape: (1, H, W, 1)."""
    if image_array is None or image_array.size == 0:
        raise ValueError("Input image array is empty.")

    gray = crop_largest_face(image_array) if detect_face else image_array
    if gray.ndim == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)

    resized = cv2.resize(gray, target_size, interpolation=cv2.INTER_AREA)
    equalized = cv2.equalizeHist(resized.astype("uint8"))
    normalized = equalized.astype("float32") / 255.0
    return np.expand_dims(normalized, axis=(0, -1))


def preprocess_image_file(
    image_path: str | Path, target_size: tuple[int, int] = (48, 48)
) -> np.ndarray:
    """Read an image from disk and transform it for CNN inference."""
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not read image file: {image_path}")
    return preprocess_image_array(image, target_size=target_size, detect_face=True)
