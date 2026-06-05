"""Capture image samples from a webcam for manual testing."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2


def capture_images(output_dir: Path, num_images: int = 5) -> None:
    """Capture images from webcam when the user presses 's'."""
    output_dir.mkdir(parents=True, exist_ok=True)
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        raise RuntimeError("Could not access webcam.")

    saved = 0
    try:
        while saved < num_images:
            success, frame = camera.read()
            if not success:
                raise RuntimeError("Could not read frame from webcam.")

            cv2.imshow("Capture image - press 's' to save, 'q' to quit", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("s"):
                saved += 1
                cv2.imwrite(str(output_dir / f"sample_{saved}.png"), frame)
            if key == ord("q"):
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Capture webcam images for testing.")
    parser.add_argument("--output-dir", type=Path, default=Path("data/manual_samples"))
    parser.add_argument("--num-images", type=int, default=5)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    capture_images(args.output_dir, args.num_images)
