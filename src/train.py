"""Train the facial emotion recognition model."""

from __future__ import annotations

import argparse
from pathlib import Path

from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from app.config import get_settings
from src.dataset import load_emotion_csv
from src.evaluate import save_classification_report, save_training_history
from src.model import build_cnn_model


def train(
    dataset_path: Path, model_output: Path, report_dir: Path, epochs: int, batch_size: int
) -> None:
    """Run training, evaluation, and artifact export."""
    settings = get_settings()
    x, y, num_classes = load_emotion_csv(dataset_path, image_size=settings.image_size)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42, stratify=y.argmax(axis=1)
    )

    datagen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
    )

    model = build_cnn_model(num_classes=num_classes)
    model_output.parent.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        ModelCheckpoint(filepath=str(model_output), monitor="val_loss", save_best_only=True),
    ]

    history = model.fit(
        datagen.flow(x_train, y_train, batch_size=batch_size),
        validation_data=(x_test, y_test),
        epochs=epochs,
        callbacks=callbacks,
    )

    save_training_history(history, report_dir)
    save_classification_report(model, x_test, y_test, settings.emotion_labels, report_dir)
    model.save(model_output)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train facial emotion recognition CNN.")
    parser.add_argument("--dataset", type=Path, default=Path("dataset/cohn-kanade-dataset.csv"))
    parser.add_argument(
        "--model-output", type=Path, default=Path("artifacts/models/emotion_detection_model.keras")
    )
    parser.add_argument("--report-dir", type=Path, default=Path("reports"))
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=32)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args.dataset, args.model_output, args.report_dir, args.epochs, args.batch_size)
