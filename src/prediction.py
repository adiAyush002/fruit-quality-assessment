import json
from pathlib import Path

import cv2
import numpy as np

from .config import IMG_SIZE, TASKS, classes_path, model_path

SUPPORTED_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")


class PredictionError(Exception):
    """Base class for errors that should be shown to the user."""


class ModelNotFoundError(PredictionError):
    pass


class InvalidImageError(PredictionError):
    pass


def load_models():
    missing = []
    for task in TASKS:
        for path in (model_path(task), classes_path(task)):
            if not path.exists():
                missing.append(str(path))
    if missing:
        raise ModelNotFoundError("Trained model files not found:\n- " + "\n- ".join(missing))

    from tensorflow import keras
    models = {}
    for task in TASKS:
        try:
            model = keras.models.load_model(model_path(task))
        except Exception as e:
            raise ModelNotFoundError(f"Could not load {model_path(task).name}: {e}")
        models[task] = (model, json.loads(classes_path(task).read_text()))
    return models


def decode_image(data, filename="image"):
    ext = Path(filename).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise InvalidImageError(f"Unsupported file format '{ext or 'unknown'}'. "
                                f"Please upload: {', '.join(SUPPORTED_EXTENSIONS)}")
    if not data:
        raise InvalidImageError("The uploaded file is empty.")
    bgr = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        raise InvalidImageError("The file could not be read as an image. It may be corrupted.")
    if min(bgr.shape[:2]) < 32:
        raise InvalidImageError("The image is too small (minimum 32x32 pixels).")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def preprocess(rgb):
    resized = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
    return np.expand_dims(resized.astype("float32"), axis=0)  # 0-255; the model rescales internally


def predict_image(rgb, models):
    batch = preprocess(rgb)
    results = {}
    for task, (model, class_names) in models.items():
        probs = model.predict(batch, verbose=0)[0]
        idx = int(np.argmax(probs))
        results[task] = {
            "label": class_names[idx],
            "confidence": float(probs[idx]),
            "probabilities": {c: float(p) for c, p in zip(class_names, probs)},
        }
    return results
