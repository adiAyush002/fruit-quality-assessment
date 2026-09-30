from pathlib import Path

import numpy as np

from .config import (BATCH_SIZE, DATASET_DIR, IMAGE_EXTENSIONS, IMG_SIZE,
                     MIN_IMAGES_PER_CLASS, SEED, SPLITS, TASKS)


class DatasetError(Exception):
    """Raised when the dataset folder is missing or wrongly organised."""


def images_in(folder):
    return sorted(p for p in Path(folder).rglob("*")
                  if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)


def validate_dataset(task):
    if task not in TASKS:
        raise DatasetError(f"Unknown task '{task}'. Choose from: {list(TASKS)}")
    cfg = TASKS[task]
    task_dir = DATASET_DIR / task
    layout = f"dataset/{task}/<train|validation|test>/<class_name>/image.jpg"
    if not task_dir.is_dir():
        raise DatasetError(f"Dataset folder not found: {task_dir}\nExpected layout: {layout}")

    train_classes = None
    for split in SPLITS:
        split_dir = task_dir / split
        if not split_dir.is_dir():
            raise DatasetError(f"Missing folder: {split_dir}\nExpected layout: {layout}")
        classes = sorted(p.name for p in split_dir.iterdir() if p.is_dir())
        if train_classes is None:
            train_classes = classes
            unknown = set(classes) - set(cfg["classes"])
            if len(classes) < 2:
                raise DatasetError(f"{split_dir} needs at least 2 class folders; found {classes}. "
                                   f"Allowed names: {cfg['classes']}")
            if unknown and cfg["strict"]:
                raise DatasetError(f"{split_dir} has unexpected class folders: {sorted(unknown)}. "
                                   f"Allowed names: {cfg['classes']}")
        elif classes != train_classes:
            raise DatasetError(f"Class folders in '{split}' {classes} differ from 'train' {train_classes}.")
        for c in classes:
            n = len(images_in(split_dir / c))
            if n < MIN_IMAGES_PER_CLASS:
                raise DatasetError(f"{split_dir / c} has only {n} usable images "
                                   f"(minimum {MIN_IMAGES_PER_CLASS}; formats: {IMAGE_EXTENSIONS}).")
    return train_classes


def build_augmentation():
    from tensorflow import keras
    from tensorflow.keras import layers

    # No hue/saturation changes: colour is the main cue for ripeness and defects.
    return keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.08),
        layers.RandomZoom(0.15),
        layers.RandomBrightness(0.15, value_range=(0, 255)),
        layers.RandomContrast(0.15),
    ], name="augmentation")


def load_datasets(task, batch_size=BATCH_SIZE):
    import tensorflow as tf
    from tensorflow import keras
    from sklearn.utils.class_weight import compute_class_weight

    class_names = validate_dataset(task)
    task_dir = DATASET_DIR / task

    def _load(split, shuffle):
        return keras.utils.image_dataset_from_directory(
            task_dir / split, label_mode="int", class_names=class_names,
            image_size=(IMG_SIZE, IMG_SIZE), batch_size=batch_size,
            shuffle=shuffle, seed=SEED)

    train_raw = _load("train", True)
    val_ds = _load("validation", False)
    test_ds = _load("test", False)

    labels = [class_names.index(Path(p).relative_to(task_dir / "train").parts[0])
              for p in train_raw.file_paths]
    weights = compute_class_weight("balanced", classes=np.arange(len(class_names)), y=labels)
    class_weight = {i: float(w) for i, w in enumerate(weights)}

    augment = build_augmentation()
    auto = tf.data.AUTOTUNE
    train_ds = train_raw.map(lambda x, y: (augment(x, training=True), y),
                             num_parallel_calls=auto).prefetch(auto)
    return train_ds, val_ds.prefetch(auto), test_ds.prefetch(auto), class_names, class_weight
