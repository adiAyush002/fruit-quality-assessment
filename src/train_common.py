import argparse
import json
import sys

from .config import IMG_SIZE, MODELS_DIR, SEED, classes_path, model_path
from .evaluate import evaluate_model
from .preprocessing import DatasetError, load_datasets


def build_model(num_classes, dropout=0.3, lr=1e-3):
    from tensorflow import keras
    from tensorflow.keras import layers

    base = keras.applications.MobileNetV2(input_shape=(IMG_SIZE, IMG_SIZE, 3),
                                          include_top=False, weights="imagenet")
    base.trainable = False

    inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = layers.Rescaling(1 / 127.5, offset=-1)(inputs)  # pixels 0-255 -> -1..1 (MobileNetV2 input range)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs, outputs)
    model.compile(optimizer=keras.optimizers.Adam(lr),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model, base


def unfreeze_top_layers(model, base, n_layers=30, lr=1e-5):
    from tensorflow import keras
    base.trainable = True
    for layer in base.layers[:-n_layers]:
        layer.trainable = False
    model.compile(optimizer=keras.optimizers.Adam(lr),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])


def make_callbacks(checkpoint_path):
    from tensorflow import keras
    return [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True),
        keras.callbacks.ModelCheckpoint(str(checkpoint_path), monitor="val_loss", save_best_only=True),
    ]


def train_task(task, epochs=15, fine_tune_epochs=10):
    from tensorflow import keras

    keras.utils.set_random_seed(SEED)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    train_ds, val_ds, test_ds, class_names, class_weight = load_datasets(task)
    print(f"\nTask: {task} | classes: {class_names} | class weights: {class_weight}")

    best_path = model_path(task)
    model, base = build_model(len(class_names))

    print("\n--- Phase 1: training the new classification head (base frozen) ---")
    h1 = model.fit(train_ds, validation_data=val_ds, epochs=epochs,
                   class_weight=class_weight, callbacks=make_callbacks(best_path))
    history = {k: list(v) for k, v in h1.history.items()}
    best_val_loss = min(h1.history["val_loss"])

    if fine_tune_epochs > 0:
        print("\n--- Phase 2: fine-tuning the top MobileNetV2 layers (low learning rate) ---")
        tmp_path = MODELS_DIR / f"{task}_finetune_tmp.keras"
        unfreeze_top_layers(model, base)
        h2 = model.fit(train_ds, validation_data=val_ds, epochs=fine_tune_epochs,
                       class_weight=class_weight, callbacks=make_callbacks(tmp_path))
        for k, v in h2.history.items():
            history[k] += list(v)
        if min(h2.history["val_loss"]) < best_val_loss and tmp_path.exists():
            tmp_path.replace(best_path)
        else:
            tmp_path.unlink(missing_ok=True)

    classes_path(task).write_text(json.dumps(class_names))
    print(f"\nBest model saved to: {best_path}")
    best_model = keras.models.load_model(best_path)
    return evaluate_model(task, best_model, test_ds, class_names, history)


def main(task):
    parser = argparse.ArgumentParser(description=f"Train the {task} model")
    parser.add_argument("--epochs", type=int, default=15, help="epochs for phase 1 (default 15)")
    parser.add_argument("--fine-tune-epochs", type=int, default=10,
                        help="epochs for phase 2 fine-tuning; 0 to skip (default 10)")
    args = parser.parse_args()
    try:
        train_task(task, args.epochs, args.fine_tune_epochs)
    except DatasetError as e:
        print(f"\nDATASET ERROR:\n{e}")
        sys.exit(1)
