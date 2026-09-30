from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT_DIR / "dataset"
MODELS_DIR = ROOT_DIR / "models"
RESULTS_DIR = ROOT_DIR / "results"

IMG_SIZE = 224
BATCH_SIZE = 32
SEED = 42
SPLITS = ("train", "validation", "test")
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")
MIN_IMAGES_PER_CLASS = 5

# "classes" = the ALLOWED class-folder names. A dataset may use a subset (at least 2).
# "strict" = folders with other names are rejected (the quality rules depend on
# these names). Fruit is not strict, so extra fruit folders can be added later.
TASKS = {
    "fruit": {
        "title": "Fruit type",
        "classes": ["apple", "banana", "mango", "orange"],
        "strict": False,
        "model_file": "fruit_model.keras",
    },
    "ripeness": {
        "title": "Ripeness",
        "classes": ["overripe", "ripe", "unripe"],
        "strict": True,
        "model_file": "ripeness_model.keras",
    },
    "defect": {
        "title": "Defect",
        "classes": ["bruised", "healthy", "rotten", "spotted"],
        "strict": True,
        "model_file": "defect_model.keras",
    },
}


def model_path(task):
    return MODELS_DIR / TASKS[task]["model_file"]


def classes_path(task):
    return MODELS_DIR / f"{task}_classes.json"
