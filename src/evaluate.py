import argparse
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, precision_recall_fscore_support)

from .config import RESULTS_DIR, TASKS, model_path
from .preprocessing import DatasetError, load_datasets


def plot_history(task, history):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    for metric, title in [("accuracy", "Accuracy"), ("loss", "Loss")]:
        plt.figure(figsize=(7, 4))
        plt.plot(history[metric], label="Training")
        plt.plot(history["val_" + metric], label="Validation")
        plt.title(f"{task.capitalize()} model - Training vs Validation {title}")
        plt.xlabel("Epoch")
        plt.ylabel(title)
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / f"{task}_{metric}.png", dpi=150)
        plt.close()


def plot_confusion_matrix(task, cm, class_names):
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, cmap="Blues")
    plt.title(f"{task.capitalize()} model - Confusion Matrix (test set)")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    ticks = np.arange(len(class_names))
    plt.xticks(ticks, class_names, rotation=45, ha="right")
    plt.yticks(ticks, class_names)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j], ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / f"{task}_confusion_matrix.png", dpi=150)
    plt.close()


def evaluate_model(task, model, test_ds, class_names, history=None):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    labels = list(range(len(class_names)))
    y_true = np.concatenate([y.numpy() for _, y in test_ds])
    y_pred = model.predict(test_ds, verbose=0).argmax(axis=1)

    print(f"\n=== {task} model: test-set results ===")
    print(classification_report(y_true, y_pred, labels=labels, target_names=class_names, zero_division=0))
    report = classification_report(y_true, y_pred, labels=labels, target_names=class_names,
                                   output_dict=True, zero_division=0)
    pd.DataFrame(report).transpose().round(4).to_csv(RESULTS_DIR / f"{task}_classification_report.csv")

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plot_confusion_matrix(task, cm, class_names)
    if history is not None:
        plot_history(task, history)
        pd.DataFrame(history).to_csv(RESULTS_DIR / f"{task}_history.csv", index_label="epoch")

    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels,
                                                  average="macro", zero_division=0)
    summary = {"task": task, "test_images": int(len(y_true)),
               "accuracy": float(accuracy_score(y_true, y_pred)),
               "macro_precision": float(p), "macro_recall": float(r), "macro_f1": float(f1)}
    (RESULTS_DIR / f"{task}_metrics.json").write_text(json.dumps(summary, indent=2))
    print(f"Saved plots and metrics to {RESULTS_DIR}")
    return summary


def main():
    from tensorflow import keras
    parser = argparse.ArgumentParser(description="Evaluate a trained model on the test set")
    parser.add_argument("--task", required=True, choices=list(TASKS))
    args = parser.parse_args()
    try:
        _, _, test_ds, class_names, _ = load_datasets(args.task)
        path = model_path(args.task)
        if not path.exists():
            raise DatasetError(f"Trained model not found: {path}. Train it first.")
        evaluate_model(args.task, keras.models.load_model(path), test_ds, class_names)
    except DatasetError as e:
        print(f"\nERROR:\n{e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
