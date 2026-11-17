import csv
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score
from threadpoolctl import threadpool_limits

from model5_scaled_mlp import build_model
from utils import load_data


DATASET_KEYS = ("20", "30", "40")
TRAIN_FRACTIONS = (0.25, 0.5, 0.75, 1.0)
SEED = 42


def stratified_subset_indices(labels, fraction, seed):
    rng = np.random.RandomState(seed)
    selected = []
    for label in np.unique(labels):
        class_indices = np.flatnonzero(labels == label)
        rng.shuffle(class_indices)
        count = max(1, int(round(len(class_indices) * fraction)))
        selected.extend(class_indices[:count])
    return rng.permutation(selected)


def main():
    repository_root = Path(__file__).resolve().parents[1]
    os.chdir(repository_root)
    output_dir = Path(__file__).resolve().parent / "results"
    output_dir.mkdir(exist_ok=True)
    rows = []

    with threadpool_limits(limits=1):
        for key in DATASET_KEYS:
            _, _, X_train, X_val, y_train, y_val = load_data(
                key, random_state=SEED
            )
            for fraction in TRAIN_FRACTIONS:
                indices = stratified_subset_indices(y_train, fraction, SEED)
                model = build_model(SEED)
                model.fit(X_train[indices], y_train[indices])
                accuracy = accuracy_score(y_val, model.predict(X_val))
                rows.append(
                    {
                        "dataset": key,
                        "training_fraction": fraction,
                        "training_examples": len(indices),
                        "validation_accuracy": accuracy,
                    }
                )
                print(
                    f"Kryptonite-{key}, {fraction:.0%} of training split "
                    f"({len(indices)} rows): {accuracy * 100:.2f}%"
                )

    csv_path = output_dir / "mlp_learning_curve.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    figure, axes = plt.subplots(1, len(DATASET_KEYS), figsize=(12, 4), sharey=True)
    for axis, key in zip(axes, DATASET_KEYS):
        dataset_rows = [row for row in rows if row["dataset"] == key]
        axis.plot(
            [row["training_examples"] for row in dataset_rows],
            [row["validation_accuracy"] * 100 for row in dataset_rows],
            marker="o",
        )
        axis.axhline(90, color="gray", linestyle="--", linewidth=1)
        axis.set_title(f"Kryptonite-{key}")
        axis.set_xlabel("Training examples")
        axis.grid(True, alpha=0.3)
    axes[0].set_ylabel("Validation accuracy (%)")
    figure.tight_layout()
    figure_path = output_dir / "mlp_learning_curve.png"
    figure.savefig(figure_path, dpi=200)
    plt.close(figure)
    print(f"Saved {csv_path} and {figure_path}")


if __name__ == "__main__":
    main()
