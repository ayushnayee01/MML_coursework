import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from utils import load_data

DATASET_KEYS = ("20", "30", "40")
SPLIT_SEEDS = (42, 123, 2026)
HIDDEN_LAYER_SIZES = (512, 256, 128)
ALPHA = 0.0001
LEARNING_RATE_INIT = 0.001


def build_model(
    seed,
    *,
    max_iter=400,
    early_stopping=True,
    n_iter_no_change=20,
    tol=0.0001,
):
    return make_pipeline(
        StandardScaler(),
        MLPClassifier(
            hidden_layer_sizes=HIDDEN_LAYER_SIZES,
            max_iter=max_iter,
            early_stopping=early_stopping,
            n_iter_no_change=n_iter_no_change,
            tol=tol,
            alpha=ALPHA,
            learning_rate_init=LEARNING_RATE_INIT,
            random_state=seed,
        ),
    )


def main():
    print("=== MODEL 5: TUNED SCALED MLP (REPEATED STRATIFIED SPLITS) ===")
    with threadpool_limits(limits=1):
        for key in DATASET_KEYS:
            accuracies = []
            epochs = []
            for seed in SPLIT_SEEDS:
                _, _, X_train, X_val, y_train, y_val = load_data(
                    key, random_state=seed
                )
                model = build_model(seed)
                model.fit(X_train, y_train)
                acc = accuracy_score(y_val, model.predict(X_val))
                accuracies.append(acc)
                epochs.append(model[-1].n_iter_)
                print(
                    f"  Kryptonite-{key} seed={seed}: "
                    f"{acc * 100:.2f}% ({model[-1].n_iter_} epochs)"
                )

            print(
                f"  Kryptonite-{key} mean: "
                f"{np.mean(accuracies) * 100:.2f}% "
                f"+/- {np.std(accuracies, ddof=1) * 100:.2f}% "
                f"(sample SD; mean epochs={np.mean(epochs):.1f})"
            )

if __name__ == "__main__":
    main()
