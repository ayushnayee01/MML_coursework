import os
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from model5_scaled_mlp import build_model
from utils import load_data, load_hidden_X


DATASET_OUTPUTS = {
    "20": "y_predicted_20.npy",
    "30": "y_predicted_30.npy",
    "40": "y_predicted_40.npy",
}
SEED = 42


def main():
    repository_root = Path(__file__).resolve().parents[1]
    os.chdir(repository_root)
    output_dir = repository_root / "hiddenlabels"
    output_dir.mkdir(exist_ok=True)

    with threadpool_limits(limits=1):
        for key, output_name in DATASET_OUTPUTS.items():
            X, y, *_ = load_data(key)
            # Use validation-selected epochs while fitting every available label.
            model = build_model(
                SEED,
                max_iter=50,
                early_stopping=False,
                n_iter_no_change=1000,
                tol=0.0,
            )
            model.fit(X, y)

            X_hidden = load_hidden_X(key)
            predictions = model.predict(X_hidden).astype(np.int64)
            if predictions.shape != (len(X_hidden),):
                raise ValueError(
                    f"Kryptonite-{key} produced shape {predictions.shape}; "
                    f"expected ({len(X_hidden)},)"
                )
            if not np.isin(predictions, (0, 1)).all():
                raise ValueError(f"Kryptonite-{key} predictions are not binary")

            output_path = output_dir / output_name
            np.save(output_path, predictions)
            print(
                f"Kryptonite-{key}: trained on {len(X)} labels; "
                f"saved {predictions.shape} predictions to {output_path}"
            )

    print("Kryptonite-40-Adapt was not changed; the LoRA experiment owns that output.")


if __name__ == "__main__":
    main()
