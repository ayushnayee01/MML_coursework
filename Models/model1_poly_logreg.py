import numpy as np
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score
from utils import load_data

def main():
    print("=== MODEL 1: POLYNOMIAL (DEG=2) LOGISTIC REGRESSION ===")
    for key in ['20', '30', '40', '40-adapt']:
        _, _, X_train, X_val, y_train, y_val = load_data(key)
        sub_idx = np.random.RandomState(42).choice(len(X_train), min(len(X_train), 12000), replace=False)

        model = make_pipeline(
            StandardScaler(),
            PolynomialFeatures(degree=2, include_bias=False),
            LogisticRegression(max_iter=300)
        )
        model.fit(X_train[sub_idx], y_train[sub_idx])
        acc = accuracy_score(y_val, model.predict(X_val))
        print(f"  Kryptonite-{key:8s} | Val Accuracy: {acc*100:.2f}%")

if __name__ == "__main__":
    main()
