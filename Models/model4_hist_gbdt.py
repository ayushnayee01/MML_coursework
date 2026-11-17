from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score
from utils import load_data

def main():
    print("=== MODEL 4: HISTOGRAM GRADIENT BOOSTING ===")
    for key in ['20', '30', '40', '40-adapt']:
        _, _, X_train, X_val, y_train, y_val = load_data(key)

        model = HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.05, max_leaf_nodes=63, random_state=42
        )
        model.fit(X_train, y_train)
        acc = accuracy_score(y_val, model.predict(X_val))
        print(f"  Kryptonite-{key:8s} | Val Accuracy: {acc*100:.2f}%")

if __name__ == "__main__":
    main()
