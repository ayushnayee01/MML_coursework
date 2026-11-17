from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score
from utils import load_data

def main():
    print("=== MODEL 3: EXTRA TREES CLASSIFIER ===")
    for key in ['20', '30', '40', '40-adapt']:
        _, _, X_train, X_val, y_train, y_val = load_data(key)

        model = ExtraTreesClassifier(n_estimators=200, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        acc = accuracy_score(y_val, model.predict(X_val))
        print(f"  Kryptonite-{key:8s} | Val Accuracy: {acc*100:.2f}%")

if __name__ == "__main__":
    main()
