import os
import numpy as np
from sklearn.model_selection import train_test_split

def find_file(filename):
    """Recursively finds a file matching filename anywhere in the repository."""
    for root, _, files in os.walk('.'):
        if filename in files:
            return os.path.join(root, filename)
    return None

def load_data(key, test_size=0.2, random_state=42):
    """Loads full X, y and a stratified train/validation split."""
    x_path = find_file(f"kryptonite-{key}-X.npy")
    y_path = find_file(f"kryptonite-{key}-y.npy")
    if not x_path or not y_path:
        raise FileNotFoundError(f"Missing files for Kryptonite-{key}")
    X = np.load(x_path)
    y = np.load(y_path)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X, y, X_train, X_val, y_train, y_val

def load_hidden_X(key):
    """Loads unlabeled hidden test features for a given dataset key."""
    x_path = find_file(f"hidden-kryptonite-{key}-X.npy")
    if not x_path:
        raise FileNotFoundError(f"Missing hidden features for Kryptonite-{key}")
    return np.load(x_path)
