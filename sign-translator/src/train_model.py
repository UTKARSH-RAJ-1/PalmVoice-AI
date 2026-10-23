from __future__ import annotations

import csv
from pathlib import Path

import joblib
import numpy as np

from src.landmarks import ensure_landmarks_csv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MODEL_PATH = PROJECT_ROOT / "models" / "gesture_model.pkl"
LANDMARKS_PATH = DATA_DIR / "landmarks.csv"


class CentroidClassifier:
    def fit(self, X: np.ndarray, y: np.ndarray) -> "CentroidClassifier":
        self.classes_ = sorted(set(y.tolist()))
        self.centroids_ = {}
        for label in self.classes_:
            class_rows = X[np.asarray(y) == label]
            self.centroids_[label] = class_rows.mean(axis=0)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        results = []
        for sample in X:
            distances = {}
            for label, centroid in self.centroids_.items():
                distances[label] = float(np.linalg.norm(sample - centroid))
            results.append(min(distances, key=distances.get))
        return np.asarray(results)


def train_model(csv_path: Path = LANDMARKS_PATH, model_path: Path = MODEL_PATH) -> dict:
    csv_path = ensure_landmarks_csv(csv_path)
    with csv_path.open("r", newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        rows = list(reader)

    labels = []
    features = []
    for row in rows:
        labels.append(str(row["label"]))
        features.append([float(row[f"f{i}"]) for i in range(1, 64)])

    X = np.asarray(features, dtype=float)
    y = np.asarray(labels)
    indices = np.arange(len(X))
    np.random.default_rng(42).shuffle(indices)
    X, y = X[indices], y[indices]
    train_size = max(1, int(len(X) * 0.8))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    model = CentroidClassifier().fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = float(np.mean(predictions == y_test))

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    return {"accuracy": accuracy, "model_path": str(model_path)}


if __name__ == "__main__":
    result = train_model()
    print(f"Model trained and saved to {result['model_path']}")
    print(f"Accuracy: {result['accuracy']:.3f}")
