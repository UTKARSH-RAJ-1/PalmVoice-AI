from __future__ import annotations

from pathlib import Path

from smart_gestures.alphabet.asl_model.model import ASLModel

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "gesture_model.pkl"


def _make_sample_hand(label: str = "A") -> list[dict[str, float]]:
    landmarks: list[dict[str, float]] = []
    for idx in range(21):
        offset = (ord(label[0]) if label else 65) % 26
        x = ((idx % 5) * 0.13) + (offset * 0.003) - 0.25
        y = ((idx // 5) * 0.12) + (offset * 0.004) - 0.35
        z = 0.02 * ((idx % 3) - 1) + (0.1 if label == "Z" else 0.0)
        landmarks.append({"x": x, "y": y, "z": z})
    return landmarks


def predict_label(landmarks: list[dict[str, float]]) -> tuple[str, float]:
    model = ASLModel()
    return model.predict(landmarks)


def demo_prediction(label: str = "A") -> tuple[str, float]:
    return predict_label(_make_sample_hand(label))


if __name__ == "__main__":
    for label in ["A", "B", "C", "M", "Y", "Z"]:
        predicted, confidence = demo_prediction(label)
        print(f"Prompt: {label} -> Predicted: {predicted} (confidence={confidence:.3f})")
