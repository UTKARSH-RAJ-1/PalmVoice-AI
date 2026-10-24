from __future__ import annotations

import json
import os
import threading
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from flask import Flask, jsonify, request, send_file, send_from_directory
from smart_gestures.alphabet.asl_model.model import ASLModel

from src.landmarks import ensure_landmarks_csv
from src.text_to_sign import export_poses
from src.train_model import train_model

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
LANDMARKS_PATH = DATA_DIR / "landmarks.csv"
WEB_DIR = ROOT / "web"
BUNDLED_VRM_PATH = ROOT / "assets" / "avatar.vrm"
DEFAULT_VRM_PATH = Path(os.environ.get("PALMVOICE_VRM_PATH", str(BUNDLED_VRM_PATH)))

app = Flask(__name__, static_folder=str(WEB_DIR), static_url_path="")
_recognizer_lock = threading.Lock()
_recognizer_model: ASLModel | None = None
_recognizer_hands = None


def _get_recognizer():
    global _recognizer_model, _recognizer_hands
    if _recognizer_model is None:
        _recognizer_model = ASLModel()
    if _recognizer_hands is None:
        _recognizer_hands = mp.solutions.hands.Hands(
            static_image_mode=True,
            max_num_hands=1,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6,
        )
    return _recognizer_model, _recognizer_hands


@app.route("/")
def index():
    return send_from_directory(str(WEB_DIR), "index.html")


@app.route("/health")
def health():
    return jsonify(
        {
            "status": "ok",
            "project": "sign-translator",
            "avatar_available": DEFAULT_VRM_PATH.is_file(),
        }
    )


@app.route("/api/avatar")
def avatar_route():
    """Serve the locally configured VRM avatar."""
    if not DEFAULT_VRM_PATH.is_file():
        return (
            jsonify(
                {
                    "error": "Avatar file not found.",
                    "path": str(DEFAULT_VRM_PATH),
                    "hint": "Set PALMVOICE_VRM_PATH to a valid .vrm file.",
                }
            ),
            404,
        )
    if DEFAULT_VRM_PATH.suffix.lower() != ".vrm":
        return jsonify({"error": "PALMVOICE_VRM_PATH must point to a .vrm file."}), 400
    return send_file(DEFAULT_VRM_PATH, mimetype="model/gltf-binary", conditional=True)


@app.route("/api/poses")
def poses_route():
    export_poses(LANDMARKS_PATH, WEB_DIR / "poses.json")
    return send_from_directory(str(WEB_DIR), "poses.json")


@app.route("/api/train")
def train_route():
    result = train_model(LANDMARKS_PATH)
    return jsonify(result)


@app.route("/api/demo")
def demo_route():
    ensure_landmarks_csv(LANDMARKS_PATH)
    export_poses(LANDMARKS_PATH, WEB_DIR / "poses.json")
    return jsonify({"sample_text": "HELLO", "letters": list("HELLO")})


@app.post("/api/recognize")
def recognize_route():
    """Recognize one sampled webcam frame using the installed ASL model."""
    image_file = request.files.get("frame")
    if image_file is None:
        return jsonify({"error": "A multipart 'frame' image is required."}), 400

    encoded = image_file.read()
    image = cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        return jsonify({"error": "The uploaded frame is not a valid image."}), 400

    with _recognizer_lock:
        model, hands = _get_recognizer()
        results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if not results.multi_hand_landmarks:
            return jsonify({"detected": False, "label": None, "confidence": 0})

        landmarks = [
            {"x": point.x, "y": point.y, "z": point.z}
            for point in results.multi_hand_landmarks[0].landmark
        ]
        label, confidence = model.predict(landmarks)

    return jsonify(
        {
            "detected": True,
            "label": str(label).upper(),
            "confidence": round(float(confidence), 4),
        }
    )


if __name__ == "__main__":
    ensure_landmarks_csv(LANDMARKS_PATH)
    export_poses(LANDMARKS_PATH, WEB_DIR / "poses.json")
    train_model(LANDMARKS_PATH)
    app.run(host="0.0.0.0", port=5000, debug=True)
