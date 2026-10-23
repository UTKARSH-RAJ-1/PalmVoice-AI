from __future__ import annotations

import json
from pathlib import Path

from flask import Flask, jsonify, send_from_directory

from src.landmarks import ensure_landmarks_csv
from src.text_to_sign import export_poses
from src.train_model import train_model

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
LANDMARKS_PATH = DATA_DIR / "landmarks.csv"
WEB_DIR = ROOT / "web"

app = Flask(__name__, static_folder=str(WEB_DIR), static_url_path="")


@app.route("/")
def index():
    return send_from_directory(str(WEB_DIR), "index.html")


@app.route("/health")
def health():
    return jsonify({"status": "ok", "project": "sign-translator"})


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


if __name__ == "__main__":
    ensure_landmarks_csv(LANDMARKS_PATH)
    export_poses(LANDMARKS_PATH, WEB_DIR / "poses.json")
    train_model(LANDMARKS_PATH)
    app.run(host="0.0.0.0", port=5000, debug=True)
