from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from src.landmarks import ensure_landmarks_csv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LANDMARKS_PATH = PROJECT_ROOT / "data" / "landmarks.csv"
POSES_PATH = PROJECT_ROOT / "web" / "poses.json"


def export_poses(csv_path: Path = LANDMARKS_PATH, output_path: Path = POSES_PATH) -> dict:
    csv_path = ensure_landmarks_csv(csv_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    groups: dict[str, list[list[float]]] = {}

    with csv_path.open("r", newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            label = str(row["label"]).upper()
            values = [float(row[f"f{i}"]) for i in range(1, 64)]
            groups.setdefault(label, []).append(values)

    poses = {}
    for label, samples in groups.items():
        arr = np.median(np.asarray(samples, dtype=float).reshape(len(samples), 21, 3), axis=0)
        poses[label] = [[round(float(x), 4) for x in row] for row in arr]

    output_path.write_text(json.dumps(poses, indent=2), encoding="utf-8")
    return poses


if __name__ == "__main__":
    poses = export_poses()
    print(f"Exported {len(poses)} poses to {POSES_PATH}")
