from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import List

import numpy as np

DEFAULT_LETTERS = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "SPACE"]


def _label_values() -> List[str]:
    return [*DEFAULT_LETTERS]


def _synthetic_pose(label: str, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    label_index = _label_values().index(label) if label in _label_values() else 0
    features = np.zeros(63, dtype=float)

    anchor_1 = (label_index * 7 + 5) % 63
    anchor_2 = (label_index * 13 + 19) % 63
    anchor_3 = (label_index * 17 + 31) % 63
    features[anchor_1] = 40.0 + label_index * 1.5
    features[anchor_2] = 26.0 + label_index * 1.1
    features[anchor_3] = 18.0 + label_index * 0.8
    for i in range(63):
        features[i] += 0.5 * math.sin((i + 1) * 0.7 + label_index + seed / 12.0)
    features += rng.normal(0.0, 0.2, size=63)

    if label == "SPACE":
        features[:] = 0.0
    return features.reshape(21, 3)


def build_landmarks_csv(output_path: Path, samples_per_class: int = 60) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for letter in _label_values():
        for sample_idx in range(samples_per_class):
            pose = _synthetic_pose(letter, seed=sample_idx + len(letter) * 17)
            values = pose.reshape(-1).tolist()
            rows.append([letter] + values)

    headers = ["label"] + [f"f{i}" for i in range(1, 63 + 1)]
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(headers)
        writer.writerows(rows)
    return output_path


def ensure_landmarks_csv(csv_path: Path, dataset_dir: Path | None = None) -> Path:
    if csv_path.exists():
        return csv_path
    if dataset_dir is not None and dataset_dir.exists():
        return csv_path
    return build_landmarks_csv(csv_path)


def export_poses_json(csv_path: Path, output_path: Path) -> Path:
    csv_path = Path(csv_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("r", newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        groups: dict[str, list[list[float]]] = {}
        for row in reader:
            label = str(row["label"]).upper()
            values = [float(row[f"f{i}"]) for i in range(1, 64)]
            groups.setdefault(label, []).append(values)

    poses: dict[str, list[list[float]]] = {}
    for label, samples in groups.items():
        arr = np.median(np.asarray(samples, dtype=float).reshape(len(samples), 21, 3), axis=0)
        poses[label] = np.round(arr, 4).tolist()

    output_path.write_text(json.dumps(poses, indent=2), encoding="utf-8")
    return output_path


def average_letter_from_csv(csv_path: Path, label: str) -> List[List[float]]:
    with csv_path.open("r", newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        samples: list[list[float]] = []
        for row in reader:
            if str(row["label"]).upper() == str(label).upper():
                values = [float(row[f"f{i}"]) for i in range(1, 64)]
                samples.append(values)

    if not samples:
        raise ValueError(f"No samples found for label {label!r}")
    arr = np.median(np.asarray(samples, dtype=float).reshape(len(samples), 21, 3), axis=0)
    return np.round(arr, 4).tolist()


def sample_classifier_features(label: str, noise: float = 0.1) -> List[float]:
    pose = _synthetic_pose(label, seed=11)
    jitter = np.random.default_rng(7).normal(0.0, noise, size=pose.shape)
    return np.round((pose + jitter).reshape(-1).tolist(), 4)
