"""MediaPipe Tasks HandLandmarker wrapper."""
from __future__ import annotations

import urllib.request
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .config import TrackerConfig


@dataclass
class HandObservation:
    landmarks: np.ndarray        # (21, 3) normalised image coords (x, y in [0, 1], z relative)
    world: np.ndarray | None     # (21, 3) metres, origin at hand centre
    handedness: str              # as reported by MediaPipe (assumes a mirrored image)
    handedness_score: float


def ensure_model(cfg: TrackerConfig) -> Path:
    path = Path(cfg.model_path)
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[tracker] downloading model -> {path}")
    tmp = path.with_suffix(".part")
    urllib.request.urlretrieve(cfg.model_url, tmp)
    tmp.replace(path)
    return path


class HandTracker:
    def __init__(self, cfg: TrackerConfig) -> None:
        from mediapipe.tasks.python import BaseOptions, vision

        model = ensure_model(cfg)
        options = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=cfg.num_hands,
            min_hand_detection_confidence=cfg.min_detection_confidence,
            min_hand_presence_confidence=cfg.min_presence_confidence,
            min_tracking_confidence=cfg.min_tracking_confidence,
        )
        self._landmarker = vision.HandLandmarker.create_from_options(options)
        self._last_ts = -1

    def detect(self, frame_bgr: np.ndarray, t_ms: float) -> list[HandObservation]:
        import mediapipe as mp

        ts = int(t_ms)
        if ts <= self._last_ts:  # VIDEO mode requires strictly increasing timestamps
            ts = self._last_ts + 1
        self._last_ts = ts
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect_for_video(image, ts)

        hands: list[HandObservation] = []
        for i, lms in enumerate(result.hand_landmarks):
            arr = np.array([[p.x, p.y, p.z] for p in lms], dtype=np.float64)
            world = None
            if result.hand_world_landmarks and i < len(result.hand_world_landmarks):
                world = np.array([[p.x, p.y, p.z] for p in result.hand_world_landmarks[i]], dtype=np.float64)
            label, score = "?", 0.0
            if result.handedness and i < len(result.handedness) and result.handedness[i]:
                cat = result.handedness[i][0]
                label, score = cat.category_name, float(cat.score)
            hands.append(HandObservation(arr, world, label, score))
        hands.sort(key=lambda h: h.handedness_score, reverse=True)
        return hands

    def close(self) -> None:
        self._landmarker.close()
