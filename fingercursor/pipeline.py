"""Landmarks -> smoothing -> features -> recognizer. Shared by the live app and log replay."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .config import Config
from .features import Features, compute_features, to_display_px
from .filters import OneEuroFilter
from .recognizer import Event, GestureRecognizer


@dataclass
class FrameResult:
    t_ms: float
    px_raw: np.ndarray | None = None       # (21, 3) display px, unfiltered
    px: np.ndarray | None = None           # (21, 3) display px, filtered
    features: Features | None = None
    position: np.ndarray | None = None     # thumb position in the direction frame
    events: list[Event] = field(default_factory=list)
    debug: dict[str, Any] = field(default_factory=dict)


class Pipeline:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        fc = cfg.filter
        self.filter = OneEuroFilter(fc.min_cutoff, fc.beta, fc.d_cutoff)
        self.recognizer = GestureRecognizer(cfg.contact, cfg.gesture, cfg.direction)

    def process(
        self,
        t_ms: float,
        landmarks: np.ndarray | None,
        world: np.ndarray | None,
        width: int,
        height: int,
    ) -> FrameResult:
        """landmarks: MediaPipe normalised (21, 3) of the camera image (not mirrored), or None."""
        res = FrameResult(t_ms=t_ms)
        if landmarks is None:
            self.filter.reset()
            res.events = self.recognizer.update(t_ms, None)
            res.debug = self.recognizer.debug()
            return res

        res.px_raw = to_display_px(landmarks, width, height, self.cfg.display.mirror)
        res.px = self.filter(res.px_raw, t_ms) if self.cfg.filter.enabled else res.px_raw.copy()
        world_arr = None
        if world is not None:
            world_arr = np.asarray(world, dtype=np.float64).copy()
            if self.cfg.display.mirror:
                world_arr[:, 0] = -world_arr[:, 0]
        res.features = compute_features(
            res.px, world_arr, self.cfg.contact.segments, self.cfg.gesture.anchor_landmark)
        res.position = self.recognizer.position(res.features)
        res.events = self.recognizer.update(t_ms, res.features)
        res.debug = self.recognizer.debug()
        return res
