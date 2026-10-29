"""Synthetic hand landmark sequences for tests (image 1000x1000, mirror disabled)."""
from __future__ import annotations

import numpy as np

from fingercursor.config import Config

W = H = 1000
DT = 1000.0 / 30.0

# A curled-index right hand, normalised coords. Only the points used by the code matter.
BASE = np.array([
    [0.50, 0.80, 0.0],   # 0 wrist
    [0.42, 0.74, 0.0],   # 1 thumb cmc
    [0.36, 0.66, 0.0],   # 2 thumb mcp
    [0.33, 0.58, 0.0],   # 3 thumb ip
    [0.32, 0.50, 0.0],   # 4 thumb tip (overwritten)
    [0.45, 0.55, 0.0],   # 5 index mcp
    [0.35, 0.55, 0.0],   # 6 index pip   (5-6 horizontal, 100 px)
    [0.35, 0.62, 0.0],   # 7 index dip
    [0.38, 0.66, 0.0],   # 8 index tip
    [0.52, 0.55, 0.0],   # 9 middle mcp
    [0.45, 0.50, 0.0], [0.45, 0.58, 0.0], [0.47, 0.62, 0.0],
    [0.57, 0.56, 0.0], [0.52, 0.52, 0.0], [0.52, 0.60, 0.0], [0.54, 0.63, 0.0],
    [0.61, 0.58, 0.0], [0.58, 0.55, 0.0], [0.58, 0.61, 0.0], [0.59, 0.64, 0.0],
])
# Hand scale S in px: 0.5 * (|0-5| + |5-17|)
S = 0.5 * (np.hypot(50, 250) + np.hypot(160, 30))
# Contact point on segment 5-6 (midpoint), thumb just above it.
CONTACT = np.array([0.40, 0.55 - 0.1 * S / W])   # d = 0.1 S
AWAY = np.array([0.40, 0.55 - 0.9 * S / W])      # d = 0.9 S


def hand(thumb_xy: np.ndarray) -> np.ndarray:
    lm = BASE.copy()
    lm[4, :2] = thumb_xy
    return lm


def offset(dx_s: float, dy_s: float) -> np.ndarray:
    """Displacement in S units -> normalised coords."""
    return np.array([dx_s * S / W, dy_s * S / H])


def make_config(**overrides) -> Config:
    cfg = Config()
    cfg.display.mirror = False
    cfg.filter.enabled = False
    for k, v in overrides.items():
        section, key = k.split("__")
        setattr(getattr(cfg, section), key, v)
    return cfg


def run(pipeline, seq: list[np.ndarray | None], t0: float = 0.0):
    events = []
    t = t0
    for thumb in seq:
        lm = None if thumb is None else hand(thumb)
        res = pipeline.process(t, lm, None, W, H)
        events.extend(res.events)
        t += DT
    return events
