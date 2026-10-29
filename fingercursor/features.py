"""Pure geometric feature extraction from hand landmarks.

Coordinate convention: "display pixels" — x to the right, y down, as seen on the
(optionally mirrored) screen. z is MediaPipe's relative depth scaled like x.
All distances that matter for gestures are normalised by the hand scale S.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

WRIST = 0
THUMB_IP, THUMB_TIP = 3, 4
INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP = 5, 6, 7, 8
MIDDLE_MCP = 9
PINKY_MCP = 17

INDEX_SEGMENTS = [(INDEX_MCP, INDEX_PIP), (INDEX_PIP, INDEX_DIP), (INDEX_DIP, INDEX_TIP)]

# Joint angle triplets (a, joint, c); 180 deg = straight.
ANGLES = {
    "index_mcp": (WRIST, 5, 6),
    "index_pip": (5, 6, 7),
    "index_dip": (6, 7, 8),
    "middle_pip": (9, 10, 11),
    "ring_pip": (13, 14, 15),
    "pinky_pip": (17, 18, 19),
    "thumb_mcp": (1, 2, 3),
    "thumb_ip": (2, 3, 4),
}


def to_display_px(landmarks: np.ndarray, width: int, height: int, mirror: bool) -> np.ndarray:
    """MediaPipe normalised landmarks (N, 3) -> display pixel coordinates (N, 3)."""
    lm = np.asarray(landmarks, dtype=np.float64)
    out = np.empty_like(lm)
    out[:, 0] = (1.0 - lm[:, 0]) * width if mirror else lm[:, 0] * width
    out[:, 1] = lm[:, 1] * height
    out[:, 2] = lm[:, 2] * width
    return out


def point_segment(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> tuple[float, float, np.ndarray]:
    """Distance from p to segment ab (2D), the projection parameter t in [0, 1], and the closest point."""
    ab = b - a
    denom = float(np.dot(ab, ab))
    t = 0.0 if denom < 1e-12 else float(np.clip(np.dot(p - a, ab) / denom, 0.0, 1.0))
    closest = a + t * ab
    return float(np.linalg.norm(p - closest)), t, closest


def joint_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    v1, v2 = a - b, c - b
    n = np.linalg.norm(v1) * np.linalg.norm(v2)
    if n < 1e-12:
        return float("nan")
    return float(np.degrees(np.arccos(np.clip(np.dot(v1, v2) / n, -1.0, 1.0))))


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-9 else np.array([0.0, -1.0])


@dataclass
class Features:
    scale_px: float                 # hand scale S in display px
    contact_dist: float             # min thumb-tip to index-segment distance / S
    contact_dist_px: float
    nearest_segment: int            # index into INDEX_SEGMENTS
    segment_t: float                # projection parameter along the nearest segment
    seg_dists: list[float]          # per-segment distance / S (nan if segment not used)
    dz: float                       # (z_thumb - z_closest) / S
    thumb_px: np.ndarray            # thumb tip (x, y) in display px
    closest_px: np.ndarray          # closest point on index skeleton
    rel_image: np.ndarray           # (thumb - anchor) / S in image frame (x right, y down)
    rel_hand: np.ndarray            # same in hand-upright frame (x right, y down)
    hand_up: np.ndarray             # unit vector wrist -> middle MCP (display px frame)
    hand_right: np.ndarray
    angles: dict[str, float] = field(default_factory=dict)

    def to_log(self, nd: int = 5) -> dict[str, Any]:
        r = lambda v: round(float(v), nd)  # noqa: E731
        return {
            "S_px": r(self.scale_px),
            "d": r(self.contact_dist),
            "d_px": r(self.contact_dist_px),
            "seg": self.nearest_segment,
            "seg_t": r(self.segment_t),
            "seg_d": [None if np.isnan(x) else r(x) for x in self.seg_dists],
            "dz": r(self.dz),
            "thumb_px": [r(x) for x in self.thumb_px],
            "closest_px": [r(x) for x in self.closest_px],
            "rel_img": [r(x) for x in self.rel_image],
            "rel_hand": [r(x) for x in self.rel_hand],
            "hand_up": [r(x) for x in self.hand_up],
            "ang": {k: (None if np.isnan(v) else round(v, 2)) for k, v in self.angles.items()},
        }


def compute_features(
    px: np.ndarray,
    world: np.ndarray | None = None,
    segments: list[int] | None = None,
    anchor_landmark: int = INDEX_MCP,
) -> Features:
    """px: (21, 3) landmarks in display pixels. world: optional (21, 3) metric world landmarks."""
    p2 = px[:, :2]
    scale = 0.5 * (np.linalg.norm(p2[WRIST] - p2[INDEX_MCP]) + np.linalg.norm(p2[INDEX_MCP] - p2[PINKY_MCP]))
    scale = max(float(scale), 1e-6)

    segments = list(range(len(INDEX_SEGMENTS))) if segments is None else segments
    thumb = p2[THUMB_TIP]
    seg_dists = [float("nan")] * len(INDEX_SEGMENTS)
    best = (float("inf"), 0, 0.0, p2[INDEX_PIP])
    for si in segments:
        a_i, b_i = INDEX_SEGMENTS[si]
        d, t, closest = point_segment(thumb, p2[a_i], p2[b_i])
        seg_dists[si] = d / scale
        if d < best[0]:
            best = (d, si, t, closest)
    d_px, seg, t, closest = best
    a_i, b_i = INDEX_SEGMENTS[seg]
    z_closest = px[a_i, 2] + t * (px[b_i, 2] - px[a_i, 2])
    dz = (px[THUMB_TIP, 2] - z_closest) / scale

    anchor = np.zeros(2) if anchor_landmark < 0 else p2[anchor_landmark]
    rel = (thumb - anchor) / scale
    up = _unit(p2[MIDDLE_MCP] - p2[WRIST])
    right = np.array([-up[1], up[0]])
    rel_hand = np.array([float(np.dot(rel, right)), -float(np.dot(rel, up))])

    pts = world if world is not None else px
    angles = {name: joint_angle(pts[a], pts[b], pts[c]) for name, (a, b, c) in ANGLES.items()}

    return Features(
        scale_px=scale,
        contact_dist=d_px / scale,
        contact_dist_px=d_px,
        nearest_segment=seg,
        segment_t=t,
        seg_dists=seg_dists,
        dz=float(dz),
        thumb_px=thumb.copy(),
        closest_px=np.asarray(closest, dtype=np.float64).copy(),
        rel_image=rel,
        rel_hand=rel_hand,
        hand_up=up,
        hand_right=right,
        angles=angles,
    )
