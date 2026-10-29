"""Contact state machine and gesture classification.

Input per frame: timestamp (ms) and Features (or None when no hand is tracked).
Output: a list of Events. Internal state is exposed through `debug()` for logging.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .config import ContactConfig, DirectionConfig, GestureConfig
from .features import Features

IDLE = "IDLE"
CANDIDATE = "CANDIDATE"
TOUCH = "TOUCH"
HOLDING = "HOLDING"
SWIPED = "SWIPED"
CONTACT_STATES = (TOUCH, HOLDING, SWIPED)

# Gesture events (what the user would map to PC operations).
GESTURE_EVENTS = (
    "SWIPE_LEFT", "SWIPE_RIGHT", "SWIPE_UP", "SWIPE_DOWN",
    "TAP", "LONG_PRESS_START", "LONG_PRESS_END",
)


@dataclass
class Event:
    type: str
    t_ms: float
    data: dict[str, Any] = field(default_factory=dict)

    @property
    def is_gesture(self) -> bool:
        return self.type in GESTURE_EVENTS

    def to_log(self) -> dict[str, Any]:
        return {"type": self.type, "t_ms": round(self.t_ms, 2), "gesture": self.is_gesture, **self.data}


def _r(v: float, nd: int = 4) -> float | None:
    return None if v is None or (isinstance(v, float) and math.isnan(v)) else round(float(v), nd)


class GestureRecognizer:
    def __init__(self, contact: ContactConfig, gesture: GestureConfig, direction: DirectionConfig) -> None:
        self.c = contact
        self.g = gesture
        self.dcfg = direction
        self.state = IDLE
        self._reset_contact()
        self.tracking = False
        self.lost_since: float | None = None
        self.last_t: float | None = None
        self.last_pos: np.ndarray | None = None
        self.frame_reason = ""

    # ------------------------------------------------------------------ helpers
    def _reset_contact(self) -> None:
        self.cand_count = 0
        self.cand_t = 0.0
        self.cand_pos: np.ndarray | None = None
        self.start_t = 0.0
        self.start_pos: np.ndarray | None = None
        self.off_count = 0
        self.release_t = 0.0
        self.disp = np.zeros(2)
        self.max_disp = 0.0
        self.path_len = 0.0
        self.peak_speed = 0.0
        self.min_d = float("inf")
        self.d_sum = 0.0
        self.d_n = 0
        self.swipe_dir: str | None = None
        self.hold_t: float | None = None

    def position(self, f: Features) -> np.ndarray:
        """Thumb position relative to the anchor, in the configured direction frame (x right, y down)."""
        p = (f.rel_hand if self.dcfg.frame == "hand" else f.rel_image).astype(np.float64).copy()
        if self.dcfg.swap_axes:
            p = p[::-1].copy()
        if self.dcfg.invert_x:
            p[0] = -p[0]
        if self.dcfg.invert_y:
            p[1] = -p[1]
        return p

    def _gates(self, f: Features) -> tuple[bool, list[str]]:
        failed = []
        if self.c.max_abs_dz is not None and abs(f.dz) > self.c.max_abs_dz:
            failed.append("dz")
        if self.c.pose_gate:
            ang = f.angles.get("index_pip", float("nan"))
            if not (ang <= self.c.pose_max_index_pip_deg):
                failed.append("pose")
        return not failed, failed

    def _contact_summary(self, t_end: float) -> dict[str, Any]:
        return {
            "start_t_ms": round(self.start_t, 2),
            "duration_ms": round(t_end - self.start_t, 2),
            "disp": [_r(self.disp[0]), _r(self.disp[1])],
            "disp_mag": _r(float(np.linalg.norm(self.disp))),
            "max_disp": _r(self.max_disp),
            "path_len": _r(self.path_len),
            "peak_speed": _r(self.peak_speed),
            "min_d": _r(self.min_d),
            "mean_d": _r(self.d_sum / self.d_n) if self.d_n else None,
        }

    # ------------------------------------------------------------------ main
    def update(self, t_ms: float, f: Features | None) -> list[Event]:
        events: list[Event] = []
        self.frame_reason = ""
        if f is None:
            self._on_lost(t_ms, events)
            self.last_t = t_ms
            return events

        if not self.tracking:
            self.tracking = True
            events.append(Event("TRACKING_FOUND", t_ms))
        self.lost_since = None

        pos = self.position(f)
        d = f.contact_dist
        gates_ok, failed = self._gates(f)
        below_on = d < self.c.touch_on and gates_ok
        above_off = d > self.c.touch_off or not gates_ok
        speed = 0.0
        if self.last_pos is not None and self.last_t is not None and t_ms > self.last_t:
            speed = float(np.linalg.norm(pos - self.last_pos)) / ((t_ms - self.last_t) / 1000.0)

        if self.state == IDLE:
            if below_on:
                self.state = CANDIDATE
                self.cand_count = 1
                self.cand_t = t_ms
                self.cand_pos = pos.copy()
                self.frame_reason = "below_on"
                if self.cand_count >= self.c.on_frames:
                    self._start_touch(events)
            elif d < self.c.touch_on:
                self.frame_reason = "gate_blocked:" + ",".join(failed)
        elif self.state == CANDIDATE:
            if below_on:
                self.cand_count += 1
                if self.cand_count >= self.c.on_frames:
                    self._start_touch(events)
            else:
                events.append(Event("CANDIDATE_DROPPED", t_ms, {
                    "frames": self.cand_count, "d": _r(d), "gates_failed": failed}))
                self.state = IDLE
                self._reset_contact()
        elif self.state in CONTACT_STATES:
            if above_off:
                self.off_count += 1
                if self.off_count == 1:
                    self.release_t = t_ms
                self.frame_reason = "above_off" if gates_ok else "gate_failed:" + ",".join(failed)
                if self.off_count >= self.c.off_frames:
                    self._release(self.release_t, events)
            else:
                self.off_count = 0
                self._track(pos, d, speed)
                self._classify(t_ms, d, events)

        self.last_t = t_ms
        self.last_pos = pos
        return events

    def _start_touch(self, events: list[Event]) -> None:
        self.state = TOUCH
        self.start_t = self.cand_t
        self.start_pos = self.cand_pos.copy() if self.cand_pos is not None else None
        self.off_count = 0
        events.append(Event("TOUCH_START", self.start_t, {"confirm_frames": self.cand_count}))

    def _track(self, pos: np.ndarray, d: float, speed: float) -> None:
        if self.start_pos is None:
            self.start_pos = pos.copy()
        if self.last_pos is not None:
            self.path_len += float(np.linalg.norm(pos - self.last_pos))
        self.disp = pos - self.start_pos
        self.max_disp = max(self.max_disp, float(np.linalg.norm(self.disp)))
        self.peak_speed = max(self.peak_speed, speed)
        self.min_d = min(self.min_d, d)
        self.d_sum += d
        self.d_n += 1

    def _classify(self, t_ms: float, d: float, events: list[Event]) -> None:
        if self.state != TOUCH:
            return
        elapsed = t_ms - self.start_t
        dx, dy = float(self.disp[0]), float(self.disp[1])
        mag = math.hypot(dx, dy)
        major, minor = (abs(dx), abs(dy)) if abs(dx) >= abs(dy) else (abs(dy), abs(dx))
        ratio = major / (minor + 1e-6)
        if mag >= self.g.swipe_min_dist:
            if elapsed > self.g.swipe_max_ms:
                self.frame_reason = "swipe_too_slow"
            elif ratio < self.g.axis_ratio:
                self.frame_reason = "swipe_ambiguous_axis"
            elif d > self.g.swipe_max_contact_d:
                self.frame_reason = "swipe_contact_too_weak"
            else:
                if abs(dx) >= abs(dy):
                    direction = "RIGHT" if dx > 0 else "LEFT"
                else:
                    direction = "DOWN" if dy > 0 else "UP"
                self.swipe_dir = direction
                self.state = SWIPED
                events.append(Event(f"SWIPE_{direction}", t_ms, {
                    **self._contact_summary(t_ms), "axis_ratio": _r(ratio), "elapsed_ms": round(elapsed, 2)}))
                return
        if elapsed >= self.g.long_press_ms:
            if self.max_disp < self.g.move_tolerance:
                self.state = HOLDING
                self.hold_t = t_ms
                events.append(Event("LONG_PRESS_START", t_ms, self._contact_summary(t_ms)))
            elif not self.frame_reason:
                self.frame_reason = "hold_moved_too_much"

    def _release(self, t_rel: float, events: list[Event]) -> None:
        summary = self._contact_summary(t_rel)
        if self.state == TOUCH:
            dur = t_rel - self.start_t
            reasons = []
            if dur < self.g.tap_min_ms:
                reasons.append("too_short")
            if dur > self.g.tap_max_ms:
                reasons.append("too_long_for_tap")
            if self.max_disp > self.g.tap_max_move:
                reasons.append("moved_too_much")
            if reasons:
                events.append(Event("TOUCH_REJECTED", t_rel, {**summary, "reasons": reasons}))
            else:
                events.append(Event("TAP", t_rel, summary))
        elif self.state == HOLDING:
            events.append(Event("LONG_PRESS_END", t_rel, summary))
        events.append(Event("TOUCH_END", t_rel, {**summary, "from_state": self.state, "swipe": self.swipe_dir}))
        self.state = IDLE
        self._reset_contact()

    def _on_lost(self, t_ms: float, events: list[Event]) -> None:
        if self.tracking:
            self.tracking = False
            self.lost_since = t_ms
            events.append(Event("TRACKING_LOST", t_ms, {"state": self.state}))
        self.last_pos = None
        if self.state == CANDIDATE:
            self.state = IDLE
            self._reset_contact()
        elif self.state in CONTACT_STATES and self.lost_since is not None:
            if t_ms - self.lost_since > self.c.lost_grace_ms:
                summary = self._contact_summary(self.lost_since)
                if self.state == HOLDING:
                    events.append(Event("LONG_PRESS_END", self.lost_since, {**summary, "aborted": True}))
                events.append(Event("TOUCH_ABORTED", t_ms, {**summary, "reason": "tracking_lost",
                                                            "from_state": self.state}))
                self.state = IDLE
                self._reset_contact()

    def debug(self) -> dict[str, Any]:
        in_contact = self.state in CONTACT_STATES
        return {
            "state": self.state,
            "reason": self.frame_reason,
            "cand": self.cand_count,
            "off": self.off_count,
            "disp": [_r(self.disp[0]), _r(self.disp[1])] if in_contact else None,
            "max_disp": _r(self.max_disp) if in_contact else None,
            "elapsed_ms": round(self.last_t - self.start_t, 2) if in_contact and self.last_t is not None else None,
        }
