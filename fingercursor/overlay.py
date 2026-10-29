"""OpenCV overlay drawing."""
from __future__ import annotations

from collections import deque

import cv2
import numpy as np

from .config import Config
from .features import INDEX_SEGMENTS, THUMB_TIP
from .pipeline import FrameResult
from .recognizer import CANDIDATE, CONTACT_STATES, HOLDING, SWIPED, TOUCH

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
]

STATE_COLORS = {
    "IDLE": (180, 180, 180),
    CANDIDATE: (0, 200, 255),
    TOUCH: (0, 255, 0),
    HOLDING: (255, 0, 255),
    SWIPED: (255, 200, 0),
}

FONT = cv2.FONT_HERSHEY_SIMPLEX


def _pt(p: np.ndarray) -> tuple[int, int]:
    return int(round(float(p[0]))), int(round(float(p[1])))


def _text(img: np.ndarray, text: str, org: tuple[int, int], scale: float = 0.55,
          color: tuple[int, int, int] = (255, 255, 255), thick: int = 1) -> None:
    cv2.putText(img, text, org, FONT, scale, (0, 0, 0), thick + 3, cv2.LINE_AA)
    cv2.putText(img, text, org, FONT, scale, color, thick, cv2.LINE_AA)


class Overlay:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.trail: deque[tuple[int, int]] = deque(maxlen=cfg.display.trail_length)
        self.history: deque[str] = deque(maxlen=cfg.display.history_lines)
        self.last_gesture: tuple[str, float] | None = None
        self.show_debug = cfg.display.show_debug_panel

    def note_event(self, ev_type: str, t_ms: float, text: str) -> None:
        self.history.appendleft(text)
        self.last_gesture = (ev_type, t_ms)

    def draw(self, img: np.ndarray, res: FrameResult, status: dict) -> None:
        h, w = img.shape[:2]
        state = res.debug.get("state", "IDLE")
        color = STATE_COLORS.get(state, (255, 255, 255))

        if res.px is not None and res.features is not None:
            px = res.px
            for a, b in HAND_CONNECTIONS:
                cv2.line(img, _pt(px[a]), _pt(px[b]), (200, 200, 200), 1, cv2.LINE_AA)
            for i in range(21):
                cv2.circle(img, _pt(px[i]), 3, (255, 255, 255), -1, cv2.LINE_AA)
            # Index segments used for contact
            for si in self.cfg.contact.segments:
                a, b = INDEX_SEGMENTS[si]
                thick = 4 if si == res.features.nearest_segment else 2
                cv2.line(img, _pt(px[a]), _pt(px[b]), (0, 220, 255), thick, cv2.LINE_AA)
            f = res.features
            cv2.line(img, _pt(f.thumb_px), _pt(f.closest_px), color, 2, cv2.LINE_AA)
            cv2.circle(img, _pt(f.closest_px), 4, (0, 220, 255), -1, cv2.LINE_AA)
            # Contact radius around the thumb tip (touch_on / touch_off in px)
            cv2.circle(img, _pt(f.thumb_px), int(self.cfg.contact.touch_on * f.scale_px), (0, 255, 0), 1, cv2.LINE_AA)
            cv2.circle(img, _pt(f.thumb_px), int(self.cfg.contact.touch_off * f.scale_px), (0, 0, 255), 1, cv2.LINE_AA)
            cv2.circle(img, _pt(px[THUMB_TIP]), 9, color, -1, cv2.LINE_AA)
            # Hand-frame axes at the wrist
            if self.cfg.direction.frame == "hand":
                o = px[0, :2]
                L = 0.6 * f.scale_px
                cv2.arrowedLine(img, _pt(o), _pt(o + f.hand_up * L), (0, 0, 255), 2, tipLength=0.2)
                cv2.arrowedLine(img, _pt(o), _pt(o + f.hand_right * L), (255, 0, 0), 2, tipLength=0.2)
            if state in CONTACT_STATES:
                self.trail.append(_pt(f.thumb_px))
            elif state != CANDIDATE:
                self.trail.clear()
        else:
            self.trail.clear()
        pts = list(self.trail)
        for i in range(1, len(pts)):
            cv2.line(img, pts[i - 1], pts[i], (255, 200, 0), 2, cv2.LINE_AA)

        # Top-left status
        y = 24
        _text(img, f"FPS {status.get('fps', 0):5.1f}  infer {status.get('infer_ms', 0):5.1f}ms  "
                   f"frame {status.get('frame', 0)}", (10, y))
        y += 26
        hand = status.get("hand", "no hand")
        _text(img, f"hand: {hand}", (10, y))
        y += 30
        _text(img, f"STATE: {state}", (10, y), 0.8, color, 2)
        if res.debug.get("reason"):
            y += 24
            _text(img, f"reason: {res.debug['reason']}", (10, y), 0.5, (200, 200, 255))
        if status.get("recording"):
            _text(img, "REC", (w - 70, 30), 0.8, (0, 0, 255), 2)
        if status.get("paused"):
            _text(img, "PAUSED", (w // 2 - 60, h // 2), 1.2, (0, 0, 255), 3)

        if self.show_debug:
            self._debug_panel(img, res)

        # Big last-gesture text (fades after 1 s)
        if self.last_gesture is not None:
            ev, t0 = self.last_gesture
            age = res.t_ms - t0
            if 0 <= age < 1000:
                size = cv2.getTextSize(ev, FONT, 1.6, 4)[0]
                _text(img, ev, ((w - size[0]) // 2, h - 60), 1.6, (0, 255, 255), 4)

        # Event history (bottom-left)
        for i, line in enumerate(self.history):
            _text(img, line, (10, h - 100 - i * 22), 0.5, (230, 230, 230))

        guided = status.get("guided")
        if guided:
            size = cv2.getTextSize(guided, FONT, 1.1, 3)[0]
            _text(img, guided, ((w - size[0]) // 2, 90), 1.1, (0, 165, 255), 3)

        _text(img, "q:quit p:pause d:debug  labels: arrows/IJKL=swipe t=tap h=hold space=mark",
              (10, h - 12), 0.45, (200, 200, 200))

    def _debug_panel(self, img: np.ndarray, res: FrameResult) -> None:
        h, w = img.shape[:2]
        c, g = self.cfg.contact, self.cfg.gesture
        # Contact distance bar
        x0, y0, bw, bh = w - 60, 60, 24, 260
        vmax = max(1.0, c.touch_off * 2.0)
        cv2.rectangle(img, (x0, y0), (x0 + bw, y0 + bh), (60, 60, 60), -1)
        f = res.features
        if f is not None:
            v = min(f.contact_dist, vmax)
            fill = int(bh * v / vmax)
            col = STATE_COLORS.get(res.debug.get("state", "IDLE"), (255, 255, 255))
            cv2.rectangle(img, (x0, y0 + bh - fill), (x0 + bw, y0 + bh), col, -1)
            _text(img, f"d={f.contact_dist:.3f}", (x0 - 70, y0 + bh + 22), 0.5)
        for thr, col in ((c.touch_on, (0, 255, 0)), (c.touch_off, (0, 0, 255))):
            yy = y0 + bh - int(bh * thr / vmax)
            cv2.line(img, (x0 - 6, yy), (x0 + bw + 6, yy), col, 2)
        _text(img, "d/S", (x0, y0 - 8), 0.5)

        # Displacement pad (direction frame)
        cx, cy, R = w - 150, y0 + bh + 140, 90
        scale = R / max(g.swipe_min_dist * 1.6, 1e-6)
        cv2.rectangle(img, (cx - R, cy - R), (cx + R, cy + R), (40, 40, 40), -1)
        cv2.circle(img, (cx, cy), int(g.swipe_min_dist * scale), (255, 200, 0), 1, cv2.LINE_AA)
        cv2.circle(img, (cx, cy), int(g.tap_max_move * scale), (0, 255, 0), 1, cv2.LINE_AA)
        cv2.circle(img, (cx, cy), int(g.move_tolerance * scale), (255, 0, 255), 1, cv2.LINE_AA)
        cv2.line(img, (cx - R, cy), (cx + R, cy), (90, 90, 90), 1)
        cv2.line(img, (cx, cy - R), (cx, cy + R), (90, 90, 90), 1)
        disp = res.debug.get("disp")
        if disp and disp[0] is not None:
            end = (int(cx + disp[0] * scale), int(cy + disp[1] * scale))
            cv2.arrowedLine(img, (cx, cy), end, (0, 255, 255), 2, tipLength=0.2)
        _text(img, f"disp ({self.cfg.direction.frame})", (cx - R, cy - R - 8), 0.45)
        el = res.debug.get("elapsed_ms")
        if el is not None:
            _text(img, f"{el:.0f} ms  max {res.debug.get('max_disp') or 0:.3f}", (cx - R, cy + R + 18), 0.45)
        if f is not None:
            a = f.angles.get("index_pip", float("nan"))
            _text(img, f"S={f.scale_px:.0f}px dz={f.dz:+.3f} seg={f.nearest_segment} t={f.segment_t:.2f}",
                  (w - 380, cy + R + 40), 0.45)
            _text(img, f"index PIP {a:.0f}deg", (w - 380, cy + R + 60), 0.45)
