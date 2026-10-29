"""Session logging: session.json + frames.jsonl + events.jsonl + labels.jsonl."""
from __future__ import annotations

import datetime as _dt
import json
import platform
import sys
from pathlib import Path
from typing import Any

import numpy as np

from .config import Config
from .pipeline import FrameResult


def _round_arr(arr: np.ndarray | None, nd: int) -> list | None:
    if arr is None:
        return None
    return np.round(np.asarray(arr, dtype=np.float64), nd).tolist()


def _versions() -> dict[str, str]:
    out = {"python": sys.version.split()[0], "platform": platform.platform()}
    for mod in ("mediapipe", "cv2", "numpy"):
        try:
            out[mod] = __import__(mod).__version__
        except Exception:  # noqa: BLE001
            out[mod] = "?"
    return out


class SessionLogger:
    def __init__(self, cfg: Config, meta: dict[str, Any], session_dir: Path | None = None) -> None:
        self.cfg = cfg
        self.nd = cfg.logging.float_digits
        stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.dir = Path(session_dir) if session_dir else Path(cfg.logging.dir) / stamp
        self.dir.mkdir(parents=True, exist_ok=True)
        self.session = {
            "created": _dt.datetime.now().isoformat(timespec="seconds"),
            "versions": _versions(),
            "config": cfg.to_dict(),
            **meta,
        }
        self._write_session()
        self._frames = open(self.dir / "frames.jsonl", "w", encoding="utf-8") if cfg.logging.frames else None
        self._events = open(self.dir / "events.jsonl", "w", encoding="utf-8")
        self._labels = open(self.dir / "labels.jsonl", "w", encoding="utf-8")
        self.counts: dict[str, int] = {}

    def _write_session(self) -> None:
        with open(self.dir / "session.json", "w", encoding="utf-8") as f:
            json.dump(self.session, f, ensure_ascii=False, indent=2)

    def update_meta(self, **kwargs: Any) -> None:
        self.session.update(kwargs)
        self._write_session()

    def log_frame(self, frame_idx: int, res: FrameResult, hand: Any | None, timing: dict[str, float],
                  n_hands: int, landmarks: np.ndarray | None, world: np.ndarray | None) -> None:
        if self._frames is None:
            return
        nd = self.nd
        rec: dict[str, Any] = {
            "i": frame_idx,
            "t_ms": round(res.t_ms, 2),
            "timing": {k: round(v, 2) for k, v in timing.items()},
            "n_hands": n_hands,
            "hand": None,
            "rec": res.debug,
        }
        if hand is not None:
            rec["hand"] = {"label": hand.handedness, "score": round(hand.handedness_score, 4)}
            rec["lm"] = _round_arr(landmarks, nd)
            if self.cfg.logging.world_landmarks:
                rec["world"] = _round_arr(world, nd)
            # Filtered key points (display px): wrist, thumb IP/tip, index MCP..TIP, middle MCP
            if res.px is not None:
                rec["px_f"] = {str(k): _round_arr(res.px[k, :2], 2) for k in (0, 3, 4, 5, 6, 7, 8, 9)}
            if res.features is not None:
                rec["f"] = res.features.to_log(nd)
            if res.position is not None:
                rec["pos"] = _round_arr(res.position, nd)
        if res.events:
            rec["ev"] = [e.type for e in res.events]
        self._frames.write(json.dumps(rec, separators=(",", ":")) + "\n")

    def log_event(self, frame_idx: int, event: Any) -> None:
        rec = {"i": frame_idx, **event.to_log()}
        self.counts[event.type] = self.counts.get(event.type, 0) + 1
        self._events.write(json.dumps(rec, ensure_ascii=False) + "\n")
        self._events.flush()

    def log_label(self, frame_idx: int, t_ms: float, label: dict[str, Any]) -> None:
        rec = {"i": frame_idx, "t_ms": round(t_ms, 2), **label}
        self._labels.write(json.dumps(rec, ensure_ascii=False) + "\n")
        self._labels.flush()

    def close(self, summary: dict[str, Any] | None = None) -> None:
        for fh in (self._frames, self._events, self._labels):
            if fh is not None:
                fh.close()
        self.update_meta(event_counts=self.counts, summary=summary or {})
