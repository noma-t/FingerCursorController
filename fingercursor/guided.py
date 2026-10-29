"""Guided test mode: prompts gestures one by one and records ground truth + outcome."""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

from .config import GuidedConfig

# Expected gesture label -> recognizer event type that satisfies it.
LABEL_TO_EVENT = {
    "swipe_left": "SWIPE_LEFT",
    "swipe_right": "SWIPE_RIGHT",
    "swipe_up": "SWIPE_UP",
    "swipe_down": "SWIPE_DOWN",
    "tap": "TAP",
    "long_press": "LONG_PRESS_START",
}
EVENT_TO_LABEL = {v: k for k, v in LABEL_TO_EVENT.items()}


@dataclass
class Trial:
    index: int
    expected: str
    t_start: float = 0.0
    t_end: float = 0.0
    detected: list[dict[str, Any]] = field(default_factory=list)
    result: str = ""


class GuidedSession:
    def __init__(self, cfg: GuidedConfig, t_ms: float) -> None:
        self.cfg = cfg
        labels = [g for g in cfg.gestures for _ in range(cfg.repeats)]
        if cfg.shuffle:
            random.shuffle(labels)
        self.trials = [Trial(i, lab) for i, lab in enumerate(labels)]
        self.pos = 0
        self.active = False
        self.next_t = t_ms + 2000.0
        self.done = False
        self.gap_events: list[dict[str, Any]] = []
        self.results = {"ok": 0, "wrong": 0, "miss": 0}

    @property
    def current(self) -> Trial | None:
        return self.trials[self.pos] if self.pos < len(self.trials) else None

    def prompt(self, t_ms: float) -> str:
        if self.done:
            r = self.results
            return f"GUIDED DONE  ok {r['ok']}  wrong {r['wrong']}  miss {r['miss']}"
        tr = self.current
        assert tr is not None
        head = f"[{self.pos + 1}/{len(self.trials)}] "
        if self.active:
            left = (tr.t_start + self.cfg.timeout_ms - t_ms) / 1000.0
            return head + f"DO: {tr.expected.upper().replace('_', ' ')}  ({left:.1f}s)"
        return head + f"next: {tr.expected.upper().replace('_', ' ')} ..."

    def update(self, t_ms: float) -> list[dict[str, Any]]:
        """Advance timers. Returns label records to log."""
        out: list[dict[str, Any]] = []
        if self.done:
            return out
        tr = self.current
        assert tr is not None
        if not self.active and t_ms >= self.next_t:
            self.active = True
            tr.t_start = t_ms
            out.append({"type": "trial_start", "trial": tr.index, "expected": tr.expected})
        elif self.active and t_ms - tr.t_start >= self.cfg.timeout_ms:
            out.append(self._finish(tr, t_ms, "miss"))
        return out

    def on_event(self, ev_type: str, t_ms: float, data: dict[str, Any]) -> list[dict[str, Any]]:
        if self.done or ev_type not in EVENT_TO_LABEL:
            return []
        rec = {"type": ev_type, "t_ms": round(t_ms, 2)}
        if not self.active:
            self.gap_events.append(rec)
            return [{"type": "gap_event", "event": ev_type}]
        tr = self.current
        assert tr is not None
        tr.detected.append(rec)
        result = "ok" if EVENT_TO_LABEL[ev_type] == tr.expected else "wrong"
        return [self._finish(tr, t_ms, result)]

    def _finish(self, tr: Trial, t_ms: float, result: str) -> dict[str, Any]:
        tr.t_end = t_ms
        tr.result = result
        self.results[result] += 1
        self.active = False
        self.pos += 1
        self.next_t = t_ms + self.cfg.gap_ms
        if self.pos >= len(self.trials):
            self.done = True
        return {
            "type": "trial",
            "trial": tr.index,
            "expected": tr.expected,
            "t_start": round(tr.t_start, 2),
            "t_end": round(tr.t_end, 2),
            "result": result,
            "detected": tr.detected,
        }

    def summary(self) -> dict[str, Any]:
        per: dict[str, dict[str, int]] = {}
        for tr in self.trials[: self.pos]:
            d = per.setdefault(tr.expected, {"ok": 0, "wrong": 0, "miss": 0})
            d[tr.result] += 1
        return {"results": self.results, "per_gesture": per, "gap_events": len(self.gap_events),
                "completed": self.pos, "total": len(self.trials)}
