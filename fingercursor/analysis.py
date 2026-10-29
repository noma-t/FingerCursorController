"""Offline analysis helpers: load a session, replay it with other parameters, score against ground truth."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .config import Config, apply_override, config_from_dict, load_config
from .guided import EVENT_TO_LABEL, LABEL_TO_EVENT
from .pipeline import Pipeline
from .recognizer import Event

RESET_GAP_MS = 500.0  # frame-time gap (e.g. pause) after which the pipeline is reset


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


@dataclass
class Session:
    dir: Path
    meta: dict[str, Any]
    frames: list[dict[str, Any]]
    events: list[dict[str, Any]]
    labels: list[dict[str, Any]]

    @property
    def size(self) -> tuple[int, int]:
        w, h = self.meta.get("frame_size", [1280, 720])
        return int(w), int(h)


def load_session(path: str | Path) -> Session:
    d = Path(path)
    with open(d / "session.json", encoding="utf-8") as f:
        meta = json.load(f)
    return Session(d, meta, _read_jsonl(d / "frames.jsonl"), _read_jsonl(d / "events.jsonl"),
                   _read_jsonl(d / "labels.jsonl"))


def session_config(session: Session, config_path: str | None = None, overrides: list[str] | None = None) -> Config:
    if config_path:
        cfg = load_config(config_path)
    else:
        cfg = config_from_dict(session.meta.get("config", {}))
    for o in overrides or []:
        apply_override(cfg, o)
    return cfg


def replay(session: Session, cfg: Config) -> list[Event]:
    w, h = session.size
    pipeline = Pipeline(cfg)
    events: list[Event] = []
    last_t = None
    for fr in session.frames:
        t = float(fr["t_ms"])
        if last_t is not None and t - last_t > RESET_GAP_MS:
            pipeline = Pipeline(cfg)
        last_t = t
        lm = np.array(fr["lm"]) if fr.get("lm") is not None else None
        world = np.array(fr["world"]) if fr.get("world") is not None else None
        res = pipeline.process(t, lm, world, w, h)
        events.extend(res.events)
    return events


def events_from_log(session: Session) -> list[Event]:
    out = []
    for e in session.events:
        data = {k: v for k, v in e.items() if k not in ("type", "t_ms", "gesture", "i")}
        out.append(Event(e["type"], float(e["t_ms"]), data))
    return out


@dataclass
class Score:
    trials: list[dict[str, Any]] = field(default_factory=list)
    manual: list[dict[str, Any]] = field(default_factory=list)
    false_positives: list[dict[str, Any]] = field(default_factory=list)

    def counts(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        if self.trials:
            per: dict[str, dict[str, int]] = {}
            for t in self.trials:
                per.setdefault(t["expected"], {"ok": 0, "wrong": 0, "miss": 0})[t["result"]] += 1
            tot = {k: sum(p[k] for p in per.values()) for k in ("ok", "wrong", "miss")}
            out["trials"] = {"total": tot, "per_gesture": per,
                             "accuracy": round(tot["ok"] / max(1, len(self.trials)), 3)}
        if self.manual:
            ok = sum(1 for m in self.manual if m["result"] == "ok")
            out["manual_labels"] = {"n": len(self.manual), "ok": ok}
        out["false_positives"] = len(self.false_positives)
        return out


def score(session: Session, events: list[Event], manual_window: tuple[float, float] = (2500.0, 1000.0)) -> Score:
    """Score gesture events against guided trials and manual labels.

    Guided trial window = [trial_start, trial_start + timeout]. The first gesture event in the
    window decides ok/wrong; none -> miss. Gesture events outside every window and not matched
    to a manual label are counted as false positives.
    """
    sc = Score()
    gestures = [e for e in events if e.type in EVENT_TO_LABEL]
    used: set[int] = set()
    timeout = float(session.meta.get("config", {}).get("guided", {}).get("timeout_ms", 5000.0))

    windows = []
    for lab in session.labels:
        if lab.get("type") == "trial_start":
            windows.append((lab["trial"], lab["expected"], float(lab["t_ms"]), float(lab["t_ms"]) + timeout))
    for trial, expected, t0, t1 in windows:
        hit = next((i for i, e in enumerate(gestures) if t0 <= e.t_ms <= t1 and i not in used), None)
        if hit is None:
            sc.trials.append({"trial": trial, "expected": expected, "result": "miss", "t_start": t0})
        else:
            used.add(hit)
            got = EVENT_TO_LABEL[gestures[hit].type]
            sc.trials.append({"trial": trial, "expected": expected, "got": got, "t_start": t0,
                              "latency_ms": round(gestures[hit].t_ms - t0, 1),
                              "result": "ok" if got == expected else "wrong"})
    # Other gesture events inside a trial window (after the first) are extras, not false positives.
    for i, e in enumerate(gestures):
        if any(t0 <= e.t_ms <= t1 for _, _, t0, t1 in windows):
            used.add(i)

    before, after = manual_window
    for lab in session.labels:
        if lab.get("type") != "label" or lab.get("label") not in LABEL_TO_EVENT:
            continue
        t = float(lab["t_ms"])
        cands = [(abs(e.t_ms - t), i) for i, e in enumerate(gestures)
                 if i not in used and t - before <= e.t_ms <= t + after]
        if not cands:
            sc.manual.append({"label": lab["label"], "t_ms": t, "result": "miss"})
            continue
        _, i = min(cands)
        used.add(i)
        got = EVENT_TO_LABEL[gestures[i].type]
        sc.manual.append({"label": lab["label"], "t_ms": t, "got": got,
                          "result": "ok" if got == lab["label"] else "wrong"})

    for i, e in enumerate(gestures):
        if i not in used:
            sc.false_positives.append({"type": e.type, "t_ms": round(e.t_ms, 1)})
    return sc


def percentiles(values: list[float], ps=(1, 5, 10, 25, 50, 75, 90, 95, 99)) -> dict[str, float]:
    if not values:
        return {}
    arr = np.asarray(values, dtype=np.float64)
    return {f"p{p}": round(float(np.percentile(arr, p)), 4) for p in ps}


def ascii_hist(values: list[float], lo: float, hi: float, bins: int = 30, width: int = 50,
               marks: dict[str, float] | None = None) -> str:
    if not values:
        return "(no data)"
    arr = np.clip(np.asarray(values), lo, hi)
    counts, edges = np.histogram(arr, bins=bins, range=(lo, hi))
    peak = max(1, counts.max())
    lines = []
    for c, a, b in zip(counts, edges[:-1], edges[1:]):
        tag = ""
        for name, v in (marks or {}).items():
            if a <= v < b:
                tag += f" <- {name}={v}"
        lines.append(f"{a:6.3f}-{b:6.3f} | {'#' * int(width * c / peak):<{width}} {c}{tag}")
    return "\n".join(lines)
