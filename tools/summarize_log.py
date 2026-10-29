"""Summarise a recorded session: tracking quality, contact-distance distribution, contacts, events, ground truth.

Example:
  python tools/summarize_log.py logs/20261029_120000 > summary.txt
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fingercursor.analysis import ascii_hist, events_from_log, load_session, percentiles, score  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("session")
    args = ap.parse_args()

    s = load_session(args.session)
    cfg = s.meta.get("config", {})
    c, g = cfg.get("contact", {}), cfg.get("gesture", {})
    frames = s.frames
    print(f"# session {s.dir}")
    print(f"source={s.meta.get('source')} size={s.meta.get('frame_size')} versions={s.meta.get('versions')}")
    print(f"summary={json.dumps(s.meta.get('summary', {}), ensure_ascii=False)}")
    print(f"camera_actual={s.meta.get('camera_actual')}")
    print(f"camera_final={s.meta.get('camera_final')}")

    # ---- tracking
    n = len(frames)
    with_hand = [f for f in frames if f.get("f")]
    print("\n## tracking")
    print(f"frames={n} with_hand={len(with_hand)} ({100 * len(with_hand) / max(1, n):.1f}%)")
    if n > 1:
        dts = np.diff([f["t_ms"] for f in frames])
        print(f"frame dt ms: {percentiles(dts.tolist(), (5, 50, 95, 99))}  max={dts.max():.1f}")
        print(f"measured fps (frames / duration): {(n - 1) / max(1e-6, (frames[-1]['t_ms'] - frames[0]['t_ms']) / 1000):.1f}")
    for key in ("infer_ms", "read_ms", "proc_ms"):
        vals = [f["timing"][key] for f in frames if key in f.get("timing", {})]
        print(f"{key}: {percentiles(vals, (50, 95, 99))}")
    cam_changes = [l for l in s.labels if l.get("type") == "camera_change"]
    for ch in cam_changes:
        print(f"camera change @ {ch['t_ms'] / 1000:.2f}s: {ch}")
    hands = Counter(f["hand"]["label"] for f in frames if f.get("hand"))
    print(f"handedness: {dict(hands)}")
    scores = [f["hand"]["score"] for f in frames if f.get("hand")]
    print(f"handedness score: {percentiles(scores, (5, 50))}")
    lost_runs, run = [], 0
    for f in frames:
        if f.get("f"):
            if run:
                lost_runs.append(run)
            run = 0
        else:
            run += 1
    print(f"lost runs (frames): count={len(lost_runs)} {percentiles(lost_runs, (50, 90, 100)) if lost_runs else ''}")

    # ---- features
    d = [f["f"]["d"] for f in with_hand]
    print("\n## contact distance d/S")
    print(percentiles(d))
    print(ascii_hist(d, 0.0, max(1.0, c.get("touch_off", 0.4) * 2.5), bins=30,
                     marks={"touch_on": c.get("touch_on"), "touch_off": c.get("touch_off")}))
    seg = Counter(f["f"]["seg"] for f in with_hand if f["f"]["d"] < c.get("touch_off", 0.4))
    print(f"nearest segment while d<touch_off: {dict(seg)}")
    for key in ("S_px", "dz"):
        print(f"{key}: {percentiles([f['f'][key] for f in with_hand], (5, 50, 95))}")
    for ang in ("index_pip", "index_mcp", "thumb_ip"):
        vals = [f["f"]["ang"][ang] for f in with_hand if f["f"]["ang"].get(ang) is not None]
        print(f"angle {ang}: {percentiles(vals, (5, 50, 95))}")

    # d per state
    by_state: dict[str, list[float]] = {}
    for f in with_hand:
        by_state.setdefault(f["rec"]["state"], []).append(f["f"]["d"])
    print("\nd by recognizer state:")
    for st, vals in sorted(by_state.items()):
        print(f"  {st:<10} n={len(vals):<6} {percentiles(vals, (5, 50, 95))}")

    # d during guided trial windows by expected gesture
    trial_starts = [l for l in s.labels if l.get("type") == "trial_start"]
    trial_ends = {l["trial"]: l for l in s.labels if l.get("type") == "trial"}
    if trial_starts:
        print("\nmin d inside each guided trial window (per expected gesture):")
        per: dict[str, list[float]] = {}
        for ts in trial_starts:
            te = trial_ends.get(ts["trial"])
            t1 = te["t_end"] if te else ts["t_ms"] + cfg.get("guided", {}).get("timeout_ms", 5000)
            ds = [f["f"]["d"] for f in with_hand if ts["t_ms"] <= f["t_ms"] <= t1]
            if ds:
                per.setdefault(ts["expected"], []).append(min(ds))
        for k, v in per.items():
            print(f"  {k:<12} {np.round(v, 3).tolist()}")

    # ---- events
    evs = events_from_log(s)
    print("\n## events")
    print(dict(Counter(e.type for e in evs)))
    rej = Counter(r for e in evs if e.type == "TOUCH_REJECTED" for r in e.data.get("reasons", []))
    print(f"touch rejected reasons: {dict(rej)}")
    ends = [e for e in evs if e.type == "TOUCH_END"]
    if ends:
        print("contacts (TOUCH_END):")
        for key in ("duration_ms", "max_disp", "path_len", "peak_speed", "min_d"):
            vals = [e.data[key] for e in ends if e.data.get(key) is not None]
            print(f"  {key:<12} {percentiles(vals, (10, 50, 90))}")
    print(f"thresholds: contact={c}\n            gesture={g}")

    # ---- ground truth
    sc = score(s, evs)
    print("\n## ground truth")
    print(json.dumps(sc.counts(), ensure_ascii=False))
    for t in sc.trials:
        if t["result"] != "ok":
            print("  trial", t)
    for m in sc.manual:
        if m["result"] != "ok":
            print("  label", m)
    for fp in sc.false_positives:
        print("  false positive", fp)


if __name__ == "__main__":
    main()
