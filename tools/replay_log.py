"""Re-run gesture recognition on a recorded session's raw landmarks with different parameters.

Examples:
  python tools/replay_log.py logs/20261029_120000
  python tools/replay_log.py logs/20261029_120000 --set contact.touch_on=0.35 --events
  python tools/replay_log.py logs/20261029_120000 --sweep contact.touch_on=0.25,0.3,0.35 --sweep gesture.swipe_min_dist=0.2,0.3
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fingercursor.analysis import events_from_log, load_session, replay, score, session_config  # noqa: E402
from fingercursor.recognizer import GESTURE_EVENTS  # noqa: E402


def fmt_event(e) -> str:
    d = e.data
    extra = []
    for k in ("duration_ms", "disp", "max_disp", "peak_speed", "min_d", "reasons", "reason"):
        if d.get(k) is not None:
            extra.append(f"{k}={d[k]}")
    return f"{e.t_ms / 1000:9.3f}s {e.type:<17} " + " ".join(extra)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("session", help="session directory (logs/YYYYMMDD_HHMMSS)")
    ap.add_argument("--config", help="use this TOML instead of the config stored in the session")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    ap.add_argument("--sweep", action="append", default=[], metavar="KEY=V1,V2,...",
                    help="grid-search a parameter (repeatable)")
    ap.add_argument("--events", action="store_true", help="print all replayed events")
    ap.add_argument("--all-events", action="store_true", help="include debug events (TOUCH_START, ...)")
    ap.add_argument("--json", action="store_true", help="print results as JSON")
    args = ap.parse_args()

    session = load_session(args.session)
    print(f"session: {session.dir}  frames={len(session.frames)}  labels={len(session.labels)}")

    original = events_from_log(session)
    orig_score = score(session, original)
    print(f"original run : {json.dumps(orig_score.counts(), ensure_ascii=False)}")

    if args.sweep:
        keys, values = [], []
        for s in args.sweep:
            k, v = s.split("=", 1)
            keys.append(k)
            values.append(v.split(","))
        rows = []
        for combo in itertools.product(*values):
            overrides = args.set + [f"{k}={v}" for k, v in zip(keys, combo)]
            cfg = session_config(session, args.config, overrides)
            evs = replay(session, cfg)
            c = score(session, evs).counts()
            n_g = sum(1 for e in evs if e.type in GESTURE_EVENTS)
            rows.append((combo, c, n_g))
            acc = c.get("trials", {}).get("accuracy")
            print(f"{' '.join(f'{k}={v}' for k, v in zip(keys, combo)):<60} "
                  f"acc={acc} fp={c['false_positives']} gestures={n_g} "
                  f"{json.dumps(c.get('trials', {}).get('total', c.get('manual_labels', {})))}")
        return

    cfg = session_config(session, args.config, args.set)
    evs = replay(session, cfg)
    sc = score(session, evs)
    print(f"replayed run : {json.dumps(sc.counts(), ensure_ascii=False)}")
    if args.json:
        print(json.dumps({"trials": sc.trials, "manual": sc.manual, "false_positives": sc.false_positives},
                         ensure_ascii=False, indent=1))
    else:
        bad = [t for t in sc.trials if t["result"] != "ok"] + [m for m in sc.manual if m["result"] != "ok"]
        if bad:
            print("\nfailed trials / labels:")
            for b in bad:
                print("  ", b)
        if sc.false_positives:
            print("\nfalse positives:")
            for fp in sc.false_positives:
                print("  ", fp)
    if args.events or args.all_events:
        print("\nevents:")
        for e in evs:
            if args.all_events or e.type in GESTURE_EVENTS or e.type in ("TOUCH_REJECTED", "TOUCH_ABORTED"):
                print("  " + fmt_event(e))


if __name__ == "__main__":
    main()
