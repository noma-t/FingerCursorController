"""Compare camera capture methods (backend x FOURCC x FOURCC order) and measure real fps and exposure.

Examples:
  python tools/probe_camera.py
  python tools/probe_camera.py --seconds 5 --backends dshow --exposure -6
"""
from __future__ import annotations

import argparse
import copy
import itertools
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fingercursor.camera import open_camera, read_back  # noqa: E402
from fingercursor.config import load_config  # noqa: E402


def probe(cam, seconds: float, warmup: float) -> dict:
    t_open = time.perf_counter()
    cap = open_camera(cam)
    open_s = time.perf_counter() - t_open
    row = {"backend": cam.backend, "fourcc_req": cam.fourcc or "-", "order": cam.fourcc_order,
           "open_s": open_s}
    if not cap.isOpened():
        row["error"] = "open failed"
        return row
    try:
        actual = read_back(cap)
        row.update(size=f"{actual['width']}x{actual['height']}", fourcc=actual["fourcc"] or "-",
                   fps_rep=actual["fps"], exposure=actual["exposure"], auto_exp=actual["auto_exposure"])
        t_end = time.perf_counter() + warmup
        while time.perf_counter() < t_end:  # let auto exposure settle
            if not cap.read()[0]:
                break
        reads, stamps, means, clipped = [], [], [], []
        t_end = time.perf_counter() + seconds
        while time.perf_counter() < t_end:
            t0 = time.perf_counter()
            ok, frame = cap.read()
            t1 = time.perf_counter()
            if not ok:
                row["error"] = "read failed"
                break
            reads.append((t1 - t0) * 1000.0)
            stamps.append(t1)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            means.append(float(gray.mean()))
            clipped.append(float((gray >= 250).mean()) * 100.0)
        if len(stamps) > 1:
            row["fps"] = (len(stamps) - 1) / (stamps[-1] - stamps[0])
            row["read_ms_p50"] = float(np.median(reads))
            row["read_ms_max"] = float(np.max(reads))
            row["brightness"] = float(np.mean(means))
            row["clipped_pct"] = float(np.mean(clipped))
    finally:
        cap.release()
    return row


def fmt(v, spec: str) -> str:
    return "-" if v is None else format(v, spec)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="config.toml")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    ap.add_argument("--backends", default="dshow,msmf,any")
    ap.add_argument("--fourccs", default="MJPG,YUY2,NV12,none", help="'none' = do not request a format")
    ap.add_argument("--orders", default="before_size,after_size")
    ap.add_argument("--seconds", type=float, default=3.0, help="measurement time per combination")
    ap.add_argument("--warmup", type=float, default=1.0)
    ap.add_argument("--exposure", type=float, help="fix the exposure (turns auto exposure off)")
    ap.add_argument("--auto-exposure", choices=["on", "off"])
    args = ap.parse_args()

    base = load_config(args.config, args.set).camera
    if args.exposure is not None:
        base.auto_exposure, base.exposure = False, args.exposure
    if args.auto_exposure:
        base.auto_exposure = args.auto_exposure == "on"
    print(f"camera index={base.index} requested {base.width}x{base.height}@{base.fps} "
          f"auto_exposure={base.auto_exposure} exposure={base.exposure}\n")

    fourccs = ["" if f.lower() == "none" else f for f in args.fourccs.split(",")]
    rows = []
    header = (f"{'backend':<7} {'req':<5} {'order':<12} {'open_s':>6} {'size':>10} {'fourcc':>6} "
              f"{'fps(rep)':>8} {'fps':>6} {'read50':>7} {'readmax':>7} {'bright':>6} {'clip%':>6} "
              f"{'exp':>6} {'auto':>5}")
    print(header)
    print("-" * len(header))
    for backend, fourcc, order in itertools.product(args.backends.split(","), fourccs, args.orders.split(",")):
        if not fourcc and order == "after_size":
            continue  # order is irrelevant when no format is requested
        cam = copy.deepcopy(base)
        cam.backend, cam.fourcc, cam.fourcc_order = backend, fourcc, order
        row = probe(cam, args.seconds, args.warmup)
        rows.append(row)
        if "error" in row and "fps" not in row:
            print(f"{backend:<7} {cam.fourcc or '-':<5} {order:<12} {row['open_s']:6.2f}  {row['error']}")
            continue
        print(f"{backend:<7} {cam.fourcc or '-':<5} {order:<12} {row['open_s']:6.2f} {row.get('size', '-'):>10} "
              f"{row.get('fourcc', '-'):>6} {fmt(row.get('fps_rep'), '8.1f')} {fmt(row.get('fps'), '6.1f')} "
              f"{fmt(row.get('read_ms_p50'), '7.1f')} {fmt(row.get('read_ms_max'), '7.1f')} "
              f"{fmt(row.get('brightness'), '6.1f')} {fmt(row.get('clipped_pct'), '6.1f')} "
              f"{fmt(row.get('exposure'), '6g')} {fmt(row.get('auto_exp'), '5g')}")

    want = f"{base.width}x{base.height}"
    ok = [r for r in rows if r.get("fps") and r.get("size") == want]
    if not ok:
        print(f"\nno combination delivered {want}")
        return
    best = max(ok, key=lambda r: (round(r["fps"]), -r.get("clipped_pct", 0.0), -r["open_s"]))
    print(f"\nbest at {want}: {best['fps']:.1f} fps, clipped {best.get('clipped_pct', 0):.1f}%")
    print("put this in config.toml:\n")
    print("[camera]")
    print(f'backend = "{best["backend"]}"')
    print(f'fourcc = "{"" if best["fourcc_req"] == "-" else best["fourcc_req"]}"')
    print(f'fourcc_order = "{best["order"]}"')
    print("\nclip% = share of near-white pixels (>=250). High values mean overexposure: "
          "try --exposure -6 / -7 / -8.")


if __name__ == "__main__":
    main()
