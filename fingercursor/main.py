"""Live app: webcam (or video file) -> hand tracking -> gesture recognition -> overlay + logs."""
from __future__ import annotations

import argparse
import sys
import time
from collections import deque
from pathlib import Path

import cv2

from .camera import CameraControl, mismatches, open_camera, read_back
from .config import load_config
from .guided import GuidedSession
from .logger import SessionLogger
from .overlay import Overlay
from .pipeline import FrameResult, Pipeline
from .tracker import HandTracker

# Key codes from cv2.waitKeyEx (Windows / Linux-GTK) and letter fallbacks.
LABEL_KEYS = {
    2424832: "swipe_left", 65361: "swipe_left", ord("j"): "swipe_left",
    2555904: "swipe_right", 65363: "swipe_right", ord("l"): "swipe_right",
    2490368: "swipe_up", 65362: "swipe_up", ord("i"): "swipe_up",
    2621440: "swipe_down", 65364: "swipe_down", ord("k"): "swipe_down",
    ord("t"): "tap",
    ord("h"): "long_press",
    ord(" "): "marker",
}


def open_capture(args, cfg):
    cap = cv2.VideoCapture(args.input) if args.input else open_camera(cfg.camera)
    if not cap.isOpened():
        sys.exit(f"could not open {'video ' + args.input if args.input else f'camera {cfg.camera.index}'}")
    return cap


def describe_event(ev) -> str:
    d = ev.data
    parts = [ev.type]
    if "duration_ms" in d:
        parts.append(f"dur={d['duration_ms']:.0f}ms")
    if d.get("disp"):
        parts.append(f"disp=({d['disp'][0]:+.2f},{d['disp'][1]:+.2f})")
    if d.get("peak_speed") is not None:
        parts.append(f"vmax={d['peak_speed']:.2f}S/s")
    if d.get("reasons"):
        parts.append("reasons=" + ",".join(d["reasons"]))
    if d.get("reason"):
        parts.append("reason=" + d["reason"])
    return " ".join(parts)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Thumb-on-index gesture recognizer")
    ap.add_argument("--config", default="config.toml", help="TOML config file (default: config.toml)")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="override a config value, e.g. --set gesture.swipe_min_dist=0.3")
    ap.add_argument("--input", help="process a video file instead of the camera")
    ap.add_argument("--camera", type=int, metavar="INDEX",
                    help="camera index to use (overrides camera.index; list them with tools/list_cameras.py)")
    ap.add_argument("--record", action="store_true", help="save the raw camera video into the session folder")
    ap.add_argument("--guided", action="store_true", help="guided test mode (prompts gestures, records results)")
    ap.add_argument("--no-window", action="store_true", help="run without a window (video input only)")
    ap.add_argument("--verbose", action="store_true", help="print debug events (TOUCH_START, ...) to the console")
    args = ap.parse_args(argv)

    cfg = load_config(args.config, args.set)
    if args.camera is not None:
        cfg.camera.index = args.camera
    cap = open_capture(args, cfg)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or float(cfg.camera.fps)
    camera_actual = read_back(cap)
    cam_ctl = None if args.input else CameraControl(cap, cfg.camera)
    if cam_ctl is not None:
        c = cfg.camera
        print(f"[camera] requested: backend={c.backend} fourcc={c.fourcc!r} order={c.fourcc_order} "
              f"{c.width}x{c.height}@{c.fps}")
        print(f"[camera] actual:    {camera_actual}")
        for m in mismatches(c, camera_actual):
            print(f"[camera] WARNING: {m}")

    tracker = HandTracker(cfg.tracker)
    pipeline = Pipeline(cfg)
    overlay = Overlay(cfg)
    logger = SessionLogger(cfg, {
        "source": args.input or f"camera:{cfg.camera.index}",
        "frame_size": [width, height],
        "source_fps": src_fps,
        "camera_actual": camera_actual,
        "argv": sys.argv,
        "guided": args.guided,
    })
    print(f"[main] logging to {logger.dir}  ({width}x{height} @ {src_fps:.1f}fps)")

    writer = None
    if args.record or cfg.logging.record_video:
        writer = cv2.VideoWriter(str(logger.dir / "video.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                                 src_fps, (width, height))

    show = not args.no_window
    if show:
        cv2.namedWindow(cfg.display.window_name, cv2.WINDOW_NORMAL)

    t0 = time.perf_counter()
    guided = GuidedSession(cfg.guided, 0.0) if args.guided else None
    frame_idx = 0
    paused = False
    frame_times: deque[float] = deque(maxlen=30)
    infer_hist: list[float] = []
    lost_frames = 0
    read_hist: deque[float] = deque(maxlen=30)
    cam_line = cam_ctl.summary_line() if cam_ctl is not None else ""
    prev_wall = None

    try:
        while True:
            t_read0 = time.perf_counter()
            ok, frame = cap.read()
            t_read1 = time.perf_counter()
            if not ok:
                if args.input:
                    break
                print("[main] camera read failed")
                continue
            if args.input:
                t_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
            else:
                t_ms = (t_read1 - t0) * 1000.0
            if paused:
                display = cv2.flip(frame, 1) if cfg.display.mirror else frame
                if show:
                    overlay.draw(display, _empty_result(t_ms), {"paused": True})
                    cv2.imshow(cfg.display.window_name, display)
                key = cv2.waitKeyEx(1) if show else -1
                if key in (ord("p"), ord("P")):
                    paused = False
                    logger.log_label(frame_idx, t_ms, {"type": "resume"})
                    pipeline = Pipeline(cfg)  # replay resets on the same time gap
                elif key in (ord("q"), 27):
                    break
                continue

            if writer is not None:
                writer.write(frame)

            t_inf0 = time.perf_counter()
            hands = tracker.detect(frame, t_ms)
            infer_ms = (time.perf_counter() - t_inf0) * 1000.0
            infer_hist.append(infer_ms)
            hand = hands[0] if hands else None
            if hand is None:
                lost_frames += 1

            t_p0 = time.perf_counter()
            res = pipeline.process(t_ms, hand.landmarks if hand else None,
                                   hand.world if hand else None, width, height)
            proc_ms = (time.perf_counter() - t_p0) * 1000.0
            read_hist.append((t_read1 - t_read0) * 1000.0)

            for ev in res.events:
                logger.log_event(frame_idx, ev)
                if ev.is_gesture or ev.type in ("TOUCH_REJECTED", "TOUCH_ABORTED") or args.verbose:
                    text = f"{ev.t_ms / 1000:8.3f}s  {describe_event(ev)}"
                    print(("  " if not ev.is_gesture else ">> ") + text)
                if ev.is_gesture:
                    overlay.note_event(ev.type, t_ms, f"{ev.t_ms / 1000:7.2f}s {ev.type}")
                if guided is not None:
                    for rec in guided.on_event(ev.type, ev.t_ms, ev.data):
                        logger.log_label(frame_idx, t_ms, rec)
                        if rec["type"] == "trial":
                            print(f"   [guided] {rec['expected']} -> {rec['result']}")
            if guided is not None:
                for rec in guided.update(t_ms):
                    logger.log_label(frame_idx, t_ms, rec)
                    if rec["type"] == "trial":
                        print(f"   [guided] {rec['expected']} -> {rec['result']}")

            now = time.perf_counter()
            if prev_wall is not None:
                frame_times.append(now - prev_wall)
            prev_wall = now
            fps = len(frame_times) / sum(frame_times) if frame_times else 0.0
            logger.log_frame(frame_idx, res, hand, {
                "read_ms": (t_read1 - t_read0) * 1000.0, "infer_ms": infer_ms, "proc_ms": proc_ms, "fps": fps,
            }, len(hands), hand.landmarks if hand else None, hand.world if hand else None)

            key = -1
            if show:
                display = cv2.flip(frame, 1) if cfg.display.mirror else frame
                status = {
                    "fps": fps, "infer_ms": infer_ms, "frame": frame_idx,
                    "read_ms": sum(read_hist) / len(read_hist),
                    "camera": f"{camera_actual['backend']} {camera_actual['fourcc'] or '-'} "
                              f"{width}x{height}  {cam_line}",
                    "hand": f"{hand.handedness} {hand.handedness_score:.2f}" if hand else "no hand",
                    "recording": writer is not None,
                    "guided": guided.prompt(t_ms) if guided is not None else None,
                }
                overlay.draw(display, res, status)
                cv2.imshow(cfg.display.window_name, display)
                key = cv2.waitKeyEx(1)

            if key in (ord("q"), 27):
                break
            if key in (ord("p"), ord("P")):
                paused = True
                logger.log_label(frame_idx, t_ms, {"type": "pause"})
            elif key in (ord("d"), ord("D")):
                overlay.show_debug = not overlay.show_debug
            elif cam_ctl is not None and key in (ord("e"), ord("["), ord("]"), ord("s")):
                if key == ord("e"):
                    change = cam_ctl.toggle_auto_exposure()
                elif key == ord("s"):
                    change = cam_ctl.open_settings_dialog()
                else:
                    change = cam_ctl.step_exposure(-1.0 if key == ord("[") else 1.0)
                cam_line = cam_ctl.summary_line()
                logger.log_label(frame_idx, t_ms, {"type": "camera_change", **change})
                overlay.history.appendleft(f"{t_ms / 1000:7.2f}s [camera] {cam_line}")
                print(f"   [camera] {change}")
            elif key in LABEL_KEYS:
                label = LABEL_KEYS[key]
                logger.log_label(frame_idx, t_ms, {"type": "label", "label": label})
                overlay.history.appendleft(f"{t_ms / 1000:7.2f}s [label] {label}")
                print(f"   [label] {label} @ {t_ms / 1000:.3f}s")
            frame_idx += 1
            if show and cv2.getWindowProperty(cfg.display.window_name, cv2.WND_PROP_VISIBLE) < 1:
                break
    except KeyboardInterrupt:
        pass
    finally:
        if cam_ctl is not None:
            final_cam = cam_ctl.state()
            logger.update_meta(camera_final=final_cam)
            print("[camera] final settings (copy into config.toml to keep them):")
            print(cam_ctl.config_snippet())
        cap.release()
        if writer is not None:
            writer.release()
        tracker.close()
        if show:
            cv2.destroyAllWindows()
        infer_sorted = sorted(infer_hist)
        summary = {
            "frames": frame_idx,
            "frames_without_hand": lost_frames,
            "duration_s": round(time.perf_counter() - t0, 2),
            "infer_ms_median": round(infer_sorted[len(infer_sorted) // 2], 2) if infer_sorted else None,
        }
        if guided is not None:
            summary["guided"] = guided.summary()
        logger.close(summary)
        print(f"[main] session saved: {Path(logger.dir).resolve()}")
        print(f"[main] events: {logger.counts}")
        if guided is not None:
            print(f"[main] guided: {summary['guided']}")


def _empty_result(t_ms: float) -> FrameResult:
    return FrameResult(t_ms=t_ms, debug={"state": "IDLE"})


if __name__ == "__main__":
    main()
