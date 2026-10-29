"""Camera opening with configurable capture method, read-back of what the driver actually applied,
and live exposure controls."""
from __future__ import annotations

from typing import Any

import cv2

from .config import CameraConfig

BACKENDS = {"dshow": cv2.CAP_DSHOW, "msmf": cv2.CAP_MSMF, "any": cv2.CAP_ANY}


def fourcc_to_str(value: float) -> str:
    v = int(value)
    if v <= 0:
        return ""
    s = "".join(chr((v >> (8 * i)) & 0xFF) for i in range(4))
    return s if s.isprintable() else f"0x{v:08x}"


def open_camera(cam: CameraConfig) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(cam.index, BACKENDS.get(cam.backend, cv2.CAP_ANY))
    if not cap.isOpened():
        return cap

    def set_fourcc() -> None:
        if cam.fourcc:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*cam.fourcc.ljust(4)[:4]))

    if cam.fourcc_order == "before_size":
        set_fourcc()
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam.height)
    cap.set(cv2.CAP_PROP_FPS, cam.fps)
    if cam.fourcc_order == "after_size":
        set_fourcc()
    if cam.buffer_size is not None:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, cam.buffer_size)
    apply_image_controls(cap, cam)
    return cap


def apply_image_controls(cap: cv2.VideoCapture, cam: CameraConfig) -> None:
    if cam.auto_exposure is not None:
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE,
                cam.auto_exposure_on_value if cam.auto_exposure else cam.auto_exposure_off_value)
    if cam.exposure is not None:
        cap.set(cv2.CAP_PROP_EXPOSURE, cam.exposure)
    if cam.gain is not None:
        cap.set(cv2.CAP_PROP_GAIN, cam.gain)
    if cam.brightness is not None:
        cap.set(cv2.CAP_PROP_BRIGHTNESS, cam.brightness)


def read_back(cap: cv2.VideoCapture) -> dict[str, Any]:
    """Values the driver reports after configuration (may differ from what was requested)."""
    try:
        backend = cap.getBackendName()
    except cv2.error:
        backend = "?"
    return {
        "backend": backend,
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": round(cap.get(cv2.CAP_PROP_FPS), 3),
        "fourcc": fourcc_to_str(cap.get(cv2.CAP_PROP_FOURCC)),
        "buffer_size": cap.get(cv2.CAP_PROP_BUFFERSIZE),
        "auto_exposure": cap.get(cv2.CAP_PROP_AUTO_EXPOSURE),
        "exposure": cap.get(cv2.CAP_PROP_EXPOSURE),
        "gain": cap.get(cv2.CAP_PROP_GAIN),
        "brightness": cap.get(cv2.CAP_PROP_BRIGHTNESS),
    }


def mismatches(cam: CameraConfig, actual: dict[str, Any]) -> list[str]:
    out = []
    if (actual["width"], actual["height"]) != (cam.width, cam.height):
        out.append(f"resolution {actual['width']}x{actual['height']} (requested {cam.width}x{cam.height})")
    if actual["fps"] and abs(actual["fps"] - cam.fps) > 0.5:
        out.append(f"fps {actual['fps']} (requested {cam.fps})")
    if cam.fourcc and actual["fourcc"] and actual["fourcc"].strip() != cam.fourcc.strip():
        out.append(f"fourcc {actual['fourcc']!r} (requested {cam.fourcc!r})")
    return out


class CameraControl:
    """Live exposure adjustments from key presses."""

    def __init__(self, cap: cv2.VideoCapture, cam: CameraConfig) -> None:
        self.cap = cap
        self.cam = cam

    def _is_auto(self) -> bool:
        v = self.cap.get(cv2.CAP_PROP_AUTO_EXPOSURE)
        return abs(v - self.cam.auto_exposure_on_value) < abs(v - self.cam.auto_exposure_off_value)

    def toggle_auto_exposure(self) -> dict[str, Any]:
        auto = not self._is_auto()
        self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE,
                     self.cam.auto_exposure_on_value if auto else self.cam.auto_exposure_off_value)
        return {"action": "auto_exposure", "requested": auto, **self.state()}

    def step_exposure(self, delta: float) -> dict[str, Any]:
        if self._is_auto():
            self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, self.cam.auto_exposure_off_value)
        value = self.cap.get(cv2.CAP_PROP_EXPOSURE) + delta
        self.cap.set(cv2.CAP_PROP_EXPOSURE, value)
        return {"action": "exposure", "requested": value, **self.state()}

    def open_settings_dialog(self) -> dict[str, Any]:
        ok = self.cap.set(cv2.CAP_PROP_SETTINGS, 1)  # DirectShow only
        return {"action": "settings_dialog", "ok": bool(ok), **self.state()}

    def state(self) -> dict[str, Any]:
        return {
            "auto_exposure": self.cap.get(cv2.CAP_PROP_AUTO_EXPOSURE),
            "exposure": self.cap.get(cv2.CAP_PROP_EXPOSURE),
            "gain": self.cap.get(cv2.CAP_PROP_GAIN),
            "brightness": self.cap.get(cv2.CAP_PROP_BRIGHTNESS),
        }

    def summary_line(self) -> str:
        s = self.state()
        return (f"auto_exp={s['auto_exposure']:g} exp={s['exposure']:g} "
                f"gain={s['gain']:g} bright={s['brightness']:g}")

    def config_snippet(self) -> str:
        s = self.state()
        v = s["auto_exposure"]
        lines = ["[camera]"]
        if min(abs(v - self.cam.auto_exposure_on_value), abs(v - self.cam.auto_exposure_off_value)) > 0.01:
            # The driver did not report a usable value (DirectShow returns -1 when it can't read it).
            lines.append(f"# auto_exposure / exposure: not reported by the driver (read {v:g})")
        else:
            auto = self._is_auto()
            lines.append(f"auto_exposure = {'true' if auto else 'false'}")
            if not auto:
                lines.append(f"exposure = {s['exposure']:g}")
        lines += [f"gain = {s['gain']:g}", f"brightness = {s['brightness']:g}"]
        return "\n".join(lines)
