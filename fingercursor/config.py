"""Configuration: dataclass defaults, overridable by a TOML file and `--set key=value`."""
from __future__ import annotations

import ast
import dataclasses
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class CameraConfig:
    index: int = 0
    width: int = 1280
    height: int = 720
    fps: int = 30
    # "dshow" / "msmf" / "any". DirectShow opens fastest on Windows.
    backend: str = "dshow"
    # Request MJPG from the camera (often needed for 720p@30+ over USB).
    mjpg: bool = True


@dataclass
class TrackerConfig:
    model_path: str = "models/hand_landmarker.task"
    model_url: str = (
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
        "hand_landmarker/float16/latest/hand_landmarker.task"
    )
    num_hands: int = 1
    min_detection_confidence: float = 0.5
    min_presence_confidence: float = 0.5
    min_tracking_confidence: float = 0.5


@dataclass
class FilterConfig:
    """One Euro Filter applied to landmarks in display-pixel units."""
    enabled: bool = True
    min_cutoff: float = 1.5   # Hz. Lower = smoother at rest, more lag.
    beta: float = 0.01        # Speed coefficient. Higher = less lag on fast motion.
    d_cutoff: float = 1.0     # Hz, cutoff for the derivative.


@dataclass
class ContactConfig:
    """Thumb-tip to index-finger contact detection (all distances in hand-scale units S)."""
    touch_on: float = 0.30    # d/S below this -> contact candidate
    touch_off: float = 0.40   # d/S above this -> release candidate (hysteresis)
    on_frames: int = 2        # consecutive frames below touch_on to confirm contact
    off_frames: int = 2       # consecutive frames above touch_off to confirm release
    # Index segments used for distance: 0 = MCP-PIP(5-6), 1 = PIP-DIP(6-7), 2 = DIP-TIP(7-8)
    segments: list[int] = field(default_factory=lambda: [0, 1, 2])
    # Optional gate on MediaPipe relative depth |z_thumb - z_index_nearest| / S. None = disabled.
    max_abs_dz: float | None = None
    # Optional pose gate: index PIP joint must be bent (angle in degrees, 180 = straight).
    pose_gate: bool = False
    pose_max_index_pip_deg: float = 150.0
    # Tracking lost longer than this during contact -> contact aborted.
    lost_grace_ms: float = 150.0


@dataclass
class GestureConfig:
    """Gesture classification (distances in S units, times in ms)."""
    swipe_min_dist: float = 0.25    # displacement magnitude that fires a swipe
    axis_ratio: float = 1.5         # |major axis| / |minor axis| needed to fire a swipe
    swipe_max_ms: float = 1000.0    # swipe must reach threshold within this time from touch start
    # Swipe only fires while contact distance d/S is at most this. Prevents the thumb lifting
    # off the finger (d rising towards touch_off) from being read as a swipe.
    swipe_max_contact_d: float = 0.35
    tap_min_ms: float = 30.0
    tap_max_ms: float = 350.0
    tap_max_move: float = 0.15      # max displacement during a tap
    long_press_ms: float = 500.0
    move_tolerance: float = 0.12    # max displacement allowed before long press fires
    # Reference point for thumb motion: thumb position is measured relative to this
    # landmark so whole-hand motion is cancelled. 5 = index MCP. -1 = absolute.
    anchor_landmark: int = 5


@dataclass
class DirectionConfig:
    # "image": directions as seen on the (mirrored) screen.
    # "hand":  directions in a hand-upright frame (wrist -> middle MCP = up), robust to hand roll.
    frame: str = "image"
    swap_axes: bool = False
    invert_x: bool = False
    invert_y: bool = False


@dataclass
class DisplayConfig:
    mirror: bool = True
    window_name: str = "FingerCursorController"
    trail_length: int = 40
    history_lines: int = 8
    show_debug_panel: bool = True


@dataclass
class LoggingConfig:
    dir: str = "logs"
    frames: bool = True          # per-frame JSONL (raw landmarks + features + state)
    world_landmarks: bool = True
    float_digits: int = 5
    record_video: bool = False   # also enabled by --record


@dataclass
class GuidedConfig:
    """Guided test mode: prompts gestures and records ground truth."""
    repeats: int = 5
    gestures: list[str] = field(default_factory=lambda: [
        "swipe_left", "swipe_right", "swipe_up", "swipe_down", "tap", "long_press"])
    timeout_ms: float = 5000.0
    gap_ms: float = 1200.0
    shuffle: bool = True


@dataclass
class Config:
    camera: CameraConfig = field(default_factory=CameraConfig)
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    filter: FilterConfig = field(default_factory=FilterConfig)
    contact: ContactConfig = field(default_factory=ContactConfig)
    gesture: GestureConfig = field(default_factory=GestureConfig)
    direction: DirectionConfig = field(default_factory=DirectionConfig)
    display: DisplayConfig = field(default_factory=DisplayConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    guided: GuidedConfig = field(default_factory=GuidedConfig)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def _apply_dict(obj: Any, data: dict[str, Any], prefix: str = "") -> None:
    for key, value in data.items():
        if not hasattr(obj, key):
            raise KeyError(f"unknown config key: {prefix}{key}")
        current = getattr(obj, key)
        if dataclasses.is_dataclass(current):
            if not isinstance(value, dict):
                raise TypeError(f"config section {prefix}{key} must be a table")
            _apply_dict(current, value, prefix=f"{prefix}{key}.")
        else:
            setattr(obj, key, value)


def config_from_dict(data: dict[str, Any]) -> Config:
    cfg = Config()
    _apply_dict(cfg, data)
    return cfg


def _parse_value(text: str) -> Any:
    lowered = text.strip().lower()
    if lowered in ("true", "false"):
        return lowered == "true"
    if lowered in ("none", "null"):
        return None
    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return text


def apply_override(cfg: Config, assignment: str) -> None:
    """Apply `section.key=value`, e.g. `gesture.swipe_min_dist=0.3`."""
    if "=" not in assignment:
        raise ValueError(f"override must be key=value: {assignment!r}")
    path, value = assignment.split("=", 1)
    parts = path.strip().split(".")
    nested: dict[str, Any] = {parts[-1]: _parse_value(value)}
    for part in reversed(parts[:-1]):
        nested = {part: nested}
    _apply_dict(cfg, nested)


def load_config(path: str | Path | None = None, overrides: list[str] | None = None) -> Config:
    cfg = Config()
    if path is not None and Path(path).exists():
        with open(path, "rb") as f:
            _apply_dict(cfg, tomllib.load(f))
    for assignment in overrides or []:
        apply_override(cfg, assignment)
    return cfg
