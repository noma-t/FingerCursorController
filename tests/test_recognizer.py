import numpy as np
import pytest

from fingercursor.analysis import load_session, replay
from fingercursor.config import Config, apply_override, load_config
from fingercursor.features import compute_features, to_display_px
from fingercursor.logger import SessionLogger
from fingercursor.pipeline import Pipeline
from fingercursor.recognizer import GESTURE_EVENTS
from fingercursor.tracker import HandObservation

from synth import AWAY, CONTACT, DT, H, S, W, hand, offset, run, make_config


def gestures(events):
    return [e.type for e in events if e.type in GESTURE_EVENTS]


def test_contact_distance_normalised():
    f = compute_features(to_display_px(hand(CONTACT), W, H, False))
    assert f.contact_dist == pytest.approx(0.1, abs=1e-6)
    assert f.nearest_segment == 0
    f = compute_features(to_display_px(hand(AWAY), W, H, False))
    assert f.contact_dist > 0.8


def test_tap():
    p = Pipeline(make_config())
    ev = run(p, [AWAY] * 10 + [CONTACT] * 5 + [AWAY] * 10)
    assert gestures(ev) == ["TAP"]


def test_too_short_contact_is_not_a_tap():
    p = Pipeline(make_config())
    ev = run(p, [AWAY] * 10 + [CONTACT] * 1 + [AWAY] * 10)
    assert gestures(ev) == []
    assert "CANDIDATE_DROPPED" in [e.type for e in ev]


def test_long_press():
    p = Pipeline(make_config())
    ev = run(p, [AWAY] * 5 + [CONTACT] * 30 + [AWAY] * 5)
    assert gestures(ev) == ["LONG_PRESS_START", "LONG_PRESS_END"]
    start = next(e for e in ev if e.type == "LONG_PRESS_START")
    assert start.data["duration_ms"] >= 500


def test_medium_contact_rejected():
    p = Pipeline(make_config())
    ev = run(p, [AWAY] * 5 + [CONTACT] * 12 + [AWAY] * 5)  # ~400 ms: too long for tap, too short for hold
    assert gestures(ev) == []
    rej = next(e for e in ev if e.type == "TOUCH_REJECTED")
    assert "too_long_for_tap" in rej.data["reasons"]


@pytest.mark.parametrize("dx,dy,expected", [
    (-0.5, 0, "SWIPE_LEFT"), (0.5, 0, "SWIPE_RIGHT"), (0, -0.5, "SWIPE_UP"), (0, 0.5, "SWIPE_DOWN"),
])
def test_swipes(dx, dy, expected):
    # Vertical motion leaves the index segment, so widen the release threshold for this geometry test.
    p = Pipeline(make_config(contact__touch_off=0.7, gesture__swipe_max_contact_d=0.7))
    steps = [CONTACT + offset(dx * k / 6, dy * k / 6) for k in range(7)]
    ev = run(p, [AWAY] * 5 + [CONTACT] * 2 + steps + [steps[-1]] * 2 + [AWAY] * 5)
    assert gestures(ev) == [expected]


def test_diagonal_is_ambiguous():
    p = Pipeline(make_config(contact__touch_off=0.7, gesture__swipe_max_contact_d=0.7))
    steps = [CONTACT + offset(-0.4 * k / 6, -0.4 * k / 6) for k in range(7)]
    ev = run(p, [AWAY] * 5 + [CONTACT] * 2 + steps + [AWAY] * 5)
    assert not any(t.startswith("SWIPE") for t in gestures(ev))


def test_mirror_flips_horizontal_direction():
    cfg = make_config(contact__touch_off=0.7, gesture__swipe_max_contact_d=0.7)
    cfg.display.mirror = True
    p = Pipeline(cfg)
    steps = [CONTACT + offset(0.5 * k / 6, 0) for k in range(7)]  # moves right in the camera image
    ev = run(p, [AWAY] * 5 + [CONTACT] * 2 + steps + [AWAY] * 5)
    assert gestures(ev) == ["SWIPE_LEFT"]  # ...which is left on the mirrored screen


def test_whole_hand_motion_is_cancelled_by_anchor():
    p = Pipeline(make_config())
    ev = []
    t = 0.0
    for k in range(30):
        shift = np.array([0.01 * k, 0.0, 0.0])  # rigid hand translation, ~0.5 S per 10 frames
        lm = hand(CONTACT if 5 <= k < 25 else AWAY) + shift
        ev.extend(p.process(t, lm, None, W, H).events)
        t += DT
    assert not any(e.startswith("SWIPE") for e in gestures(ev))


def test_hysteresis_keeps_contact():
    p = Pipeline(make_config())
    near_off = np.array([0.40, 0.55 - 0.35 * S / W])  # d = 0.35, between touch_on and touch_off
    seq = [AWAY] * 3 + [CONTACT] * 2 + [near_off, CONTACT] * 15 + [AWAY] * 3
    ev = run(p, seq)
    assert [e.type for e in ev].count("TOUCH_START") == 1


def test_tracking_lost_aborts_contact():
    p = Pipeline(make_config())
    ev = run(p, [AWAY] * 3 + [CONTACT] * 4 + [None] * 10 + [AWAY] * 3)
    types = [e.type for e in ev]
    assert "TOUCH_ABORTED" in types
    assert gestures(ev) == []


def test_short_dropout_is_tolerated():
    p = Pipeline(make_config())
    ev = run(p, [AWAY] * 3 + [CONTACT] * 2 + [None] * 2 + [CONTACT] * 2 + [AWAY] * 3)
    assert gestures(ev) == ["TAP"]


def test_config_override_and_file(tmp_path):
    cfg = Config()
    apply_override(cfg, "gesture.swipe_min_dist=0.33")
    apply_override(cfg, "contact.max_abs_dz=none")
    apply_override(cfg, "direction.frame=hand")
    assert cfg.gesture.swipe_min_dist == 0.33
    assert cfg.contact.max_abs_dz is None
    assert cfg.direction.frame == "hand"
    toml = tmp_path / "c.toml"
    toml.write_text("[contact]\ntouch_on = 0.2\n")
    assert load_config(toml).contact.touch_on == 0.2
    with pytest.raises(KeyError):
        apply_override(cfg, "contact.nope=1")


def test_log_replay_roundtrip(tmp_path):
    cfg = make_config()
    p = Pipeline(cfg)
    logger = SessionLogger(cfg, {"frame_size": [W, H]}, session_dir=tmp_path / "s")
    seq = [AWAY] * 5 + [CONTACT] * 5 + [AWAY] * 5 + [None] * 3 + [AWAY] * 3 + [CONTACT] * 30 + [AWAY] * 5
    t = 0.0
    live = []
    for i, thumb in enumerate(seq):
        lm = None if thumb is None else hand(thumb)
        obs = None if lm is None else HandObservation(lm, None, "Right", 0.99)
        res = p.process(t, lm, None, W, H)
        for e in res.events:
            logger.log_event(i, e)
        live.extend(res.events)
        logger.log_frame(i, res, obs, {"infer_ms": 5.0}, 0 if obs is None else 1, lm, None)
        t += DT
    logger.close()
    s = load_session(tmp_path / "s")
    replayed = replay(s, cfg)
    assert [e.type for e in replayed] == [e.type for e in live]
    assert gestures(replayed) == ["TAP", "LONG_PRESS_START", "LONG_PRESS_END"]


def test_lift_off_is_not_a_swipe():
    # Thumb peels away from the finger (d rises past swipe_max_contact_d but stays below touch_off).
    p = Pipeline(make_config(contact__touch_off=0.7))
    steps = [CONTACT + offset(0, -0.5 * k / 6) for k in range(7)]
    ev = run(p, [AWAY] * 5 + [CONTACT] * 2 + steps + [AWAY] * 5)
    assert not any(t.startswith("SWIPE") for t in gestures(ev))
