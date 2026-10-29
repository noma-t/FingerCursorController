from fingercursor.analysis import Session, score
from fingercursor.config import GuidedConfig
from fingercursor.guided import GuidedSession
from fingercursor.recognizer import Event


def test_guided_flow_and_scoring(tmp_path):
    cfg = GuidedConfig(repeats=1, gestures=["tap", "swipe_left", "long_press"], shuffle=False,
                       timeout_ms=1000, gap_ms=500)
    g = GuidedSession(cfg, 0.0)
    labels = []
    log = lambda recs, t: labels.extend({"t_ms": t, **r} for r in recs)  # noqa: E731
    log(g.update(2000.0), 2000.0)                        # trial 0 (tap) starts
    log(g.on_event("TAP", 2300.0, {}), 2300.0)           # ok
    log(g.update(2800.0), 2800.0)                        # trial 1 (swipe_left) starts
    log(g.on_event("SWIPE_RIGHT", 3000.0, {}), 3000.0)   # wrong
    log(g.update(3500.0), 3500.0)                        # trial 2 (long_press) starts
    log(g.update(4600.0), 4600.0)                        # timeout -> miss
    assert g.done
    assert g.results == {"ok": 1, "wrong": 1, "miss": 1}

    session = Session(tmp_path, {"config": {"guided": {"timeout_ms": 1000}}}, [], [], labels)
    events = [Event("TAP", 2300.0), Event("SWIPE_RIGHT", 3000.0), Event("TAP", 9000.0)]
    c = score(session, events).counts()
    assert c["trials"]["total"] == {"ok": 1, "wrong": 1, "miss": 1}
    assert c["false_positives"] == 1
