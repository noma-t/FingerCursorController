import cv2

from fingercursor import camera
from fingercursor.config import CameraConfig, Config, apply_override


class FakeCap:
    def __init__(self, *args):
        self.args = args
        self.calls = []
        self.props = {cv2.CAP_PROP_AUTO_EXPOSURE: 0.75, cv2.CAP_PROP_EXPOSURE: -5.0,
                      cv2.CAP_PROP_FRAME_WIDTH: 1280, cv2.CAP_PROP_FRAME_HEIGHT: 720, cv2.CAP_PROP_FPS: 30,
                      cv2.CAP_PROP_FOURCC: cv2.VideoWriter_fourcc(*"YUY2")}

    def isOpened(self):
        return True

    def set(self, prop, value):
        self.calls.append((prop, value))
        self.props[prop] = value
        return True

    def get(self, prop):
        return self.props.get(prop, 0.0)

    def getBackendName(self):
        return "DSHOW"


def _open(monkeypatch, **kw):
    monkeypatch.setattr(camera.cv2, "VideoCapture", FakeCap)
    return camera.open_camera(CameraConfig(**kw))


def test_fourcc_order(monkeypatch):
    props = lambda cap: [p for p, _ in cap.calls]  # noqa: E731
    cap = _open(monkeypatch, fourcc="MJPG", fourcc_order="before_size")
    assert props(cap).index(cv2.CAP_PROP_FOURCC) < props(cap).index(cv2.CAP_PROP_FRAME_WIDTH)
    cap = _open(monkeypatch, fourcc="MJPG", fourcc_order="after_size")
    assert props(cap).index(cv2.CAP_PROP_FOURCC) > props(cap).index(cv2.CAP_PROP_FPS)
    cap = _open(monkeypatch, fourcc="")
    assert cv2.CAP_PROP_FOURCC not in props(cap)
    assert cv2.CAP_PROP_EXPOSURE not in props(cap)  # untouched by default


def test_exposure_settings_and_readback(monkeypatch):
    cap = _open(monkeypatch, auto_exposure=False, exposure=-7, fourcc="YUY2")
    assert cap.get(cv2.CAP_PROP_AUTO_EXPOSURE) == 0.25
    assert cap.get(cv2.CAP_PROP_EXPOSURE) == -7
    actual = camera.read_back(cap)
    assert actual["fourcc"] == "YUY2" and actual["backend"] == "DSHOW"
    assert camera.mismatches(CameraConfig(fourcc="YUY2"), actual) == []
    cap.props[cv2.CAP_PROP_FPS] = 5
    assert any("fps" in m for m in camera.mismatches(CameraConfig(), camera.read_back(cap)))


def test_camera_control(monkeypatch):
    cap = _open(monkeypatch)
    ctl = camera.CameraControl(cap, CameraConfig())
    ch = ctl.step_exposure(-1)
    assert ch["auto_exposure"] == 0.25 and ch["exposure"] == -6.0
    assert ctl.toggle_auto_exposure()["auto_exposure"] == 0.75
    assert "auto_exposure = true" in ctl.config_snippet()


def test_camera_overrides():
    cfg = Config()
    apply_override(cfg, "camera.fourcc=''")
    apply_override(cfg, "camera.backend=msmf")
    apply_override(cfg, "camera.exposure=-6")
    assert (cfg.camera.fourcc, cfg.camera.backend, cfg.camera.exposure) == ("", "msmf", -6)


def test_snippet_skips_unreported_exposure(monkeypatch):
    cap = _open(monkeypatch)
    cap.props[cv2.CAP_PROP_AUTO_EXPOSURE] = -1.0
    snippet = camera.CameraControl(cap, CameraConfig()).config_snippet()
    assert "auto_exposure =" not in snippet and "not reported" in snippet
