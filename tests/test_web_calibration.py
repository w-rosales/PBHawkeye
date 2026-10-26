"""Calibration API state and geometry validation regressions."""

import importlib
import json
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.homography import CourtHomography
from src.tracker import PlayerDetection

web = importlib.import_module("src.web.app")
CORNERS = [[240, 80], [400, 90], [600, 330], [30, 320]]


class Capture:
    def __init__(self):
        self.released = False
        self.frames_read = 0
        self.seeks = []

    def read(self):
        self.frames_read += 1
        return True, np.zeros((360, 640, 3), dtype=np.uint8)

    def isOpened(self):
        return True

    def get(self, prop):
        return {
            cv2.CAP_PROP_FRAME_WIDTH: 640,
            cv2.CAP_PROP_FRAME_HEIGHT: 360,
            cv2.CAP_PROP_FPS: 30,
            cv2.CAP_PROP_FRAME_COUNT: 5,
        }.get(prop, 0)

    def set(self, *args):
        self.seeks.append(args)
        return True

    def release(self):
        self.released = True


@pytest.fixture
def studio(tmp_path, monkeypatch):
    videos = tmp_path / "videos"
    videos.mkdir()
    (videos / "match.mp4").write_bytes(b"video fixture")
    calibration_dir = tmp_path / "calibration"
    monkeypatch.setattr(web, "RAW_VIDEOS_DIR", videos)
    monkeypatch.setattr(web, "CALIBRATION_DIR", calibration_dir)
    monkeypatch.setattr(web, "OUTPUTS_DIR", tmp_path / "outputs")
    captures = []

    def capture(_path):
        cap = Capture()
        captures.append(cap)
        return cap

    monkeypatch.setattr(web.cv2, "VideoCapture", capture)
    web.active_stream_state["is_running"] = False
    return TestClient(web.app), calibration_dir, captures


def test_manual_save_records_dimensions_confirmation_and_video_identity(studio):
    client, calibration_dir, captures = studio
    response = client.post("/api/calibration", json={"video_name": "match.mp4", "corners": CORNERS})
    assert response.status_code == 200
    result = response.json()
    assert result["confirmed"] is True
    assert result["calibrated"] is True
    assert result["image_size"] == [640, 360]
    saved = json.loads((calibration_dir / "match_calibration.json").read_text())
    assert saved["calibration"]["confirmed"] is True
    assert saved["calibration"]["video_signature"]["size_bytes"] == len(b"video fixture")
    assert not (calibration_dir / "default_court_calibration.json").exists()
    assert captures[0].released


@pytest.mark.parametrize("corners", [
    CORNERS[:3],
    [[10, 10], [20, 20], [30, 30], [40, 40]],
    [CORNERS[0], CORNERS[2], CORNERS[1], CORNERS[3]],
    [[-10, 80], [400, 90], [600, 330], [30, 320]],
])
def test_invalid_manual_geometry_is_rejected_without_saving(studio, corners):
    client, calibration_dir, captures = studio
    response = client.post("/api/calibration", json={"video_name": "match.mp4", "corners": corners})
    assert response.status_code == 400
    assert not calibration_dir.exists()
    assert captures[0].released


def test_missing_video_cannot_create_calibration(studio):
    client, calibration_dir, _ = studio
    response = client.post("/api/calibration", json={"video_name": "missing.mp4", "corners": CORNERS})
    assert response.status_code == 404
    assert not calibration_dir.exists()


def test_saved_automatic_proposal_is_not_human_confirmed(studio, monkeypatch):
    client, _, _ = studio
    proposal = CourtHomography(CORNERS, image_size=(640, 360), metadata={
        "method": "auto_detected", "confidence": 0.87, "confirmed": False,
    })
    monkeypatch.setattr(web, "get_or_create_court_calibration", lambda *args, **kwargs: (proposal, "saved"))
    result = client.get("/api/calibration", params={"video_name": "match.mp4"}).json()
    assert result["confirmed"] is False
    assert result["calibrated"] is True
    assert result["confidence"] == 0.87
    response = client.get("/api/stream", params={"video_name": "match.mp4", "mode": "A"})
    assert response.status_code == 409
    assert web.active_stream_state["is_running"] is False


def test_low_confidence_is_reported_as_needing_review(studio, monkeypatch):
    client, _, _ = studio
    proposal = CourtHomography(CORNERS, image_size=(640, 360), metadata={
        "method": "auto_detected", "confidence": 0.2, "confirmed": False,
    })
    monkeypatch.setattr(web, "get_or_create_court_calibration", lambda *args, **kwargs: (proposal, "needs_review"))
    response = client.post("/api/auto_detect_court", params={"video_name": "match.mp4"})
    assert response.status_code == 200
    assert response.json()["calibrated"] is False
    assert response.json()["confirmed"] is False


def test_stream_uses_video_stem_and_releases_resources(studio, monkeypatch):
    client, _, captures = studio
    homography = CourtHomography(CORNERS, image_size=(640, 360), metadata={"confirmed": True})
    monkeypatch.setattr(web, "get_or_create_court_calibration", lambda *args, **kwargs: (homography, "saved"))
    monkeypatch.setattr(web, "PickleballDetector", lambda **kwargs: SimpleNamespace(
        detect=lambda *args, **kwargs: (None, 0.0, []), reset_ball_history=lambda: None))
    monkeypatch.setattr(web, "PickleVisualizer", lambda **kwargs: SimpleNamespace(render_frame=lambda **kwargs: kwargs["frame"]))
    writers = []

    class Writer:
        def __init__(self, path, *_args):
            self.path = path
            self.released = False
            self.count = 0
            writers.append(self)

        def isOpened(self):
            return True

        def write(self, _frame):
            self.count += 1

        def release(self):
            self.released = True

    monkeypatch.setattr(web.cv2, "VideoWriter", Writer)
    response = client.get("/api/stream", params={"video_name": "match.mp4", "mode": "A", "max_frames": 1})
    assert response.status_code == 200
    assert b"Content-Type: image/jpeg" in response.content
    assert writers[0].path.endswith("annotated_match.mp4")
    assert writers[0].count == 1
    assert writers[0].released and captures[-1].released
    assert web.active_stream_state["is_running"] is False


def test_stream_setup_failure_clears_running_state(studio, monkeypatch):
    client, _, captures = studio
    homography = CourtHomography(CORNERS, image_size=(640, 360), metadata={"confirmed": True})
    monkeypatch.setattr(web, "get_or_create_court_calibration", lambda *args, **kwargs: (homography, "saved"))

    def fail_detector(**kwargs):
        raise RuntimeError("detector setup failed")

    monkeypatch.setattr(web, "PickleballDetector", fail_detector)
    with pytest.raises(RuntimeError, match="detector setup failed"):
        client.get("/api/stream", params={"video_name": "match.mp4", "mode": "A"})
    assert captures[-1].released
    assert web.active_stream_state["is_running"] is False


def test_stream_reconstructs_3d_and_results_preserve_numeric_types(studio, monkeypatch):
    from tests.test_ball_3d import known_flight

    client, _, _ = studio
    court, observations, _, average = known_flight()
    court.metadata["confirmed"] = True
    monkeypatch.setattr(web, "get_or_create_court_calibration", lambda *a, **kw: (court, "saved"))
    monkeypatch.setattr(web, "CourtMotionTracker", lambda *a: SimpleNamespace(
        update=lambda frame: court, tracking_ok=True, prev_to_curr_transform=np.eye(3)))
    detections = iter(observations)
    monkeypatch.setattr(web, "PickleballDetector", lambda **kw: SimpleNamespace(
        detect=lambda *a, **kw: (next(detections).measurement_pixel, .95, []),
        reset_ball_history=lambda: None))
    monkeypatch.setattr(web, "PickleVisualizer", lambda **kw: SimpleNamespace(
        render_frame=lambda **kwargs: kwargs["frame"]))
    monkeypatch.setattr(web.cv2, "VideoWriter", lambda *a: SimpleNamespace(
        isOpened=lambda: True, write=lambda frame: None, release=lambda: None))
    capture = Capture()
    capture.get = lambda prop: {
        cv2.CAP_PROP_FRAME_WIDTH: 1920, cv2.CAP_PROP_FRAME_HEIGHT: 1080,
        cv2.CAP_PROP_FPS: 30, cv2.CAP_PROP_FRAME_COUNT: len(observations),
        cv2.CAP_PROP_POS_MSEC: max(0, capture.frames_read-2)*1000/30,
    }.get(prop, 0)
    monkeypatch.setattr(web.cv2, "VideoCapture", lambda *a: capture)
    response = client.get("/api/stream", params={"video_name": "match.mp4", "mode": "A",
                                                "max_frames": len(observations)})
    assert response.status_code == 200
    assert web.active_stream_state["speed_status"] == "estimated"
    assert web.active_stream_state["current_speed_mph"] == pytest.approx(average, abs=.05)
    data = client.get("/api/results", params={"video_name": "match.mp4"}).json()
    assert isinstance(data["shots"][0]["avg_mph"], float)
    assert data["shots"][0]["avg_mph"] == pytest.approx(average, abs=.05)
    assert len(data["trajectory_3d"]) == len(observations)
    assert capture.released


@pytest.mark.parametrize("failure_stage", ["write", "release"])
def test_final_encoder_failure_closes_capture_and_clears_state(studio, monkeypatch, failure_stage):
    client, _, captures = studio
    court = CourtHomography(CORNERS, image_size=(640, 360), metadata={"confirmed": True})
    monkeypatch.setattr(web, "get_or_create_court_calibration", lambda *a, **kw: (court, "saved"))
    monkeypatch.setattr(web, "PickleballDetector", lambda **kw: SimpleNamespace(
        detect=lambda *a, **kwargs: (None, 0., []), reset_ball_history=lambda: None))
    monkeypatch.setattr(web, "CourtMotionTracker", lambda *a: SimpleNamespace(
        update=lambda frame: court, tracking_ok=True, prev_to_curr_transform=np.eye(3)))
    monkeypatch.setattr(web, "PickleVisualizer", lambda **kw: SimpleNamespace(
        render_frame=lambda **kwargs: kwargs["frame"]))
    writers = []

    class FailingWriter:
        def __init__(self, *args):
            self.released = False
            writers.append(self)

        def isOpened(self):
            return True

        def write(self, frame):
            if failure_stage == "write":
                raise IOError("final encoder failure")

        def release(self):
            self.released = True
            if failure_stage == "release":
                raise IOError("final encoder failure")

    monkeypatch.setattr(web.cv2, "VideoWriter", FailingWriter)
    with pytest.raises(IOError, match="final encoder failure"):
        client.get("/api/stream", params={"video_name": "match.mp4", "mode": "A", "max_frames": 1})
    assert captures[-1].released and writers[-1].released
    assert web.active_stream_state["is_running"] is False
    assert web.active_stream_state["error"] == "final encoder failure"


def test_stream_passes_camera_motion_into_ball_prediction(studio, monkeypatch):
    client, _, _ = studio
    court = CourtHomography(CORNERS, image_size=(640, 360), metadata={"confirmed": True})
    monkeypatch.setattr(web, "get_or_create_court_calibration", lambda *a, **kw: (court, "saved"))
    shift = np.float64([[1, 0, 10], [0, 1, 5], [0, 0, 1]])

    class Motion:
        tracking_ok = True

        def __init__(self, *args):
            self.calls = 0
            self.prev_to_curr_transform = np.eye(3)

        def update(self, frame):
            self.prev_to_curr_transform = np.eye(3) if self.calls == 0 else shift
            self.calls += 1
            return court

    predictions = []

    def detect(frame, predicted, homography=None, camera_transform=None):
        predictions.append(predicted)
        index = len(predictions)-1
        np.testing.assert_array_equal(camera_transform, np.eye(3) if index == 0 else shift)
        return (320.+10*index, 180.+5*index), .9, []

    monkeypatch.setattr(web, "CourtMotionTracker", Motion)
    monkeypatch.setattr(web, "PickleballDetector", lambda **kw: SimpleNamespace(
        detect=detect, reset_ball_history=lambda: None))
    monkeypatch.setattr(web, "PickleVisualizer", lambda **kw: SimpleNamespace(
        render_frame=lambda **kwargs: kwargs["frame"]))
    monkeypatch.setattr(web.cv2, "VideoWriter", lambda *a: SimpleNamespace(
        isOpened=lambda: True, write=lambda frame: None, release=lambda: None))
    response = client.get("/api/stream", params={"video_name": "match.mp4", "mode": "A", "max_frames": 2})
    assert response.status_code == 200
    assert predictions[0] is None
    np.testing.assert_allclose(predictions[1], [330., 185.])
    assert web.active_stream_state["error"] is None


@pytest.mark.parametrize("speed", [4, 5])
def test_fast_stream_respects_sampling_and_range(studio, monkeypatch, speed):
    client, _, _ = studio
    court = CourtHomography(CORNERS, image_size=(640, 360), metadata={"confirmed": True})
    monkeypatch.setattr(web, "get_or_create_court_calibration", lambda *a, **kw: (court, "saved"))
    monkeypatch.setattr(web, "CourtMotionTracker", lambda *a: SimpleNamespace(
        update=lambda frame: court, tracking_ok=True, prev_to_curr_transform=np.eye(3)))
    monkeypatch.setattr(web, "PickleballDetector", lambda **kw: SimpleNamespace(
        detect=lambda *a, **kwargs: (None, 0., []), reset_ball_history=lambda: None))
    monkeypatch.setattr(web, "PickleVisualizer", lambda **kw: SimpleNamespace(
        render_frame=lambda **kwargs: kwargs["frame"]))
    monkeypatch.setattr(web.cv2, "VideoWriter", lambda *a: SimpleNamespace(
        isOpened=lambda: True, write=lambda frame: None, release=lambda: None))
    capture = Capture()
    capture.get = lambda prop: {cv2.CAP_PROP_FRAME_WIDTH: 640, cv2.CAP_PROP_FRAME_HEIGHT: 360,
                                cv2.CAP_PROP_FPS: 30, cv2.CAP_PROP_FRAME_COUNT: 100}.get(prop, 0)
    capture.grab = lambda: True
    monkeypatch.setattr(web.cv2, "VideoCapture", lambda *a: capture)
    response = client.get("/api/stream", params={"video_name": "match.mp4", "mode": "A", "max_frames": 7, "speed": speed})
    assert response.status_code == 200
    assert capture.frames_read == 3  # Reference plus the two sampled analysis frames.
    assert web.active_stream_state["current_frame"] == 7
    assert web.active_stream_state["total_frames"] == 7


def test_stream_tracks_from_opening_view_and_pauses_unmatched_geometry(studio, monkeypatch):
    client, _, captures = studio
    original = CourtHomography(CORNERS, image_size=(640, 360), metadata={"confirmed": True})
    moved = CourtHomography(np.asarray(CORNERS)+[8, 3], image_size=(640, 360), metadata={"confirmed": True})
    monkeypatch.setattr(web, "get_or_create_court_calibration", lambda *args, **kwargs: (original, "saved"))
    initialization = []

    class Motion:
        def __init__(self, frame, court):
            initialization.append((frame.shape, court))
            self.calls = 0
            self.tracking_ok = True

        def update(self, _frame):
            self.calls += 1
            self.tracking_ok = self.calls > 1
            return moved

    supplied_courts, rendered_courts = [], []

    def detect(_frame, _prediction, homography=None, camera_transform=None):
        supplied_courts.append(homography)
        return (320., 180.), .9, [PlayerDetection(1, (300, 100, 350, 200), feet_pixel=(325, 200))]

    class Visualizer:
        def __init__(self, **kwargs):
            self.homography = kwargs["homography"]

        def render_frame(self, **kwargs):
            rendered_courts.append(self.homography)
            if self.homography is None:
                assert kwargs["ball_pt"].court_x_ft is None
                assert kwargs["ball_trail_court"] == []
                assert kwargs["players"][0].court_feet is None
            return kwargs["frame"]

    monkeypatch.setattr(web, "CourtMotionTracker", Motion)
    monkeypatch.setattr(web, "PickleballDetector", lambda **kwargs: SimpleNamespace(
        detect=detect, reset_ball_history=lambda: None))
    monkeypatch.setattr(web, "PickleVisualizer", Visualizer)
    monkeypatch.setattr(web.cv2, "VideoWriter", lambda *args: SimpleNamespace(
        isOpened=lambda: True, write=lambda frame: None, release=lambda: None))
    response = client.get("/api/stream", params={"video_name": "match.mp4", "mode": "A", "start_frame": 2, "max_frames": 2})
    assert response.status_code == 200
    assert initialization == [((360, 640, 3), original)]
    assert captures[-1].seeks[0] == (cv2.CAP_PROP_POS_FRAMES, 2)
    assert captures[-1].frames_read == 3  # Opening frame plus two analyzed frames.
    assert supplied_courts == [None, moved]
    assert rendered_courts == [None, moved]
    assert web.active_stream_state["court_tracking_ok"] is True
    assert web.active_stream_state["current_frame"] == 2


@pytest.mark.parametrize("corners", [CORNERS * 2, [[0, 0]] * 4, [[0, 0], [1, 0], [1, float("nan")], [0, 1]]])
def test_homography_rejects_invalid_points(corners):
    with pytest.raises(ValueError):
        CourtHomography(corners)


def test_homography_metadata_roundtrip_and_legacy_compatibility(tmp_path):
    path = tmp_path / "court.json"
    homography = CourtHomography(CORNERS, metadata={"method": "auto_detected", "confirmed": False, "confidence": 0.8})
    homography.save_to_file(path)
    assert CourtHomography.load_from_file(path).metadata == homography.metadata
    legacy = homography.to_dict()
    legacy.pop("calibration")
    path.write_text(json.dumps(legacy))
    assert CourtHomography.load_from_file(path).metadata == {}
