"""Regression tests for Task 2: Separating generated annotations from confirmed references."""

import json
from pathlib import Path
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
import numpy as np
import cv2

from src.homography import CourtHomography
from src.web.ball_testing import BallTestStore, TestClip as ClipRequest, router


@pytest.fixture
def store_client(tmp_path, monkeypatch):
    videos = tmp_path / "videos"
    videos.mkdir()
    video = videos / "match.avi"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (64, 48))
    for index in range(50):
        writer.write(np.full((48, 64, 3), index * 3, np.uint8))
    writer.release()

    models_dir = tmp_path / "data" / "models" / "ball_v4"
    models_dir.mkdir(parents=True)
    (models_dir / "best_court.pt").write_bytes(b"mock qualified model")

    court = CourtHomography([(20, 5), (45, 5), (60, 42), (5, 42)], image_size=(64, 48), metadata={"confirmed": True})

    def complete(session):
        folder = store.folder(session)
        metadata = store.metadata(session)
        (folder / "frames").mkdir(parents=True, exist_ok=True)
        for i in range(metadata["frame_start"], metadata["frame_end_inclusive"] + 1):
            cv2.imwrite(str(folder / "frames" / f"{i:06d}.png"), np.zeros((48, 64, 3), np.uint8))
        records = [
            {"frame": i, "classical": [12.0, 14.0], "neural": [12.0, 14.0], "classical_score": 0.85, "neural_score": 0.90, "court_ok": True}
            for i in range(metadata["frame_start"], metadata["frame_end_inclusive"] + 1)
        ]
        (folder / "records.json").write_text(json.dumps(records))
        metadata.update(status="completed", processed=len(records))
        (folder / "metadata.json").write_text(json.dumps(metadata))
        if store.lease is not None:
            store.lease.close()
            store.lease = None

    store = BallTestStore(tmp_path, lambda _: video, lambda _: (court, "fixture"), lambda: False)
    monkeypatch.setattr(store, "run", complete)
    app = FastAPI()
    app.include_router(router(store))
    client = TestClient(app)
    return client, store, video


def create_session(client, store):
    res = client.post("/api/ball-tests", json={"video_name": "match.avi", "start_seconds": 1.0, "duration_seconds": 0.5, "split": "train"})
    assert res.status_code == 200
    store.worker.join(10)
    return res.json()["id"]


def test_prefill_alone_does_not_increase_verified_metrics_or_review_count(store_client):
    client, store, _ = store_client
    session = create_session(client, store)

    # Prefill from classical
    res = client.post(f"/api/ball-tests/{session}/labels/prefill", json={
        "source": "classical",
        "min_confidence": 0.25,
        "only_unreviewed": True,
        "expected_revision": 0
    })
    assert res.status_code == 200
    data = res.json()
    assert data["prefilled_count"] == 5
    # Reviewed count must remain 0!
    assert data["reviewed"] == 0

    detail = store.detail(session)
    # Suggestions exist with confirmed=False
    labels = detail["annotations"]["labels"]
    assert len(labels) == 5
    for l in labels:
        assert l["origin"] == "prefilled"
        assert l["confirmed"] is False
        assert l["reviewed"] is False

    # Metrics must NOT count unconfirmed suggestions as verified truth!
    metrics = detail["metrics"]
    assert metrics["classical"]["visible"] == 0
    assert metrics["neural"]["visible"] == 0

    # Explicit confirmation of frames 10 and 11
    conf_res = client.post(f"/api/ball-tests/{session}/labels/confirm", json={
        "frames": [10, 11],
        "expected_revision": 1
    })
    assert conf_res.status_code == 200
    conf_data = conf_res.json()
    assert conf_data["confirmed_count"] == 2
    assert conf_data["reviewed"] == 2

    # Verified metrics should now reflect the 2 confirmed frames
    detail_after = store.detail(session)
    assert detail_after["metrics"]["classical"]["visible"] == 2
    assert detail_after["metrics"]["classical"]["correct"] == 2


def test_interpolate_does_not_overwrite_manual_labels_by_default(store_client):
    client, store, _ = store_client
    session = create_session(client, store)

    # Set keyframes at frame 10 and 14
    client.post(f"/api/ball-tests/{session}/labels", json={"frame": 10, "visibility": "visible", "x": 10.0, "y": 10.0, "expected_revision": 0})
    client.post(f"/api/ball-tests/{session}/labels", json={"frame": 14, "visibility": "visible", "x": 18.0, "y": 18.0, "expected_revision": 1})

    # Set an intermediate manual label at frame 12 marking ball as absent
    client.post(f"/api/ball-tests/{session}/labels", json={"frame": 12, "visibility": "absent", "expected_revision": 2})

    # Run interpolate without overwrite_existing
    interp_res = client.post(f"/api/ball-tests/{session}/labels/interpolate", json={
        "start_frame": 10,
        "end_frame": 14,
        "expected_revision": 3,
        "overwrite_existing": False
    })
    assert interp_res.status_code == 200

    detail = store.detail(session)
    lbl_map = {l["frame"]: l for l in detail["annotations"]["labels"]}

    # Frame 12 MUST remain absent (manual label preserved!)
    assert lbl_map[12]["visibility"] == "absent"
    assert lbl_map[12]["origin"] == "manual"
    assert lbl_map[12]["confirmed"] is True

    # Frames 11 and 13 should be interpolated suggestions (confirmed=False)
    assert lbl_map[11]["origin"] == "interpolated"
    assert lbl_map[11]["confirmed"] is False
    assert lbl_map[13]["origin"] == "interpolated"
    assert lbl_map[13]["confirmed"] is False


def test_export_excludes_unconfirmed_suggestions(store_client):
    client, store, _ = store_client
    session = create_session(client, store)

    # Add 1 confirmed manual label at frame 10
    client.post(f"/api/ball-tests/{session}/labels", json={"frame": 10, "visibility": "visible", "x": 10.0, "y": 10.0, "expected_revision": 0})

    # Prefill frames 11-14 (unconfirmed)
    client.post(f"/api/ball-tests/{session}/labels/prefill", json={"source": "classical", "min_confidence": 0.25, "expected_revision": 1})

    # Export
    exp_res = client.post(f"/api/ball-tests/{session}/export")
    assert exp_res.status_code == 200
    exp_data = exp_res.json()

    # Only 1 confirmed reference should be exported!
    assert exp_data["labels"] == 1
    assert exp_data["trainable_visible_frames"] == 1

    # Check export directory
    dest = store.project / exp_data["dataset"]
    labels_file = json.loads((dest / "manual_labels.json").read_text())
    assert len(labels_file["labels"]) == 1
    assert labels_file["labels"][0]["frame"] == 10
    assert labels_file["labels"][0]["origin"] == "manual"

