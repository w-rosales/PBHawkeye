"""Durable human review, source-frame extraction, split isolation and exports."""

import json
from pathlib import Path
import threading

import cv2
from fastapi import FastAPI
from fastapi.testclient import TestClient
import numpy as np
import pytest

from src.calibration import video_signature
from src.homography import CourtHomography
from src.web.ball_testing import BallTestStore, TestClip as ClipRequest, router


@pytest.fixture
def review(tmp_path,monkeypatch):
    videos=tmp_path/"videos";videos.mkdir();video=videos/"match.avi"
    writer=cv2.VideoWriter(str(video),cv2.VideoWriter_fourcc(*"MJPG"),10.,(64,48))
    assert writer.isOpened()
    for index in range(80):
        image=np.full((48,64,3),index*3,np.uint8);writer.write(image)
    writer.release()
    checkpoint=tmp_path/"data/models/ball_v1/best_court.pt";checkpoint.parent.mkdir(parents=True);checkpoint.write_bytes(b"fixture model unchanged")
    court=CourtHomography([(20,5),(45,5),(60,42),(5,42)],image_size=(64,48),metadata={"confirmed":True})

    def video_path(name):
        from fastapi import HTTPException
        path=(videos/name).resolve()
        if path.parent!=videos or not path.is_file():raise HTTPException(404,"Video not found")
        return path

    def complete(session):
        folder=store.folder(session);metadata=store.metadata(session)
        cap=cv2.VideoCapture(str(video));cap.set(cv2.CAP_PROP_POS_FRAMES,max(0,metadata["frame_start"]-2))
        for i in range(max(0,metadata["frame_start"]-2),metadata["frame_end_inclusive"]+1):
            ok,frame=cap.read();assert ok
            cv2.imwrite(str(folder/"frames"/f"{i:06d}.png"),frame)
        cap.release()
        records=[{"frame":i,"classical":[10.,10.],"neural":None,"classical_score":.8,"neural_score":.4,"court_ok":True} for i in range(metadata["frame_start"],metadata["frame_end_inclusive"]+1)]
        (folder/"records.json").write_text(json.dumps(records))
        metadata.update(status="completed",processed=len(records));(folder/"metadata.json").write_text(json.dumps(metadata))
        if store.lease is not None:store.lease.close();store.lease=None

    store=BallTestStore(tmp_path,video_path,lambda _:(court,"fixture"),lambda:False)
    monkeypatch.setattr(store,"run",complete)
    app=FastAPI();app.include_router(router(store));client=TestClient(app)
    yield client,store,video,checkpoint,court
    if store.worker is not None:store.worker.join(10)


def create(review,**changes):
    client,store,*_=review
    request={"video_name":"match.avi","start_seconds":1.,"duration_seconds":.5,"split":"train",**changes}
    response=client.post("/api/ball-tests",json=request)
    assert response.status_code==200,response.text
    store.worker.join(15);assert not store.worker.is_alive()
    return response.json()["id"]


def test_labels_persist_with_edit_history_and_stale_updates_cannot_overwrite(review):
    client,store,video,checkpoint,_=review;original=checkpoint.read_bytes();session=create(review)
    payload={"frame":10,"visibility":"visible","x":10.,"y":10.,"expected_revision":0}
    response=client.post(f"/api/ball-tests/{session}/labels",json=payload)
    assert response.status_code==200 and response.json()["revision"]==1
    stale=client.post(f"/api/ball-tests/{session}/labels",json={**payload,"x":40.})
    assert stale.status_code==409
    response=client.post(f"/api/ball-tests/{session}/labels",json={**payload,"x":11.,"expected_revision":1})
    assert response.status_code==200
    reopened=BallTestStore(store.project,store.video_path,store.resolve_court,lambda:False)
    result=reopened.detail(session)
    assert result["annotations"]["labels"][0]["x"]==11.
    assert len(result["annotations"]["history"])==2
    assert result["annotations"]["history"][1]["previous"]["x"]==10.
    assert result["metrics"]["classical"]["correct"]==1
    assert result["metrics"]["neural"]["missing"]==1
    assert checkpoint.read_bytes()==original  # Reviewing never updates model weights.


@pytest.mark.parametrize("label,status",[
    ({"frame":10,"visibility":"visible"},422),
    ({"frame":10,"visibility":"visible","x":float('inf'),"y":1},422),
    ({"frame":10,"visibility":"visible","x":64.,"y":1},400),
    ({"frame":9,"visibility":"uncertain"},400),
    ({"frame":10,"visibility":"occluded","x":1,"y":2},422),
])
def test_invalid_or_invented_labels_are_rejected(review,label,status):
    client,*_=review;session=create(review)
    # Raw JSON intentionally exercises nonfinite JSON parsing as well.
    response=client.post(f"/api/ball-tests/{session}/labels",content=json.dumps({**label,"expected_revision":0}),headers={"Content-Type":"application/json"})
    assert response.status_code==status


def test_known_absent_is_scored_separately_from_hidden_or_uncertain(review):
    client,store,*_=review;session=create(review)
    for revision,(frame,visibility) in enumerate([(10,"absent"),(11,"occluded"),(12,"uncertain")]):
        response=client.post(f"/api/ball-tests/{session}/labels",json={"frame":frame,"visibility":visibility,"expected_revision":revision})
        assert response.status_code==200
    stats=store.detail(session)["metrics"]
    assert stats["classical"]["absent_reviewed"]==1
    assert stats["classical"]["absent_false_detections"]==1
    assert stats["neural"]["absent_false_detections"]==0
    assert stats["classical"]["visible"]==0


def test_export_contains_context_and_frozen_revision_and_can_prepare_training(review):
    from tools.train_ball_model import prepare
    client,store,_,checkpoint,_=review;session=create(review)
    response=client.post(f"/api/ball-tests/{session}/labels",json={"frame":10,"visibility":"visible","x":10,"y":10,"expected_revision":0})
    assert response.status_code==200
    exported=client.post(f"/api/ball-tests/{session}/export").json()
    assert exported["trained"] is False and exported["split"]=="train"
    destination=store.project/exported["dataset"]
    assert {p.stem for p in (destination/"frames").iterdir()}=={"000008","000009","000010"}
    prepare(destination,store.project/"prepared",width=64)
    manifest=json.loads((store.project/"prepared/manifest.json").read_text())
    assert len(manifest["samples"])==1 and manifest["samples"][0]["split"]=="train"
    client.post(f"/api/ball-tests/{session}/labels",json={"frame":10,"visibility":"visible","x":12,"y":12,"expected_revision":1})
    assert json.loads((destination/"manual_labels.json").read_text())["labels"][0]["x"]==10
    second=client.post(f"/api/ball-tests/{session}/export").json()
    assert second["dataset"]!=exported["dataset"]
    assert checkpoint.read_bytes()==b"fixture model unchanged"


def test_split_check_includes_two_context_frames_and_protects_seed_training(review):
    client,store,video,*_=review;create(review)
    response=client.post("/api/ball-tests",json={"video_name":"match.avi","start_seconds":1.6,"duration_seconds":.3,"split":"test"})
    assert response.status_code==409 and "context" in response.json()["detail"]
    seed=store.project/"data/learning/ball_v1/manifest.json";seed.parent.mkdir(parents=True)
    identity=video_signature(video)
    seed.write_text(json.dumps({"sources":[{"metadata":{"video":str(video),"source_size":identity["size_bytes"],"source_mtime_ns":identity["mtime_ns"]}}],"samples":[{"source":"audit0","frame":42,"split":"train"}]}))
    response=client.post("/api/ball-tests",json={"video_name":"match.avi","start_seconds":4.,"duration_seconds":.3,"split":"test"})
    assert response.status_code==409 and "seed" in response.json()["detail"]


def test_api_rejects_traversal_unconfirmed_court_and_out_of_range_clips(review,monkeypatch):
    client,store,_,_,court=review
    assert client.get('/api/ball-tests/../anything').status_code==404
    assert client.post('/api/ball-tests',json={"video_name":"../match.avi"}).status_code==404
    assert client.post('/api/ball-tests',json={"video_name":"match.avi","start_seconds":99}).status_code==400
    assert client.post('/api/ball-tests',json={"video_name":"match.avi","duration_seconds":11}).status_code==422
    court.metadata["confirmed"]=False
    assert client.post('/api/ball-tests',json={"video_name":"match.avi"}).status_code==409


def test_background_worker_extracts_exact_source_frames_and_runs_both_backends(review,monkeypatch):
    client,store,video,_,court=review
    import src.ball_model
    import src.court_motion
    import src.detector

    class Camera:
        tracking_ok=True;status="fixture";prev_to_curr_transform=np.eye(3)
        def __init__(self,*args):pass
        def update(self,frame):return court
    class Detector:
        def __init__(self,**kwargs):pass
        def reset_ball_history(self):pass
        def detect(self,frame,*args,**kwargs):return (10.,10.),.8,[]
        def _get_court_mask(self,shape,court):return np.full(shape,255,np.uint8)
    class Neural:
        threshold=.9
        def __init__(self,path):self.calls=0
        def reset(self):self.calls=0
        def update(self,frame,**kwargs):
            self.calls+=1
            return ((11.,11.),.98) if self.calls>=3 else (None,0.)

    monkeypatch.setattr(src.court_motion,"CourtMotionTracker",Camera)
    monkeypatch.setattr(src.detector,"PickleballDetector",Detector)
    monkeypatch.setattr(src.ball_model,"NeuralBallLocator",Neural)
    monkeypatch.setattr(store,"run",BallTestStore.run.__get__(store))
    session=create(review)
    result=store.detail(session)
    assert result["metadata"]["status"]=="completed",result["metadata"]
    assert len(result["records"])==5
    assert result["records"][0]["frame"]==10
    assert result["records"][0]["neural"]==[11.,11.]
    cap=cv2.VideoCapture(str(video));cap.set(cv2.CAP_PROP_POS_FRAMES,10);ok,expected=cap.read();cap.release();assert ok
    saved=cv2.imread(str(store.folder(session)/"frames/000010.png"))
    np.testing.assert_array_equal(saved,expected)
    assert not store.busy()


def test_new_store_does_not_interrupt_another_live_worker_but_recovers_stale_job(review):
    _,store,*_=review;session=create(review)
    metadata=store.metadata(session);metadata["status"]="running"
    path=store.folder(session)/"metadata.json";path.write_text(json.dumps(metadata))
    from src.web.ball_testing import job_lease
    lease=job_lease(store.root/"worker.lock");assert lease is not None
    other=BallTestStore(store.project,store.video_path,store.resolve_court,lambda:False)
    assert other.metadata(session)["status"]=="running" and other.busy()
    lease.close()
    restarted=BallTestStore(store.project,store.video_path,store.resolve_court,lambda:False)
    assert restarted.metadata(session)["status"]=="interrupted"


def test_failed_job_creation_releases_worker_and_does_not_publish_partial_session(review,monkeypatch):
    import src.web.ball_testing as module
    client,store,*_=review
    original=module.write_json
    def failed_save(path,value):
        if path.name=="manual_labels.json":raise OSError("Disk full")
        original(path,value)
    monkeypatch.setattr(module,"write_json",failed_save)
    with pytest.raises(OSError,match="Disk full"):
        store.create(ClipRequest(video_name="match.avi",start_seconds=1,duration_seconds=.5))
    assert not store.busy()
    assert store.sessions()==[]


def test_range_label_marks_multiple_frames(review):
    client, store, *_ = review
    session = create(review)
    res = client.post(f"/api/ball-tests/{session}/labels/range", json={
        "start_frame": 10,
        "end_frame": 12,
        "visibility": "absent",
        "expected_revision": 0,
    })
    assert res.status_code == 200
    data = res.json()
    assert data["revision"] == 1
    assert data["count"] == 3
    assert data["reviewed"] == 3
    detail = store.detail(session)
    labels = {l["frame"]: l for l in detail["annotations"]["labels"]}
    assert labels[10]["visibility"] == "absent" and labels[10]["x"] is None
    assert labels[11]["visibility"] == "absent"
    assert labels[12]["visibility"] == "absent"


def test_interpolate_fills_trajectory_between_keyframes(review):
    client, store, *_ = review
    session = create(review)
    # Set keyframes at frame 10 (x=10, y=20) and frame 14 (x=18, y=28)
    client.post(f"/api/ball-tests/{session}/labels", json={
        "frame": 10, "visibility": "visible", "x": 10.0, "y": 20.0, "expected_revision": 0
    })
    client.post(f"/api/ball-tests/{session}/labels", json={
        "frame": 14, "visibility": "visible", "x": 18.0, "y": 28.0, "expected_revision": 1
    })
    res = client.post(f"/api/ball-tests/{session}/labels/interpolate", json={
        "start_frame": 10,
        "end_frame": 14,
        "expected_revision": 2,
    })
    assert res.status_code == 200
    data = res.json()
    assert data["revision"] == 3
    assert data["interpolated_frames"] == 3
    detail = store.detail(session)
    labels = {l["frame"]: l for l in detail["annotations"]["labels"]}
    # Frame 12 should be halfway: x=14, y=24
    assert labels[12]["visibility"] == "visible"
    assert abs(labels[12]["x"] - 14.0) < 1e-4
    assert abs(labels[12]["y"] - 24.0) < 1e-4


def test_prefill_populates_detections(review):
    client, store, *_ = review
    session = create(review)
    # Prefill from classical detector (which detected ball in session fixture)
    res = client.post(f"/api/ball-tests/{session}/labels/prefill", json={
        "source": "classical",
        "min_confidence": 0.0,
        "only_unreviewed": True,
        "expected_revision": 0,
    })
    assert res.status_code == 200
    data = res.json()
    assert data["revision"] == 1
    assert data["prefilled_count"] > 0
    detail = store.detail(session)
    assert len(detail["annotations"]["labels"]) == data["prefilled_count"]

