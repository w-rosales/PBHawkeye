"""Persistent, local ball review sessions. Corrections never train a model implicitly."""

from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import threading
import time
from typing import List, Literal, Optional
from uuid import uuid4

import cv2
import numpy as np
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, field_validator, model_validator

from src.calibration import video_signature
from src.model_registry import get_model_registry


def now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    temporary=path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False),encoding="utf-8")
    temporary.replace(path)


def job_lease(path):
    """A process-held file lock also makes restart recovery safe during tests."""
    handle=path.open("a+b")
    try:
        if path.stat().st_size==0:handle.write(b"0");handle.flush()
        handle.seek(0)
        if os.name=="nt":
            import msvcrt
            msvcrt.locking(handle.fileno(),msvcrt.LK_NBLCK,1)
        else:
            import fcntl
            fcntl.flock(handle.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        return handle
    except OSError:
        handle.close();return None


class ReviewPayload(BaseModel):
    @field_validator("*",mode="before")
    @classmethod
    def finite_json_numbers(cls,value):
        # Reject nonstandard NaN/Infinity JSON without echoing those values in
        # FastAPI's otherwise non-serializable validation error payload.
        if isinstance(value,float) and not math.isfinite(value):
            raise HTTPException(422,"Review values must be finite numbers")
        return value


def is_confirmed(label_dict: dict) -> bool:
    if "confirmed" in label_dict:
        return bool(label_dict["confirmed"])
    origin = label_dict.get("origin", "manual")
    if origin in ("prefilled", "interpolated"):
        return False
    notes = str(label_dict.get("notes", ""))
    if notes.startswith("prefill:") or notes.startswith("interpolated:") or notes.startswith("suggestion:"):
        return False
    return bool(label_dict.get("reviewed", True))


class TestClip(ReviewPayload):
    video_name: str
    start_seconds: float = Field(default=30,ge=0,allow_inf_nan=False)
    duration_seconds: float = Field(default=3,ge=.2,le=10,allow_inf_nan=False)
    split: Literal["train","validation","test"] = "train"
    ball_color: Literal["optic_yellow","neon_green","orange","white"] = "optic_yellow"


class Label(ReviewPayload):
    frame: int = Field(ge=0)
    visibility: Literal["visible","partial","occluded","absent","uncertain"]
    x: Optional[float] = Field(default=None,allow_inf_nan=False)
    y: Optional[float] = Field(default=None,allow_inf_nan=False)
    notes: str = Field(default="",max_length=500)
    expected_revision: int = Field(ge=0)
    origin: Literal["manual", "prefilled", "interpolated"] = "manual"
    confirmed: bool = True

    @model_validator(mode="after")
    def coordinates(self):
        if self.visibility in ("visible","partial"):
            if self.x is None or self.y is None:
                raise ValueError("Click the ball center before saving a visible or partial label")
        elif self.x is not None or self.y is not None:
            raise ValueError("Hidden, absent and uncertain labels must have no invented coordinates")
        return self


class RangeLabel(ReviewPayload):
    start_frame: int = Field(ge=0)
    end_frame: int = Field(ge=0)
    visibility: Literal["absent", "occluded", "uncertain"]
    notes: str = Field(default="", max_length=500)
    expected_revision: int = Field(ge=0)

    @model_validator(mode="after")
    def range_order(self):
        if self.start_frame > self.end_frame:
            raise ValueError("start_frame must be less than or equal to end_frame")
        return self


class InterpolateLabel(ReviewPayload):
    start_frame: int = Field(ge=0)
    end_frame: int = Field(ge=0)
    expected_revision: int = Field(ge=0)
    overwrite_existing: bool = False

    @model_validator(mode="after")
    def range_order(self):
        if self.start_frame >= self.end_frame:
            raise ValueError("start_frame must be less than end_frame")
        return self


class PrefillPayload(ReviewPayload):
    source: Literal["classical", "neural"] = "classical"
    min_confidence: float = Field(default=0.25, ge=0.0, le=1.0)
    only_unreviewed: bool = True
    expected_revision: int = Field(ge=0)


class ConfirmPayload(ReviewPayload):
    frames: List[int] = Field(default_factory=list)
    confirm_all: bool = False
    expected_revision: int = Field(ge=0)


class BallTestStore:
    def __init__(self, project_root, video_path, resolve_court, live_running):
        self.project=Path(project_root)
        self.root=self.project/"data"/"ball_tests"
        self.root.mkdir(parents=True,exist_ok=True)
        self.video_path=video_path
        self.resolve_court=resolve_court
        self.live_running=live_running
        self.lock=threading.RLock()
        self.worker=None
        self.lease=None
        self.cancel=threading.Event()
        # An interrupted process cannot leave an unfinishable "running" job.
        lease=job_lease(self.root/"worker.lock")
        if lease is not None:
            try:
                for path in self.root.glob("*/metadata.json"):
                    if not re.fullmatch(r"[0-9a-f]{32}",path.parent.name):continue
                    metadata=json.loads(path.read_text(encoding="utf-8"))
                    if metadata["status"] in ("queued","running"):
                        metadata.update(status="interrupted",error="Studio restarted before this test finished. Create a new test; existing reviews are preserved.")
                        write_json(path,metadata)
            finally:lease.close()

    def busy(self):
        if self.worker is not None and self.worker.is_alive():return True
        lease=job_lease(self.root/"worker.lock")
        if lease is None:return True
        lease.close();return False

    def folder(self,session):
        if not re.fullmatch(r"[0-9a-f]{32}",session):
            raise HTTPException(404,"Test session not found")
        folder=self.root/session
        if not (folder/"metadata.json").is_file():
            raise HTTPException(404,"Test session not found")
        return folder

    def metadata(self,session):
        return json.loads((self.folder(session)/"metadata.json").read_text(encoding="utf-8"))

    def labels(self,session):
        return json.loads((self.folder(session)/"manual_labels.json").read_text(encoding="utf-8"))

    def sessions(self):
        with self.lock:
            return sorted([json.loads(p.read_text(encoding="utf-8")) for p in self.root.glob("*/metadata.json") if re.fullmatch(r"[0-9a-f]{32}",p.parent.name)],key=lambda m:m["created_at"],reverse=True)

    def check_split(self,identity,start,end,split):
        """Include causal inputs; use video identity rather than folder names."""
        ranges=[]
        manifest=self.project/"data"/"learning"/"ball_v1"/"manifest.json"
        if manifest.exists():
            data=json.loads(manifest.read_text(encoding="utf-8"))
            for source in data["sources"]:
                meta=source["metadata"]
                signature={"name":Path(meta["video"].replace("\\","/")).name,"size_bytes":meta.get("source_size"),"mtime_ns":meta.get("source_mtime_ns")}
                if signature!=identity:continue
                for sample in data["samples"]:
                    if sample["source"]==f'audit{data["sources"].index(source)}':
                        ranges.append((sample["frame"]-2,sample["frame"],sample["split"],"seed experiment"))
        # The later seed holdout was evaluated after training, outside its manifest.
        holdout=self.project/"data"/"validation"/"gold_ball_holdout"/"metadata.json"
        if holdout.exists():
            meta=json.loads(holdout.read_text(encoding="utf-8"))
            if Path(meta["video"].replace("\\","/")).name==identity["name"]:
                ranges.append((max(0,meta["frame_start"]-2),meta["frame_end_inclusive"],"test","seed holdout"))
        for session in self.sessions():
            if session["video_signature"]==identity:
                ranges.append((max(0,session["frame_start"]-2),session["frame_end_inclusive"],session["split"],"saved review"))
        role=lambda value:"test" if value=="external" else value
        for first,last,purpose,source in ranges:
            if role(purpose)!=role(split) and max(max(0,start-2),first)<=min(end,last):
                raise HTTPException(409,f"This range overlaps {source} footage reserved for {role(purpose)}. Choose a different range or the same purpose. Two context frames are included in this check.")

    def create(self,request):
        with self.lock:
            if self.busy() or self.live_running():
                raise HTTPException(409,"Finish or stop the current analysis/test before starting another test")
            video=self.video_path(request.video_name)
            registry = get_model_registry()
            try:
                active_model = registry.get_deployable_model()
                checkpoint = active_model.checkpoint_file
                model_version = active_model.name
            except Exception as exc:
                raise HTTPException(409, f"The experimental ball model is not available: {exc}")
            court,_=self.resolve_court(video)
            if court.metadata.get("confirmed") is not True:
                raise HTTPException(409,"Confirm this video's court in Match Analysis first")
            cap=cv2.VideoCapture(str(video))
            try:
                if not cap.isOpened():raise HTTPException(400,"Could not open the source video")
                fps=float(cap.get(cv2.CAP_PROP_FPS));total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH));height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            finally:cap.release()
            if not math.isfinite(fps) or fps<=0 or min(total,width,height)<=0:
                raise HTTPException(400,"This video has no usable frame timing or dimensions")
            start=round(request.start_seconds*fps)
            if start>=total:raise HTTPException(400,"Start time is beyond the video")
            end=min(total-1,start+max(1,round(request.duration_seconds*fps))-1)
            if end-start+1>600:raise HTTPException(400,"Use a shorter clip (maximum 600 frames)")
            identity=video_signature(video)
            self.check_split(identity,start,end,request.split)
            self.lease=job_lease(self.root/"worker.lock")
            if self.lease is None:raise HTTPException(409,"Another test is preparing frames. Wait for it to finish.")
            try:
                return self.schedule(request,video,court,checkpoint,identity,fps,width,height,start,end,model_version=model_version)
            except Exception:
                self.lease.close();self.lease=None;raise

    def schedule(self,request,video,court,checkpoint,identity,fps,width,height,start,end,model_version="ball_v4"):
        session=uuid4().hex;folder=self.root/(session+".tmp");(folder/"frames").mkdir(parents=True)
        shutil.copyfile(checkpoint,folder/"model.pt")
        court.save_to_file(folder/"calibration.json")
        calibration_hash=hashlib.sha256((folder/"calibration.json").read_bytes()).hexdigest()
        metadata={"id":session,"created_at":now(),"status":"queued","error":None,"video":str(video),
                  "video_name":video.name,"video_signature":identity,"fps":fps,"width":width,"height":height,
                  "frame_start":start,"frame_end_inclusive":end,"split":request.split,"ball_color":request.ball_color,
                  "model_version":model_version,"model_sha256":hashlib.sha256((folder/"model.pt").read_bytes()).hexdigest(),
                  "calibration_sha256":calibration_hash,"processed":0,"total":end-start+1,"reviewed":0,
                  "classical_warmup_frames":min(start,62),"source_sha256":{name:hashlib.sha256((self.project/name).read_bytes()).hexdigest() for name in ["src/detector.py","src/ball_model.py","src/court_motion.py","src/web/ball_testing.py"] if (self.project/name).is_file()}}
        write_json(folder/"metadata.json",metadata)
        write_json(folder/"manual_labels.json",{"method":"Human corrections on unmarked source frames","revision":0,"metadata":metadata,"labels":[]})
        folder.replace(self.root/session)
        self.cancel.clear()
        self.worker=threading.Thread(target=self.run,args=(session,),daemon=True)
        self.worker.start()
        return metadata

    def run(self,session):
        folder=self.folder(session);metadata=self.metadata(session);cap=None;records=[];started=time.perf_counter()
        def progress(**changes):
            with self.lock:
                metadata.update(changes);write_json(folder/"metadata.json",metadata)
        try:
            progress(status="running")
            from src.ball_model import NeuralBallLocator
            from src.court_motion import CourtMotionTracker
            from src.detector import PickleballDetector
            from src.homography import CourtHomography
            from src.tracker import BallKalmanTracker
            video=Path(metadata["video"])
            if video_signature(video)!=metadata["video_signature"]:raise ValueError("Source video changed since this test was created")
            court=CourtHomography.load_from_file(folder/"calibration.json")
            cap=cv2.VideoCapture(str(video));cap.set(cv2.CAP_PROP_POS_FRAMES,int(court.metadata.get("frame_index",0)))
            ok,reference=cap.read()
            if not ok:raise ValueError("Could not decode the calibration reference")
            camera=CourtMotionTracker(reference,court)
            presets={"optic_yellow":((20,35,85),(70,255,255),32.),"neon_green":((35,40,80),(85,255,255),50.),"orange":((5,60,95),(22,255,255),14.),"white":((0,0,160),(180,45,255),None)}
            lower,upper,hue=presets[metadata["ball_color"]]
            classical=PickleballDetector(mode="B",hsv_lower=lower,hsv_upper=upper,ball_hue_hint=hue)
            neural=NeuralBallLocator(folder/"model.pt")
            tracker=BallKalmanTracker(metadata["fps"])
            start=metadata["frame_start"];context_start=max(0,start-2)
            first=max(0,start-metadata["classical_warmup_frames"])
            cap.set(cv2.CAP_PROP_POS_FRAMES,first)
            previous_ok=True
            for frame_index in range(first,metadata["frame_end_inclusive"]+1):
                if self.cancel.is_set():
                    progress(status="cancelled",error="Test stopped. Completed reviews in other sessions are unchanged.");return
                ok,frame=cap.read()
                if not ok:raise ValueError(f"Could not decode source frame {frame_index}")
                timestamp=float(cap.get(cv2.CAP_PROP_POS_MSEC))/1000.
                current=camera.update(frame);court_ok=camera.tracking_ok
                current=current if court_ok else None
                if not court_ok or not previous_ok:
                    tracker.reset();classical.reset_ball_history();neural.reset()
                previous_ok=court_ok
                transform=camera.prev_to_curr_transform
                prediction=tracker.predict_position(frame_index,camera_transform=transform)
                t=time.perf_counter()
                point,confidence,_=classical.detect(frame,prediction,homography=current,camera_transform=transform)
                classical_ms=(time.perf_counter()-t)*1000
                tracker.update(frame_index,point,confidence,homography=current,camera_transform=transform)
                neural_point=None;score=0.;neural_ms=0.
                if frame_index>=context_start:
                    if not cv2.imwrite(str(folder/"frames"/f"{frame_index:06d}.png"),frame):raise OSError("Could not save a source frame")
                    if court_ok:
                        t=time.perf_counter()
                        neural_point,score=neural.update(frame,court_mask=classical._get_court_mask(frame.shape[:2],current),homography=current)
                        neural_ms=(time.perf_counter()-t)*1000
                if frame_index>=start:
                    records.append({"frame":frame_index,"timestamp_s":timestamp,"nominal_timestamp_s":frame_index/metadata["fps"],
                                    "court_ok":bool(court_ok),"court_status":str(camera.status),
                                    "classical":list(point) if point is not None else None,"classical_score":float(confidence),
                                    "neural":list(neural_point) if neural_point is not None else None,"neural_score":float(score),
                                    "classical_ms":classical_ms,"neural_ms":neural_ms})
                    if len(records)%10==0:progress(processed=len(records))
            write_json(folder/"records.json",records)
            if video_signature(video)!=metadata["video_signature"]:raise ValueError("Source video changed while this test was running")
            progress(status="completed",processed=len(records),elapsed_seconds=time.perf_counter()-started,
                     neural_threshold=neural.threshold,completed_at=now())
        except Exception as exc:
            progress(status="failed",error=str(exc),processed=len(records))
        finally:
            if cap is not None:cap.release()
            if self.lease is not None:self.lease.close();self.lease=None

    def detail(self,session):
        with self.lock:
            metadata=self.metadata(session);labels=self.labels(session)
            path=self.folder(session)/"records.json"
            records=json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
            return {"metadata":metadata,"annotations":labels,"records":records,"metrics":self.metrics(records,labels["labels"])}

    @staticmethod
    def metrics(records,labels):
        truth={label["frame"]:label for label in labels if is_confirmed(label)}
        result={}
        for detector in ("classical","neural"):
            stats={"visible":0,"correct":0,"wrong":0,"missing":0,"absent_reviewed":0,"absent_false_detections":0}
            for record in records:
                label=truth.get(record["frame"])
                if label is None:continue
                point=record[detector]
                if label["visibility"]=="visible":
                    stats["visible"]+=1
                    if point is None:stats["missing"]+=1
                    elif np.linalg.norm(np.asarray(point)-[label["x"],label["y"]])<=10:stats["correct"]+=1
                    else:stats["wrong"]+=1
                elif label["visibility"]=="absent":
                    stats["absent_reviewed"]+=1;stats["absent_false_detections"]+=int(point is not None)
            result[detector]=stats
        return result

    def save_label(self,session,label):
        with self.lock:
            metadata=self.metadata(session)
            if metadata["status"]!="completed":raise HTTPException(409,"Wait for this test to finish before reviewing")
            if not metadata["frame_start"]<=label.frame<=metadata["frame_end_inclusive"]:raise HTTPException(400,"Frame is outside this review")
            if label.x is not None and not (0<=label.x<metadata["width"] and 0<=label.y<metadata["height"]):raise HTTPException(400,"Ball position is outside the source image")
            payload=self.labels(session)
            if label.expected_revision!=payload["revision"]:raise HTTPException(409,"This review changed in another window. Reopen it before saving again; your existing labels are preserved.")
            revision=payload["revision"]+1
            annotation={k:v for k,v in label.model_dump().items() if k!="expected_revision"}
            annotation.update(origin=label.origin,confirmed=label.confirmed,split=metadata["split"],reviewed=label.confirmed,updated_at=now(),revision=revision,
                              uncertainty_px=3 if label.visibility=="visible" else 8 if label.visibility=="partial" else None)
            previous=next((r for r in payload["labels"] if r["frame"]==label.frame),None)
            payload["labels"]=[r for r in payload["labels"] if r["frame"]!=label.frame]+[annotation]
            payload["labels"].sort(key=lambda r:r["frame"]);payload["revision"]=revision
            # One atomic authoritative file; audit history is included in that transaction.
            payload.setdefault("history",[]).append({"revision":revision,"previous":previous,"saved":annotation})
            write_json(self.folder(session)/"manual_labels.json",payload)
            metadata["reviewed"]=sum(1 for r in payload["labels"] if is_confirmed(r));write_json(self.folder(session)/"metadata.json",metadata)
            return {"revision":revision,"label":annotation,"reviewed":metadata["reviewed"]}

    def save_range_label(self, session, payload: RangeLabel):
        with self.lock:
            metadata = self.metadata(session)
            if metadata["status"] != "completed":
                raise HTTPException(409, "Wait for this test to finish before reviewing")
            if not (metadata["frame_start"] <= payload.start_frame <= metadata["frame_end_inclusive"] and
                    metadata["frame_start"] <= payload.end_frame <= metadata["frame_end_inclusive"]):
                raise HTTPException(400, "Frame range is outside this review")
            data = self.labels(session)
            if payload.expected_revision != data["revision"]:
                raise HTTPException(409, "This review changed in another window. Reopen it before saving again; your existing labels are preserved.")

            revision = data["revision"] + 1
            updated_frames = set(range(payload.start_frame, payload.end_frame + 1))
            new_labels = [r for r in data["labels"] if r["frame"] not in updated_frames]
            for f in sorted(updated_frames):
                new_labels.append({
                    "frame": f,
                    "visibility": payload.visibility,
                    "x": None,
                    "y": None,
                    "notes": payload.notes,
                    "split": metadata["split"],
                    "reviewed": True,
                    "confirmed": True,
                    "origin": "manual",
                    "updated_at": now(),
                    "revision": revision,
                    "uncertainty_px": None,
                })
            new_labels.sort(key=lambda r: r["frame"])
            data["labels"] = new_labels
            data["revision"] = revision
            data.setdefault("history", []).append({
                "revision": revision,
                "action": "range_label",
                "start_frame": payload.start_frame,
                "end_frame": payload.end_frame,
                "visibility": payload.visibility,
            })
            write_json(self.folder(session) / "manual_labels.json", data)
            total_confirmed = sum(1 for r in data["labels"] if is_confirmed(r))
            metadata["reviewed"] = total_confirmed
            write_json(self.folder(session) / "metadata.json", metadata)
            return {"revision": revision, "count": len(updated_frames), "reviewed": total_confirmed}

    def interpolate_labels(self, session, payload: InterpolateLabel):
        with self.lock:
            metadata = self.metadata(session)
            if metadata["status"] != "completed":
                raise HTTPException(409, "Wait for this test to finish before reviewing")
            if not (metadata["frame_start"] <= payload.start_frame < payload.end_frame <= metadata["frame_end_inclusive"]):
                raise HTTPException(400, "Frame range is outside this review")
            data = self.labels(session)
            if payload.expected_revision != data["revision"]:
                raise HTTPException(409, "This review changed in another window. Reopen it before saving again; your existing labels are preserved.")

            label_by_frame = {r["frame"]: r for r in data["labels"]}
            start_lbl = label_by_frame.get(payload.start_frame)
            end_lbl = label_by_frame.get(payload.end_frame)
            if not start_lbl or start_lbl.get("visibility") not in ("visible", "partial") or start_lbl.get("x") is None:
                raise HTTPException(400, f"Start frame {payload.start_frame} must have a visible or partial ball position")
            if not end_lbl or end_lbl.get("visibility") not in ("visible", "partial") or end_lbl.get("x") is None:
                raise HTTPException(400, f"End frame {payload.end_frame} must have a visible or partial ball position")

            intermediate_frames = list(range(payload.start_frame + 1, payload.end_frame))
            if not intermediate_frames:
                raise HTTPException(400, "There are no intermediate frames between start and end frames")

            revision = data["revision"] + 1
            x0, y0 = float(start_lbl["x"]), float(start_lbl["y"])
            x1, y1 = float(end_lbl["x"]), float(end_lbl["y"])
            span = payload.end_frame - payload.start_frame
            target_vis = "visible" if (start_lbl["visibility"] == "visible" and end_lbl["visibility"] == "visible") else "partial"

            frames_to_interpolate = []
            for f in intermediate_frames:
                existing_lbl = label_by_frame.get(f)
                if existing_lbl is not None and not payload.overwrite_existing:
                    continue
                frames_to_interpolate.append(f)

            interp_set = set(frames_to_interpolate)
            new_labels = [r for r in data["labels"] if r["frame"] not in interp_set]
            for f in frames_to_interpolate:
                t = (f - payload.start_frame) / span
                xi = x0 + t * (x1 - x0)
                yi = y0 + t * (y1 - y0)
                new_labels.append({
                    "frame": f,
                    "visibility": target_vis,
                    "x": float(xi),
                    "y": float(yi),
                    "notes": f"interpolated:{payload.start_frame}->{payload.end_frame}",
                    "split": metadata["split"],
                    "reviewed": False,
                    "confirmed": False,
                    "origin": "interpolated",
                    "updated_at": now(),
                    "revision": revision,
                    "uncertainty_px": 5 if target_vis == "visible" else 10,
                })
            new_labels.sort(key=lambda r: r["frame"])
            data["labels"] = new_labels
            data["revision"] = revision
            data.setdefault("history", []).append({
                "revision": revision,
                "action": "interpolate",
                "start_frame": payload.start_frame,
                "end_frame": payload.end_frame,
                "count": len(frames_to_interpolate),
                "overwrite_existing": payload.overwrite_existing,
            })
            write_json(self.folder(session) / "manual_labels.json", data)
            total_confirmed = sum(1 for r in data["labels"] if is_confirmed(r))
            metadata["reviewed"] = total_confirmed
            write_json(self.folder(session) / "metadata.json", metadata)
            return {"revision": revision, "interpolated_frames": len(intermediate_frames), "reviewed": total_confirmed}

    def prefill_predictions(self, session, payload: PrefillPayload):
        with self.lock:
            metadata = self.metadata(session)
            if metadata["status"] != "completed":
                raise HTTPException(409, "Wait for this test to finish before reviewing")
            data = self.labels(session)
            if payload.expected_revision != data["revision"]:
                raise HTTPException(409, "This review changed in another window. Reopen it before saving again; your existing labels are preserved.")

            path = self.folder(session) / "records.json"
            if not path.exists():
                raise HTTPException(404, "Session records not found")
            records = json.loads(path.read_text(encoding="utf-8"))

            existing = {r["frame"]: r for r in data["labels"]}
            revision = data["revision"] + 1
            prefilled = 0
            score_key = f"{payload.source}_score"

            for r in records:
                frame_idx = r["frame"]
                if payload.only_unreviewed and frame_idx in existing:
                    continue
                point = r.get(payload.source)
                score = r.get(score_key, 0.0)
                if point is not None and score >= payload.min_confidence:
                    existing[frame_idx] = {
                        "frame": frame_idx,
                        "visibility": "visible",
                        "x": float(point[0]),
                        "y": float(point[1]),
                        "notes": f"prefill:{payload.source}",
                        "split": metadata["split"],
                        "reviewed": False,
                        "confirmed": False,
                        "origin": "prefilled",
                        "updated_at": now(),
                        "revision": revision,
                        "uncertainty_px": 5,
                    }
                    prefilled += 1

            if prefilled > 0:
                data["labels"] = sorted(existing.values(), key=lambda r: r["frame"])
                data["revision"] = revision
                data.setdefault("history", []).append({
                    "revision": revision,
                    "action": "prefill",
                    "source": payload.source,
                    "count": prefilled,
                })
                write_json(self.folder(session) / "manual_labels.json", data)
                total_confirmed = sum(1 for r in data["labels"] if is_confirmed(r))
                metadata["reviewed"] = total_confirmed
                write_json(self.folder(session) / "metadata.json", metadata)
            return {"revision": data["revision"], "prefilled_count": prefilled, "reviewed": sum(1 for r in data["labels"] if is_confirmed(r))}

    def confirm_labels(self, session, payload: ConfirmPayload):
        with self.lock:
            metadata = self.metadata(session)
            if metadata["status"] != "completed":
                raise HTTPException(409, "Wait for this test to finish before reviewing")
            data = self.labels(session)
            if payload.expected_revision != data["revision"]:
                raise HTTPException(409, "This review changed in another window. Reopen it before saving again; your existing labels are preserved.")

            revision = data["revision"] + 1
            if payload.confirm_all:
                target_frames = {r["frame"] for r in data["labels"]}
            else:
                target_frames = set(payload.frames)

            confirmed_count = 0
            for r in data["labels"]:
                if r["frame"] in target_frames:
                    r["confirmed"] = True
                    r["reviewed"] = True
                    r["updated_at"] = now()
                    r["revision"] = revision
                    confirmed_count += 1

            data["revision"] = revision
            data.setdefault("history", []).append({
                "revision": revision,
                "action": "confirm",
                "frames": sorted(list(target_frames)),
                "count": confirmed_count,
            })
            write_json(self.folder(session) / "manual_labels.json", data)
            total_confirmed = sum(1 for r in data["labels"] if is_confirmed(r))
            metadata["reviewed"] = total_confirmed
            write_json(self.folder(session) / "metadata.json", metadata)
            return {"revision": revision, "confirmed_count": confirmed_count, "reviewed": total_confirmed}

    def export(self,session):
        with self.lock:
            metadata=self.metadata(session);payload=self.labels(session)
            confirmed_labels=[label for label in payload["labels"] if is_confirmed(label)]
            if metadata["status"]!="completed" or not confirmed_labels:raise HTTPException(409,"Finish the test and save at least one reviewed frame first")
            destination=self.project/"data"/"learning"/"reviews"/f'{session}_r{payload["revision"]}'
            if not destination.exists():
                staging=destination.with_name(destination.name+".tmp")
                (staging/"frames").mkdir(parents=True,exist_ok=True)
                needed={i for label in confirmed_labels for i in range(max(0,label["frame"]-2),label["frame"]+1)}
                for frame in needed:
                    src_frame = self.folder(session)/"frames"/f"{frame:06d}.png"
                    if src_frame.is_file():
                        shutil.copyfile(src_frame, staging/"frames"/f"{frame:06d}.png")
                write_json(staging/"metadata.json",metadata)
                export_payload=dict(payload)
                export_payload["labels"]=confirmed_labels
                write_json(staging/"manual_labels.json",export_payload)
                write_json(staging/"provenance.json",{"review_id":session,"annotation_revision":payload["revision"],"created_at":now(),
                           "labels_sha256":hashlib.sha256((staging/"manual_labels.json").read_bytes()).hexdigest(),"model_sha256":metadata["model_sha256"],
                           "note":"Saved dataset snapshot only; no training or model activation performed. Hidden/absent/partial labels are preserved; the current trainer consumes visible labels only."})
                staging.replace(destination)
            return {"dataset":str(destination.relative_to(self.project)),"split":metadata["split"],"revision":payload["revision"],"labels":len(confirmed_labels),
                    "trainable_visible_frames":sum(label["visibility"]=="visible" and label["frame"]>=2 for label in confirmed_labels),"trained":False}


def router(store):
    routes=APIRouter(prefix="/api/ball-tests")

    @routes.get("")
    def sessions():return {"sessions":store.sessions(),"busy":store.busy()}

    @routes.post("")
    def create(payload:TestClip):return store.create(payload)

    @routes.get("/{session}")
    def detail(session:str):return store.detail(session)

    @routes.post("/{session}/labels")
    def label(session:str,payload:Label):return store.save_label(session,payload)

    @routes.post("/{session}/labels/range")
    def range_label(session:str,payload:RangeLabel):return store.save_range_label(session,payload)

    @routes.post("/{session}/labels/interpolate")
    def interpolate(session:str,payload:InterpolateLabel):return store.interpolate_labels(session,payload)

    @routes.post("/{session}/labels/prefill")
    def prefill(session:str,payload:PrefillPayload):return store.prefill_predictions(session,payload)

    @routes.post("/{session}/labels/confirm")
    def confirm(session:str,payload:ConfirmPayload):return store.confirm_labels(session,payload)

    @routes.post("/{session}/export")
    def export(session:str):return store.export(session)

    @routes.post("/{session}/stop")
    def stop(session:str):
        with store.lock:
            if store.metadata(session)["status"] in ("queued","running"):store.cancel.set()
        return {"stopping":True}

    @routes.get("/{session}/frames/{frame}")
    def frame(session:str,frame:int):
        metadata=store.metadata(session)
        if not metadata["frame_start"]<=frame<=metadata["frame_end_inclusive"]:raise HTTPException(404,"Frame is outside this review")
        path=store.folder(session)/"frames"/f"{frame:06d}.png"
        if not path.is_file():raise HTTPException(404,"Frame is not available yet")
        return FileResponse(path,media_type="image/png")

    @routes.get("/{session}/annotations")
    def annotations(session:str):return FileResponse(store.folder(session)/"manual_labels.json",media_type="application/json",filename=f"ball_review_{session}.json")

    return routes
