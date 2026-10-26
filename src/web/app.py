"""PickleTrack-Local Web Studio API.

Provides:
  - Real-time MJPEG live video streaming while rendering/analyzing
  - Simplified, human-friendly settings (ball color, lighting, smart AI vs fast mode)
  - Interactive browser-based court 4-corner calibration
  - Live broadcast telemetry (speed gauge, shot classifications, player kitchen stats)
  - Output video download, CSV shot logs, and JSON analytics reports
"""

import asyncio
import queue
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, Query, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.analytics import TrajectoryAnalytics
from src.ball_3d import ObservationRecorder, LiveFlightSpeed
from src.calibration import get_or_create_court_calibration, video_signature
from src.config import (
    CALIBRATION_DIR,
    COURT_CORNERS_FEET,
    DEFAULT_FPS,
    MAX_BALL_SPEED_MPH,
    OUTPUTS_DIR,
    PROJECT_ROOT,
    RAW_VIDEOS_DIR,
)
from src.detector import PickleballDetector
from src.court_motion import CourtMotionTracker
from src.homography import CourtHomography
from src.model_registry import get_model_registry
from src.tracker import BallKalmanTracker, PlayerDetection
from src.visualizer import PickleVisualizer
from src.web.ball_testing import BallTestStore, router as ball_test_router

app = FastAPI(title="PickleTrack-Local Live Studio")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount outputs for direct video playback
app.mount("/outputs", StaticFiles(directory=str(OUTPUTS_DIR)), name="outputs")

# Local, reproducible frame-audit reports; kept separate from match telemetry.
validation_dir = PROJECT_ROOT / "data" / "validation"
validation_dir.mkdir(parents=True, exist_ok=True)
app.mount("/validation", StaticFiles(directory=str(validation_dir), html=True), name="validation")

# Friendly preset mapping to technical parameters
BALL_COLOR_PRESETS = {
    "optic_yellow": (20, 70, 35, 255, 85, 255),  # (h_min, h_max, s_min, s_max, v_min, v_max)
    "neon_green": (35, 85, 40, 255, 80, 255),
    "orange": (5, 22, 60, 255, 95, 255),
    "white": (0, 180, 0, 45, 160, 255),
}

# Live state tracking
latest_annotated_jpeg: Optional[bytes] = None

active_stream_state = {
    "is_running": False,
    "should_stop": False,
    "current_frame": 0,
    "total_frames": 0,
    "current_speed_mph": None,
    "speed_status": "pending",
    "speed_reason": "Collecting actual ball observations for a 3D speed estimate",
    "peak_speed_mph": None,
    "current_shot": "Rally",
    "total_shots": 0,
    "video_name": None,
    "mode": "B",
    "players": {},
    "recent_shots": [],
    "error": None,
    "court_tracking_ok": None,
    "court_tracking_error": None,
}


class AsyncVideoWriter:
    """Threaded lossless video writer wrapper that preserves recorded frames without stalling analysis."""

    def __init__(self, writer, max_queue_size: int = 16):
        self._writer = writer
        self._queue: queue.Queue = queue.Queue(maxsize=max_queue_size)
        self._exception: Optional[Exception] = None
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def _worker(self):
        while True:
            frame = self._queue.get()
            if frame is None:
                self._queue.task_done()
                break
            try:
                if self._exception is None:
                    self._writer.write(frame)
            except Exception as exc:
                self._exception = exc
            finally:
                self._queue.task_done()

    def isOpened(self) -> bool:
        return self._writer.isOpened() if hasattr(self._writer, "isOpened") else True

    def write(self, frame: np.ndarray) -> None:
        if self._exception is not None:
            raise self._exception
        # Lossless enqueue: pace frame analysis to video writer to prevent dropping frames
        self._queue.put(frame, block=True)

    def release(self) -> None:
        self._queue.put(None)
        self._queue.join()
        if self._thread.is_alive():
            self._thread.join()
        if hasattr(self._writer, "release"):
            self._writer.release()
        if self._exception is not None:
            raise self._exception


class CalibrationPayload(BaseModel):
    video_name: str
    corners: List[List[float]]  # [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]


def _video_path(video_name: str) -> Path:
    """Resolve a single recording from the video library."""
    root = RAW_VIDEOS_DIR.resolve()
    path = (root / video_name).resolve()
    if path.parent != root or not path.is_file():
        raise HTTPException(status_code=404, detail="Video not found")
    return path


def _calibration_response(homography: CourtHomography, source: str) -> dict:
    metadata = homography.metadata
    confirmed = metadata.get("confirmed") is True
    confidence = metadata.get("confidence")
    if not isinstance(confidence, (int, float)) or not np.isfinite(confidence):
        confidence = None
    else:
        confidence = max(0.0, min(1.0, float(confidence)))
    return {
        "calibrated": confirmed or (
            metadata.get("method") == "auto_detected"
            and confidence is not None and confidence >= 0.70
        ),
        "confirmed": confirmed,
        "confidence": confidence,
        "source": source,
        "corners": homography.pixel_corners.tolist(),
        "image_size": list(homography.image_size) if homography.image_size else None,
    }


def _resolve_calibration(video_path: Path, force_auto: bool = False):
    try:
        return get_or_create_court_calibration(video_path, force_auto_detect=force_auto)
    except (ValueError, OSError, cv2.error) as exc:
        raise HTTPException(status_code=400, detail=f"Court calibration failed: {exc}") from exc


ball_tests = BallTestStore(PROJECT_ROOT, _video_path, _resolve_calibration,
                           lambda: active_stream_state["is_running"])
app.include_router(ball_test_router(ball_tests))


@app.get("/ball-testing", response_class=HTMLResponse)
def ball_testing_page():
    return (PROJECT_ROOT / "src" / "web" / "ball_testing.html").read_text(encoding="utf-8")


@app.get("/api/videos")
def list_videos():
    """Returns available match clips in data/raw_videos."""
    supported = {".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm"}
    videos = []
    for f in RAW_VIDEOS_DIR.iterdir():
        if f.is_file() and f.suffix.lower() in supported:
            cap = cv2.VideoCapture(str(f))
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
            fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
            count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            cap.release()
            videos.append({
                "name": f.name,
                "size_mb": round(f.stat().st_size / (1024 * 1024), 1),
                "width": w,
                "height": h,
                "fps": round(fps, 1),
                "total_frames": count,
                "duration_sec": round(count / max(1.0, fps), 1),
            })
    return {"videos": videos}


@app.get("/api/frame")
def get_video_frame(video_name: str, frame_idx: int = 0):
    """Retrieves a single JPEG frame for visual calibration or scrubbing."""
    vpath = _video_path(video_name)

    cap = cv2.VideoCapture(str(vpath))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 1)
    safe_idx = max(0, min(frame_idx, total - 1))
    cap.set(cv2.CAP_PROP_POS_FRAMES, safe_idx)
    ret, frame = cap.read()
    if not ret or frame is None:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, frame = cap.read()
    cap.release()

    if not ret or frame is None:
        raise HTTPException(status_code=400, detail="Could not read frame")

    _, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return Response(content=encoded.tobytes(), media_type="image/jpeg")


@app.get("/api/calibration")
def get_calibration(video_name: str, force_auto: bool = False):
    """Fetch a calibration proposal and its actual validation state."""
    homography, source = _resolve_calibration(_video_path(video_name), force_auto)
    return _calibration_response(homography, source)


@app.post("/api/auto_detect_court")
def auto_detect_court_endpoint(video_name: str):
    """Triggers instant computer vision auto-detection of court corners."""
    homography, source = _resolve_calibration(_video_path(video_name), force_auto=True)
    return {"status": "success", **_calibration_response(homography, source)}


@app.post("/api/calibration")
def save_calibration(payload: CalibrationPayload):
    """Save an explicitly reviewed calibration for this recording only."""
    vpath = _video_path(payload.video_name)
    cap = cv2.VideoCapture(str(vpath))
    try:
        ret, frame = cap.read()
        if not ret or frame is None:
            raise HTTPException(status_code=400, detail="Could not read video for calibration")
        h, w = frame.shape[:2]
    finally:
        cap.release()
    try:
        homography = CourtHomography(
            pixel_corners=payload.corners,
            court_corners_ft=COURT_CORNERS_FEET,
            image_size=(w, h),
            metadata={
                "method": "manual",
                "confirmed": True,
                "confidence": None,
                "detector_version": "manual-v1",
                "video_signature": video_signature(vpath),
                "frame_index": 0,
            },
        )
        if (np.any(homography.pixel_corners < 0)
                or np.any(homography.pixel_corners[:, 0] >= w)
                or np.any(homography.pixel_corners[:, 1] >= h)):
            raise ValueError("Court corners must be inside the video frame")
    except (ValueError, TypeError, cv2.error) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    calib_file = CALIBRATION_DIR / f"{vpath.stem}_calibration.json"
    homography.save_to_file(calib_file)
    return {"status": "success", **_calibration_response(homography, "saved")}


@app.post("/api/stop")
def stop_stream():
    """Signals the live analysis stream to gracefully stop."""
    global active_stream_state
    active_stream_state["should_stop"] = True
    return {"status": "stopping"}


@app.get("/api/live_stats")
def get_live_stats():
    """Returns real-time telemetry of current speed, shot type, frame, and players."""
    return active_stream_state


@app.get("/api/stream")
def stream_analysis(
    video_name: str,
    mode: str = "B",
    start_frame: int = 0,
    max_frames: Optional[int] = 300,
    ball_color: str = "optic_yellow",
    lighting: str = "daylight",
    speed: int = 1,
    ball_tracker: str = "neural",
):
    """Real-time MJPEG live stream generator that plays and analyzes the video simultaneously."""
    global active_stream_state, latest_annotated_jpeg

    if ball_tests.busy():
        raise HTTPException(status_code=409, detail="Finish or stop the ball tracking test before starting match analysis")

    vpath = _video_path(video_name)
    stem = vpath.stem
    if speed not in range(1, 6):
        raise HTTPException(status_code=400, detail="Analysis speed must be between 1 and 5")
    stride_val = int(speed)
    homography, _ = _resolve_calibration(vpath)

    ball_model_path = None
    if ball_tracker.lower() in ("neural", "ai", "true"):
        registry = get_model_registry()
        try:
            active_model = registry.get_deployable_model()
            ball_model_path = str(active_model.checkpoint_file)
        except Exception as exc:
            raise HTTPException(status_code=409, detail=f"No deployable neural ball model: {exc}")
    if homography.metadata.get("confirmed") is not True:
        raise HTTPException(status_code=409, detail="Review and confirm the court before starting analysis")
    if start_frame < 0 or (max_frames is not None and max_frames < 0):
        raise HTTPException(status_code=400, detail="Frame ranges must be nonnegative")

    # Immediate initialization so polling returns live status immediately
    active_stream_state["is_running"] = True
    active_stream_state["should_stop"] = False
    active_stream_state["video_name"] = video_name
    active_stream_state["mode"] = mode.upper()
    active_stream_state["current_frame"] = 0
    active_stream_state["current_speed_mph"] = None
    active_stream_state["speed_status"] = "pending"
    active_stream_state["speed_reason"] = "Collecting actual ball observations for a 3D speed estimate"
    active_stream_state["peak_speed_mph"] = None
    active_stream_state["current_shot"] = "Rally"
    active_stream_state["total_shots"] = 0
    active_stream_state["total_flight_segments"] = 0
    active_stream_state["recent_shots"] = []
    active_stream_state["players"] = {}
    active_stream_state["error"] = None
    active_stream_state["court_tracking_ok"] = None
    active_stream_state["court_tracking_error"] = None
    latest_annotated_jpeg = None

    def frame_generator():
        global active_stream_state, latest_annotated_jpeg

        cap = None
        out_writer = None
        tracker = None
        player_history = {}
        try:
            cap = cv2.VideoCapture(str(vpath))
            if not cap.isOpened():
                raise ValueError("Could not open video for analysis")
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)
            fps = float(cap.get(cv2.CAP_PROP_FPS) or DEFAULT_FPS)
            total_vid_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 1000)
            if start_frame >= total_vid_frames:
                raise ValueError("The start frame is beyond the end of this video")
            ret, reference_frame = cap.read()
            if not ret or reference_frame is None:
                raise ValueError("Could not read the opening court reference frame")
            camera_tracker = CourtMotionTracker(reference_frame, homography)
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            target_count = min(total_vid_frames - start_frame, max_frames) if max_frames else (total_vid_frames - start_frame)
            active_stream_state["total_frames"] = target_count

            preset_vals = BALL_COLOR_PRESETS.get(ball_color, BALL_COLOR_PRESETS["optic_yellow"])
            h_min, h_max, s_min, s_max, v_min, v_max = preset_vals
            if lighting == "indoor":
                s_min = max(20, s_min - 10)
            elif lighting == "overcast":
                v_min = max(60, v_min - 20)
            detector = PickleballDetector(
                mode=mode.upper(),
                hsv_lower=(h_min, s_min, v_min),
                hsv_upper=(h_max, s_max, v_max),
                ball_hue_hint={"optic_yellow": 32., "neon_green": 50., "orange": 14.}.get(ball_color),
                ball_model_path=ball_model_path,
            )
            tracker = BallKalmanTracker(fps=fps)
            visualizer = PickleVisualizer(homography=homography, detection_mode=mode.upper())
            analytics = TrajectoryAnalytics(fps=fps, frame_stride=stride_val)
            observations = ObservationRecorder(homography, fps)
            live_speed = LiveFlightSpeed(fps, stride_val)

            out_path = OUTPUTS_DIR / f"annotated_{stem}.mp4"
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            raw_writer = cv2.VideoWriter(str(out_path), fourcc, fps / stride_val, (w, h))
            if not raw_writer.isOpened():
                raise ValueError("Could not create the annotated output video")
            out_writer = AsyncVideoWriter(raw_writer)
            frame_idx = start_frame
            processed = 0
            preview_w = min(w, 960)
            preview_h = int(h * (preview_w / w)) if w > 0 else 540
            last_yield_time = 0.0
            previous_camera_ok = True

            while not active_stream_state["should_stop"]:
                if max_frames and processed >= max_frames:
                    break

                ret, frame = cap.read()
                if not ret or frame is None:
                    break

                current_court = camera_tracker.update(frame)
                frame_homography = current_court if camera_tracker.tracking_ok else None
                cam_transform = getattr(camera_tracker, "prev_to_curr_transform", None)
                camera_ok = camera_tracker.tracking_ok
                if not camera_ok or not previous_camera_ok:
                    tracker.reset()
                    detector.reset_ball_history()
                previous_camera_ok = camera_ok
                active_stream_state["court_tracking_ok"] = camera_tracker.tracking_ok
                active_stream_state["court_tracking_error"] = (
                    None if camera_tracker.tracking_ok else
                    "The camera view could not be matched. Court measurements are paused until tracking recovers."
                )
                visualizer.homography = frame_homography

                # Kalman Prediction for spatial gating
                pred_pos = tracker.predict_position(frame_idx, camera_transform=cam_transform)

                # Detect Ball & Players (using camera-motion compensation and court mask)
                ball_meas, conf, players = detector.detect(
                    frame, pred_pos, homography=frame_homography, camera_transform=cam_transform
                )

                # Update Tracker with camera motion compensation
                ball_pt = tracker.update(
                    frame_idx=frame_idx,
                    detection=ball_meas,
                    confidence=conf,
                    homography=frame_homography,
                    camera_transform=cam_transform,
                )

                timestamp_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
                timestamp = (timestamp_ms/1000. if np.isfinite(timestamp_ms) and
                             (timestamp_ms > 0 or frame_idx == 0) else None)
                observations.record(ball_pt, frame_homography, timestamp)

                # Map player positions
                active_players = {}
                for p in players:
                    p_court = None
                    in_kitch = False
                    if p.feet_pixel is not None and frame_homography is not None:
                        cx, cy = frame_homography.pixel_to_court(p.feet_pixel[0], p.feet_pixel[1])
                        p_court = (float(cx), float(cy))
                        in_kitch = bool(frame_homography.is_inside_kitchen(cx, cy))
                    p.court_feet = p_court
                    p.in_kitchen = in_kitch

                    if p.tracker_id not in player_history:
                        player_history[p.tracker_id] = []
                    setattr(p, "frame_idx", frame_idx)

                    # Store an independent snapshot in player history to prevent mutating past records
                    snapshot = PlayerDetection(
                        tracker_id=p.tracker_id,
                        bbox_xyxy=tuple(p.bbox_xyxy),
                        confidence=p.confidence,
                        feet_pixel=p.feet_pixel,
                        court_feet=p_court,
                        in_kitchen=in_kitch,
                    )
                    setattr(snapshot, "frame_idx", frame_idx)
                    if frame_homography is not None and p_court is not None:
                        player_history[p.tracker_id].append(snapshot)

                    # Update live state
                    active_players[str(p.tracker_id)] = {
                        "id": p.tracker_id,
                        "in_kitchen": in_kitch,
                        "zone": frame_homography.get_court_zone(p_court[0], p_court[1]) if frame_homography is not None and p_court else "unavailable",
                    }
                active_stream_state["players"] = active_players

                # Use the same 3D checks live and in the final report. No
                # fallback to floor-projected distance when a fit is uncertain.
                curr_spd = live_speed.update(tracker.trajectory, frame_idx, camera_ok)
                active_stream_state["current_speed_mph"] = round(curr_spd, 1) if curr_spd is not None else None
                active_stream_state["peak_speed_mph"] = round(live_speed.peak_mph, 1) if live_speed.peak_mph is not None else None
                active_stream_state["speed_status"] = live_speed.status
                active_stream_state["speed_reason"] = live_speed.reason
                live_shot = "Tracking"
                active_stream_state["current_shot"] = live_shot
                active_stream_state["current_frame"] = processed

                # Render annotated frame
                annotated = visualizer.render_frame(
                    frame=frame,
                    ball_pt=ball_pt,
                    ball_trail_pixels=tracker.get_recent_trail(15),
                    ball_trail_court=tracker.get_recent_court_trail(15) if frame_homography is not None else [],
                    players=players,
                    current_speed_mph=curr_spd,
                    current_shot_type=live_shot,
                    shot_count=max(1, len(active_stream_state["recent_shots"])),
                    frame_idx=frame_idx,
                )

                if frame_homography is None:
                    cv2.putText(annotated, "Court tracking uncertain - measurements paused", (20, 90),
                                cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 180, 255), 2, cv2.LINE_AA)

                out_writer.write(annotated)

                # Stream live preview updates at 8 FPS max to prevent browser flow-control backpressure
                now = time.perf_counter()
                if now - last_yield_time >= 0.12 or processed == 0:
                    if w > 960:
                        preview_frame = cv2.resize(annotated, (preview_w, preview_h), interpolation=cv2.INTER_AREA)
                    else:
                        preview_frame = annotated

                    _, buffer = cv2.imencode(".jpg", preview_frame, [cv2.IMWRITE_JPEG_QUALITY, 65])
                    frame_bytes = buffer.tobytes()
                    latest_annotated_jpeg = frame_bytes
                    last_yield_time = now

                    yield (b"--frame\r\n"
                           b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")

                if stride_val > 1:
                    remaining = max(0, target_count - processed - 1)
                    for _ in range(min(stride_val - 1, remaining)):
                        if not cap.grab():
                            break
                        frame_idx += 1
                        processed += 1

                frame_idx += 1
                processed += 1
                active_stream_state["current_frame"] = processed
        except Exception as exc:
            active_stream_state["error"] = str(exc)
            raise
        finally:
            try:
                cleanup_error = None
                for resource in (cap, out_writer):
                    if resource is not None:
                        try:
                            resource.release()
                        except Exception as exc:
                            cleanup_error = cleanup_error or exc
                # Preserve an earlier processing error, while still closing both
                # resources. A final buffered encoder failure must be visible too.
                if cleanup_error is not None and active_stream_state["error"] is None:
                    raise cleanup_error

                if active_stream_state["error"] is None and tracker is not None:
                    shots, p_reports, summary = analytics.analyze_rally(
                        trajectory=tracker.trajectory,
                        player_history=player_history,
                        homography=homography,
                    )
                    csv_path = OUTPUTS_DIR / f"shot_log_{stem}.csv"
                    json_path = OUTPUTS_DIR / f"analytics_{stem}.json"
                    analytics.export_to_csv(shots, csv_path)
                    analytics.export_to_json(summary, shots, p_reports, json_path, tracker.trajectory)
                    active_stream_state["total_shots"] = 0
                    active_stream_state["total_flight_segments"] = len(shots)
                    active_stream_state["recent_shots"] = [vars(s) for s in shots[-10:]]
                    active_stream_state["peak_speed_mph"] = summary.get("max_rally_speed_mph")
                    valid = [s for s in shots if s.avg_mph is not None]
                    active_stream_state["current_speed_mph"] = valid[-1].avg_mph if valid else None
                    active_stream_state["speed_status"] = "estimated" if valid else "unavailable"
                    active_stream_state["speed_reason"] = (
                        "Estimated average over the last observed 3D flight; see results for confidence"
                        if valid else summary["speed_display_reason"])
            except Exception as exc:
                active_stream_state["error"] = active_stream_state["error"] or str(exc)
                raise
            finally:
                active_stream_state["is_running"] = False

    return StreamingResponse(frame_generator(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/api/live_frame")
def get_live_frame():
    """Returns the most recent annotated frame for real-time canvas updates."""
    global latest_annotated_jpeg
    if latest_annotated_jpeg is not None:
        return Response(
            content=latest_annotated_jpeg,
            media_type="image/jpeg",
            headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
        )
    raise HTTPException(status_code=404, detail="No live frame available")


@app.get("/api/results")
def get_results(video_name: str):
    """Retrieves analytics summary and shot log for the video."""
    stem = Path(video_name).stem
    json_path = OUTPUTS_DIR / f"analytics_{stem}.json"
    csv_path = OUTPUTS_DIR / f"shot_log_{stem}.csv"

    if not json_path.exists():
        raise HTTPException(status_code=404, detail="No analytics results found yet for this video")

    import json
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return {
        "summary": data.get("summary", {}),
        "players": data.get("players", {}),
        # JSON preserves numeric values and null. CSV turns both into strings.
        "shots": data.get("shots", []),
        "trajectory_3d": data.get("trajectory_3d", []),
        "video_url": f"/outputs/annotated_{stem}.mp4",
        "csv_url": f"/outputs/shot_log_{stem}.csv",
    }


@app.get("/", response_class=HTMLResponse)
def index():
    """Serves the user-friendly live analysis dashboard."""
    html_file = PROJECT_ROOT / "src" / "web" / "index.html"
    if html_file.exists():
        with open(html_file, "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>PickleTrack-Local Live Studio</h1>"
