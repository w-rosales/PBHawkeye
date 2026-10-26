"""PickleTrack-Local: Local Pickleball Vision & Analytics Prototype.

Main CLI Entry Point:
Ingests recorded pickleball match footage (.mp4), tracks players and the pickleball,
calibrates court geometry, estimates 3D flight speed after tracking, and outputs
an annotated video along with a JSON/CSV analytics report.

Usage Examples:
  # Mode A (Pure Classical CV) with existing calibration:
  python main.py --video data/raw_videos/sample_rally.mp4 --mode A

  # Mode B (YOLOv8 + ByteTrack):
  python main.py --video data/raw_videos/sample_rally.mp4 --mode B

  # Run interactive calibration first:
  python main.py --video data/raw_videos/sample_rally.mp4 --mode A --calibrate

  # Generate and analyze a realistic synthetic sample rally:
  python main.py --generate-sample --mode A
"""

import argparse
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from src.analytics import TrajectoryAnalytics
from src.ball_3d import ObservationRecorder, LiveFlightSpeed
from src.calibration import calibrate_video, get_or_create_court_calibration
from src.config import (
    CALIBRATION_DIR,
    DEFAULT_FPS,
    HSV_PRESETS,
    OUTPUTS_DIR,
    RAW_VIDEOS_DIR,
)
from src.detector import PickleballDetector
from src.court_motion import CourtMotionTracker
from src.homography import CourtHomography
from src.tracker import BallKalmanTracker, PlayerDetection
from src.visualizer import PickleVisualizer


def generate_sample_clip(
    output_path: Path,
    num_frames: int = 150,
    fps: float = 30.0,
    width: int = 1280,
    height: int = 720,
) -> Tuple[Path, CourtHomography]:
    """Generates a realistic synthetic pickleball rally clip with court lines,

    moving optic-yellow ball, and simulated players.
    """
    from tests.test_synthetic_cv import create_synthetic_match_clip

    print(f"Generating synthetic rally clip ({num_frames} frames) at: {output_path}...")
    video_path, homography = create_synthetic_match_clip(
        output_path=output_path,
        num_frames=num_frames,
        fps=fps,
        width=width,
        height=height,
    )
    # Save default calibration JSON for this clip
    from src.calibration import video_signature
    homography.metadata = {"method": "synthetic", "confirmed": True,
                           "video_signature": video_signature(video_path), "frame_index": 0}
    calib_json = CALIBRATION_DIR / f"{video_path.stem}_calibration.json"
    homography.save_to_file(calib_json)
    print(f"Saved calibration file at: {calib_json}")
    return video_path, homography


def process_video(
    video_path: Path,
    mode: str = "A",
    calibration_file: Optional[Path] = None,
    interactive_calibrate: bool = False,
    output_video_path: Optional[Path] = None,
    output_csv_path: Optional[Path] = None,
    output_json_path: Optional[Path] = None,
    preview: bool = False,
    hsv_preset: str = "optic_yellow_default",
    start_frame: int = 0,
    max_frames: Optional[int] = None,
    ball_model: Optional[Path] = None,
) -> None:
    """Core processing pipeline for PickleTrack-Local."""
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Input video not found: {video_path}")

    # Determine default output paths
    stem = video_path.stem
    if output_video_path is None:
        output_video_path = OUTPUTS_DIR / f"annotated_{stem}.mp4"
    if output_csv_path is None:
        output_csv_path = OUTPUTS_DIR / f"shot_log_{stem}.csv"
    if output_json_path is None:
        output_json_path = OUTPUTS_DIR / f"analytics_{stem}.json"

    # Calibration resolution
    homography: Optional[CourtHomography] = None

    if interactive_calibrate:
        print(f"\n[Calibration] Launching interactive court 4-corner calibration for: {video_path.name}")
        try:
            calib_dest = calibration_file or (CALIBRATION_DIR / f"{stem}_calibration.json")
            homography = calibrate_video(video_path, output_json_path=calib_dest)
            print(f"[Calibration] Successfully saved to {calib_dest}")
        except Exception as e:
            print(f"[Calibration Notice] Interactive GUI failed or was canceled ({e}).")

    if homography is None:
        if calibration_file is not None and Path(calibration_file).exists():
            print(f"[Calibration] Loading court corners from: {calibration_file}")
            homography = CourtHomography.load_from_file(calibration_file)
        else:
            homography, source = get_or_create_court_calibration(video_path)
            if source == "needs_review":
                raise ValueError("Court lines could not be matched reliably. Review the court in the web studio or use --calibrate.")
            print(f"[Calibration] Court corners ready ({source}): {CALIBRATION_DIR / f'{stem}_calibration.json'}")

    # Open video capture
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video capture: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or DEFAULT_FPS
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    reference_index = int(homography.metadata.get("frame_index", 0))
    cap.set(cv2.CAP_PROP_POS_FRAMES, reference_index)
    ok, reference_frame = cap.read()
    if not ok:
        cap.release()
        raise ValueError("Could not read the court calibration reference frame")
    camera_tracker = CourtMotionTracker(reference_frame, homography)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    frames_to_process = (
        min(total_frames - start_frame, max_frames)
        if max_frames is not None
        else (total_frames - start_frame)
    )

    print(f"\n[Video Info] {video_path.name}: {width}x{height} @ {fps:.1f} FPS, {total_frames} total frames")
    if start_frame > 0 or max_frames is not None:
        print(f"[Processing Range] Starting at frame {start_frame}, processing {frames_to_process} frames")
    print(f"[Pipeline] Mode: {'Mode A (Pure Classical CV)' if mode.upper() == 'A' else 'Mode B (YOLOv8 + ByteTrack)'}")

    # Initialize Engine Components
    hsv_bounds = HSV_PRESETS.get(hsv_preset, HSV_PRESETS["optic_yellow_default"])
    detector = PickleballDetector(
        mode=mode,
        hsv_lower=hsv_bounds[0],
        hsv_upper=hsv_bounds[1],
        ball_hue_hint=32.,
        ball_model_path=str(ball_model) if ball_model is not None else None,
    )
    tracker = BallKalmanTracker(fps=fps)
    visualizer = PickleVisualizer(homography=homography, detection_mode=mode)
    analytics = TrajectoryAnalytics(fps=fps, frame_stride=1)

    # Initialize Video Writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_writer = cv2.VideoWriter(str(output_video_path), fourcc, fps, (width, height))

    player_history: Dict[int, List[PlayerDetection]] = defaultdict(list)
    frame_idx = start_frame
    processed_count = 0
    start_time = time.time()
    last_known_speed = None
    observations = ObservationRecorder(homography, fps)
    live_speed = LiveFlightSpeed(fps)
    previous_camera_ok = True
    current_shot_label = "Rally"

    print("\n[Processing] Running frame-by-frame analysis...")

    while True:
        if max_frames is not None and processed_count >= max_frames:
            break

        ret, frame = cap.read()
        if not ret or frame is None:
            break

        current_court = camera_tracker.update(frame)
        frame_homography = current_court if camera_tracker.tracking_ok else None
        camera_transform = camera_tracker.prev_to_curr_transform
        # Generated fixtures have a known static camera and no background texture.
        if homography.metadata.get("method") == "synthetic":
            frame_homography = homography
            camera_transform = None
        camera_ok = frame_homography is not None
        if not camera_ok or not previous_camera_ok:
            tracker.reset()
            detector.reset_ball_history()
        previous_camera_ok = camera_ok
        visualizer.homography = frame_homography

        # Predict next ball location with Kalman Filter for gating
        predicted_pos = tracker.predict_position(frame_idx, camera_transform=camera_transform)

        # Detect Ball and Players
        ball_meas, conf, players = detector.detect(frame, predicted_pos, homography=frame_homography,
                                                  camera_transform=camera_transform)

        # Update Kalman Tracker
        ball_pt = tracker.update(
            frame_idx=frame_idx,
            detection=ball_meas,
            confidence=conf,
            homography=frame_homography,
            camera_transform=camera_transform,
        )

        # Map Player Feet to Court Coordinates
        for p in players:
            if p.feet_pixel is not None and frame_homography is not None:
                cx_ft, cy_ft = frame_homography.pixel_to_court(p.feet_pixel[0], p.feet_pixel[1])
                p.court_feet = (cx_ft, cy_ft)
                p.in_kitchen = frame_homography.is_inside_kitchen(cx_ft, cy_ft)
            setattr(p, "frame_idx", frame_idx)
            if frame_homography is not None:
                player_history[p.tracker_id].append(p)

        timestamp_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
        timestamp = (timestamp_ms/1000. if np.isfinite(timestamp_ms) and
                     (timestamp_ms > 0 or frame_idx == 0) else None)
        observations.record(ball_pt, frame_homography, timestamp)
        last_known_speed = live_speed.update(tracker.trajectory, frame_idx, camera_ok)

        # Render Annotated Frame
        annotated_frame = visualizer.render_frame(
            frame=frame,
            ball_pt=ball_pt,
            ball_trail_pixels=tracker.get_recent_trail(12),
            ball_trail_court=tracker.get_recent_court_trail(12) if frame_homography is not None else [],
            players=players,
            current_speed_mph=last_known_speed,
            current_shot_type=current_shot_label,
            shot_count=1,
            frame_idx=frame_idx,
        )

        if frame_homography is None:
            cv2.putText(annotated_frame, "Court tracking lost - measurements paused", (20, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 180, 255), 2, cv2.LINE_AA)

        out_writer.write(annotated_frame)

        if preview:
            cv2.imshow("PickleTrack-Local Preview", annotated_frame)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                print("\n[Notice] Preview aborted by user.")
                break

        frame_idx += 1
        processed_count += 1
        if processed_count % 30 == 0 or processed_count == frames_to_process:
            pct = (processed_count / frames_to_process * 100.0) if frames_to_process > 0 else 0
            sys.stdout.write(f"\rProgress: [{processed_count}/{frames_to_process}] {pct:5.1f}% completed")
            sys.stdout.flush()

    cap.release()
    out_writer.release()
    if preview:
        cv2.destroyAllWindows()

    elapsed = time.time() - start_time
    proc_fps = processed_count / max(1e-3, elapsed)
    print(f"\n[Completed] Processed {processed_count} frames in {elapsed:.1f}s ({proc_fps:.1f} FPS)")

    # -----------------------------------------------------------------------
    # Comprehensive Analytics & Shot Classification
    # -----------------------------------------------------------------------
    print("\n[Analytics] Computing shot trajectories, speeds, and kitchen occupancy...")
    shots, player_reports, summary = analytics.analyze_rally(
        trajectory=tracker.trajectory,
        player_history=player_history,
        homography=homography,
    )

    # Export outputs
    analytics.export_to_csv(shots, output_csv_path)
    analytics.export_to_json(summary, shots, player_reports, output_json_path, tracker.trajectory)

    # Print Formatted Executive Report
    print("\n" + "=" * 60)
    print("           PICKLETRACK-LOCAL ANALYTICS REPORT             ")
    print("=" * 60)
    print(f"Annotated Video : {output_video_path}")
    print(f"Shot Log (CSV)  : {output_csv_path}")
    print(f"Report (JSON)   : {output_json_path}")
    print("-" * 60)
    print(f"Rally Duration  : {summary.get('rally_duration_sec', 0.0):.2f} seconds")
    print(f"Flight Segments : {summary.get('total_flight_segments', 0)}")
    peak = summary.get('max_rally_speed_mph')
    print(f"Peak Ball Speed : {f'{peak:.1f} MPH (estimated)' if peak is not None else 'Unavailable'}")
    average = summary.get('avg_rally_speed_mph')
    print(f"Avg Flight Speed: {f'{average:.1f} MPH (estimated)' if average is not None else 'Unavailable'}")
    print("-" * 60)
    print("FLIGHT SEGMENTS (contacts unverified):")
    for st, count in summary.get("shot_type_distribution", {}).items():
        print(f"  * {st:15s}: {count} segment(s)")
    print("-" * 60)
    print("PLAYER KITCHEN OCCUPANCY & STATS:")
    for pid, report in player_reports.items():
        print(
            f"  Player #{pid}: Kitchen {report.kitchen_percentage:4.1f}% | "
            f"Transition {report.transition_percentage:4.1f}% | "
            f"Baseline {report.baseline_percentage:4.1f}% | "
            f"Shots Hit: {report.shots_hit_count} | Peak: {report.max_shot_mph if report.max_shot_mph is not None else 'Unavailable'}"
        )
    print("=" * 60 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="PickleTrack-Local: 100% Local Pickleball Vision & Analytics Prototype"
    )
    parser.add_argument(
        "--video",
        type=str,
        default=None,
        help="Path to input match clip (.mp4, .mov, .avi, .mkv, etc.)",
    )
    parser.add_argument(
        "--start-frame",
        type=int,
        default=0,
        help="Frame index to start processing from (default: 0)",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Maximum number of frames to process (default: all)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["A", "B", "a", "b"],
        default="A",
        help="Detection Mode: 'A' (Classical CV: HSV+MOG2) or 'B' (YOLOv8n + ByteTrack)",
    )
    parser.add_argument(
        "--calibrate",
        action="store_true",
        help="Run interactive 4-corner court calibration before video processing",
    )
    parser.add_argument(
        "--ball-model", type=Path, default=None,
        help="Experimental local temporal-ball checkpoint; requires a validation-selected threshold",
    )
    parser.add_argument(
        "--calibration-file",
        type=str,
        default=None,
        help="Path to saved calibration JSON file",
    )
    parser.add_argument(
        "--output-video",
        type=str,
        default=None,
        help="Custom output annotated MP4 path",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default=None,
        help="Custom output shot log CSV path",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Custom output analytics JSON path",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Show live processing window",
    )
    parser.add_argument(
        "--hsv-preset",
        type=str,
        default="optic_yellow_default",
        choices=list(HSV_PRESETS.keys()),
        help="HSV color preset for ball detection in Mode A",
    )
    parser.add_argument(
        "--generate-sample",
        action="store_true",
        help="Generate a synthetic test match clip and run analysis immediately",
    )
    parser.add_argument(
        "--ui",
        action="store_true",
        help="Launch the interactive Web Dashboard in your browser (http://localhost:8000)",
    )

    args = parser.parse_args()

    if args.ui:
        import uvicorn
        print("\nStarting PickleTrack-Local Web UI at http://127.0.0.1:8000 ...")
        uvicorn.run("src.web.app:app", host="127.0.0.1", port=8000, log_level="info")
        return

    # Supported video formats
    supported_exts = {".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm"}

    # If --generate-sample is requested or no video supplied
    found_videos = [f for f in RAW_VIDEOS_DIR.iterdir() if f.is_file() and f.suffix.lower() in supported_exts]

    if args.generate_sample or (args.video is None and not found_videos):
        sample_video_path = RAW_VIDEOS_DIR / "sample_rally.mp4"
        generate_sample_clip(sample_video_path, num_frames=120)
        target_video = sample_video_path
    elif args.video is not None:
        target_video = Path(args.video)
    else:
        # Pick first available video in raw_videos
        target_video = found_videos[0]
        print(f"Discovered match clip in raw_videos: {target_video.name}")

    calib_path = Path(args.calibration_file) if args.calibration_file else None
    out_video = Path(args.output_video) if args.output_video else None
    out_csv = Path(args.output_csv) if args.output_csv else None
    out_json = Path(args.output_json) if args.output_json else None

    process_video(
        video_path=target_video,
        mode=args.mode.upper(),
        calibration_file=calib_path,
        interactive_calibrate=args.calibrate,
        output_video_path=out_video,
        output_csv_path=out_csv,
        output_json_path=out_json,
        preview=args.preview,
        hsv_preset=args.hsv_preset,
        start_frame=args.start_frame,
        max_frames=args.max_frames,
        ball_model=args.ball_model,
    )


if __name__ == "__main__":
    main()

