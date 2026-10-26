"""Synthetic Computer Vision End-to-End Test for Mode A.

Generates a synthetic pickleball rally clip with an optic yellow ball moving
across a rendered court, runs Mode A (MOG2 + HSV + Kalman) tracking,
and verifies trajectory extraction, homography mapping, and report generation.
"""

from pathlib import Path
import cv2
import numpy as np
import pytest

from src.analytics import TrajectoryAnalytics
from src.config import COURT_CORNERS_FEET
from src.detector import PickleballDetector
from src.homography import CourtHomography
from src.tracker import BallKalmanTracker, PlayerDetection
from src.visualizer import PickleVisualizer


def generate_synthetic_court_frame(
    width: int = 1280,
    height: int = 720,
    court_quad: tuple = ((300, 160), (980, 160), (1160, 660), (120, 660)),
) -> np.ndarray:
    """Renders a static perspective pickleball court."""
    frame = np.full((height, width, 3), (35, 45, 35), dtype=np.uint8)  # Surroundings

    # Draw court polygon
    pts = np.array(court_quad, dtype=np.int32).reshape((-1, 1, 2))
    cv2.fillPoly(frame, [pts], (140, 90, 40))  # Court surface (blueish-slate in BGR)

    # Court lines
    cv2.polylines(frame, [pts], isClosed=True, color=(255, 255, 255), thickness=3)

    return frame


def create_synthetic_match_clip(
    output_path: Path,
    num_frames: int = 60,
    fps: float = 30.0,
    width: int = 1280,
    height: int = 720,
) -> tuple[Path, CourtHomography]:
    """Generates an MP4 video of an optic-yellow ball bouncing back and forth across a synthetic court."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 4 Court corners in pixels: [Top-Left, Top-Right, Bottom-Right, Bottom-Left]
    pixel_corners = [
        (300.0, 160.0),   # Far-Left (0.0, 44.0)
        (980.0, 160.0),   # Far-Right (20.0, 44.0)
        (1160.0, 660.0),  # Near-Right (20.0, 0.0)
        (120.0, 660.0),   # Near-Left (0.0, 0.0)
    ]

    homography = CourtHomography(
        pixel_corners=pixel_corners,
        court_corners_ft=COURT_CORNERS_FEET,
        image_size=(width, height),
    )

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    # Optic yellow color in BGR: (0, 240, 240) -> in HSV is H approx 30 (yellow), S=255, V=240
    ball_color_bgr = (0, 240, 240)
    ball_radius = 8

    # Trajectory on 2D court:
    # Shot 1: from Near-Court (x=10, y=5) to Far-Court (x=10, y=38)
    # Shot 2: returns from (x=10, y=38) to Near-Court (x=10, y=8)
    half_frames = num_frames // 2
    y_shot1 = np.linspace(5.0, 38.0, half_frames)
    y_shot2 = np.linspace(38.0, 8.0, num_frames - half_frames)
    all_y = np.concatenate([y_shot1, y_shot2])

    for f_idx in range(num_frames):
        frame = generate_synthetic_court_frame(width, height)

        # Draw a synthetic player at baseline
        cv2.rectangle(frame, (600, 580), (680, 680), (120, 120, 120), -1)

        # Map current court position to pixel coordinate
        c_x = 10.0 + 1.5 * np.sin(f_idx * 0.2)
        c_y = float(all_y[f_idx])
        u_px, v_px = homography.court_to_pixel(c_x, c_y)

        # Draw yellow ball circle
        cv2.circle(frame, (int(u_px), int(v_px)), ball_radius, ball_color_bgr, -1)
        # Subtle white shine
        cv2.circle(frame, (int(u_px) - 2, int(v_px) - 2), 2, (255, 255, 255), -1)

        out.write(frame)

    out.release()
    return output_path, homography


def test_synthetic_mode_a_tracking(tmp_path):
    """Verifies that Mode A detects, tracks, and extracts trajectory from synthetic video."""
    video_path = tmp_path / "synthetic_test_clip.mp4"
    video_path, homography = create_synthetic_match_clip(video_path, num_frames=45)

    assert video_path.exists()

    cap = cv2.VideoCapture(str(video_path))
    assert cap.isOpened()

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    detector = PickleballDetector(mode="A")
    tracker = BallKalmanTracker(fps=fps)
    visualizer = PickleVisualizer(homography=homography, detection_mode="A")

    detected_ball_count = 0
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        predicted_pos = (
            (float(tracker.kf.x[0, 0]), float(tracker.kf.x[1, 0]))
            if tracker.is_initialized
            else None
        )

        ball_detection, conf, players = detector.detect(frame, predicted_pos)
        ball_pt = tracker.update(
            frame_idx=frame_idx,
            detection=ball_detection,
            confidence=conf,
            homography=homography,
        )

        if ball_pt is not None:
            detected_ball_count += 1
            assert ball_pt.court_x_ft is not None
            assert ball_pt.court_y_ft is not None
            # Court coordinates must be within physical court bounds + buffer
            assert -5.0 <= ball_pt.court_x_ft <= 25.0
            assert -5.0 <= ball_pt.court_y_ft <= 49.0

        # Verify visualizer rendering runs without exceptions
        annotated = visualizer.render_frame(
            frame=frame,
            ball_pt=ball_pt,
            ball_trail_pixels=tracker.get_recent_trail(10),
            ball_trail_court=tracker.get_recent_court_trail(10),
            players=players,
            current_speed_mph=ball_pt.speed_mph if ball_pt else 0.0,
            current_shot_type="Rally",
            shot_count=1,
            frame_idx=frame_idx,
        )
        assert annotated.shape == frame.shape

        frame_idx += 1

    cap.release()

    # Mode A should have tracked the yellow ball in majority of the frames
    assert detected_ball_count >= 25, f"Only detected {detected_ball_count} ball frames out of {frame_idx}"
    assert len(tracker.trajectory) >= 25


def test_shoe_filter_prevents_false_positives():
    """Verifies that yellow objects inside the bottom 25% of player bbox are rejected."""
    detector = PickleballDetector(mode="A")

    # Player bbox: x1=200, y1=100, x2=300, y2=400 (height = 300)
    # Bottom 25% starts at y = 100 + 0.75 * 300 = 325 to 400
    player_bboxes = [(200.0, 100.0, 300.0, 400.0)]

    # Yellow shoe at (250, 360) -> Inside bottom 25%
    shoe_pt = (250.0, 360.0)
    assert detector.is_inside_shoe_zone(shoe_pt, player_bboxes) is True

    # Real ball at waist/paddle level at (250, 220) -> Outside shoe zone
    ball_pt = (250.0, 220.0)
    assert detector.is_inside_shoe_zone(ball_pt, player_bboxes) is False

    # Real ball to the side at (350, 360) -> Outside horizontally
    outside_pt = (350.0, 360.0)
    assert detector.is_inside_shoe_zone(outside_pt, player_bboxes) is False


def test_mode_b_initialization_and_inference(tmp_path):
    """Verifies that Mode B (YOLOv8 + ByteTrack) initializes and runs inference without errors."""
    detector = PickleballDetector(mode="B")
    assert detector.mode == "B"
    assert detector.yolo_model is not None

    # Run inference on a blank synthetic test frame
    test_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    ball, conf, players = detector.detect(test_frame)

    assert isinstance(players, list)
    assert ball is None or isinstance(ball, tuple)
    assert isinstance(conf, float)
