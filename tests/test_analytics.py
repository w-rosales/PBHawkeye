"""Unit tests for the analytics engine (MPH calculation, inflection hit detection,

shot classification, and kitchen occupancy).
"""

import numpy as np
import pytest

from src.analytics import TrajectoryAnalytics
from src.config import COURT_LENGTH_FT, COURT_WIDTH_FT, FPS_TO_MPH
from src.tracker import BallPoint, PlayerDetection


@pytest.fixture
def analytics():
    return TrajectoryAnalytics(fps=30.0)


def test_velocity_and_mph_calculation(analytics):
    """Test that known constant velocity in feet/second converts accurately to MPH."""
    # Constant velocity: Ball travels from y=0 to y=44 ft over 22 frames at 30 fps
    # dt per frame = 1/30 s
    # Total distance = 44 ft
    # Total time = 22 * (1/30) = 0.7333 s
    # True velocity vy = 44 / (22/30) = 60 ft/s
    # Expected speed in MPH = 60 * (3600 / 5280) = 60 * 0.681818 = 40.909 MPH
    num_frames = 25
    y_vals = np.linspace(0.0, 44.0, num_frames)
    court_coords = [(10.0, float(y)) for y in y_vals]

    vx, vy, speed_mph = analytics.calculate_velocities_and_speeds(court_coords, smoothing_window=5)

    # Ignore edge boundary effects of Savitzky-Golay by checking central frames
    mid_speeds = speed_mph[4:-4]
    expected_mph = (44.0 / ((num_frames - 1) / 30.0)) * FPS_TO_MPH
    for s in mid_speeds:
        assert s == pytest.approx(expected_mph, rel=0.08)


def test_inflection_point_hit_detection(analytics):
    """Test that a V-shaped ball trajectory (reversal across net) triggers an inflection hit."""
    # Rally:
    # Shot 1: Ball travels from y=5 to y=35 ft (frames 0 to 20, vy > 0)
    # Inflection: Paddle strike at frame 20
    # Shot 2: Ball travels from y=35 back to y=5 ft (frames 20 to 40, vy < 0)
    frames_shot1 = np.linspace(5.0, 35.0, 21)
    frames_shot2 = np.linspace(35.0, 5.0, 21)[1:]  # avoid duplicate point

    trajectory: list[BallPoint] = []
    for f_idx, y in enumerate(list(frames_shot1) + list(frames_shot2)):
        trajectory.append(
            BallPoint(
                frame_idx=f_idx,
                pixel_x=500.0,
                pixel_y=500.0,
                court_x_ft=10.0,
                court_y_ft=float(y),
            )
        )

    inflection_indices = analytics.detect_inflection_points(trajectory, min_shot_gap_frames=6)

    # Must contain initial hit (0), reversal near index 20, and final index
    assert len(inflection_indices) >= 2
    assert 0 in inflection_indices
    # The reversal should be within 2 frames of frame 20
    reversals = [idx for idx in inflection_indices if 18 <= idx <= 22]
    assert len(reversals) >= 1


def test_shot_classification_dink(analytics):
    """Test classification of a low-speed dink landing in the kitchen."""
    shot_type = analytics.classify_shot(
        start_pt=(10.0, 16.0),
        end_pt=(12.0, 27.0),
        peak_mph=21.5,
        avg_mph=18.0,
        duration_sec=0.75,
        start_zone="near_kitchen",
        landing_zone="far_kitchen",
    )
    assert shot_type == "Dink"


def test_shot_classification_drive(analytics):
    """Test classification of a hard drive hit deep into the court."""
    shot_type = analytics.classify_shot(
        start_pt=(5.0, 4.0),
        end_pt=(15.0, 40.0),
        peak_mph=48.2,
        avg_mph=42.0,
        duration_sec=0.55,
        start_zone="near_baseline",
        landing_zone="far_baseline",
    )
    assert shot_type == "Drive"


def test_shot_classification_third_shot_drop(analytics):
    """Test classification of a third-shot drop originating from baseline into NVZ."""
    shot_type = analytics.classify_shot(
        start_pt=(10.0, 3.0),
        end_pt=(10.0, 26.0),
        peak_mph=28.0,
        avg_mph=24.0,
        duration_sec=0.95,
        start_zone="near_baseline",
        landing_zone="far_kitchen",
    )
    assert shot_type == "Drop"


def test_shot_classification_speedup(analytics):
    """Test classification of an aggressive attack shot from the kitchen/transition."""
    shot_type = analytics.classify_shot(
        start_pt=(12.0, 16.0),
        end_pt=(8.0, 30.0),
        peak_mph=36.5,
        avg_mph=32.0,
        duration_sec=0.40,
        start_zone="near_kitchen",
        landing_zone="far_transition",
    )
    assert shot_type == "Speedup"


def test_kitchen_occupancy_percentage(analytics):
    """Test calculation of player kitchen occupancy percentage."""
    # 20 frames total: 12 frames in kitchen (y=18), 8 frames in baseline (y=4)
    player_history = {
        1: [
            PlayerDetection(tracker_id=1, bbox_xyxy=(0, 0, 10, 10), court_feet=(10.0, 18.0))
            for _ in range(12)
        ]
        + [
            PlayerDetection(tracker_id=1, bbox_xyxy=(0, 0, 10, 10), court_feet=(10.0, 4.0))
            for _ in range(8)
        ]
    }

    dummy_trajectory = [
        BallPoint(frame_idx=i, pixel_x=100.0, pixel_y=100.0, court_x_ft=10.0, court_y_ft=20.0)
        for i in range(10)
    ]

    _, player_reports, _ = analytics.analyze_rally(dummy_trajectory, player_history)

    assert 1 in player_reports
    report = player_reports[1]
    assert report.total_frames_tracked == 20
    assert report.kitchen_frames == 12
    # 12 / 20 = 60.0%
    assert report.kitchen_percentage == pytest.approx(60.0, abs=1e-1)
    # 8 / 20 = 40.0%
    assert report.baseline_percentage == pytest.approx(40.0, abs=1e-1)


def _court_point(frame, y, x=10.0):
    return BallPoint(frame_idx=frame, pixel_x=100.0, pixel_y=100.0,
                     court_x_ft=x, court_y_ft=y)


def test_lost_court_mapping_does_not_invent_ball_speed(analytics):
    trajectory = [_court_point(i, 5.0) if not 10 <= i < 20
                  else _court_point(i, None, x=None) for i in range(30)]
    shots, _, summary = analytics.analyze_rally(trajectory, {})
    assert all(point.speed_mph is None for point in trajectory)
    assert summary["max_rally_speed_mph"] is None
    assert all(shot.shot_type != "Drive" for shot in shots)
    assert all(not (shot.frame_start < 10 and shot.frame_end >= 20) for shot in shots)
    assert all(point.speed_mph is None for point in trajectory[10:20])


def test_floor_motion_after_mapping_recovers_is_not_airborne_speed(analytics):
    trajectory = [_court_point(i, 5.0 + 0.5*i) if not 10 <= i < 20
                  else _court_point(i, None, x=None) for i in range(30)]
    shots, _, _ = analytics.analyze_rally(trajectory, {})
    # Floor coordinates alone cannot measure airborne speed, even after recovery.
    assert all(point.speed_mph is None for point in trajectory)
    assert shots == []


def test_missing_video_frames_do_not_compress_measurement_time(analytics):
    trajectory = [_court_point(i, 5.0 + 0.5*i) for i in list(range(10)) + list(range(30, 40))]
    shots, _, summary = analytics.analyze_rally(trajectory, {})
    assert all(point.speed_mph is None for point in trajectory)
    assert shots == []
    assert summary["rally_duration_sec"] == pytest.approx(1.3)


@pytest.mark.parametrize("invalid", [None, float("nan"), float("inf")])
def test_nonfinite_court_positions_break_measurement_runs(analytics, invalid):
    trajectory = [_court_point(i, 5.0) for i in range(20)]
    trajectory[10].court_x_ft = invalid
    shots, _, summary = analytics.analyze_rally(trajectory, {})
    assert all(point.speed_mph is None for point in trajectory)
    assert summary["max_rally_speed_mph"] is None
    assert all(not (shot.frame_start < 10 < shot.frame_end) for shot in shots)


def test_unmapped_players_do_not_reduce_kitchen_occupancy(analytics):
    positions = [(10.0, 18.0), (10.0, 18.0), (10.0, 4.0), (10.0, 4.0),
                 None, None, (float("nan"), 18.0)]
    history = {1: [PlayerDetection(tracker_id=1, bbox_xyxy=(0, 0, 10, 10), court_feet=position)
                   for position in positions]}
    _, reports, _ = analytics.analyze_rally([_court_point(i, 20.0) for i in range(10)], history)
    assert reports[1].total_frames_tracked == 4
    assert reports[1].kitchen_percentage == 50.0
    assert reports[1].baseline_percentage == 50.0

