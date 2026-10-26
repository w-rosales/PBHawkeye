"""Regression tests for tracking correctness, camera compensation, and lossless writing.

Covers:
  - Elapsed-time handling and unphysical jump rejection (not concealing via clipping)
  - Player detection history independence (cached objects not mutating earlier history)
  - AsyncVideoWriter lossless queue drain and exception propagation
  - Ball filter confidence gating, Q matrix grouping, and tentative track confirmation
  - Camera-motion compensation rejecting stationary yellow landmarks
  - Small 2-pixel radius ball detection
"""

import numpy as np
import pytest
import cv2

from src.analytics import TrajectoryAnalytics
from src.config import MAX_BALL_SPEED_MPH, FPS_TO_MPH
from src.detector import PickleballDetector
from src.tracker import BallKalmanTracker, BallPoint, PlayerDetection
from src.web.app import AsyncVideoWriter


@pytest.mark.parametrize("frame_gap", [1, 2, 3, 4, 5])
def test_prediction_and_process_noise_use_elapsed_time(frame_gap):
    tracker = BallKalmanTracker(fps=30.)
    tracker.update(0, (100., 100.), .9)
    tracker.update(frame_gap, (101., 100.), .9)
    dt = frame_gap/30.
    variance = tracker.process_noise_std**2
    assert tracker.kf.F[0, 2] == pytest.approx(dt)
    assert tracker.kf.Q[0, 0] == pytest.approx(variance*dt**4/4)
    assert tracker.kf.Q[0, 2] == pytest.approx(variance*dt**3/2)
    assert tracker.kf.Q[2, 2] == pytest.approx(variance*dt**2)


def test_camera_zoom_rotation_transports_state_and_covariance():
    tracker = BallKalmanTracker(fps=30.)
    tracker.update(0, (100., 100.), .9)
    tracker.is_confirmed = True
    tracker.kf.x[2:, 0] = [30., -15.]
    tracker.kf.P = np.diag([4., 9., 16., 25.])
    state, covariance = tracker.kf.x.copy(), tracker.kf.P.copy()
    camera = np.float64([[0, -2, 300], [2, 0, 50], [0, 0, 1]])
    search_center = tracker.predict_position(1, camera_transform=camera)
    np.testing.assert_allclose(search_center, [101., 252.])
    np.testing.assert_array_equal(tracker.kf.x, state)
    np.testing.assert_array_equal(tracker.kf.P, covariance)
    point = tracker.update(1, None, camera_transform=camera)
    np.testing.assert_allclose([point.pixel_x, point.pixel_y], search_center)
    np.testing.assert_allclose([point.velocity_px_x, point.velocity_px_y], [30., 60.])
    transition = np.eye(4)
    transition[0, 2] = transition[1, 3] = 1/30.
    expected = transition @ np.diag([36., 16., 100., 64.]) @ transition.T + tracker.kf.Q
    np.testing.assert_allclose(tracker.kf.P, expected)


def test_projective_camera_covariance_matches_numerical_transport():
    camera = np.float64([[1.1, .02, 12], [-.01, .95, 5], [.0004, -.0002, 1]])
    state = np.float64([200., 100., 30., -15.])
    covariance = np.diag([4., 9., 16., 25.])

    def position(point):
        projected = camera @ np.r_[point, 1.]
        return projected[:2]/projected[2]

    def mapped(values):
        epsilon = 1e-4
        velocity = (position(values[:2] + epsilon*values[2:])
                    - position(values[:2] - epsilon*values[2:]))/(2*epsilon)
        return np.r_[position(values[:2]), velocity]

    epsilon = 1e-3
    jacobian = np.column_stack([(mapped(state + delta) - mapped(state - delta))/(2*epsilon)
                                for delta in epsilon*np.eye(4)])
    transported, actual_covariance = BallKalmanTracker._camera_state(
        state.reshape(4, 1), covariance, camera)
    np.testing.assert_allclose(transported[:, 0], mapped(state), atol=1e-6)
    np.testing.assert_allclose(actual_covariance, jacobian @ covariance @ jacobian.T,
                               rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("latest_translation", [0., 10.])
def test_two_frame_history_composes_camera_steps(latest_translation):
    detector = PickleballDetector(mode="A")
    frames = []
    # The current ball overlaps the incorrectly single-warped t-2 image. Its
    # true t-2 position in this view differs by the intervening camera step.
    for x in [200, 180, int(200+latest_translation)]:
        frame = np.full((360, 640, 3), 40, np.uint8)
        cv2.circle(frame, (x, 150), 4, (0, 240, 240), -1)
        frames.append(frame)
    previous = np.float64([[1, 0, 10], [0, 1, 0], [0, 0, 1]])
    current = np.float64([[1, 0, latest_translation], [0, 1, 0], [0, 0, 1]])
    detector.detect(frames[0])
    detector.detect(frames[1], camera_transform=previous)
    ball, _, _ = detector.detect(frames[2], camera_transform=current)
    assert ball is not None
    np.testing.assert_allclose(ball, [200+latest_translation, 150], atol=1.)


def test_history_transform_order_for_rotation_then_translation(monkeypatch):
    detector = PickleballDetector(mode="A")
    detector.prev_gray1 = detector.prev_gray2 = np.zeros((360, 640), np.uint8)
    detector._previous_camera_transform = np.float64([[0, -1, 200], [1, 0, 0], [0, 0, 1]])
    current = np.float64([[1, 0, 30], [0, 1, -10], [0, 0, 1]])
    transforms = []
    original = cv2.warpPerspective

    def record(image, transform, size):
        transforms.append(transform.copy())
        return original(image, transform, size)

    monkeypatch.setattr(cv2, "warpPerspective", record)
    detector._extract_moving_ball(np.zeros((360, 640, 3), np.uint8), [], camera_transform=current)
    # Old point (50,100) rotates into (100,50), then translates into (130,40).
    projected = transforms[1] @ np.array([50., 100., 1.])
    np.testing.assert_allclose(projected[:2]/projected[2], [130., 40.])


@pytest.mark.parametrize("stride", [1, 2, 3, 4, 5])
def test_all_sampling_speeds_keep_timestamps_and_reports(stride):
    analytics = TrajectoryAnalytics(fps=30., frame_stride=stride)
    frames = [i*stride for i in range(12)]
    trajectory = [BallPoint(frame_idx=i, pixel_x=100., pixel_y=100.,
                            court_x_ft=10., court_y_ft=5.+.1*i) for i in frames]
    shots, _, _ = analytics.analyze_rally(trajectory, {})
    assert shots == []
    assert all(p.speed_mph is None for p in trajectory)
    # The unit conversion utility still respects original source frame indices.
    _, _, speeds = analytics.calculate_velocities_and_speeds(
        [(p.court_x_ft, p.court_y_ft) for p in trajectory], frame_indices=frames)
    np.testing.assert_allclose(speeds, 3.*FPS_TO_MPH)


@pytest.mark.parametrize("stride", [1, 2, 3, 4, 5])
def test_sampling_does_not_bridge_a_missing_observation(stride):
    frames = [i*stride for i in range(25) if i != 12]
    trajectory = [BallPoint(frame_idx=i, pixel_x=100., pixel_y=100.,
                            court_x_ft=10., court_y_ft=5.+.1*i) for i in frames]
    analytics = TrajectoryAnalytics(fps=30., frame_stride=stride)
    runs = analytics._continuous_court_runs(trajectory, fps=30., expected_frame_gap=stride)
    assert [(trajectory[r[0]].frame_idx, trajectory[r[-1]].frame_idx) for r in runs] == [(0, 11*stride), (13*stride, 24*stride)]
    shots, _, _ = analytics.analyze_rally(trajectory, {})
    assert shots == []


def test_process_noise_matrix_structure():
    """Verify that the 4-state Kalman Filter process noise matrix Q is correctly grouped for [x, y, vx, vy]."""
    tracker = BallKalmanTracker(fps=30.0)
    Q = tracker.kf.Q
    assert Q.shape == (4, 4)
    # Diagonal variances must be strictly positive
    assert Q[0, 0] > 0.0  # x pos
    assert Q[1, 1] > 0.0  # y pos
    assert Q[2, 2] > 0.0  # vx vel
    assert Q[3, 3] > 0.0  # vy vel

    # Cross-terms must couple x with vx (0 with 2) and y with vy (1 with 3)
    assert Q[0, 2] > 0.0 and Q[2, 0] > 0.0
    assert Q[1, 3] > 0.0 and Q[3, 1] > 0.0

    # Cross-terms between decoupled dimensions x and y must be exactly 0
    assert Q[0, 1] == 0.0 and Q[1, 0] == 0.0
    assert Q[0, 3] == 0.0 and Q[3, 0] == 0.0
    assert Q[1, 2] == 0.0 and Q[2, 1] == 0.0
    assert Q[2, 3] == 0.0 and Q[3, 2] == 0.0


def test_zero_confidence_does_not_initialize_tracker():
    """Verify that a detection with zero or low confidence does not initialize tracking."""
    tracker = BallKalmanTracker(fps=30.0)

    # Zero confidence detection
    pt = tracker.update(frame_idx=0, detection=(500.0, 500.0), confidence=0.0)
    assert pt is None
    assert not tracker.is_initialized

    # Low confidence (< 0.25) detection
    pt = tracker.update(frame_idx=1, detection=(500.0, 500.0), confidence=0.10)
    assert pt is None
    assert not tracker.is_initialized

    # Valid confident detection initializes
    pt = tracker.update(frame_idx=2, detection=(500.0, 500.0), confidence=0.80)
    assert pt is not None
    assert tracker.is_initialized


def test_tentative_track_requires_confirmation():
    """Verify that a single isolated detection is tentative and resets if missed immediately."""
    tracker = BallKalmanTracker(fps=30.0)

    # Frame 0: first hit (tentative)
    pt0 = tracker.update(frame_idx=0, detection=(500.0, 500.0), confidence=0.8)
    assert pt0 is not None
    assert not tracker.is_confirmed

    # Frame 1: missed immediately (noise glitch) -> tracker should reset, not coast
    pt1 = tracker.update(frame_idx=1, detection=None, confidence=0.0)
    assert pt1 is None
    assert not tracker.is_initialized


def test_confirmed_track_coasts_up_to_max_missed():
    """Verify that once confirmed (2+ consecutive detections), the tracker interpolates missed frames."""
    tracker = BallKalmanTracker(fps=30.0)

    pt0 = tracker.update(frame_idx=0, detection=(500.0, 500.0), confidence=0.8)
    pt1 = tracker.update(frame_idx=1, detection=(505.0, 502.0), confidence=0.8)
    assert tracker.is_confirmed

    # Frame 2: missed detection -> should interpolate
    pt2 = tracker.update(frame_idx=2, detection=None, confidence=0.0)
    assert pt2 is not None
    assert pt2.is_interpolated


def test_unphysical_speed_rejected_not_clipped_to_80():
    """Verify that an impossible jump (e.g. 266 MPH) is rejected (0.0 MPH) rather than clipped to 80 MPH."""
    analytics = TrajectoryAnalytics(fps=30.0)

    # Coordinate jump: 0 to 39 ft in 1 frame (1/30 s) -> ~797 ft/s ~= 543 MPH!
    coords = [(10.0, 0.0), (10.0, 39.0)]
    vx, vy, speed_mph = analytics.calculate_velocities_and_speeds(coords, smoothing_window=3)

    # Neither point should be clipped to 80.0 MPH; unphysical speed must be rejected (0.0)
    for s in speed_mph:
        assert s == 0.0


def test_two_pixel_radius_ball_detected():
    """Verify that a small 2-pixel radius ball (area ~8 px) is detected and not rejected by BALL_MIN_AREA."""
    detector = PickleballDetector(mode="A")

    # Synthetic frame with a 2-pixel radius optic yellow ball moving across frames
    # optic yellow HSV ~ (35, 180, 220)
    hsv_ball = np.uint8([[[35, 200, 220]]])
    bgr_ball = cv2.cvtColor(hsv_ball, cv2.COLOR_HSV2BGR)[0, 0]
    color = (int(bgr_ball[0]), int(bgr_ball[1]), int(bgr_ball[2]))

    f1 = np.full((360, 640, 3), 40, dtype=np.uint8)
    f2 = np.full((360, 640, 3), 40, dtype=np.uint8)
    f3 = np.full((360, 640, 3), 40, dtype=np.uint8)

    # Ball moves by 6 pixels each frame
    cv2.circle(f1, (300, 200), 2, color, -1)
    cv2.circle(f2, (306, 200), 2, color, -1)
    cv2.circle(f3, (312, 200), 2, color, -1)

    detector.detect(f1)
    detector.detect(f2)
    ball, conf, _ = detector.detect(f3)

    assert ball is not None
    assert np.hypot(ball[0] - 312, ball[1] - 200) < 3.0


def test_camera_motion_compensation_rejects_static_yellow_landmark():
    """Verify that a stationary yellow landmark is not mistaken for a ball when the camera translates."""
    detector = PickleballDetector(mode="A")

    hsv_landmark = np.uint8([[[35, 200, 220]]])
    bgr_landmark = cv2.cvtColor(hsv_landmark, cv2.COLOR_HSV2BGR)[0, 0]
    color = (int(bgr_landmark[0]), int(bgr_landmark[1]), int(bgr_landmark[2]))

    f1 = np.full((360, 640, 3), 40, dtype=np.uint8)
    f2 = np.full((360, 640, 3), 40, dtype=np.uint8)
    f3 = np.full((360, 640, 3), 40, dtype=np.uint8)

    # Camera moves by dx = +10, dy = +5 each frame:
    # A stationary landmark on court shifts by +10, +5 in the image:
    cv2.circle(f1, (200, 150), 4, color, -1)
    cv2.circle(f2, (210, 155), 4, color, -1)
    cv2.circle(f3, (220, 160), 4, color, -1)

    # Camera transform from t-1 to t is translation by (+10, +5)
    T = np.array([[1.0, 0.0, 10.0], [0.0, 1.0, 5.0], [0.0, 0.0, 1.0]], dtype=np.float64)

    detector.detect(f1, camera_transform=T)
    detector.detect(f2, camera_transform=T)
    ball, conf, _ = detector.detect(f3, camera_transform=T)

    # Motion compensation aligns historical frames, so the landmark has 0 apparent motion and is rejected
    assert ball is None


def test_player_history_independence_from_cached_detections():
    """Verify that modifying a PlayerDetection in subsequent frames does not mutate earlier history snapshots."""
    detector = PickleballDetector(mode="B")
    detector.yolo_frame_interval = 5

    # Provide an initial detection
    p_orig = PlayerDetection(tracker_id=1, bbox_xyxy=(100, 100, 150, 200), feet_pixel=(125, 200), court_feet=(10.0, 18.0))
    detector.last_player_detections = [p_orig]
    detector.last_player_bbox_tuples = [(100, 100, 150, 200)]
    detector._frame_count = 1  # intermediate frame (reusing cache)

    frame = np.zeros((360, 640, 3), dtype=np.uint8)
    _, _, players_cached = detector.detect_mode_b(frame)

    assert len(players_cached) == 1
    p_cached = players_cached[0]

    # Mutating the cached detection must NOT alter p_orig
    p_cached.court_feet = (5.0, 4.0)
    p_cached.in_kitchen = True
    assert p_orig.court_feet == (10.0, 18.0)
    assert p_orig.in_kitchen is False


def test_async_video_writer_lossless_drain_and_exception_propagation():
    """Verify that AsyncVideoWriter flushes all frames losslessly and propagates exceptions."""
    class MockWriter:
        def __init__(self, should_fail=False):
            self.written = []
            self.released = False
            self.should_fail = should_fail

        def isOpened(self):
            return True

        def write(self, frame):
            if self.should_fail:
                raise IOError("Disk full error")
            self.written.append(frame)

        def release(self):
            self.released = True

    # 1. Lossless drain test
    mock = MockWriter()
    async_writer = AsyncVideoWriter(mock, max_queue_size=8)
    for i in range(25):
        async_writer.write(f"frame_{i}")
    async_writer.release()

    assert mock.released is True
    assert len(mock.written) == 25
    assert mock.written == [f"frame_{i}" for i in range(25)]

    # 2. Exception propagation test
    fail_mock = MockWriter(should_fail=True)
    async_fail_writer = AsyncVideoWriter(fail_mock, max_queue_size=8)
    with pytest.raises(IOError, match="Disk full error"):
        try:
            for i in range(10):
                async_fail_writer.write(f"frame_{i}")
        finally:
            async_fail_writer.release()
    assert fail_mock.released

