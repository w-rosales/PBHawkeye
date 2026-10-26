"""Known camera/3D flights validate physical scale, height and rejection gates."""

import json

import cv2
import numpy as np
import pytest

from src.analytics import TrajectoryAnalytics
from src.ball_3d import CameraModel, ObservationRecorder, LiveFlightSpeed, fit_flight, flight_segments
from src.config import COURT_CORNERS_FEET, FPS_TO_MPH
from src.homography import CourtHomography
from src.tracker import BallKalmanTracker, BallPoint


def camera_fixture(center=(10., -12., 12.), supplied=True, distortion=None):
    center = np.array(center)
    forward = np.array([10., 22., 0.])-center
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0, 0, 1.])
    right /= np.linalg.norm(right)
    rotation = np.array([right, np.cross(forward, right), forward])
    intrinsic = np.array([[1200., 0, 960.], [0, 1200., 540.], [0, 0, 1.]])
    projection = intrinsic @ np.column_stack([rotation, -rotation @ center])
    world = np.column_stack([COURT_CORNERS_FEET, np.zeros(4)]).astype(float)
    distortion = np.zeros(5) if distortion is None else np.array(distortion, float)
    corners, _ = cv2.projectPoints(world, cv2.Rodrigues(rotation)[0], -rotation @ center,
                                   intrinsic, distortion)
    metadata = {"camera_intrinsics": {"matrix": intrinsic.tolist(), "distortion": distortion.tolist()}} if supplied else {}
    court = CourtHomography(corners.reshape(-1, 2), image_size=(1920, 1080), metadata=metadata)
    return court, projection, rotation, intrinsic, center


def known_flight(stride=1, noise=0., duration=1.2, nominal_fps=30., variable_timing=False):
    court, projection, _, _, _ = camera_fixture()
    frames = np.arange(0, round(duration*30)+1, stride)
    t = frames/30.
    if variable_timing:
        t = t + .02*np.sin(t*5)
    # Independent analytic linear-drag trajectory (not calling production motion).
    drag = .4
    a = (1-np.exp(-drag*t))/drag
    b = (t-a)/drag
    initial = np.array([8., 8., 3.])
    velocity = np.array([3., 24., 18.])
    gravity = np.array([0., 0., -32.174])
    world = initial+a[:, None]*velocity+b[:, None]*gravity
    pixels_h = np.column_stack([world, np.ones(len(t))]) @ projection.T
    pixels = pixels_h[:, :2]/pixels_h[:, 2:]
    pixels += np.random.default_rng(42).normal(0, noise, pixels.shape)
    points = []
    recorder = ObservationRecorder(court, nominal_fps)
    for f, timestamp, pixel in zip(frames, t, pixels):
        point = BallPoint(frame_idx=int(f), pixel_x=float(pixel[0]), pixel_y=float(pixel[1]),
                          measurement_pixel=tuple(pixel), confidence=.95)
        recorder.record(point, court, float(timestamp))
        points.append(point)
    dense_t = np.linspace(t[0], t[-1], 10000)
    dense_v = np.exp(-drag*dense_t[:, None])*velocity + ((1-np.exp(-drag*dense_t[:, None]))/drag)*gravity
    average = np.trapezoid(np.linalg.norm(dense_v, axis=1), dense_t)/(dense_t[-1]-dense_t[0])*FPS_TO_MPH
    return court, points, world, average


@pytest.mark.parametrize("supplied", [True, False])
def test_camera_recovers_known_pose_and_ground_projection(supplied):
    court, projection, _, intrinsic, center = camera_fixture(supplied=supplied)
    camera = CameraModel.from_court(court)
    np.testing.assert_allclose(camera.center, center, atol=1e-4)
    np.testing.assert_allclose(camera.matrix, intrinsic, atol=.005)
    np.testing.assert_allclose(camera.projection, projection, atol=.01)
    assert camera.source == ("supplied_intrinsics" if supplied else "estimated_intrinsics")


def test_lens_distortion_is_removed_from_ball_observations():
    court, _, rotation, intrinsic, center = camera_fixture(distortion=[-.15, .03, 0, 0, 0])
    camera = CameraModel.from_court(court)
    point = np.array([[12., 15., 6.]])
    distorted, _ = cv2.projectPoints(point, cv2.Rodrigues(rotation)[0], -rotation @ center, intrinsic, camera.distortion)
    expected, _ = cv2.projectPoints(point, cv2.Rodrigues(rotation)[0], -rotation @ center, intrinsic, None)
    np.testing.assert_allclose(camera.undistort(distorted), expected.reshape(2), atol=.001)


@pytest.mark.parametrize("stride", [1, 2, 3])
def test_true_3d_speed_height_and_sampled_timing(stride):
    _, points, world, average = known_flight(stride=stride)
    estimate = fit_flight(points, 30.)
    assert estimate.status == "estimated", estimate.reason
    assert estimate.average_mph == pytest.approx(average, rel=.005)
    np.testing.assert_allclose(estimate.positions, world, atol=.001)
    assert estimate.distance_ft > np.linalg.norm(world[-1]-world[0])


def test_real_timestamps_override_incorrect_nominal_fps():
    _, points, world, average = known_flight(nominal_fps=60., variable_timing=True)
    result = fit_flight(points, 60.)
    assert result.status == "estimated", result.reason
    assert result.average_mph == pytest.approx(average, rel=.005)
    np.testing.assert_allclose(result.positions, world, atol=.001)


def test_small_detection_noise_retains_useful_speed():
    _, points, _, average = known_flight(noise=.35)
    estimate = fit_flight(points, 30.)
    assert estimate.status == "estimated", estimate.reason
    assert estimate.average_mph == pytest.approx(average, rel=.08)


def test_short_partial_arc_is_rejected_for_depth_ambiguity():
    _, points, _, _ = known_flight(duration=.8)
    estimate = fit_flight(points, 30.)
    assert estimate.status == "unavailable"
    assert estimate.relative_uncertainty > .20
    assert estimate.average_mph is None


def test_live_mph_uses_the_final_report_checks_and_expires():
    court, points, _, average = known_flight()
    live = LiveFlightSpeed(30.)
    for i, point in enumerate(points):
        live.update(points[:i+1], point.frame_idx)
    # Tick after the next bounded reconstruction interval without inventing data.
    speed = live.update(points, 40)
    assert speed == pytest.approx(average, abs=.05)
    assert live.status == "estimated"
    shots, _, _ = TrajectoryAnalytics(30.).analyze_rally(points, {}, court)
    assert speed == pytest.approx(shots[0].avg_mph, abs=.05)
    assert live.update(points, 86) is None
    assert live.status == "unavailable"
    assert "expired" in live.reason


def test_live_mph_keeps_ambiguous_flight_unavailable():
    _, points, _, _ = known_flight(duration=.8)
    live = LiveFlightSpeed(30.)
    assert live.update(points, 30) is None
    assert live.status == "unavailable"
    assert "uncertainty" in live.reason
    assert live.peak_mph is None


def test_live_mph_is_cleared_on_court_loss_or_new_track():
    _, points, _, average = known_flight()
    live = LiveFlightSpeed(30.)
    assert live.update(points, 40) == pytest.approx(average, abs=.05)
    assert live.update(points, 41, court_ok=False) is None
    assert "court tracking" in live.reason
    points.append(BallPoint(42, 100., 100., track_generation=1))
    assert live.update(points, 50) is None


def test_rejected_report_includes_visible_reason_counts():
    court, points, _, _ = known_flight(duration=.8)
    shots, _, summary = TrajectoryAnalytics(30.).analyze_rally(points, {}, court)
    assert summary["max_rally_speed_mph"] is None
    assert summary["estimated_speed_segments"] == 0
    assert summary["speed_rejection_counts"]["Depth or aerodynamic uncertainty is too large"] == len(shots)
    assert "none of" in summary["speed_display_reason"]
    assert "ambiguous" in summary["speed_display_reason"]


def test_camera_zoom_does_not_become_ball_speed():
    court, points, world, average = known_flight()
    recorder = ObservationRecorder(court, 30.)
    for i, point in enumerate(points):
        scale = 1.+.12*i/(len(points)-1)
        warp = np.array([[scale, 0, 960*(1-scale)], [0, scale, 540*(1-scale)], [0, 0, 1.]])
        corners = cv2.perspectiveTransform(court.pixel_corners.reshape(1, 4, 2), warp).reshape(4, 2)
        intrinsic = warp @ np.array(court.metadata["camera_intrinsics"]["matrix"])
        current = CourtHomography(corners, image_size=court.image_size,
                                  metadata={"camera_intrinsics": {"matrix": intrinsic.tolist()}})
        pixel = warp @ np.r_[point.measurement_pixel, 1.]
        point.measurement_pixel = tuple(pixel[:2]/pixel[2])
        recorder.record(point, current, point.timestamp_s)
    result = fit_flight(points, 30.)
    assert result.status == "estimated", result.reason
    assert result.average_mph == pytest.approx(average, abs=.02)
    np.testing.assert_allclose(result.positions, world, atol=.001)


def test_abrupt_reversal_is_split_before_fitting():
    _, points, _, _ = known_flight()
    # Repeat the path backwards, representing a strong direction discontinuity.
    backward = []
    for i, original in enumerate(reversed(points[:-1]), len(points)):
        backward.append(BallPoint(i, original.pixel_x, original.pixel_y,
                                   observation_pixel=original.observation_pixel,
                                   reference_pixel=original.reference_pixel,
                                   camera_projection=original.camera_projection,
                                   timestamp_s=i/30., timing_source="video_timestamps",
                                   camera_source=original.camera_source))
    combined = points+backward
    segments = flight_segments(combined, 30., 1)
    assert len(segments) >= 2
    assert any(abs(indices[-1]-(len(points)-1)) <= 2 for indices, _ in segments)
    assert all(not (indices[0] < len(points)-3 and indices[-1] > len(points)+2)
               for indices, _ in segments)


@pytest.mark.parametrize("fault", ["short", "predicted", "bad_pixels", "timestamps", "missing_camera", "stationary"])
def test_bad_flights_do_not_emit_mph(fault):
    _, points, _, _ = known_flight(duration=.2 if fault == "short" else 1.)
    if fault == "predicted":
        points[10].is_interpolated = True
    elif fault == "bad_pixels":
        points[10].observation_pixel = (500., 100.)
    elif fault == "timestamps":
        points[10].timestamp_s = points[9].timestamp_s
    elif fault == "missing_camera":
        points[10].camera_projection = None
    elif fault == "stationary":
        for point in points:
            point.observation_pixel = points[0].observation_pixel
    estimate = fit_flight(points, 30.)
    assert estimate.status == "unavailable"
    assert estimate.average_mph is None


def test_ball_filter_does_not_present_floor_speed_as_airborne_speed():
    court, _, _, _, _ = camera_fixture()
    tracker = BallKalmanTracker(fps=30.)
    for i in range(10):
        point = tracker.update(i, (960., 700.-i), .9, homography=court)
        assert point.speed_mph is None
        assert point.measurement_pixel == (960., 700.-i)


def test_constant_floor_motion_cannot_masquerade_as_a_gravity_driven_flight():
    court, _, _, _, _ = camera_fixture()
    recorder = ObservationRecorder(court, 30.)
    points = []
    for i in range(37):
        xy = court.court_to_pixel(10., 5.+.5*i)
        point = BallPoint(i, *xy, measurement_pixel=xy)
        recorder.record(point, court, i/30.)
        points.append(point)
    estimate = fit_flight(points, 30.)
    assert estimate.status == "unavailable"
    assert estimate.average_mph is None


def test_missing_observation_splits_flights():
    _, points, _, _ = known_flight()
    points[15].is_interpolated = True
    segments = flight_segments(points, 30., frame_stride=1)
    assert len(segments) >= 2
    assert all(not (indices[0] < 15 < indices[-1]) for indices, _ in segments)


def test_brief_occlusion_uses_only_actual_observations_and_preserves_time():
    court, points, _, average = known_flight()
    points[15].is_interpolated = True
    points[15].speed_reason = "Predicted position is not a ball observation"
    segments = flight_segments(points, 30., frame_stride=1)
    assert len(segments) == 1
    indices, estimate = segments[0]
    assert 15 not in indices
    assert estimate.status == "estimated", estimate.reason
    assert estimate.average_mph == pytest.approx(average, abs=.05)
    TrajectoryAnalytics(30., frame_stride=1).analyze_rally(points, {}, court)
    assert points[15].speed_mph is None


def test_tracker_reset_always_starts_a_separate_flight():
    _, points, _, _ = known_flight()
    for point in points[15:]:
        point.track_generation += 1
    segments = flight_segments(points, 30., frame_stride=1)
    assert all(not (indices[0] < 15 <= indices[-1]) for indices, _ in segments)


def test_analytics_exports_3d_and_nulls_not_zero(tmp_path):
    court, points, world, average = known_flight()
    analytics = TrajectoryAnalytics(fps=30.)
    shots, players, summary = analytics.analyze_rally(points, {}, court)
    assert len(shots) == 1
    assert shots[0].avg_mph == pytest.approx(average, abs=.05)
    assert shots[0].start_height_ft == pytest.approx(world[0, 2], abs=.001)
    assert summary["estimated_speed_segments"] == 1
    assert summary["speed_metric"] == "average_over_observed_flight"
    analytics.export_to_csv(shots, tmp_path/"flights.csv")
    analytics.export_to_json(summary, shots, players, tmp_path/"flights.json", points)
    report = json.loads((tmp_path/"flights.json").read_text())
    assert len(report["trajectory_3d"]) == len(points)
    assert report["shots"][0]["camera_source"] == "supplied_intrinsics"
    unknown = [BallPoint(i, 100., 100., court_x_ft=10., court_y_ft=float(i)) for i in range(30)]
    _, _, summary = analytics.analyze_rally(unknown, {})
    assert summary["max_rally_speed_mph"] is None
    assert all(p.speed_mph is None for p in unknown)


def test_rally_average_weights_observed_flight_time(monkeypatch):
    from src.ball_3d import FlightEstimate
    import src.analytics as analytics_module

    points = [BallPoint(i, 100., 100., timestamp_s=t) for i, t in enumerate([0., 1., 2., 5.])]
    estimates = []
    for distance, speed in [(10., 10.), (90., 30.)]:
        estimates.append(FlightEstimate(status="estimated", average_mph=speed*FPS_TO_MPH,
                                        peak_mph=speed*FPS_TO_MPH, distance_ft=distance,
                                        positions=np.array([[10., 0., 2.], [10., distance, 2.]]),
                                        velocities=np.array([[0., speed, 0.], [0., speed, 0.]])))
    monkeypatch.setattr(analytics_module, "flight_segments", lambda *a: [([0, 1], estimates[0]), ([2, 3], estimates[1])])
    _, _, summary = TrajectoryAnalytics(30.).analyze_rally(points, {})
    assert summary["avg_rally_speed_mph"] == round(100./4.*FPS_TO_MPH, 1)
    assert summary["measured_flight_duration_sec"] == 4.


def test_impossible_camera_is_rejected_without_disrupting_tracking():
    court = CourtHomography([(100., 100.), (500., 100.), (500., 500.), (100., 500.)], image_size=(640, 640))
    recorder = ObservationRecorder(court, 30.)
    point = BallPoint(0, 200., 200., measurement_pixel=(200., 200.))
    recorder.record(point, court)
    assert point.camera_projection is None
    assert point.speed_mph is None
