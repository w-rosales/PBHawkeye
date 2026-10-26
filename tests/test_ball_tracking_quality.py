"""Regressions for marker lag, tiny balls, clothing and broken trails."""

import cv2
import numpy as np
import pytest

from src.detector import PickleballDetector
from src.tracker import BallKalmanTracker


def test_fast_observations_are_not_displaced_by_filter_lag():
    tracker = BallKalmanTracker(fps=30.)
    for i in range(20):
        measurement = (100.+30*i, 300.-4*i)
        point = tracker.update(i, measurement, .95)
        assert point is not None
        np.testing.assert_allclose([point.pixel_x, point.pixel_y], measurement)
        if i > 3:
            np.testing.assert_allclose(tracker.predict_position(i+1), [measurement[0]+30, measurement[1]-4], atol=3.)


def test_contact_reversal_does_not_leave_marker_following_old_velocity():
    tracker = BallKalmanTracker(fps=30.)
    positions = [100+25*i for i in range(8)]+[275-25*i for i in range(1, 9)]
    for i, x in enumerate(positions):
        point = tracker.update(i, (float(x), 200.), .95)
        assert point is not None and not point.is_interpolated
        assert point.pixel_x == x


def test_trails_never_connect_across_a_reset_or_prediction():
    tracker = BallKalmanTracker(fps=30.)
    tracker.update(0, (100., 100.), .9)
    tracker.update(1, (110., 100.), .9)
    assert len(tracker.get_recent_trail()) == 2
    tracker.update(2, None)
    assert tracker.get_recent_trail() == []
    tracker.reset()
    tracker.update(10, (400., 400.), .9)
    assert tracker.get_recent_trail() == [(400., 400.)]


def _frame(ball, logo=None, radius=5):
    frame = np.full((240, 400, 3), 40, np.uint8)
    if ball is not None:
        cv2.circle(frame, ball, radius, (0, 240, 240), -1)
    if logo is not None:
        green = cv2.cvtColor(np.uint8([[[46, 80, 250]]]), cv2.COLOR_HSV2BGR)[0, 0]
        cv2.circle(frame, logo, 7, tuple(map(int, green)), -1)
    return frame


def test_learned_ball_color_rejects_nearby_moving_green_logo():
    detector = PickleballDetector(mode="A")
    for i in range(7):
        expected = (100+10*i, 120)
        frame = _frame(expected, (110+7*i, 145) if i >= 4 else None)
        ball, _, _ = detector.detect(frame, predicted_pos=expected)
        if i >= 2:
            assert ball is not None
            np.testing.assert_allclose(ball, expected, atol=1.)


def test_slow_two_pixel_ball_survives_temporal_crescent_and_erosion():
    detector = PickleballDetector(mode="A")
    for i in range(8):
        expected = (100+i, 120)
        ball, _, _ = detector.detect(_frame(expected, radius=2), predicted_pos=expected)
        if i >= 2:
            assert ball is not None
            np.testing.assert_allclose(ball, expected, atol=1.)


def test_local_recovery_does_not_lock_onto_a_static_yellow_object():
    detector = PickleballDetector(mode="A")
    for _ in range(5):
        ball, confidence, _ = detector.detect(_frame((100, 120), radius=2), predicted_pos=(100, 120))
    assert ball is None
    assert confidence == 0.


def test_selected_yellow_profile_can_recover_from_wrong_learned_appearance():
    detector = PickleballDetector(mode="A", ball_hue_hint=32.)
    # Reproduce a false green acquisition before the real yellow ball appears.
    detector._ball_hue = 49.
    detector._ball_saturation = 70.
    detector._hue_observations = 20
    for i in range(5):
        expected = (120+12*i, 110)
        ball, _, _ = detector.detect(_frame(expected), predicted_pos=expected)
        if i >= 2:
            assert ball is not None
            np.testing.assert_allclose(ball, expected, atol=1.)


@pytest.mark.parametrize("hue", [14., 32., 50.])
def test_selected_color_profile_respects_orange_yellow_and_green(hue):
    lower, upper = (5, 22) if hue == 14 else (20, 70)
    detector = PickleballDetector(mode="A", hsv_lower=(lower, 35, 80),
                                  hsv_upper=(upper, 255, 255), ball_hue_hint=hue)
    color = cv2.cvtColor(np.uint8([[[hue, 150, 250]]]), cv2.COLOR_HSV2BGR)[0, 0]
    for i in range(5):
        expected = (100+15*i, 120)
        frame = np.full((240, 400, 3), 40, np.uint8)
        cv2.circle(frame, expected, 5, tuple(map(int, color)), -1)
        ball, _, _ = detector.detect(frame, predicted_pos=expected)
        if i >= 2:
            assert ball is not None
            np.testing.assert_allclose(ball, expected, atol=1.)


def test_calibrated_action_mask_does_not_clip_a_lob_at_top_of_frame():
    from src.homography import CourtHomography

    court = CourtHomography([(240.,80.),(400.,90.),(600.,330.),(30.,320.)], image_size=(640,360))
    mask = PickleballDetector._get_court_mask((360,640), court)
    assert mask[2,320] == 255
