"""Camera movement must move court coordinates without inventing new boundaries."""

import cv2
import numpy as np
import pytest

from src.court_motion import CourtMotionTracker
from src.homography import CourtHomography


@pytest.fixture
def textured_court():
    """A deterministic static background with an intentionally plain playing area."""
    rng = np.random.default_rng(28)
    frame = np.full((540, 960, 3), (65, 88, 61), dtype=np.uint8)
    for _ in range(750):
        x, y = rng.integers([8, 8], [952, 532])
        color = tuple(int(value) for value in rng.integers(15, 245, 3))
        cv2.rectangle(frame, (int(x), int(y)), (int(x + 5), int(y + 5)), color, -1)
    for x in range(25, 950, 80):
        cv2.putText(frame, str(x), (x, 65 + (x % 3) * 32), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (220, 220, 220), 2)
    corners = np.float32([[325, 180], [645, 175], [890, 490], [75, 500]])
    cv2.fillConvexPoly(frame, corners.astype(np.int32), (150, 85, 40))
    cv2.polylines(frame, [corners.astype(np.int32)], True, (240, 240, 240), 3)
    court = CourtHomography(corners, image_size=(960, 540),
                           metadata={"confirmed": True, "method": "manual", "nested": {"value": 1}})
    for y in (15, 29):
        left = np.rint(court.court_to_pixel(0, y)).astype(int)
        right = np.rint(court.court_to_pixel(20, y)).astype(int)
        cv2.line(frame, tuple(left), tuple(right), (240, 240, 240), 2)
    return frame, court


def warped(frame, matrix):
    return cv2.warpPerspective(frame, matrix, (frame.shape[1], frame.shape[0]))


def transformed_corners(court, matrix):
    return cv2.perspectiveTransform(court.pixel_corners.reshape(-1, 1, 2), matrix).reshape(-1, 2)


def test_zoom_tracks_a_baseline_outside_the_image(textured_court):
    frame, court = textured_court
    matrix = np.float64([[1.65, 0, -310], [0, 1.65, -160], [0, 0, 1]])
    tracker = CourtMotionTracker(frame, court)
    current = tracker.update(warped(frame, matrix))
    assert tracker.tracking_ok, tracker.status
    expected = transformed_corners(court, matrix)
    assert np.max(np.linalg.norm(current.pixel_corners - expected, axis=1)) < 6.0
    assert current.pixel_corners[2, 1] > frame.shape[0]
    assert current.court_corners_ft.tolist() == court.court_corners_ft.tolist()
    assert current.metadata == court.metadata
    current.metadata["nested"]["value"] = 2
    assert court.metadata["nested"]["value"] == 1
    # Original calibration stays anchored and is never overwritten by camera motion.
    assert np.array_equal(court.pixel_corners, tracker._anchor_corners)


def test_perspective_motion_is_followed_between_orb_updates(textured_court):
    frame, court = textured_court
    tracker = CourtMotionTracker(frame, court, update_interval=5)
    for index in range(12):
        zoom = 1.0 + index * 0.012
        matrix = np.float64([[zoom, index * 0.0005, -index * 2.1],
                             [index * 0.0002, zoom, -index * 1.5],
                             [index * 0.000001, -index * 0.0000005, 1]])
        current = tracker.update(warped(frame, matrix))
        assert tracker.tracking_ok, (index, tracker.status)
        expected = transformed_corners(court, matrix)
        assert np.max(np.linalg.norm(current.pixel_corners - expected, axis=1)) < 5.0


def test_blank_frame_retains_last_geometry_and_valid_frame_recovers(textured_court):
    frame, court = textured_court
    tracker = CourtMotionTracker(frame, court)
    matrix = np.float64([[1.2, 0, -75], [0, 1.2, -30], [0, 0, 1]])
    previous = tracker.update(warped(frame, matrix))
    assert tracker.tracking_ok
    failed = tracker.update(np.zeros_like(frame))
    assert not tracker.tracking_ok
    assert tracker.status == "insufficient_features"
    assert failed is previous
    assert tracker.inlier_count == 0
    recovered = tracker.update(frame)
    assert tracker.tracking_ok, tracker.status
    assert np.max(np.linalg.norm(recovered.pixel_corners - court.pixel_corners, axis=1)) < 3.0


def test_changed_frame_dimensions_are_rejected(textured_court):
    frame, court = textured_court
    tracker = CourtMotionTracker(frame, court)
    previous = tracker.homography
    assert tracker.update(frame[:300, :500]) is previous
    assert not tracker.tracking_ok
    assert tracker.status == "frame_size_changed"
    with pytest.raises(ValueError, match="dimensions"):
        CourtMotionTracker(frame[:300, :500], court)


def test_textureless_opening_frame_cannot_claim_tracking():
    frame = np.zeros((540, 960, 3), dtype=np.uint8)
    court = CourtHomography([[300, 180], [650, 180], [900, 500], [80, 500]], image_size=(960, 540))
    tracker = CourtMotionTracker(frame, court)
    assert not tracker.tracking_ok
    assert tracker.update(frame) is court
    assert not tracker.tracking_ok


def test_painted_static_court_without_background_texture_is_verified():
    from tests.test_court_detection import _synthetic_court
    corners = [[390, 130], [610, 130], [890, 490], [100, 490]]
    frame = _synthetic_court(corners)
    court = CourtHomography(corners, image_size=(960, 540), metadata={"confirmed": True})
    tracker = CourtMotionTracker(frame, court)
    current = tracker.update(frame)
    assert tracker.tracking_ok, tracker.status
    np.testing.assert_allclose(current.pixel_corners, corners)
    blank = np.zeros_like(frame)
    assert tracker.update(blank) is current
    assert not tracker.tracking_ok
    assert tracker.update(frame) is court
    assert tracker.tracking_ok


def test_moving_player_does_not_move_the_court(textured_court):
    frame, court = textured_court
    # A richly textured person-like region inside the court moves independently.
    rng = np.random.default_rng(9)
    person = rng.integers(0, 255, (100, 45, 3), dtype=np.uint8)
    opening = frame.copy()
    opening[260:360, 420:465] = person
    current = frame.copy()
    current[320:420, 540:585] = person
    tracker = CourtMotionTracker(opening, court)
    tracked = tracker.update(current)
    assert tracker.tracking_ok, tracker.status
    assert np.max(np.linalg.norm(tracked.pixel_corners - court.pixel_corners, axis=1)) < 2.0


def test_unrelated_static_scene_cannot_replace_geometry(textured_court):
    frame, court = textured_court
    tracker = CourtMotionTracker(frame, court)
    unrelated = np.random.default_rng(333).integers(0, 255, frame.shape, dtype=np.uint8)
    assert tracker.update(unrelated) is court
    assert not tracker.tracking_ok
    assert tracker.status == "insufficient_static_matches"
