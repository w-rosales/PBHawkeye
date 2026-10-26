"""Court registration regressions using real footage and independent geometry.

The real image is the downsampled opening frame of the user's recording.
Its labels are observations of painted line intersections, not
coordinates emitted by the detector. See fixtures/real_court_960.json.
"""

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from src.calibration import auto_detect_court_corners, synthesize_clean_court_frame
from src.config import COURT_CORNERS_FEET
from src.homography import CourtHomography


FIXTURES = Path(__file__).parent / "fixtures"


def _project(points, corners):
    matrix = cv2.getPerspectiveTransform(
        np.float32(COURT_CORNERS_FEET), np.float32(corners)
    )
    return cv2.perspectiveTransform(
        np.float32(points).reshape(-1, 1, 2), matrix
    ).reshape(-1, 2)


def _synthetic_court(corners, *, shadow=False, occlusion=False, neighbor=False):
    """Render painted lines from court feet, independently of production code."""
    image = np.full((540, 960, 3), (83, 112, 88), dtype=np.uint8)

    def paint(quad):
        polygon = np.round(quad).astype(np.int32)
        cv2.fillConvexPoly(image, polygon, (155, 99, 48))
        segments = [
            ((0, 0), (20, 0)),
            ((20, 0), (20, 44)),
            ((20, 44), (0, 44)),
            ((0, 44), (0, 0)),
            ((0, 15), (20, 15)),
            ((0, 29), (20, 29)),
            ((10, 0), (10, 15)),
            ((10, 29), (10, 44)),
        ]
        for segment in segments:
            endpoints = np.round(_project(segment, quad)).astype(int)
            cv2.line(image, tuple(endpoints[0]), tuple(endpoints[1]),
                     (240, 240, 240), 4, cv2.LINE_AA)

    if neighbor:
        paint([(780, 145), (1010, 140), (1290, 480), (970, 495)])
    paint(corners)
    if shadow:
        # A straight dark shadow crosses several real lines and the surface.
        shade = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.fillConvexPoly(shade, np.int32([(0, 330), (960, 275),
                                         (960, 315), (0, 370)]), 255)
        image[shade > 0] = (image[shade > 0].astype(float) * 0.5).astype(np.uint8)
    if occlusion:
        cv2.rectangle(image, (650, 340), (710, 455), (35, 30, 25), -1)
        cv2.circle(image, (680, 325), 18, (90, 110, 145), -1)
    return image


def _assert_detected(image, expected, tolerance):
    corners, score = auto_detect_court_corners(image)
    corners = np.asarray(corners)
    assert corners.shape == (4, 2), "A court has exactly four ordered corners"
    assert np.isfinite(corners).all()
    assert 0.7 <= score <= 1.0, f"Court was not reliably matched: score={score}"
    errors = np.linalg.norm(corners - np.asarray(expected), axis=1)
    assert errors.max() <= tolerance, f"Corner errors in pixels: {errors.tolist()}"
    return corners


def test_real_footage_matches_painted_court():
    annotation = json.loads((FIXTURES / "real_court_960.json").read_text())
    image = cv2.imread(str(FIXTURES / "real_court_960.jpg"))
    assert image is not None
    corners = _assert_detected(image, annotation["corners_pixel"], tolerance=12)
    mapping = CourtHomography(corners, image_size=(960, 540))
    for landmark in annotation["landmarks"]:
        actual = mapping.court_to_pixel(*landmark["court_feet"])
        error = np.linalg.norm(np.asarray(actual) - landmark["pixel"])
        assert error <= 10, f"{landmark['name']} is {error:.1f}px off the marking"


@pytest.mark.parametrize("scale", [0.75, 1.5])
def test_real_footage_resolution_changes(scale):
    annotation = json.loads((FIXTURES / "real_court_960.json").read_text())
    image = cv2.imread(str(FIXTURES / "real_court_960.jpg"))
    resized = cv2.resize(image, None, fx=scale, fy=scale)
    _assert_detected(resized, np.asarray(annotation["corners_pixel"]) * scale,
                     tolerance=12 * scale)


@pytest.mark.parametrize("corners,options", [
    ([(390, 130), (610, 130), (890, 490), (100, 490)], {}),
    ([(260, 155), (480, 137), (795, 465), (75, 510)], {}),
    ([(380, 140), (600, 132), (860, 485), (100, 503)],
     {"shadow": True, "occlusion": True, "neighbor": True}),
])
def test_synthetic_perspective_and_distractors(corners, options):
    image = _synthetic_court(corners, **options)
    _assert_detected(image, corners, tolerance=12)


@pytest.mark.parametrize("color", [(0, 0, 0), (155, 99, 48), (220, 220, 220)])
def test_uniform_images_require_manual_review(color):
    image = np.full((540, 960, 3), color, dtype=np.uint8)
    corners, score = auto_detect_court_corners(image)
    assert np.asarray(corners).shape == (4, 2)
    assert np.isfinite(corners).all()
    assert 0.0 <= score < 0.7


def test_unrelated_lines_are_not_a_confirmed_court():
    image = np.zeros((540, 960, 3), dtype=np.uint8)
    for a, b in [((80, 430), (280, 80)), ((700, 60), (930, 480)),
                 ((20, 300), (950, 300))]:
        cv2.line(image, a, b, (240, 240, 240), 4)
    corners, score = auto_detect_court_corners(image)
    assert np.asarray(corners).shape == (4, 2)
    assert 0.0 <= score < 0.7


def test_outer_rectangle_without_court_markings_needs_review():
    image = np.full((540, 960, 3), (83, 112, 88), dtype=np.uint8)
    quad = np.int32([(390, 130), (610, 130), (890, 490), (100, 490)])
    cv2.fillConvexPoly(image, quad, (155, 99, 48))
    cv2.polylines(image, [quad], True, (240, 240, 240), 4, cv2.LINE_AA)
    corners, score = auto_detect_court_corners(image)
    assert np.asarray(corners).shape == (4, 2)
    assert 0.0 <= score < 0.7, "An outlined rectangle is insufficient court evidence"


def test_clean_frame_does_not_mix_camera_positions(monkeypatch, tmp_path):
    """A majority of zoomed samples must not move the initial court calibration."""
    video_path = tmp_path / "camera_zoom.mp4"
    video_path.write_bytes(b"mock video")
    corners = [(390, 130), (610, 130), (890, 490), (100, 490)]
    reference = _synthetic_court(corners)
    # Textured stationary background makes camera registration measurable.
    rng = np.random.default_rng(19)
    for x, y in rng.integers([0, 10], [960, 110], size=(120, 2)):
        cv2.circle(reference, (int(x), int(y)), 3, (180, 180, 180), -1)
    zoom = cv2.getRotationMatrix2D((480, 270), 0, 1.15)
    zoom[:, 2] += [18, 6]
    moved = cv2.warpAffine(reference, zoom, (960, 540))

    class Capture:
        index = 0

        def isOpened(self):
            return True

        def get(self, prop):
            return {cv2.CAP_PROP_FRAME_WIDTH: 960,
                    cv2.CAP_PROP_FRAME_HEIGHT: 540,
                    cv2.CAP_PROP_FRAME_COUNT: 300,
                    cv2.CAP_PROP_FPS: 30}.get(prop, 0)

        def set(self, prop, value):
            if prop == cv2.CAP_PROP_POS_FRAMES:
                self.index = int(value)
            return True

        def read(self):
            image = reference if self.index < 20 else moved
            self.index += 1
            return True, image.copy()

        def release(self):
            pass

    monkeypatch.setattr(cv2, "VideoCapture", lambda _: Capture())
    clean, width, height = synthesize_clean_court_frame(
        video_path, num_samples=7, sample_step=10
    )
    assert (width, height) == (960, 540)
    assert clean is not None
    difference = np.mean(np.abs(clean.astype(float) - reference))
    assert difference < 2, f"Camera movement smeared the court: mean error={difference}"
