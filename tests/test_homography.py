"""Unit tests for Homography coordinate mapping between camera pixels and court feet."""

import json
from pathlib import Path
import numpy as np
import pytest

from src.config import (
    COURT_CORNERS_FEET,
    COURT_LENGTH_FT,
    COURT_WIDTH_FT,
    KITCHEN_FAR_Y,
    KITCHEN_NEAR_Y,
    NET_Y,
)
from src.homography import CourtHomography


@pytest.fixture
def synthetic_homography():
    """Provides a calibrated CourtHomography fixture with known perspective quadrangle."""
    # Synthetic camera view (1920x1080) looking down from behind baseline
    # [top_left, top_right, bottom_right, bottom_left]
    pixel_corners = [
        (460.0, 240.0),   # Far-Left (0.0, 44.0)
        (1460.0, 240.0),  # Far-Right (20.0, 44.0)
        (1820.0, 980.0),  # Near-Right (20.0, 0.0)
        (100.0, 980.0),   # Near-Left (0.0, 0.0)
    ]
    return CourtHomography(
        pixel_corners=pixel_corners,
        court_corners_ft=COURT_CORNERS_FEET,
        image_size=(1920, 1080),
    )


def test_four_corners_exact_mapping(synthetic_homography):
    """Test that all 4 pixel corners map to the exact designated court coordinates in feet."""
    corners = synthetic_homography.pixel_corners
    expected_feet = [
        (0.0, 44.0),   # Top-Left (Far-Left)
        (20.0, 44.0),  # Top-Right (Far-Right)
        (20.0, 0.0),   # Bottom-Right (Near-Right)
        (0.0, 0.0),    # Bottom-Left (Near-Left)
    ]

    for (u, v), (exp_x, exp_y) in zip(corners, expected_feet):
        court_x, court_y = synthetic_homography.pixel_to_court(u, v)
        assert court_x == pytest.approx(exp_x, abs=1e-2), f"X mismatch for pixel ({u}, {v})"
        assert court_y == pytest.approx(exp_y, abs=1e-2), f"Y mismatch for pixel ({u}, {v})"


def test_inverse_mapping_roundtrip(synthetic_homography):
    """Test that roundtrip conversion (pixel -> court -> pixel) produces original values."""
    test_pixels = [
        (960.0, 540.0),
        (500.0, 300.0),
        (1200.0, 800.0),
        (800.0, 600.0),
    ]

    for u, v in test_pixels:
        court_x, court_y = synthetic_homography.pixel_to_court(u, v)
        u_rec, v_rec = synthetic_homography.court_to_pixel(court_x, court_y)
        assert u_rec == pytest.approx(u, abs=1e-2)
        assert v_rec == pytest.approx(v, abs=1e-2)


def test_court_center_landmark(synthetic_homography):
    """Test that court center (10.0 ft, 22.0 ft) projects within the perspective bounds."""
    # Net center: X = 10.0 ft, Y = 22.0 ft
    center_u, center_v = synthetic_homography.court_to_pixel(10.0, 22.0)

    # In our perspective, center U should be near mid-width (approx 960)
    assert 900 <= center_u <= 1020
    # In perspective with foreshortening, Y=22 is visually higher than screen midpoint
    assert 300 <= center_v <= 650

    # Project back and verify
    rec_x, rec_y = synthetic_homography.pixel_to_court(center_u, center_v)
    assert rec_x == pytest.approx(10.0, abs=1e-2)
    assert rec_y == pytest.approx(22.0, abs=1e-2)


def test_batch_transform_points(synthetic_homography):
    """Test batch transformation of multiple pixel coordinates."""
    pts = np.array([
        [460.0, 240.0],
        [1460.0, 240.0],
        [1820.0, 980.0],
        [100.0, 980.0],
    ], dtype=np.float32)

    transformed = synthetic_homography.transform_points(pts)
    assert transformed.shape == (4, 2)
    assert transformed[0][0] == pytest.approx(0.0, abs=1e-2)
    assert transformed[0][1] == pytest.approx(44.0, abs=1e-2)
    assert transformed[1][0] == pytest.approx(20.0, abs=1e-2)
    assert transformed[1][1] == pytest.approx(44.0, abs=1e-2)


def test_court_zone_queries(synthetic_homography):
    """Test classification of court coordinates into zones (baseline, transition, kitchen)."""
    assert synthetic_homography.get_court_zone(10.0, 4.0) == "near_baseline"
    assert synthetic_homography.get_court_zone(10.0, 12.0) == "near_transition"
    assert synthetic_homography.get_court_zone(10.0, 18.0) == "near_kitchen"
    assert synthetic_homography.get_court_zone(10.0, 25.0) == "far_kitchen"
    assert synthetic_homography.get_court_zone(10.0, 32.0) == "far_transition"
    assert synthetic_homography.get_court_zone(10.0, 40.0) == "far_baseline"
    assert synthetic_homography.get_court_zone(-5.0, 20.0) == "out_of_bounds"
    assert synthetic_homography.get_court_zone(25.0, 20.0) == "out_of_bounds"


def test_kitchen_boundary_queries(synthetic_homography):
    """Test exact Non-Volley Zone (Kitchen) boundary inclusion."""
    # Near kitchen: 15 <= y <= 22
    assert synthetic_homography.is_inside_kitchen(10.0, 15.0) is True
    assert synthetic_homography.is_inside_kitchen(10.0, 20.0) is True
    assert synthetic_homography.is_inside_kitchen(10.0, 22.0) is True

    # Far kitchen: 22 <= y <= 29
    assert synthetic_homography.is_inside_kitchen(10.0, 28.5) is True
    assert synthetic_homography.is_inside_kitchen(10.0, 29.0) is True

    # Outside kitchen
    assert synthetic_homography.is_inside_kitchen(10.0, 14.8) is False
    assert synthetic_homography.is_inside_kitchen(10.0, 29.5) is False
    assert synthetic_homography.is_inside_kitchen(-1.0, 20.0) is False


def test_serialization_and_deserialization(synthetic_homography, tmp_path):
    """Test saving calibration to JSON and reloading it."""
    save_file = tmp_path / "test_calibration.json"
    synthetic_homography.save_to_file(save_file)

    assert save_file.exists()
    loaded_h = CourtHomography.load_from_file(save_file)

    # Check that loaded homography produces the same court mappings
    test_u, test_v = 960.0, 540.0
    orig_x, orig_y = synthetic_homography.pixel_to_court(test_u, test_v)
    load_x, load_y = loaded_h.pixel_to_court(test_u, test_v)

    assert load_x == pytest.approx(orig_x, abs=1e-5)
    assert load_y == pytest.approx(orig_y, abs=1e-5)

