"""Calibration provenance and fallback regressions; all writes use temporary files."""

import json

import numpy as np
import pytest

from src import calibration
from src.homography import CourtHomography


CORNERS = [(160, 90), (480, 90), (580, 330), (60, 330)]
UPDATED_CORNERS = [(165, 95), (475, 95), (575, 325), (65, 325)]


@pytest.fixture
def clip(tmp_path, monkeypatch):
    video = tmp_path / "game.mp4"
    video.write_bytes(b"test source video")
    cache_dir = tmp_path / "calibration"
    cache_dir.mkdir()
    monkeypatch.setattr(calibration, "CALIBRATION_DIR", cache_dir)
    monkeypatch.setattr(calibration, "DEFAULT_CALIBRATION_FILE", cache_dir / "default.json")
    image = np.zeros((360, 640, 3), dtype=np.uint8)
    monkeypatch.setattr(calibration, "synthesize_clean_court_frame",
                        lambda *args, **kwargs: (image.copy(), 640, 360))
    monkeypatch.setattr(calibration, "auto_detect_court_corners",
                        lambda _: (UPDATED_CORNERS, 0.91))
    return video, cache_dir / "game_calibration.json"


def _metadata(video, *, confirmed=False, method="auto_detected"):
    return {
        "method": method,
        "confirmed": confirmed,
        "confidence": None if confirmed else 0.91,
        "detector_version": calibration.DETECTOR_VERSION,
        "video_signature": calibration.video_signature(video),
        "frame_index": 0,
    }


def _save(path, metadata=None):
    CourtHomography(CORNERS, image_size=(640, 360), metadata=metadata).save_to_file(path)


def test_legacy_cache_is_redetected_without_claiming_confirmation(clip):
    video, path = clip
    _save(path)
    homography, source = calibration.get_or_create_court_calibration(video)
    assert source == "auto_detected"
    np.testing.assert_allclose(homography.pixel_corners, UPDATED_CORNERS)
    assert homography.metadata["confirmed"] is False
    assert homography.metadata["video_signature"] == calibration.video_signature(video)
    saved = json.loads(path.read_text())
    assert saved["calibration"]["confirmed"] is False
    assert saved["image_size"] == [640, 360]


def test_confirmed_matching_cache_is_reused(clip, monkeypatch):
    video, path = clip
    _save(path, _metadata(video, confirmed=True, method="manual"))

    def unexpected_detection(_):
        pytest.fail("An unchanged confirmed camera view should not be overwritten")

    monkeypatch.setattr(calibration, "auto_detect_court_corners", unexpected_detection)
    homography, source = calibration.get_or_create_court_calibration(video)
    assert source == "saved"
    assert homography.metadata["confirmed"] is True
    np.testing.assert_allclose(homography.pixel_corners, CORNERS)


def test_cached_automatic_proposal_stays_unconfirmed(clip, monkeypatch):
    video, path = clip
    _save(path, _metadata(video))
    monkeypatch.setattr(calibration, "auto_detect_court_corners",
                        lambda _: pytest.fail("Current detector cache should be reusable"))
    homography, source = calibration.get_or_create_court_calibration(video)
    assert source == "auto_detected"
    assert homography.metadata["confirmed"] is False


def test_replaced_video_invalidates_previous_confirmation(clip):
    video, path = clip
    _save(path, _metadata(video, confirmed=True, method="manual"))
    video.write_bytes(b"a different video replaced the original")
    homography, source = calibration.get_or_create_court_calibration(video)
    assert source == "auto_detected"
    assert homography.metadata["confirmed"] is False
    np.testing.assert_allclose(homography.pixel_corners, UPDATED_CORNERS)


def test_failed_redetection_preserves_saved_calibration(clip, monkeypatch):
    video, path = clip
    _save(path, _metadata(video, confirmed=True, method="manual"))
    before = path.read_bytes()
    monkeypatch.setattr(calibration, "auto_detect_court_corners",
                        lambda _: (UPDATED_CORNERS, 0.25))
    proposal, source = calibration.get_or_create_court_calibration(video, force_auto_detect=True)
    assert source == "needs_review"
    assert proposal.metadata.get("confirmed", False) is False
    assert path.read_bytes() == before


def test_low_confidence_does_not_apply_unrelated_global_default(clip, monkeypatch):
    video, path = clip
    # Reproduce the historical missing image_size in a global camera profile.
    unrelated = CourtHomography(CORNERS).to_dict()
    calibration.DEFAULT_CALIBRATION_FILE.write_text(json.dumps(unrelated))
    monkeypatch.setattr(calibration, "auto_detect_court_corners",
                        lambda _: (UPDATED_CORNERS, 0.0))
    proposal, source = calibration.get_or_create_court_calibration(video)
    assert source == "needs_review"
    assert proposal.image_size == (640, 360)
    assert proposal.metadata.get("confirmed", False) is False
    assert not path.exists(), "A guessed court must not become a saved calibration"


def test_corrupt_cache_can_be_replaced_by_valid_detection(clip):
    video, path = clip
    path.write_text("{broken JSON")
    homography, source = calibration.get_or_create_court_calibration(video)
    assert source == "auto_detected"
    np.testing.assert_allclose(homography.pixel_corners, UPDATED_CORNERS)
    assert json.loads(path.read_text())["calibration"]["confirmed"] is False


@pytest.mark.parametrize("confidence", [None, "bad", float("nan")])
def test_invalid_cached_confidence_is_redetected(clip, confidence):
    video, path = clip
    metadata = _metadata(video)
    metadata["confidence"] = confidence
    _save(path, metadata)
    homography, source = calibration.get_or_create_court_calibration(video)
    assert source == "auto_detected"
    assert homography.metadata["confidence"] == pytest.approx(0.91)
    assert homography.metadata["confirmed"] is False
    np.testing.assert_allclose(homography.pixel_corners, UPDATED_CORNERS)
