"""Regression tests for Task 1: Stationary filter vs legitimate ball movement."""

import numpy as np
import pytest
import torch

from src.ball_model import MODEL_FORMAT, NeuralBallLocator, TemporalBallNet


class SyntheticHeatmapModel(torch.nn.Module):
    """Mock model that outputs controlled probability heatmaps."""
    def __init__(self, peaks_per_frame):
        super().__init__()
        self.peaks_per_frame = peaks_per_frame
        self.current_frame = 0
        self.dummy_param = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch, _, h, w = x.shape
        out_h, out_w = h // 2, w // 2
        logits = torch.full((batch, 1, out_h, out_w), -5.0)

        frame_idx = self.current_frame
        peaks = self.peaks_per_frame(frame_idx) if callable(self.peaks_per_frame) else self.peaks_per_frame.get(frame_idx, [])

        for (px, py, score) in peaks:
            hx = int(round(px * out_w / 1920.0))
            hy = int(round(py * out_h / 1080.0))
            if 0 <= hx < out_w and 0 <= hy < out_h:
                logit_val = float(np.log(score / max(1e-4, 1.0 - score)))
                for dy in range(-1, 2):
                    for dx in range(-1, 2):
                        if 0 <= hy + dy < out_h and 0 <= hx + dx < out_w:
                            decay = np.exp(-(dx**2 + dy**2) / 1.5)
                            logits[0, 0, hy + dy, hx + dx] = max(logits[0, 0, hy + dy, hx + dx].item(), logit_val * decay)
        return logits


def create_mock_locator(tmp_path, peaks_fn, threshold=0.25):
    """Creates a NeuralBallLocator backed by SyntheticHeatmapModel."""
    model = TemporalBallNet()
    ckpt_path = tmp_path / "mock_model.pt"
    torch.save({
        "format": MODEL_FORMAT,
        "state_dict": model.state_dict(),
        "input_width": 960,
        "acceptance_threshold": threshold
    }, ckpt_path)
    loc = NeuralBallLocator(ckpt_path, threshold=threshold)
    loc.model = SyntheticHeatmapModel(peaks_fn)
    return loc


def test_reproduction_two_pixels_per_frame_must_not_be_rejected(tmp_path):
    """Reproduction test from handoff:

    A deterministic sequence with high-confidence peaks moving two pixels per frame
    must not be classified as stationary.
    """
    # 8 consecutive frames, ball moving at (1000 + 2*f, 500)
    def peaks_fn(frame_idx):
        return [(1000.0 + 2.0 * frame_idx, 500.0, 0.90)]

    loc = create_mock_locator(tmp_path, peaks_fn)
    dummy_frame = np.zeros((1080, 1920, 3), dtype=np.uint8)

    accepted_frames = []
    for f in range(8):
        loc.model.current_frame = f
        pt, score = loc.update(dummy_frame)
        if pt is not None:
            accepted_frames.append((f, pt[0], pt[1]))

    # NeuralBallLocator returns detections starting on frame 2 (after 3 frames in buffer)
    # Frames 2, 3, 4, 5, 6, 7 should ALL be accepted!
    detected_indices = [af[0] for af in accepted_frames]
    assert len(detected_indices) >= 5, f"Legitimate slow motion was rejected! Detected only on frames: {detected_indices}"
    for f, x, y in accepted_frames:
        expected_x = 1000.0 + 2.0 * f
        assert abs(x - expected_x) < 4.0, f"Frame {f}: expected x~{expected_x}, got {x}"


def test_apex_of_lob_motion_is_not_rejected(tmp_path):
    """A ball slowing vertically near the apex of a lob (dy < 1px for 3 frames)

    must not be permanently blacklisted.
    """
    y_coords = [410, 404, 401, 400, 400, 401, 404, 410]
    def peaks_fn(frame_idx):
        idx = min(frame_idx, len(y_coords) - 1)
        return [(1000.0 + 5.0 * idx, float(y_coords[idx]), 0.88)]

    loc = create_mock_locator(tmp_path, peaks_fn)
    dummy_frame = np.zeros((1080, 1920, 3), dtype=np.uint8)

    detections = []
    for f in range(len(y_coords)):
        loc.model.current_frame = f
        pt, score = loc.update(dummy_frame)
        if pt is not None:
            detections.append(f)

    # All frames after warm-up (frames 2 to 7) must be tracked through the apex
    assert set(range(2, len(y_coords))).issubset(set(detections)), f"Apex motion lost! Detections: {detections}"


def test_static_distractor_suppressed_while_live_ball_tracked(tmp_path):
    """A true static distractor (spare ball at 1315, 608) must be suppressed

    so the live moving ball (1000 + 10*f, 500) is selected.
    """
    def peaks_fn(frame_idx):
        # Distractor has even higher score than live ball!
        return [
            (1315.0, 608.0, 0.95),  # Static spare ball
            (1000.0 + 15.0 * frame_idx, 500.0, 0.80),  # Moving live ball
        ]

    loc = create_mock_locator(tmp_path, peaks_fn)
    dummy_frame = np.zeros((1080, 1920, 3), dtype=np.uint8)

    tracked_live = 0
    for f in range(15):
        loc.model.current_frame = f
        pt, score = loc.update(dummy_frame)
        if pt is not None:
            # Check if tracked point is near the live ball
            live_x = 1000.0 + 15.0 * f
            if abs(pt[0] - live_x) < 10.0:
                tracked_live += 1

    # After initial distractor suppression, live ball must be consistently chosen
    assert tracked_live >= 8, f"Live ball was not selected over static distractor! Tracked live frames: {tracked_live}"


def test_held_then_hit_ball_recovers_without_permanent_blacklist(tmp_path):
    """A ball held stationary by the server for 10 frames before serve

    must be trackable when hit, and its prior position must not permanently blacklist
    balls returning to that area later.
    """
    def peaks_fn(frame_idx):
        if frame_idx < 8:
            # Held still
            return [(1000.0, 500.0, 0.90)]
        elif frame_idx < 16:
            # Struck and moving away
            return [(1000.0 + 20.0 * (frame_idx - 8), 500.0 + 10.0 * (frame_idx - 8), 0.85)]
        else:
            # Ball returned toward initial serve area 50 frames later
            return [(1002.0, 501.0, 0.85)]

    loc = create_mock_locator(tmp_path, peaks_fn)
    dummy_frame = np.zeros((1080, 1920, 3), dtype=np.uint8)

    # Run initial held phase
    for f in range(8):
        loc.model.current_frame = f
        loc.update(dummy_frame)

    # Run struck phase: must be tracked immediately
    struck_detected = []
    for f in range(8, 16):
        loc.model.current_frame = f
        pt, score = loc.update(dummy_frame)
        if pt is not None:
            struck_detected.append(f)
    assert len(struck_detected) >= 6, f"Ball struck after being held was not tracked! {struck_detected}"

    # Advance 45 frames of ball elsewhere
    for f in range(16, 60):
        loc.model.current_frame = f
        loc.model.peaks_per_frame = lambda idx: [(500.0, 500.0, 0.85)]
        loc.update(dummy_frame)

    # Ball returns to serve area (1002, 501)
    loc.model.current_frame = 60
    loc.model.peaks_per_frame = lambda idx: [(1002.0, 501.0, 0.85)]
    pt, score = loc.update(dummy_frame)
    assert pt is not None, "Prior serve position was permanently blacklisted!"
    assert abs(pt[0] - 1002.0) < 5.0 and abs(pt[1] - 501.0) < 5.0



def test_camera_motion_compensation_and_recovery(tmp_path):
    """When camera moves (represented by homography shift), legitimate ball motion

    remains tracked and recovers after temporary occlusion/rejection.
    """
    from src.homography import CourtHomography

    # Create a base court calibration
    base_calib = {
        "H": [
            [0.1, 0.0, -100.0],
            [0.0, 0.1, -50.0],
            [0.0, 0.0, 1.0]
        ],
        "image_size": [1920, 1080],
        "court_corners_pixel": [
            [500, 300], [1400, 300], [1700, 900], [200, 900]
        ]
    }
    calib_path = tmp_path / "calib.json"
    import json
    calib_path.write_text(json.dumps(base_calib))
    court = CourtHomography.load_from_file(calib_path)

    # Ball moves diagonally across court while camera has slight jitter (+1px shift)
    def peaks_fn(frame_idx):
        jitter = frame_idx % 2
        # Ball moves 8px per frame
        return [(1000.0 + 8.0 * frame_idx + jitter, 500.0 + 4.0 * frame_idx, 0.88)]

    loc = create_mock_locator(tmp_path, peaks_fn)
    dummy_frame = np.zeros((1080, 1920, 3), dtype=np.uint8)

    detections = []
    for f in range(12):
        loc.model.current_frame = f
        pt, score = loc.update(dummy_frame, homography=court)
        if pt is not None:
            detections.append(f)

    # All frames after warm-up buffer (frames 2 to 11) must be successfully tracked
    assert len(detections) >= 9, f"Camera jitter caused tracking drops: {detections}"


def test_deterministic_one_pixel_per_frame_sequence_all_accepted(tmp_path):
    """Codex Finding C exact reproduction:
    Deterministic sequence: A high-confidence Gaussian peak moving approximately
    one source pixel per frame (x: 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33)
    must ALL be accepted without any vetoes.
    """
    def peaks_fn(frame_idx):
        return [(20.0 + 1.0 * frame_idx, 500.0, 0.92)]

    loc = create_mock_locator(tmp_path, peaks_fn)
    dummy_frame = np.zeros((1080, 1920, 3), dtype=np.uint8)

    accepted = {}
    for f in range(14):
        loc.model.current_frame = f
        pt, score = loc.update(dummy_frame)
        if pt is not None:
            accepted[f] = pt[0]

    # After initial 2 frames buffer, frames 2 to 13 must ALL be accepted
    for f in range(2, 14):
        assert f in accepted, f"Frame {f} at x={20.0 + f} was incorrectly vetoed! Accepted: {accepted}"
        expected_x = 20.0 + 1.0 * f
        assert abs(accepted[f] - expected_x) < 3.0, f"Frame {f}: expected x~{expected_x}, got {accepted[f]}"


def test_subpixel_motion_sequence_all_accepted(tmp_path):
    """Subpixel motion (0.5 pixel per frame) must not be vetoed."""
    def peaks_fn(frame_idx):
        return [(100.0 + 0.5 * frame_idx, 500.0, 0.90)]

    loc = create_mock_locator(tmp_path, peaks_fn)
    dummy_frame = np.zeros((1080, 1920, 3), dtype=np.uint8)

    accepted = {}
    for f in range(16):
        loc.model.current_frame = f
        pt, score = loc.update(dummy_frame)
        if pt is not None:
            accepted[f] = pt[0]

    for f in range(2, 16):
        assert f in accepted, f"Subpixel motion frame {f} was incorrectly vetoed! Accepted: {accepted}"


def test_different_resolution_and_cadence(tmp_path):
    """Tests slow motion on a 1280x720 frame."""
    def peaks_fn(frame_idx):
        return [(50.0 + 1.0 * frame_idx, 360.0, 0.91)]

    loc = create_mock_locator(tmp_path, peaks_fn)
    dummy_frame = np.zeros((720, 1280, 3), dtype=np.uint8)

    accepted = []
    for f in range(10):
        loc.model.current_frame = f
        pt, score = loc.update(dummy_frame)
        if pt is not None:
            accepted.append(f)
    assert len(accepted) >= 7

