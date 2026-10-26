"""Interactive OpenCV mouse-click calibration UI for selecting 4 court corners.

Allows users to click the 4 corners of the pickleball court on video footage
and saves the perspective mapping into a structured JSON file.
"""

import argparse
from pathlib import Path
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np

from src.config import (
    CALIBRATION_DIR,
    COURT_CORNERS_ORDER,
    RAW_VIDEOS_DIR,
)
from src.homography import CourtHomography

CORNER_PROMPTS = [
    "Point 1/4: Far-Left (Top-Left) Court Corner",
    "Point 2/4: Far-Right (Top-Right) Court Corner",
    "Point 3/4: Near-Right (Bottom-Right) Court Corner",
    "Point 4/4: Near-Left (Bottom-Left) Court Corner",
]


class CourtCalibrator:
    """Handles manual and programmatic court 4-corner calibration."""

    def __init__(self, frame: np.ndarray, window_name: str = "PickleTrack Court Calibration"):
        self.orig_frame = frame.copy()
        self.display_frame = frame.copy()
        self.window_name = window_name
        self.clicked_points: List[Tuple[float, float]] = []
        self.h, self.w = frame.shape[:2]

    def _mouse_callback(self, event, x, y, flags, param) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(self.clicked_points) < 4:
                self.clicked_points.append((float(x), float(y)))
                self._redraw()

    def _redraw(self) -> None:
        self.display_frame = self.orig_frame.copy()

        # Draw semi-transparent header overlay
        overlay = self.display_frame.copy()
        cv2.rectangle(overlay, (0, 0), (self.w, 70), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.75, self.display_frame, 0.25, 0, self.display_frame)

        # Draw instructions
        pt_count = len(self.clicked_points)
        if pt_count < 4:
            prompt_text = CORNER_PROMPTS[pt_count]
            color = (0, 220, 255)
        else:
            prompt_text = "All 4 corners selected! Press 'S' to Save or 'R' to Reset"
            color = (50, 255, 120)

        cv2.putText(
            self.display_frame,
            prompt_text,
            (20, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            color,
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            self.display_frame,
            "[Click]: Place Point | [R]: Reset | [S/Enter]: Save | [Q/Esc]: Quit",
            (20, 58),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (200, 200, 200),
            1,
            cv2.LINE_AA,
        )

        # Draw points and connecting lines
        for i, (px, py) in enumerate(self.clicked_points):
            pt_int = (int(px), int(py))
            cv2.circle(self.display_frame, pt_int, 7, (0, 240, 255), -1)
            cv2.circle(self.display_frame, pt_int, 9, (0, 0, 0), 2)
            cv2.putText(
                self.display_frame,
                f"P{i+1}: {COURT_CORNERS_ORDER[i]}",
                (int(px) + 12, int(py) - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

        if pt_count >= 2:
            pts_poly = np.array(self.clicked_points, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(
                self.display_frame,
                [pts_poly],
                isClosed=(pt_count == 4),
                color=(0, 255, 120),
                thickness=2,
            )

        cv2.imshow(self.window_name, self.display_frame)

    def run_interactive(self) -> Optional[List[Tuple[float, float]]]:
        """Runs the interactive calibration loop until user saves or cancels."""
        try:
            cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)
            cv2.setMouseCallback(self.window_name, self._mouse_callback)
            self._redraw()

            while True:
                key = cv2.waitKey(20) & 0xFF
                if key in (27, ord("q"), ord("Q")):
                    # User quit
                    cv2.destroyWindow(self.window_name)
                    return None
                elif key in (ord("r"), ord("R")):
                    self.clicked_points.clear()
                    self._redraw()
                elif key in (13, ord("s"), ord("S")):  # Enter or S
                    if len(self.clicked_points) == 4:
                        cv2.destroyWindow(self.window_name)
                        return self.clicked_points
        except cv2.error:
            # OpenCV GUI unavailable in headless mode
            return None


DEFAULT_CALIBRATION_FILE = CALIBRATION_DIR / "default_court_calibration.json"
DETECTOR_VERSION = "court-template-v1"


def video_signature(video_path: Union[str, Path]) -> dict:
    """Invalidate cached calibration when a different file replaces the video."""
    path = Path(video_path)
    stat = path.stat()
    return {"name": path.name, "size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def synthesize_clean_court_frame(
    video_path: Union[str, Path],
    num_samples: int = 15,
    sample_step: int = 12,
) -> Tuple[Optional[np.ndarray], int, int]:
    """Extracts a temporal median frame across several seconds to remove moving players

    and reveal clean, unobstructed court lines from a stationary mounted camera.
    """
    vpath = Path(video_path)
    if not vpath.exists():
        return None, 1280, 720

    cap = cv2.VideoCapture(str(vpath))
    if not cap.isOpened():
        return None, 1280, 720

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 100)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)
    frames = []
    reference_points = reference_descriptors = reference_gray = None
    feature_detector = cv2.ORB_create(nfeatures=900)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    # Spread samples over a bounded opening window. Widely spaced frames remove
    # waiting players more reliably and avoid decoding the complete match.
    end_frame = min(total_frames - 1, num_samples * sample_step)
    indices = np.unique(np.linspace(0, max(0, end_frame), num_samples).astype(int))
    for target_f in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_f)
        ret, frame = cap.read()
        if ret and frame is not None:
            h, w = frame.shape[:2]
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (640, round(640*h/w)))
            if reference_gray is None:
                reference_gray = gray
                reference_points, reference_descriptors = feature_detector.detectAndCompute(gray, None)
                frames.append(frame)
                continue
            points, descriptors = feature_detector.detectAndCompute(gray, None)
            if reference_descriptors is None or descriptors is None:
                continue
            pairs = matcher.knnMatch(reference_descriptors, descriptors, k=2)
            matches = [a for pair in pairs if len(pair) == 2 for a, b in [pair]
                       if a.distance < .72*b.distance]
            if len(matches) < 15:
                continue
            src = np.float32([reference_points[m.queryIdx].pt for m in matches])
            dst = np.float32([points[m.trainIdx].pt for m in matches])
            transform, inliers = cv2.findHomography(src, dst, cv2.RANSAC, 2.)
            if transform is None or inliers.sum() < 12:
                continue
            gh, gw = gray.shape
            anchors = np.float32([[[gw*.2, gh*.2]], [[gw*.8, gh*.2]],
                                  [[gw*.8, gh*.8]], [[gw*.2, gh*.8]]])
            shifted = cv2.perspectiveTransform(anchors, transform)
            max_shift = float(np.max(np.linalg.norm(shifted-anchors, axis=2)))
            if max_shift <= 4.5:
                scale_x, scale_y = w / gw, h / gh
                S = np.diag([scale_x, scale_y, 1.0])
                S_inv = np.diag([1.0 / scale_x, 1.0 / scale_y, 1.0])
                full_transform = S @ transform @ S_inv
                ret_inv, inv_T = cv2.invert(full_transform)
                if ret_inv:
                    frames.append(cv2.warpPerspective(frame, inv_T, (w, h)))
                else:
                    frames.append(frame)

    cap.release()

    if not frames:
        return None, w, h

    if len(frames) == 1:
        return frames[0], w, h

    # Compute pixel-wise median to erase players & moving ball
    stack = np.stack(frames, axis=0)
    median_frame = np.median(stack, axis=0).astype(np.uint8)
    return median_frame, w, h


def auto_detect_court_corners(frame: np.ndarray) -> Tuple[List[Tuple[float, float]], float]:
    """Detect a full court locally; score measures painted-line alignment."""
    from src.court_detection import detect_court
    return detect_court(frame)


def get_or_create_court_calibration(
    video_path: Union[str, Path],
    force_auto_detect: bool = False,
) -> Tuple[CourtHomography, str]:
    """Resolve calibration for the opening camera view, with explicit provenance.

    Confirmed calibration belongs to this video, not to an unrelated global court.
    Legacy/unverified files are re-evaluated once by the current detector. A failed
    re-detection leaves any previous saved calibration intact for manual review.
    """
    vpath = Path(video_path)
    signature = video_signature(vpath)
    calib_file = CALIBRATION_DIR / f"{vpath.stem}_calibration.json"
    previous = None
    if calib_file.exists():
        try:
            previous = CourtHomography.load_from_file(calib_file)
        except (ValueError, KeyError, TypeError, cv2.error):
            previous = None
    if previous is not None:
        meta = previous.metadata
        same_video = meta.get("video_signature") == signature
        current = meta.get("detector_version") == DETECTOR_VERSION
        score = meta.get("confidence")
        valid_score = (isinstance(score, (int, float)) and not isinstance(score, bool)
                       and np.isfinite(score) and .7 <= score <= 1.)
        same_view = meta.get("frame_index", 0) == 0
        if same_video and same_view and not force_auto_detect:
            if meta.get("confirmed") is True:
                return previous, "saved"
            if current and meta.get("method") == "auto_detected" and valid_score:
                return previous, "auto_detected"

    frame, w, h = synthesize_clean_court_frame(vpath)
    if frame is None:
        raise ValueError(f"Could not read video for court calibration: {vpath.name}")
    corners, confidence = auto_detect_court_corners(frame)
    metadata = {"method": "auto_detected" if confidence >= .7 else "needs_review",
                "confidence": float(confidence), "confirmed": False,
                "detector_version": DETECTOR_VERSION, "video_signature": signature,
                "frame_index": 0}
    if (confidence < .7 and previous is not None and previous.image_size == (w, h)
            and previous.metadata.get("video_signature") == signature
            and previous.metadata.get("frame_index", 0) == 0):
        # Keep a previous usable shape as correction handles, without claiming
        # that old coordinates are verified or overwriting their saved file.
        corners = previous.pixel_corners.tolist()
    homography = CourtHomography(pixel_corners=corners, image_size=(w, h), metadata=metadata)
    if confidence >= .7:
        homography.save_to_file(calib_file)
        return homography, "auto_detected"
    return homography, "needs_review"


def calibrate_video(
    video_path: Union[str, Path],
    output_json_path: Optional[Union[str, Path]] = None,
    frame_offset: int = 0,
    preset_points: Optional[List[Tuple[float, float]]] = None,
) -> CourtHomography:
    """Calibrates court corners from a video file.

    If preset_points is provided, creates calibration directly without GUI.
    Otherwise opens interactive UI on the specified frame.
    """
    vpath = Path(video_path)
    if not vpath.exists():
        raise FileNotFoundError(f"Video file not found at: {vpath}")

    cap = cv2.VideoCapture(str(vpath))
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {vpath}")

    if frame_offset > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_offset)

    ret, frame = cap.read()
    cap.release()

    if not ret or frame is None:
        raise ValueError(f"Failed to read frame from video: {vpath}")

    h, w = frame.shape[:2]

    if preset_points is not None:
        points = preset_points
    else:
        calibrator = CourtCalibrator(frame)
        points = calibrator.run_interactive()
        if points is None:
            raise RuntimeError("Interactive calibration was canceled or GUI not supported.")

    homography = CourtHomography(
        pixel_corners=points,
        image_size=(w, h),
        metadata={"method": "manual", "confirmed": True, "confidence": None,
                  "video_signature": video_signature(vpath), "frame_index": frame_offset},
    )

    if output_json_path is None:
        output_json_path = CALIBRATION_DIR / f"{vpath.stem}_calibration.json"

    homography.save_to_file(output_json_path)
    return homography


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Interactive Pickleball Court Calibrator")
    parser.add_argument("--video", type=str, required=True, help="Path to match clip (.mp4)")
    parser.add_argument("--output", type=str, default=None, help="Output calibration JSON path")
    parser.add_argument("--frame", type=int, default=0, help="Frame index to use for calibration")
    args = parser.parse_args()

    calib = calibrate_video(args.video, args.output, frame_offset=args.frame)
    print("Calibration saved successfully!")

