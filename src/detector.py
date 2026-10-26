"""Detection engine for PickleTrack-Local.

Provides a unified interface with two toggleable modes:
  Mode A: Pure Classical CV (MOG2 background subtraction + HSV color masking + contour filtering)
  Mode B: Local Open-Source Weights (Ultralytics YOLOv8n + Supervision ByteTrack)

Includes the heuristic shoe-filter preventing neon footwear from being misclassified as the ball.
"""

from typing import List, Optional, Tuple

import cv2
import numpy as np

from src.config import (
    BALL_ASPECT_RATIO_MAX,
    BALL_ASPECT_RATIO_MIN,
    BALL_MAX_AREA,
    BALL_MIN_AREA,
    BALL_MIN_CIRCULARITY,
    COCO_PERSON_CLASS_ID,
    COCO_SPORTS_BALL_CLASS_ID,
    DEFAULT_HSV_LOWER,
    DEFAULT_HSV_UPPER,
    MOG2_DETECT_SHADOWS,
    MOG2_HISTORY,
    MOG2_VAR_THRESHOLD,
    PLAYER_MAX_AREA,
    PLAYER_MIN_AREA,
    SHOE_FILTER_BOTTOM_RATIO,
    YOLO_BALL_CONF_THRESHOLD,
    YOLO_MODEL_NAME,
    YOLO_PERSON_CONF_THRESHOLD,
)
from src.tracker import PlayerDetection


class ClassicalPlayerTracker:
    """Lightweight Euclidean centroid tracker for players in Mode A."""

    def __init__(self, max_distance: float = 120.0):
        self.next_id = 1
        self.players = {}  # {id: (cx, cy, bbox, lost_count)}
        self.max_distance = max_distance

    def update(self, bboxes: List[Tuple[float, float, float, float]]) -> List[PlayerDetection]:
        current_centroids = []
        for bbox in bboxes:
            x1, y1, x2, y2 = bbox
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            current_centroids.append((cx, cy, bbox))

        if not self.players:
            for cx, cy, bbox in current_centroids:
                self.players[self.next_id] = (cx, cy, bbox, 0)
                self.next_id += 1
        else:
            unmatched_detections = list(range(len(current_centroids)))
            updated_players = {}

            for pid, (px, py, p_bbox, lost) in self.players.items():
                best_dist = float("inf")
                best_idx = None
                for idx in unmatched_detections:
                    cx, cy, _ = current_centroids[idx]
                    d = np.hypot(cx - px, cy - py)
                    if d < best_dist and d < self.max_distance:
                        best_dist = d
                        best_idx = idx

                if best_idx is not None:
                    cx, cy, bbox = current_centroids[best_idx]
                    updated_players[pid] = (cx, cy, bbox, 0)
                    unmatched_detections.remove(best_idx)
                else:
                    if lost < 10:
                        updated_players[pid] = (px, py, p_bbox, lost + 1)

            # Assign new IDs for remaining unmatched detections (up to 4 players)
            for idx in unmatched_detections:
                if len(updated_players) < 4:
                    cx, cy, bbox = current_centroids[idx]
                    updated_players[self.next_id] = (cx, cy, bbox, 0)
                    self.next_id += 1

            self.players = updated_players

        results = []
        for pid, (cx, cy, bbox, lost) in self.players.items():
            if lost == 0:
                x1, y1, x2, y2 = bbox
                feet_px = ((x1 + x2) / 2.0, float(y2))
                results.append(
                    PlayerDetection(
                        tracker_id=pid,
                        bbox_xyxy=bbox,
                        confidence=0.9,
                        feet_pixel=feet_px,
                    )
                )
        return results


class PickleballDetector:
    """Unified detector supporting Mode A (Classical CV) and Mode B (YOLOv8 + ByteTrack)."""

    def __init__(
        self,
        mode: str = "A",
        hsv_lower: Tuple[int, int, int] = DEFAULT_HSV_LOWER,
        hsv_upper: Tuple[int, int, int] = DEFAULT_HSV_UPPER,
        yolo_model_path: str = YOLO_MODEL_NAME,
        ball_hue_hint: Optional[float] = None,
        ball_model_path: Optional[str] = None,
    ):
        """Initialize detector.

        Args:
            mode: "A" for pure Classical CV (HSV + MOG2), "B" for YOLOv8 + ByteTrack
            hsv_lower: Lower HSV bound tuple (H, S, V)
            hsv_upper: Upper HSV bound tuple (H, S, V)
            yolo_model_path: Local path or name of YOLOv8 model weights
        """
        self.mode = mode.upper()
        self.hsv_lower = np.array(hsv_lower, dtype=np.uint8)
        self.hsv_upper = np.array(hsv_upper, dtype=np.uint8)
        self.ball_hue_hint = ball_hue_hint
        self.neural_ball = None
        if ball_model_path is not None:
            from src.ball_model import NeuralBallLocator
            self.neural_ball = NeuralBallLocator(ball_model_path)

        # Mode A components
        self.mog2 = cv2.createBackgroundSubtractorMOG2(
            history=MOG2_HISTORY,
            varThreshold=MOG2_VAR_THRESHOLD,
            detectShadows=MOG2_DETECT_SHADOWS,
        )
        self.classical_player_tracker = ClassicalPlayerTracker()
        self.prev_frame1: Optional[np.ndarray] = None
        self.prev_frame2: Optional[np.ndarray] = None
        self.prev_gray1: Optional[np.ndarray] = None
        self.prev_gray2: Optional[np.ndarray] = None
        self._previous_camera_transform = np.eye(3, dtype=np.float64)
        self.temporal_threshold: int = 16
        self._ball_hue = None
        self._ball_saturation = None
        self._hue_observations = 0
        self._last_ball_point = None
        self._ball_misses = 0

        # Mode B components (lazy-loaded if Mode B is selected)
        self.yolo_model = None
        self.byte_tracker = None
        self.yolo_frame_interval: int = 6
        self._frame_count: int = 0
        self.last_player_detections: List[PlayerDetection] = []
        self.last_player_bbox_tuples: List[Tuple[float, float, float, float]] = []
        if self.mode == "B":
            self._init_yolo_and_bytetrack(yolo_model_path)

    def _init_yolo_and_bytetrack(self, model_path: str) -> None:
        """Loads local Ultralytics YOLOv8 model with native ByteTrack."""
        from ultralytics import YOLO

        self.yolo_model = YOLO(model_path)
        self.byte_tracker = "bytetrack.yaml"

    def reset_ball_history(self) -> None:
        """Discard incomparable ball images after losing the camera reference."""
        self.prev_frame1 = self.prev_frame2 = None
        self.prev_gray1 = self.prev_gray2 = None
        self._previous_camera_transform = np.eye(3, dtype=np.float64)
        self._ball_hue = None
        self._ball_saturation = None
        self._hue_observations = 0
        self._last_ball_point = None
        self._ball_misses = 0
        if self.neural_ball is not None:
            self.neural_ball.reset()

    @staticmethod
    def is_inside_shoe_zone(
        ball_pt: Tuple[float, float],
        player_bboxes: List[Tuple[float, float, float, float]],
        bottom_ratio: float = SHOE_FILTER_BOTTOM_RATIO,
    ) -> bool:
        """Heuristic shoe filter:

        Checks if the candidate ball coordinate (bx, by) falls within the bottom
        fraction (default 25%) of any player's bounding box.
        """
        bx, by = ball_pt
        for x1, y1, x2, y2 in player_bboxes:
            box_h = y2 - y1
            shoe_y_top = y1 + (1.0 - bottom_ratio) * box_h
            # Check bounding box inclusion with horizontal margin
            if (x1 - 5.0) <= bx <= (x2 + 5.0) and shoe_y_top <= by <= (y2 + 8.0):
                return True
        return False

    @staticmethod
    def is_inside_player_body(
        ball_pt: Tuple[float, float],
        player_bboxes: List[Tuple[float, float, float, float]],
    ) -> bool:
        """Checks if candidate coordinate is inside player torso/body (excluding upper paddle area)."""
        bx, by = ball_pt
        for x1, y1, x2, y2 in player_bboxes:
            box_h = y2 - y1
            # Core body: between 25% and 85% of bbox height
            body_top = y1 + 0.25 * box_h
            body_bot = y1 + 0.85 * box_h
            if (x1 + 2.0) <= bx <= (x2 - 2.0) and body_top <= by <= body_bot:
                return True
        return False

    @staticmethod
    def _get_court_mask(
        frame_shape: Tuple[int, int],
        homography=None,
    ) -> np.ndarray:
        """Builds an expanded court action mask to eliminate background tree/sky false detections."""
        h, w = frame_shape
        mask = np.zeros((h, w), dtype=np.uint8)
        if homography is not None and getattr(homography, "pixel_corners", None) is not None:
            corners = homography.pixel_corners.astype(np.float32)
            center = np.mean(corners, axis=0)
            expanded = []
            for pt in corners:
                vec = pt - center
                expanded.append(pt + vec * 0.25)
            # High lobs can reach the top edge. A fixed 5% exclusion clipped
            # visible balls and biased centroids near the apex in real footage.
            expanded[0][1] = 0.
            expanded[1][1] = 0.
            poly = np.array(expanded, dtype=np.int32)
            cv2.fillPoly(mask, [poly], 255)
        else:
            # Default fallback: ignore top 10% (sky/lights) and frame borders
            cv2.rectangle(
                mask,
                (int(0.03 * w), int(0.05 * h)),
                (int(0.97 * w), int(0.98 * h)),
                255,
                -1,
            )
        return mask

    def _extract_moving_ball(
        self,
        frame: np.ndarray,
        player_bbox_tuples: List[Tuple[float, float, float, float]],
        predicted_pos: Optional[Tuple[float, float]] = None,
        court_mask: Optional[np.ndarray] = None,
        camera_transform: Optional[np.ndarray] = None,
        homography=None,
    ) -> Tuple[Optional[Tuple[float, float]], float]:
        """High-sensitivity 3-frame temporal differencing with camera motion compensation and small-ball extraction."""
        if self.neural_ball is not None:
            if court_mask is None:
                self.neural_ball.reset()
                return None, 0.
            try:
                return self.neural_ball.update(frame, court_mask=court_mask, homography=homography)
            except TypeError:
                return self.neural_ball.update(frame, court_mask=court_mask)
        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = curr_gray.shape
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask_hsv = cv2.inRange(hsv, self.hsv_lower, self.hsv_upper)

        # Each history image needs the transform from its own camera view.
        # t-2 -> t is (t-1 -> t) @ (t-2 -> t-1), even when the latest step is identity.
        current_transform = (np.eye(3, dtype=np.float64) if camera_transform is None
                             else np.asarray(camera_transform, dtype=np.float64))
        older_transform = current_transform @ self._previous_camera_transform
        if (not np.allclose(current_transform, np.eye(3))
                or not np.allclose(older_transform, np.eye(3))):
            try:
                aligned_prev1 = cv2.warpPerspective(self.prev_gray1, current_transform, (w, h)) if self.prev_gray1 is not None else None
                aligned_prev2 = cv2.warpPerspective(self.prev_gray2, older_transform, (w, h)) if self.prev_gray2 is not None else None
            except cv2.error:
                aligned_prev1 = self.prev_gray1
                aligned_prev2 = self.prev_gray2
        else:
            aligned_prev1 = self.prev_gray1
            aligned_prev2 = self.prev_gray2

        # 1. 3-Frame Temporal Differencing using motion-compensated single-channel grayscale
        # Suppresses single-step lighting flickers, shadows, and static court noise
        ball_mask = None
        if aligned_prev1 is not None and aligned_prev2 is not None:
            diff1 = cv2.absdiff(curr_gray, aligned_prev1)
            diff2 = cv2.absdiff(curr_gray, aligned_prev2)
            _, m1 = cv2.threshold(diff1, self.temporal_threshold, 255, cv2.THRESH_BINARY)
            _, m2 = cv2.threshold(diff2, self.temporal_threshold, 255, cv2.THRESH_BINARY)
            motion_mask = cv2.bitwise_and(m1, m2)
            ball_mask = cv2.bitwise_and(mask_hsv, motion_mask)
        elif aligned_prev1 is not None:
            # 2-frame differencing fallback on frame 2
            diff1 = cv2.absdiff(curr_gray, aligned_prev1)
            _, motion_mask = cv2.threshold(diff1, self.temporal_threshold, 255, cv2.THRESH_BINARY)
            ball_mask = cv2.bitwise_and(mask_hsv, motion_mask)

        # Apply court mask constraint to exclude trees, sky, spectators
        if court_mask is not None:
            if ball_mask is not None:
                ball_mask = cv2.bitwise_and(ball_mask, court_mask)
            mask_hsv = cv2.bitwise_and(mask_hsv, court_mask)

        # Clean noise
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        if ball_mask is not None:
            clean_mask = cv2.morphologyEx(ball_mask, cv2.MORPH_OPEN, kernel_open)
            # Preserve genuinely tiny round components that opening would erase,
            # without admitting every moving grass/fence pixel into acquisition.
            tiny, _ = cv2.findContours(ball_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in tiny:
                area = cv2.contourArea(contour)
                _, _, tw, th = cv2.boundingRect(contour)
                perimeter = cv2.arcLength(contour, True)
                if (5 <= area <= 22 and max(tw, th) <= 7 and perimeter > 0
                        and 4*np.pi*area/perimeter**2 >= .55):
                    cv2.drawContours(clean_mask, [contour], -1, 255, -1)
            if predicted_pos is not None:
                # A slow, tiny ball overlaps its previous image. Difference
                # masks contain only a crescent: recover its full colour blob
                # locally, requiring actual changed pixels (never static HSV).
                px, py = map(round, predicted_pos)
                x0, y0 = max(0, px-20), max(0, py-20)
                x1, y1 = min(frame.shape[1], px+21), min(frame.shape[0], py+21)
                if x1 > x0 and y1 > y0:
                    local_mask = mask_hsv[y0:y1, x0:x1]
                    local_contours, _ = cv2.findContours(local_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    for contour in local_contours:
                        area = cv2.contourArea(contour)
                        bx, by, bw, bh = cv2.boundingRect(contour)
                        perimeter = cv2.arcLength(contour, True)
                        if (5 <= area <= 35 and max(bw, bh) <= 9 and perimeter > 0
                                and 4*np.pi*area/perimeter**2 >= .45):
                            component = np.zeros_like(local_mask)
                            cv2.drawContours(component, [contour], -1, 255, -1)
                            if cv2.countNonZero(cv2.bitwise_and(component, ball_mask[y0:y1, x0:x1])) >= 2:
                                cv2.drawContours(clean_mask, [contour+np.array([[[x0, y0]]])], -1, 255, -1)
        else:
            clean_mask = np.zeros(frame.shape[:2], dtype=np.uint8)

        contours, _ = cv2.findContours(
            clean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # Note: We do NOT fall back to raw static HSV masking when motion is absent.
        # Falling back to raw HSV causes the tracker to latch onto static yellow floor artifacts,
        # paddle logos, and spectator clothing, making the ball indicator fly wildly across the court.

        best_ball = None
        best_score = -1.0
        best_conf = 0.0
        best_hue = None
        best_saturation = None
        last_point = self._last_ball_point
        if last_point is not None:
            p = current_transform @ np.r_[last_point, 1.]
            last_point = tuple(p[:2]/p[2]) if abs(p[2]) > 1e-7 else None

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if not (BALL_MIN_AREA <= area <= BALL_MAX_AREA):
                continue

            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = float(w) / max(1.0, float(h))
            if not (BALL_ASPECT_RATIO_MIN <= aspect_ratio <= BALL_ASPECT_RATIO_MAX):
                continue

            perimeter = cv2.arcLength(cnt, True)
            if perimeter < 1.0:
                continue
            circularity = 4.0 * np.pi * area / (perimeter**2)
            if circularity < BALL_MIN_CIRCULARITY:
                continue

            # Centroid
            M = cv2.moments(cnt)
            if M["m00"] < 1e-5:
                continue
            cx = float(M["m10"] / M["m00"])
            cy = float(M["m01"] / M["m00"])
            candidate_pt = (cx, cy)
            # Keep the ball's observed colour distinct from moving neon shirt
            # logos. Only learn colour from motion-supported selected candidates.
            local = hsv[y:y+h, x:x+w]
            component = np.zeros((h, w), dtype=np.uint8)
            cv2.drawContours(component, [cnt-np.array([[[x, y]]])], -1, 255, -1)
            colored = local[(component > 0) & (local[..., 1] >= 45)]
            hue = float(np.median(colored[:, 0])) if len(colored) else None
            saturation = float(np.median(colored[:, 1])) if len(colored) else None
            hue_error = 0.
            hint_error = 0.
            if hue is not None and self.ball_hue_hint is not None:
                hint_error = min(abs(hue-self.ball_hue_hint), 180-abs(hue-self.ball_hue_hint))
                if hint_error > 12.:
                    continue
            if hue is not None and self._ball_hue is not None and self._hue_observations >= 3:
                hue_error = min(abs(hue-self._ball_hue), 180-abs(hue-self._ball_hue))
                # A selected ball colour outranks a self-learned appearance.
                # A false first acquisition must not permanently veto the ball.
                if self.ball_hue_hint is None and hue_error > 7.:
                    continue
                if (self.ball_hue_hint is None and hue_error > 4. and self._ball_saturation is not None
                        and abs(saturation-self._ball_saturation) > 20.):
                    continue

            # Heuristic Filters: Reject shoes and player torso motion
            if self.is_inside_shoe_zone(candidate_pt, player_bbox_tuples):
                continue
            if self.is_inside_player_body(candidate_pt, player_bbox_tuples):
                continue

            # Scoring: circularity + proximity to Kalman prediction if available
            score = circularity
            score += max(0., 1.-hue_error/10.)
            if self.ball_hue_hint is not None and hue is not None:
                score += 2.*np.exp(-.5*(hint_error/6.)**2)
            if predicted_pos is not None:
                dist = np.hypot(cx - predicted_pos[0], cy - predicted_pos[1])
                # A paddle contact reverses direction before a velocity model
                # can react. A measured ball near the last observation remains
                # eligible, but still needs motion, shape, colour and body checks.
                last_dist = np.hypot(cx-last_point[0], cy-last_point[1]) if last_point is not None else dist
                search_dist = min(dist, last_dist)
                if search_dist > 140.0:
                    continue  # Gate out impossible jumps across the court
                score += max(0.0, 2.0 * (1.0 - (search_dist / 140.0)))
            else:
                score += min(1.0, area / 40.0)

            if score > best_score:
                best_score = score
                best_ball = candidate_pt
                best_conf = float(min(1.0, circularity + 0.3))
                best_hue = hue
                best_saturation = saturation

        if best_ball is not None:
            self._last_ball_point = best_ball
            self._ball_misses = 0
            if best_hue is not None:
                if self._ball_hue is None or self._hue_observations < 3:
                    self._ball_hue = best_hue
                    self._ball_saturation = best_saturation
                else:
                    self._ball_hue = .95*self._ball_hue+.05*best_hue
                    self._ball_saturation = .95*self._ball_saturation+.05*best_saturation
                self._hue_observations += 1
        else:
            self._last_ball_point = last_point
            self._ball_misses += 1
            if self._ball_misses > 5:
                self._last_ball_point = None
            # Allow a new rally/lighting change to acquire a new appearance.
            if self._ball_misses > 30:
                self._ball_hue = None
                self._ball_saturation = None
                self._hue_observations = 0

        return best_ball, best_conf

    def detect_mode_a(
        self,
        frame: np.ndarray,
        predicted_pos: Optional[Tuple[float, float]] = None,
        court_mask: Optional[np.ndarray] = None,
        camera_transform: Optional[np.ndarray] = None,
        homography=None,
    ) -> Tuple[Optional[Tuple[float, float]], float, List[PlayerDetection]]:
        """Mode A: Pure Classical CV using MOG2 foreground subtraction, HSV color masking,

        and morphological contour filtering.
        """
        # Background subtraction for players
        fg_mask = self.mog2.apply(frame)
        _, fg_thresh = cv2.threshold(fg_mask, 180, 255, cv2.THRESH_BINARY)

        # Detect Players in Mode A (large moving blobs)
        player_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 15))
        player_mask = cv2.morphologyEx(fg_thresh, cv2.MORPH_CLOSE, player_kernel)
        player_contours, _ = cv2.findContours(
            player_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        player_bboxes = []
        for cnt in player_contours:
            area = cv2.contourArea(cnt)
            if PLAYER_MIN_AREA <= area <= PLAYER_MAX_AREA:
                x, y, w, h = cv2.boundingRect(cnt)
                if h >= w * 0.8:
                    player_bboxes.append((float(x), float(y), float(x + w), float(y + h)))

        tracked_players = self.classical_player_tracker.update(player_bboxes)
        player_bbox_tuples = [p.bbox_xyxy for p in tracked_players]

        best_ball, best_conf = self._extract_moving_ball(
            frame, player_bbox_tuples, predicted_pos, court_mask=court_mask, camera_transform=camera_transform, homography=homography
        )
        return best_ball, best_conf, tracked_players

    def detect_mode_b(
        self,
        frame: np.ndarray,
        predicted_pos: Optional[Tuple[float, float]] = None,
        court_mask: Optional[np.ndarray] = None,
        camera_transform: Optional[np.ndarray] = None,
        homography=None,
    ) -> Tuple[Optional[Tuple[float, float]], float, List[PlayerDetection]]:
        """Mode B: Local Open-Source Weights using YOLOv8n + ByteTrack,

        with heuristic shoe-filtering.
        """
        if self.yolo_model is None:
            self._init_yolo_and_bytetrack(YOLO_MODEL_NAME)

        # Run local YOLOv8 with ByteTrack persistence only on interval frames (or when cache empty)
        run_yolo = (
            self._frame_count % self.yolo_frame_interval == 0
            or not self.last_player_detections
        )

        player_detections: List[PlayerDetection] = []
        player_bbox_tuples: List[Tuple[float, float, float, float]] = []

        if run_yolo:
            results = self.yolo_model.track(
                frame,
                persist=True,
                tracker=self.byte_tracker,
                verbose=False,
                conf=YOLO_PERSON_CONF_THRESHOLD,
                classes=[COCO_PERSON_CLASS_ID],
            )[0]

            # Extract person detections with ByteTrack IDs
            boxes = results.boxes.xyxy.cpu().numpy() if results.boxes is not None else np.empty((0, 4))
            confs = results.boxes.conf.cpu().numpy() if results.boxes is not None else np.empty((0,))
            cls_ids = (
                results.boxes.cls.cpu().numpy().astype(int)
                if results.boxes is not None
                else np.empty((0,), dtype=int)
            )
            track_ids = (
                results.boxes.id.cpu().numpy().astype(int)
                if (results.boxes is not None and results.boxes.id is not None)
                else None
            )

            for i in range(len(boxes)):
                if cls_ids[i] == COCO_PERSON_CLASS_ID and confs[i] >= YOLO_PERSON_CONF_THRESHOLD:
                    bbox = (float(boxes[i][0]), float(boxes[i][1]), float(boxes[i][2]), float(boxes[i][3]))
                    tid = int(track_ids[i]) if track_ids is not None else (i + 1)
                    feet_px = ((bbox[0] + bbox[2]) / 2.0, bbox[3])

                    player_detections.append(
                        PlayerDetection(
                            tracker_id=tid,
                            bbox_xyxy=bbox,
                            confidence=float(confs[i]),
                            feet_pixel=feet_px,
                        )
                    )
                    player_bbox_tuples.append(bbox)

            self.last_player_detections = player_detections
            self.last_player_bbox_tuples = player_bbox_tuples
        else:
            # Return independent snapshot copies of cached player tracking bounding boxes for intermediate frames
            player_detections = [
                PlayerDetection(
                    tracker_id=p.tracker_id,
                    bbox_xyxy=tuple(p.bbox_xyxy),
                    confidence=p.confidence,
                    feet_pixel=p.feet_pixel,
                    court_feet=p.court_feet,
                    in_kitchen=p.in_kitchen,
                )
                for p in self.last_player_detections
            ]
            player_bbox_tuples = list(self.last_player_bbox_tuples)

        best_ball = None
        best_conf = 0.0

        # Always run high-sensitivity 3-frame temporal ball extractor for precision small-ball flight
        best_ball, best_conf = self._extract_moving_ball(
            frame, player_bbox_tuples, predicted_pos, court_mask=court_mask, camera_transform=camera_transform, homography=homography
        )

        return best_ball, best_conf, player_detections

    def detect(
        self,
        frame: np.ndarray,
        predicted_pos: Optional[Tuple[float, float]] = None,
        homography=None,
        camera_transform: Optional[np.ndarray] = None,
    ) -> Tuple[Optional[Tuple[float, float]], float, List[PlayerDetection]]:
        """Processes frame using configured mode ('A' or 'B')."""
        court_mask = self._get_court_mask(frame.shape[:2], homography)
        if self.neural_ball is not None and homography is None:
            court_mask = None
        if self.mode == "B":
            ball, conf, players = self.detect_mode_b(
                frame, predicted_pos, court_mask=court_mask, camera_transform=camera_transform, homography=homography
            )
        else:
            ball, conf, players = self.detect_mode_a(
                frame, predicted_pos, court_mask=court_mask, camera_transform=camera_transform, homography=homography
            )

        # Update 3-frame history FIFO: t-2 <= t-1, t-1 <= t
        self.prev_frame2 = self.prev_frame1
        self.prev_frame1 = frame.copy()
        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.prev_gray2 = self.prev_gray1
        self.prev_gray1 = curr_gray
        self._previous_camera_transform = (np.eye(3, dtype=np.float64) if camera_transform is None
                                           else np.array(camera_transform, dtype=np.float64, copy=True))
        self._frame_count += 1
        return ball, conf, players
