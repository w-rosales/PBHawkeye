"""Track a calibrated court through camera pan and zoom using local image features.

The opening-frame calibration stays the geometric anchor. ORB feature matching
periodically re-estimates its image transform; sparse optical flow follows the
same static landmarks between those matches. Failed matches never replace the
last reliable court and are exposed to callers through ``tracking_ok``.
"""

from copy import deepcopy
from typing import Optional

import cv2
import numpy as np

from src.homography import CourtHomography


class CourtMotionTracker:
    """Follow a court calibrated on ``initial_frame`` without paid models.

    ``update(frame)`` returns the current court mapping, including corners outside
    the image when a zoom crops a baseline. Callers must check ``tracking_ok``
    before using that mapping for overlays or physical court measurements.
    ``status`` explains a failed match; a later valid frame can recover tracking.
    """

    def __init__(
        self,
        initial_frame: np.ndarray,
        homography: CourtHomography,
        update_interval: int = 12,
        max_width: int = 960,
        min_inliers: int = 20,
    ):
        initial_gray = self._gray(initial_frame)
        if initial_gray is None:
            raise ValueError("The court motion tracker needs a nonempty video frame")
        self.height, self.width = initial_gray.shape
        self.image_size = (self.width, self.height)
        if homography.image_size is not None and tuple(homography.image_size) != self.image_size:
            raise ValueError("The opening frame dimensions do not match the court calibration")
        self.update_interval = max(1, int(update_interval))
        self.min_inliers = max(8, int(min_inliers))
        self.scale = min(1.0, float(max_width) / self.width)
        self.small_size = (max(1, round(self.width * self.scale)),
                           max(1, round(self.height * self.scale)))
        self._scale_matrix = np.diag([self.small_size[0] / self.width,
                                      self.small_size[1] / self.height, 1.0])
        self._anchor_corners = homography.pixel_corners.copy()
        self._court_corners = homography.court_corners_ft.copy()
        self._metadata = deepcopy(homography.metadata)
        self.homography = homography
        self.tracking_ok = False
        self.status = "insufficient_features"
        self.inlier_count = 0
        self.inlier_ratio = 0.0
        self._calls = 0
        self._anchor_to_current = np.eye(3, dtype=np.float64)
        self.full_transform = np.eye(3, dtype=np.float64)
        self._prev_full_transform = np.eye(3, dtype=np.float64)
        self.prev_to_curr_transform = np.eye(3, dtype=np.float64)
        self._orb = cv2.ORB_create(nfeatures=2000, scaleFactor=1.2, nlevels=10,
                                   edgeThreshold=19, fastThreshold=12)
        self._matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        self._reference_gray = self._resize(initial_gray)
        mask = self._background_mask(self._anchor_corners)
        keypoints, self._reference_descriptors = self._orb.detectAndCompute(self._reference_gray, mask)
        self._reference_points = np.asarray([point.pt for point in keypoints], dtype=np.float32).reshape(-1, 2)
        self._keyframe_points = self._reference_points
        self._keyframe_descriptors = self._reference_descriptors
        self._anchor_to_keyframe = np.eye(3, dtype=np.float64)
        self._previous_gray: Optional[np.ndarray] = None
        self._flow_anchor_points: Optional[np.ndarray] = None
        self._flow_image_points: Optional[np.ndarray] = None
        self._static_reference_score = None
        if len(self._reference_points) >= self.min_inliers:
            self.tracking_ok = True
            self.status = "initialized"
        else:
            # Plain surroundings can contain no usable background keypoints.
            # A complete painted court still supplies an independent check that
            # its existing geometry remains stationary; blank frames cannot pass.
            from src.court_detection import court_alignment_score
            color_frame = initial_frame if initial_frame.ndim == 3 else cv2.cvtColor(initial_frame, cv2.COLOR_GRAY2BGR)
            score = court_alignment_score(color_frame, self._anchor_corners)
            if score >= .80:
                self._static_reference_score = score
                self.tracking_ok = True
                self.status = "static_lines"

    @staticmethod
    def _gray(frame: np.ndarray) -> Optional[np.ndarray]:
        if not isinstance(frame, np.ndarray) or frame.size == 0 or frame.dtype != np.uint8:
            return None
        if frame.ndim == 2:
            return frame
        if frame.ndim == 3 and frame.shape[2] in (3, 4):
            return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY if frame.shape[2] == 3 else cv2.COLOR_BGRA2GRAY)
        return None

    def _resize(self, gray: np.ndarray) -> np.ndarray:
        return cv2.resize(gray, self.small_size, interpolation=cv2.INTER_AREA)

    def _background_mask(self, corners: np.ndarray) -> np.ndarray:
        """Avoid using the players inside the court as camera-motion landmarks."""
        mask = np.full(self.small_size[::-1], 255, dtype=np.uint8)
        small_corners = cv2.perspectiveTransform(corners.reshape(-1, 1, 2), self._scale_matrix).reshape(-1, 2)
        # Exclude the playing surface and a small border, where players' feet may lie.
        court_mask = np.zeros_like(mask)
        cv2.fillConvexPoly(court_mask, np.rint(small_corners).astype(np.int32), 255)
        court_mask = cv2.dilate(court_mask, np.ones((11, 11), dtype=np.uint8))
        mask[court_mask > 0] = 0
        return mask

    @staticmethod
    def _transform(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
        return cv2.perspectiveTransform(points.astype(np.float32).reshape(-1, 1, 2), matrix).reshape(-1, 2)

    def _fail(self, reason: str) -> CourtHomography:
        self.tracking_ok = False
        self.status = reason
        self.inlier_count = 0
        self.inlier_ratio = 0.0
        self.prev_to_curr_transform = np.eye(3, dtype=np.float64)
        return self.homography

    def _spread_ok(self, points: np.ndarray) -> bool:
        w, h = self.small_size
        span = np.ptp(points, axis=0)
        area = abs(float(cv2.contourArea(cv2.convexHull(points.astype(np.float32)))))
        return span[0] >= w * 0.16 and span[1] >= h * 0.10 and area >= w * h * 0.012

    def _estimate(self, anchor_points: np.ndarray, image_points: np.ndarray):
        if len(anchor_points) < self.min_inliers:
            return None
        matrix, mask = cv2.findHomography(anchor_points, image_points, cv2.RANSAC, 3.0,
                                           maxIters=2000, confidence=0.995)
        if matrix is None or mask is None or not np.isfinite(matrix).all() or abs(matrix[2, 2]) < 1e-8:
            return None
        matrix = matrix / matrix[2, 2]
        inliers = mask.ravel().astype(bool)
        # Pan, zoom, and modest roll are often affine. Prefer that simpler fit
        # when perspective terms add little evidence; this avoids amplifying
        # subpixel feature noise at a baseline cropped outside the frame.
        for estimator in (cv2.estimateAffine2D, cv2.estimateAffinePartial2D):
            affine, affine_mask = estimator(
                anchor_points, image_points, method=cv2.RANSAC,
                ransacReprojThreshold=3.0, maxIters=2000, confidence=0.995, refineIters=10)
            if affine is not None and affine_mask is not None and np.isfinite(affine).all():
                affine_matrix = np.vstack([affine, [0.0, 0.0, 1.0]])
                affine_inliers = affine_mask.ravel().astype(bool)
                if np.count_nonzero(affine_inliers) >= np.count_nonzero(inliers) * 0.85:
                    if np.count_nonzero(inliers) >= self.min_inliers:
                        hom_error = np.linalg.norm(self._transform(anchor_points[inliers], matrix) - image_points[inliers], axis=1)
                        affine_error = np.linalg.norm(self._transform(anchor_points[inliers], affine_matrix) - image_points[inliers], axis=1)
                        if float(np.mean(np.minimum(affine_error, 3.0))) <= float(np.mean(np.minimum(hom_error, 3.0))) + 0.15:
                            matrix, inliers = affine_matrix, affine_inliers
        count = int(np.count_nonzero(inliers))
        ratio = count / len(inliers)
        if count < self.min_inliers or ratio < 0.4:
            return None
        source, target = anchor_points[inliers], image_points[inliers]
        if not self._spread_ok(source) or not self._spread_ok(target):
            return None
        residual = np.linalg.norm(self._transform(source, matrix) - target, axis=1)
        if float(np.median(residual)) > 1.8:
            return None

        w, h = self.small_size
        grid = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        denominators = np.column_stack([grid, np.ones(4)]) @ matrix[2]
        if (np.any(denominators <= 0.15)
                or float(denominators.max() / denominators.min()) > 4):
            return None
        warped = self._transform(grid, matrix)
        try:
            CourtHomography._validate_corners(warped, "camera transform", clockwise=True)
        except ValueError:
            return None
        zoom = np.sqrt(abs(cv2.contourArea(warped)) / (w * h))
        if not 0.25 <= zoom <= 4.0 or np.max(np.abs(warped)) > max(w, h) * 6:
            return None
        full_transform = np.linalg.inv(self._scale_matrix) @ matrix @ self._scale_matrix
        court_corners = self._transform(self._anchor_corners, full_transform)
        if not np.isfinite(court_corners).all() or np.max(np.abs(court_corners)) > max(self.image_size) * 8:
            return None
        try:
            court = CourtHomography(court_corners, self._court_corners, self.image_size,
                                    metadata=deepcopy(self._metadata))
        except (ValueError, cv2.error):
            return None
        return matrix, court, source, target, count, ratio

    def _matches(self, reference_points, reference_descriptors, points, descriptors, anchor_to_reference):
        if (reference_descriptors is None or descriptors is None
                or len(reference_descriptors) < self.min_inliers or len(descriptors) < 2):
            return None
        pairs = self._matcher.knnMatch(reference_descriptors, descriptors, k=2)
        matches = [pair[0] for pair in pairs if len(pair) == 2 and pair[0].distance < 0.75 * pair[1].distance]
        # A target feature must contribute at most once to the consensus.
        unique = {}
        for match in matches:
            if match.trainIdx not in unique or match.distance < unique[match.trainIdx].distance:
                unique[match.trainIdx] = match
        matches = list(unique.values())
        if len(matches) < self.min_inliers:
            return None
        source = np.float32([reference_points[match.queryIdx] for match in matches])
        source = self._transform(source, np.linalg.inv(anchor_to_reference))
        target = np.float32([points[match.trainIdx] for match in matches])
        return self._estimate(source, target)

    def _optical_flow(self, gray: np.ndarray):
        if self._previous_gray is None or self._flow_image_points is None:
            return None
        previous = self._flow_image_points.reshape(-1, 1, 2)
        current, forward_ok, _ = cv2.calcOpticalFlowPyrLK(
            self._previous_gray, gray, previous, None, winSize=(21, 21), maxLevel=3)
        if current is None or forward_ok is None:
            return None
        backward, backward_ok, _ = cv2.calcOpticalFlowPyrLK(
            gray, self._previous_gray, current, None, winSize=(21, 21), maxLevel=3)
        if backward is None or backward_ok is None:
            return None
        valid = (forward_ok.ravel().astype(bool) & backward_ok.ravel().astype(bool)
                 & (np.linalg.norm(backward.reshape(-1, 2) - previous.reshape(-1, 2), axis=1) < 1.0))
        return self._estimate(self._flow_anchor_points[valid], current.reshape(-1, 2)[valid])

    def _refine_original_match(self, gray: np.ndarray, estimate):
        """Remove ORB's scale-pyramid rounding using aligned background patches."""
        matrix, _, source, _, _, _ = estimate
        predicted = self._transform(source, matrix)
        w, h = self.small_size
        inside = ((predicted[:, 0] >= 12) & (predicted[:, 0] < w - 12)
                  & (predicted[:, 1] >= 12) & (predicted[:, 1] < h - 12))
        if np.count_nonzero(inside) < self.min_inliers:
            return estimate
        source, predicted = source[inside], predicted[inside].reshape(-1, 1, 2)
        aligned = cv2.warpPerspective(self._reference_gray, matrix, self.small_size)
        refined, forward_ok, errors = cv2.calcOpticalFlowPyrLK(
            aligned, gray, predicted, None, winSize=(21, 21), maxLevel=2)
        if refined is None or forward_ok is None:
            return estimate
        backward, backward_ok, _ = cv2.calcOpticalFlowPyrLK(
            gray, aligned, refined, None, winSize=(21, 21), maxLevel=2)
        if backward is None or backward_ok is None:
            return estimate
        valid = (forward_ok.ravel().astype(bool) & backward_ok.ravel().astype(bool)
                 & (np.linalg.norm(backward.reshape(-1, 2) - predicted.reshape(-1, 2), axis=1) < 0.8)
                 & (errors.ravel() < 25.0))
        result = self._estimate(source[valid], refined.reshape(-1, 2)[valid])
        return result if result is not None and result[4] >= estimate[4] * 0.6 else estimate

    def update(self, frame: np.ndarray) -> CourtHomography:
        """Match this frame to the original calibration, retaining geometry on failure."""
        gray = self._gray(frame)
        if gray is None:
            return self._fail("invalid_frame")
        if gray.shape != (self.height, self.width):
            return self._fail("frame_size_changed")
        if self._static_reference_score is not None:
            from src.court_detection import court_alignment_score
            color_frame = frame if frame.ndim == 3 else cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            score = court_alignment_score(color_frame, self.homography.pixel_corners)
            if score >= max(.80, self._static_reference_score-.08):
                self.tracking_ok = True
                self.status = "static_lines"
                return self.homography
        gray = self._resize(gray)
        use_features = self._calls % self.update_interval == 0 or not self.tracking_ok
        self._calls += 1
        estimate = None if use_features else self._optical_flow(gray)
        from_original = False
        feature_points = feature_descriptors = None
        if estimate is None:
            mask = self._background_mask(self.homography.pixel_corners)
            keypoints, feature_descriptors = self._orb.detectAndCompute(gray, mask)
            feature_points = np.float32([point.pt for point in keypoints]).reshape(-1, 2)
            if len(feature_points) < self.min_inliers:
                return self._fail("insufficient_features")
            estimate = self._matches(self._reference_points, self._reference_descriptors,
                                     feature_points, feature_descriptors, np.eye(3))
            from_original = estimate is not None
            if estimate is None and self._keyframe_descriptors is not self._reference_descriptors:
                estimate = self._matches(self._keyframe_points, self._keyframe_descriptors,
                                         feature_points, feature_descriptors, self._anchor_to_keyframe)
        if estimate is None:
            return self._fail("insufficient_static_matches")

        if from_original:
            estimate = self._refine_original_match(gray, estimate)

        matrix, court, source, target, count, ratio = estimate
        self._anchor_to_current = matrix
        full_transform = np.linalg.inv(self._scale_matrix) @ matrix @ self._scale_matrix
        self.full_transform = full_transform
        if hasattr(self, "_prev_full_transform") and self._prev_full_transform is not None:
            try:
                self.prev_to_curr_transform = full_transform @ np.linalg.inv(self._prev_full_transform)
            except np.linalg.LinAlgError:
                self.prev_to_curr_transform = np.eye(3, dtype=np.float64)
        else:
            self.prev_to_curr_transform = np.eye(3, dtype=np.float64)
        self._prev_full_transform = full_transform.copy()
        self.homography = court
        self.inlier_count, self.inlier_ratio = count, ratio
        self.tracking_ok = True
        self.status = "tracking"
        self._previous_gray = gray
        self._flow_anchor_points = source
        self._flow_image_points = target
        if from_original and count >= max(35, self.min_inliers) and ratio >= 0.55:
            # Re-anchor only from a direct match to the original frame. Fallback
            # keyframe matches never replace this reference, avoiding chained drift.
            self._keyframe_points = feature_points
            self._keyframe_descriptors = feature_descriptors
            self._anchor_to_keyframe = matrix.copy()
        return self.homography
