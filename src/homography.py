"""Homography engine for mapping video pixel coordinates to 2D court feet coordinates.

Uses cv2.getPerspectiveTransform and perspective projection to transform coordinates
between the camera perspective and top-down 20x44 ft pickleball court space.
"""

import json
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

from src.config import (
    COURT_CORNERS_FEET,
    COURT_CORNERS_ORDER,
    COURT_LENGTH_FT,
    COURT_WIDTH_FT,
    KITCHEN_FAR_Y,
    KITCHEN_NEAR_Y,
    NET_Y,
)


class CourtHomography:
    """Manages homography matrix computation, forward and inverse perspective transforms,

    and court zone queries.
    """

    def __init__(
        self,
        pixel_corners: Union[List[Tuple[float, float]], np.ndarray],
        court_corners_ft: Optional[Union[List[Tuple[float, float]], np.ndarray]] = None,
        image_size: Optional[Tuple[int, int]] = None,
        metadata: Optional[dict] = None,
    ):
        """Initialize homography mapper.

        Args:
            pixel_corners: 4 pixel coordinates ordered:
                [top_left, top_right, bottom_right, bottom_left]
            court_corners_ft: Optional 4 court coordinates in feet. Defaults to
                Pickleball standard:
                [(0.0, 44.0), (20.0, 44.0), (20.0, 0.0), (0.0, 0.0)]
            image_size: Optional (width, height) tuple of calibration image/frame.
            metadata: Calibration provenance, fit score, and explicit confirmation.
        """
        self.pixel_corners = np.array(pixel_corners, dtype=np.float32)
        if court_corners_ft is None:
            self.court_corners_ft = np.array(COURT_CORNERS_FEET, dtype=np.float32)
        else:
            self.court_corners_ft = np.array(court_corners_ft, dtype=np.float32)

        self._validate_corners(self.pixel_corners, "pixel_corners", clockwise=True)
        self._validate_corners(self.court_corners_ft, "court_corners_ft")
        if image_size is not None:
            dimensions = np.asarray(image_size, dtype=np.float64)
            if (dimensions.shape != (2,) or not np.isfinite(dimensions).all()
                    or np.any(dimensions <= 0) or np.any(dimensions != np.floor(dimensions))):
                raise ValueError("image_size must contain a positive integer width and height")
            image_size = (int(dimensions[0]), int(dimensions[1]))
        if metadata is not None and not isinstance(metadata, dict):
            raise ValueError("Calibration metadata must be an object")
        self.image_size = image_size
        self.metadata = dict(metadata or {})
        self.H = None
        self.H_inv = None
        self._compute_matrices()

    @staticmethod
    def _validate_corners(points: np.ndarray, name: str, clockwise: bool = False) -> None:
        """Reject malformed, crossed, concave, and collapsed quadrilaterals."""
        if points.shape != (4, 2) or not np.isfinite(points).all():
            raise ValueError(f"{name} must contain exactly four finite (x, y) points")
        points = points.astype(np.float64)
        edges = np.roll(points, -1, axis=0) - points
        next_edges = np.roll(edges, -1, axis=0)
        crosses = edges[:, 0] * next_edges[:, 1] - edges[:, 1] * next_edges[:, 0]
        extent = max(1.0, float(np.ptp(points, axis=0).max()))
        tolerance = extent * extent * 1e-8
        convex = np.all(crosses > tolerance) or np.all(crosses < -tolerance)
        if not convex or (clockwise and not np.all(crosses > tolerance)):
            raise ValueError(
                f"{name} must form a convex quadrilateral ordered far-left, far-right, "
                "near-right, near-left"
            )
        area = abs(float(np.sum(points[:, 0] * np.roll(points[:, 1], -1)
                                - points[:, 1] * np.roll(points[:, 0], -1)))) / 2.0
        if area <= max(tolerance, 1e-4):
            raise ValueError(f"{name} quadrilateral has insufficient area")

    def _compute_matrices(self) -> None:
        """Computes the forward H and inverse H_inv perspective transform matrices."""
        if cv2 is not None:
            self.H = cv2.getPerspectiveTransform(self.pixel_corners, self.court_corners_ft)
            self.H_inv = cv2.getPerspectiveTransform(self.court_corners_ft, self.pixel_corners)
        else:
            # Fallback pure-numpy DLT (Direct Linear Transform) solver for environments without cv2
            self.H = self._compute_homography_dlt(self.pixel_corners, self.court_corners_ft)
            self.H_inv = np.linalg.inv(self.H)
        if (not np.isfinite(self.H).all() or not np.isfinite(self.H_inv).all()
                or np.linalg.matrix_rank(self.H) != 3 or np.linalg.matrix_rank(self.H_inv) != 3):
            raise ValueError("Court corners do not define an invertible perspective transform")

    @staticmethod
    def _compute_homography_dlt(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
        """Pure numpy Direct Linear Transform algorithm for 4-point homography."""
        A = []
        for i in range(4):
            x, y = src[i][0], src[i][1]
            u, v = dst[i][0], dst[i][1]
            A.append([-x, -y, -1, 0, 0, 0, u * x, u * y, u])
            A.append([0, 0, 0, -x, -y, -1, v * x, v * y, v])
        A = np.array(A, dtype=np.float64)
        _, _, Vh = np.linalg.svd(A)
        H = Vh[-1].reshape((3, 3))
        return (H / H[2, 2]).astype(np.float32)

    def pixel_to_court(
        self, u: float, v: float, clip_to_bounds: bool = False
    ) -> Tuple[float, float]:
        """Maps single pixel coordinate (u, v) to top-down court coordinate (X_ft, Y_ft).

        Args:
            u: Horizontal pixel coordinate (X in frame)
            v: Vertical pixel coordinate (Y in frame)
            clip_to_bounds: If True, clamp result within court boundaries + safety margin

        Returns:
            (x_court_ft, y_court_ft) in feet
        """
        vec = np.array([u, v, 1.0], dtype=np.float32)
        res = self.H @ vec
        w = res[2]
        if abs(w) < 1e-7:
            w = 1e-7 if w >= 0 else -1e-7
        x_ft = float(res[0] / w)
        y_ft = float(res[1] / w)

        if clip_to_bounds:
            # Clamp with 5ft out-of-bounds buffer
            x_ft = max(-5.0, min(COURT_WIDTH_FT + 5.0, x_ft))
            y_ft = max(-5.0, min(COURT_LENGTH_FT + 5.0, y_ft))

        return x_ft, y_ft

    def court_to_pixel(self, x_ft: float, y_ft: float) -> Tuple[float, float]:
        """Maps court coordinate in feet (x_ft, y_ft) back to image pixel coordinate (u, v).

        Args:
            x_ft: Lateral position on court (0 to 20 ft)
            y_ft: Longitudinal position on court (0 to 44 ft)

        Returns:
            (u_pixel, v_pixel)
        """
        vec = np.array([x_ft, y_ft, 1.0], dtype=np.float32)
        res = self.H_inv @ vec
        w = res[2]
        if abs(w) < 1e-7:
            w = 1e-7 if w >= 0 else -1e-7
        return float(res[0] / w), float(res[1] / w)

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        """Batch transforms Nx2 pixel coordinates to Nx2 court feet coordinates.

        Args:
            points: Array of shape (N, 2)

        Returns:
            Transformed points array of shape (N, 2)
        """
        if len(points) == 0:
            return np.empty((0, 2), dtype=np.float32)

        pts_reshaped = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
        if cv2 is not None:
            transformed = cv2.perspectiveTransform(pts_reshaped, self.H)
            return transformed.reshape(-1, 2)
        else:
            N = len(points)
            homog_pts = np.hstack([points, np.ones((N, 1), dtype=np.float32)])  # N x 3
            res = (self.H @ homog_pts.T).T  # N x 3
            w = res[:, 2:3]
            w = np.where(np.abs(w) < 1e-7, 1e-7, w)
            return (res[:, :2] / w).astype(np.float32)

    def is_inside_court(self, x_ft: float, y_ft: float, margin_ft: float = 0.0) -> bool:
        """Checks if a court coordinate is inside the legal playing court."""
        return (
            -margin_ft <= x_ft <= COURT_WIDTH_FT + margin_ft
            and -margin_ft <= y_ft <= COURT_LENGTH_FT + margin_ft
        )

    def is_inside_kitchen(self, x_ft: float, y_ft: float, margin_ft: float = 0.0) -> bool:
        """Checks if a court coordinate falls inside the Non-Volley Zone (Kitchen).

        Kitchen runs across full 20 ft width and between y=15.0 ft and y=29.0 ft.
        """
        return (
            -margin_ft <= x_ft <= COURT_WIDTH_FT + margin_ft
            and (KITCHEN_NEAR_Y - margin_ft) <= y_ft <= (KITCHEN_FAR_Y + margin_ft)
        )

    def get_court_zone(self, x_ft: float, y_ft: float) -> str:
        """Categorizes a court position into standard pickleball zones."""
        if not self.is_inside_court(x_ft, y_ft, margin_ft=2.0):
            return "out_of_bounds"

        if y_ft < KITCHEN_NEAR_Y:
            if y_ft < 8.0:
                return "near_baseline"
            return "near_transition"
        elif y_ft <= NET_Y:
            return "near_kitchen"
        elif y_ft <= KITCHEN_FAR_Y:
            return "far_kitchen"
        elif y_ft <= 36.0:
            return "far_transition"
        else:
            return "far_baseline"

    def to_dict(self) -> dict:
        """Serializes calibration data to dictionary."""
        return {
            "court_corners_pixel": self.pixel_corners.tolist(),
            "court_corners_feet": self.court_corners_ft.tolist(),
            "corner_order": COURT_CORNERS_ORDER,
            "court_dimensions_ft": [COURT_WIDTH_FT, COURT_LENGTH_FT],
            "image_size": list(self.image_size) if self.image_size else None,
            "homography_matrix": self.H.tolist(),
            "calibration": self.metadata,
        }

    def save_to_file(self, filepath: Union[str, Path]) -> None:
        """Saves calibration to JSON file."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load_from_file(cls, filepath: Union[str, Path]) -> "CourtHomography":
        """Loads calibration from JSON file and recomputes homography."""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Calibration file not found at: {path}")

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        pixel_corners = data["court_corners_pixel"]
        court_corners_ft = data.get("court_corners_feet", COURT_CORNERS_FEET)
        image_size = tuple(data["image_size"]) if data.get("image_size") else None

        return cls(
            pixel_corners=pixel_corners,
            court_corners_ft=court_corners_ft,
            image_size=image_size,
            metadata=data.get("calibration", {}),
        )

