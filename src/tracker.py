"""2D Kalman Filter and Trajectory Smoothing Engine for Pickleball Tracking.

Uses filterpy.kalman.KalmanFilter to track the pickleball in 2D space,
interpolate missing detections across frames (e.g. net occlusions and motion blur),
and provide smoothed position and velocity estimates.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np
from filterpy.common import Q_discrete_white_noise
from filterpy.kalman import KalmanFilter

from src.config import (
    COURT_MAX_X_FT,
    COURT_MAX_Y_FT,
    COURT_MIN_X_FT,
    COURT_MIN_Y_FT,
    DEFAULT_FPS,
    KALMAN_MEASUREMENT_NOISE_STD,
    KALMAN_PROCESS_NOISE_STD,
    KALMAN_VELOCITY_DAMPING,
    MAX_BALL_JUMP_PIXELS,
    MAX_MISSED_BALL_FRAMES,
)


@dataclass
class BallPoint:
    """Represents a single-frame ball observation or interpolated estimate."""

    frame_idx: int
    pixel_x: float
    pixel_y: float
    court_x_ft: Optional[float] = None
    court_y_ft: Optional[float] = None
    velocity_px_x: float = 0.0
    velocity_px_y: float = 0.0
    velocity_ft_x: float = 0.0
    velocity_ft_y: float = 0.0
    # Airborne speed is unavailable until a complete 3D flight has been fitted.
    speed_mph: Optional[float] = None
    is_interpolated: bool = False
    confidence: float = 1.0
    measurement_pixel: Optional[Tuple[float, float]] = None
    observation_pixel: Optional[Tuple[float, float]] = None
    reference_pixel: Optional[Tuple[float, float]] = None
    camera_projection: Optional[np.ndarray] = None
    camera_source: str = "unavailable"
    timestamp_s: Optional[float] = None
    timing_source: str = "nominal_fps"
    world_x_ft: Optional[float] = None
    world_y_ft: Optional[float] = None
    height_ft: Optional[float] = None
    velocity_ft_z: Optional[float] = None
    speed_confidence: float = 0.0
    speed_reason: str = "Awaiting calibrated 3D flight"
    track_generation: int = 0


@dataclass
class PlayerDetection:
    """Represents a player bounding box in a single frame."""

    tracker_id: int
    bbox_xyxy: Tuple[float, float, float, float]  # [x1, y1, x2, y2]
    confidence: float = 1.0
    feet_pixel: Optional[Tuple[float, float]] = None
    court_feet: Optional[Tuple[float, float]] = None
    in_kitchen: bool = False


class BallKalmanTracker:
    """2D Kalman Filter tracker for the pickleball.

    Tracks state [x, y, vx, vy]^T with constant-velocity dynamics and discrete white noise.
    Interpolates occluded or missed detections up to max_missed_frames.
    """

    def __init__(
        self,
        fps: float = DEFAULT_FPS,
        process_noise_std: float = KALMAN_PROCESS_NOISE_STD,
        measurement_noise_std: float = KALMAN_MEASUREMENT_NOISE_STD,
        max_missed_frames: int = MAX_MISSED_BALL_FRAMES,
        max_jump_pixels: float = MAX_BALL_JUMP_PIXELS,
    ):
        self.fps = fps
        self.dt = 1.0 / fps
        self.process_noise_std = process_noise_std
        self.measurement_noise_std = measurement_noise_std
        self.max_missed_frames = max_missed_frames
        self.max_jump_pixels = max_jump_pixels

        self.kf: Optional[KalmanFilter] = None
        self.is_initialized = False
        self.missed_frames_count = 0
        self.consecutive_hits = 0
        self.is_confirmed = False
        self.is_fresh_track = True
        self.last_frame_idx: Optional[int] = None
        self.trajectory: List[BallPoint] = []
        self.track_generation = 0
        self._init_kalman()

    def _init_kalman(self) -> None:
        """Configures the 4-state constant velocity Kalman Filter."""
        self.kf = KalmanFilter(dim_x=4, dim_z=2)
        # State transition matrix F: x_t = x_t-1 + vx * dt
        self.kf.F = np.array(
            [
                [1.0, 0.0, self.dt, 0.0],
                [0.0, 1.0, 0.0, self.dt],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

        # Measurement matrix H: observe x and y
        self.kf.H = np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        )

        # Measurement noise covariance R
        r_var = self.measurement_noise_std**2
        self.kf.R = np.eye(2, dtype=np.float64) * r_var

        self._set_process_noise(self.dt)

        # A first position tells us almost nothing about image velocity.
        # Giving velocity only 50 px^2/s^2 froze the filter behind a fast ball.
        self.kf.P = np.diag([16., 16., 1200.**2, 1200.**2])

    def _set_process_noise(self, step_dt: float) -> None:
        """Use the same elapsed time for uncertainty growth and prediction."""
        q_1d = Q_discrete_white_noise(
            dim=2, dt=step_dt, var=self.process_noise_std**2
        )
        self.kf.Q = np.zeros((4, 4), dtype=np.float64)
        # Correct decoupled 2D state grouping for state vector [x, y, vx, vy]^T:
        # Block [0, 2] corresponds to x and vx
        self.kf.Q[np.ix_([0, 2], [0, 2])] = q_1d
        # Block [1, 3] corresponds to y and vy
        self.kf.Q[np.ix_([1, 3], [1, 3])] = q_1d

    @staticmethod
    def _camera_state(state, covariance, transform):
        """Transport position, velocity and covariance into the new camera view.

        The local homography Jacobian maps velocity. Its position derivative is
        also included when transporting the full state covariance through zoom,
        rotation or perspective motion.
        """
        matrix = np.asarray(transform, dtype=np.float64)
        if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
            raise ValueError("Camera transform must be a finite 3x3 matrix")
        position, velocity = state[:2, 0], state[2:, 0]
        denominator = float(matrix[2, :2] @ position + matrix[2, 2])
        if abs(denominator) < 1e-8 or np.linalg.matrix_rank(matrix) < 3:
            raise ValueError("Camera transform is singular at the ball position")
        warped = cv2.perspectiveTransform(position.reshape(1, 1, 2), matrix)[0, 0]
        projective = matrix[2, :2]
        jacobian = (matrix[:2, :2] - np.outer(warped, projective))/denominator
        warped_velocity = jacobian @ velocity
        velocity_position = -(np.outer(warped_velocity, projective)
                              + (projective @ velocity)*jacobian)/denominator
        transport = np.block([[jacobian, np.zeros((2, 2))],
                              [velocity_position, jacobian]])
        new_state = np.r_[warped, warped_velocity].reshape(4, 1)
        new_covariance = transport @ covariance @ transport.T
        if not np.isfinite(new_state).all() or not np.isfinite(new_covariance).all():
            raise ValueError("Camera transform produced an invalid ball state")
        return new_state, (new_covariance + new_covariance.T)*.5

    def predict_position(self, frame_idx: int, camera_transform=None):
        """Return a current-view search center without advancing the filter."""
        if not self.is_initialized:
            return None
        state = self.kf.x
        if camera_transform is not None:
            try:
                state, _ = self._camera_state(state, self.kf.P, camera_transform)
            except (ValueError, cv2.error):
                return None
        elapsed = max(1, frame_idx-self.last_frame_idx)*self.dt
        point = state[:2, 0] + elapsed*state[2:, 0]
        return float(point[0]), float(point[1])

    def reset(self) -> None:
        """Resets the Kalman Filter tracker state."""
        self.track_generation += 1
        self.is_initialized = False
        self.missed_frames_count = 0
        self.consecutive_hits = 0
        self.is_confirmed = False
        self.is_fresh_track = True
        self.last_frame_idx = None
        self._init_kalman()

    def update(
        self,
        frame_idx: int,
        detection: Optional[Tuple[float, float]],
        confidence: float = 1.0,
        homography=None,
        dt: Optional[float] = None,
        camera_transform: Optional[np.ndarray] = None,
    ) -> Optional[BallPoint]:
        """Processes a new video frame with an optional ball detection.

        Args:
            frame_idx: Index of current video frame
            detection: (pixel_x, pixel_y) if detected, or None if missed
            confidence: Detector confidence score [0, 1]
            homography: Optional CourtHomography instance to compute court coordinates
            dt: Optional elapsed time in seconds (defaults to frame step / fps)
            camera_transform: Optional 3x3 homography mapping previous frame view to current view

        Returns:
            BallPoint with estimated position and speed, or None if tracker is uninitialized/lost.
        """
        MIN_INIT_CONFIDENCE = 0.25
        MIN_UPDATE_CONFIDENCE = 0.15

        # If uninitialized, require confident detection to initialize
        if not self.is_initialized:
            if detection is not None and confidence >= MIN_INIT_CONFIDENCE:
                px, py = detection
                self.kf.x = np.array([px, py, 0.0, 0.0], dtype=np.float64).reshape(-1, 1)
                self.is_initialized = True
                self.consecutive_hits = 1
                self.is_confirmed = False
                self.missed_frames_count = 0
                self.is_fresh_track = True
                self.last_frame_idx = frame_idx

                pt = self._create_ball_point(
                    frame_idx=frame_idx,
                    px=px,
                    py=py,
                    vx=0.0,
                    vy=0.0,
                    is_interpolated=False,
                    confidence=confidence,
                    homography=homography,
                    is_fresh_track=True,
                )
                pt.measurement_pixel = (float(px), float(py))
                self.trajectory.append(pt)
                self.is_fresh_track = False
                return pt
            return None

        # Move the complete state into the current camera coordinate system.
        if camera_transform is not None and not np.allclose(camera_transform, np.eye(3)):
            try:
                self.kf.x, self.kf.P = self._camera_state(self.kf.x, self.kf.P, camera_transform)
            except (ValueError, cv2.error):
                self.reset()
                return None

        # Determine elapsed dt for accurate prediction
        if dt is not None and np.isfinite(dt) and dt > 0:
            step_dt = float(dt)
        elif self.last_frame_idx is not None:
            gap = max(1, frame_idx - self.last_frame_idx)
            step_dt = gap * self.dt
        else:
            step_dt = self.dt
        self.last_frame_idx = frame_idx

        # Predict step using true elapsed dt
        self.kf.F[0, 2] = step_dt
        self.kf.F[1, 3] = step_dt
        self._set_process_noise(step_dt)
        self.kf.predict()
        pred_x = float(self.kf.x[0, 0])
        pred_y = float(self.kf.x[1, 0])
        vx_px = float(self.kf.x[2, 0])
        vy_px = float(self.kf.x[3, 0])

        # If predicted position drifts off frame, stop coasting immediately
        if homography is not None and homography.image_size:
            w_img, h_img = homography.image_size
            if pred_x < -20.0 or pred_x > w_img + 20.0 or pred_y < -20.0 or pred_y > h_img + 20.0:
                self.reset()
                return None

        # Validate measurement using distance gating and confidence
        is_valid_detection = False
        if detection is not None and confidence >= MIN_UPDATE_CONFIDENCE:
            meas_x, meas_y = detection
            dist_to_pred = np.hypot(meas_x - pred_x, meas_y - pred_y)

            # Adaptive gating: Allow wider jump if consecutive frames were missed to re-acquire hits
            gate_dist = self.max_jump_pixels if self.missed_frames_count < 2 else self.max_jump_pixels * 1.6
            if dist_to_pred <= gate_dist:
                is_valid_detection = True

        if is_valid_detection:
            # Scale measurement noise covariance inversely with detection confidence
            meas_conf = max(0.1, min(1.0, float(confidence)))
            r_var = (self.measurement_noise_std**2) / meas_conf
            self.kf.R = np.eye(2, dtype=np.float64) * r_var

            self.kf.update(np.array([meas_x, meas_y], dtype=np.float64).reshape(-1, 1))
            curr_x = float(self.kf.x[0, 0])
            curr_y = float(self.kf.x[1, 0])
            curr_vx = float(self.kf.x[2, 0])
            curr_vy = float(self.kf.x[3, 0])

            self.consecutive_hits += 1
            if self.consecutive_hits >= 2:
                self.is_confirmed = True

            was_fresh = self.is_fresh_track or (self.missed_frames_count >= 3)
            self.missed_frames_count = 0
            self.is_fresh_track = False

            pt = self._create_ball_point(
                frame_idx=frame_idx,
                # Render the actual accepted observation. The filter state is
                # used to predict/search, not to move a measured ball off target.
                px=float(meas_x),
                py=float(meas_y),
                vx=curr_vx,
                vy=curr_vy,
                is_interpolated=False,
                confidence=confidence,
                homography=homography,
                is_fresh_track=was_fresh,
            )
            pt.measurement_pixel = (float(meas_x), float(meas_y))
            self.trajectory.append(pt)
            return pt

        # No valid detection:
        # If track was only tentative (single hit without confirmation), reset immediately to avoid false ghost tracks
        if not self.is_confirmed:
            self.reset()
            return None

        # Interpolate confirmed track using Kalman prediction
        self.missed_frames_count += 1
        if self.missed_frames_count <= self.max_missed_frames:
            # Damp velocity on consecutive missed frames to avoid runaway coasting
            self.kf.x[2, 0] *= KALMAN_VELOCITY_DAMPING
            self.kf.x[3, 0] *= KALMAN_VELOCITY_DAMPING

            # Interpolated frame using Kalman prediction
            pt = self._create_ball_point(
                frame_idx=frame_idx,
                px=pred_x,
                py=pred_y,
                vx=vx_px,
                vy=vy_px,
                is_interpolated=True,
                confidence=max(0.1, confidence * (0.8**self.missed_frames_count)),
                homography=homography,
                is_fresh_track=False,
            )
            self.trajectory.append(pt)
            return pt
        else:
            # Lost track: Reset filter to prevent divergent predictions
            self.reset()
            return None

    def _create_ball_point(
        self,
        frame_idx: int,
        px: float,
        py: float,
        vx: float,
        vy: float,
        is_interpolated: bool,
        confidence: float,
        homography=None,
        is_fresh_track: bool = False,
    ) -> BallPoint:
        """Helper to create BallPoint and project to court feet coordinates."""
        court_x_ft = None
        court_y_ft = None
        vel_ft_x = 0.0
        vel_ft_y = 0.0
        speed_mph = None

        if homography is not None:
            c_x, c_y = homography.pixel_to_court(px, py)
            # Court envelope guard: reject horizon singularities and extreme off-court runaway values
            if COURT_MIN_X_FT <= c_x <= COURT_MAX_X_FT and COURT_MIN_Y_FT <= c_y <= COURT_MAX_Y_FT:
                court_x_ft = c_x
                court_y_ft = c_y

            # These are ray intersections with the floor, retained for legacy
            # overlays only. Differentiating them cannot measure airborne speed.

        return BallPoint(
            frame_idx=frame_idx,
            pixel_x=px,
            pixel_y=py,
            court_x_ft=court_x_ft,
            court_y_ft=court_y_ft,
            velocity_px_x=vx,
            velocity_px_y=vy,
            velocity_ft_x=vel_ft_x,
            velocity_ft_y=vel_ft_y,
            speed_mph=speed_mph,
            is_interpolated=is_interpolated,
            confidence=confidence,
            track_generation=self.track_generation,
        )

    def get_recent_trail(self, length: int = 15) -> List[Tuple[float, float]]:
        """Returns the last N pixel coordinates for rendering a motion trail."""
        trail = []
        for pt in reversed(self.trajectory[-length:]):
            if pt.track_generation != self.track_generation or pt.is_interpolated:
                break
            trail.append((pt.pixel_x, pt.pixel_y))
        return list(reversed(trail))

    def get_recent_court_trail(self, length: int = 15) -> List[Tuple[float, float]]:
        """Returns the last N court coordinates for radar overlay rendering."""
        trail = []
        for pt in self.trajectory[-length:]:
            if pt.court_x_ft is not None and pt.court_y_ft is not None:
                trail.append((pt.court_x_ft, pt.court_y_ft))
        return trail

