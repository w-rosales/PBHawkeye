"""Conservative monocular flight reconstruction, in feet and seconds.

Court geometry supplies camera pose. Gravity supplies metric depth along each
observed ray. This is a model-based estimate, not a direct height measurement.
No predicted/Kalman-only samples are treated as independent observations.
"""

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.signal import find_peaks

from src.config import FPS_TO_MPH, MAX_BALL_SPEED_MPH

GRAVITY = np.array([0., 0., -32.174])


@dataclass
class CameraModel:
    matrix: np.ndarray
    distortion: np.ndarray
    projection: np.ndarray
    center: np.ndarray
    source: str
    reprojection_error_px: float

    @classmethod
    def from_court(cls, court):
        """Recover pose; optionally use calibration.camera_intrinsics metadata.

        Automatic intrinsics assume square pixels, zero skew/distortion and a
        centered principal point. Reject views that cannot support those assumptions.
        Supplied intrinsics must describe this frame's resolution and zoom.
        """
        if court is None or court.image_size is None:
            raise ValueError("Camera calibration needs frame dimensions")
        width, height = court.image_size
        settings = court.metadata.get("camera_intrinsics")
        distortion = np.zeros(5, dtype=float)
        if settings is not None:
            matrix = np.asarray(settings["matrix"], dtype=float)
            distortion = np.asarray(settings.get("distortion", [0.] * 5), dtype=float).ravel()
            if (matrix.shape != (3, 3) or not np.isfinite(matrix).all()
                    or matrix[0, 0] <= 0 or matrix[1, 1] <= 0
                    or not np.allclose(matrix[2], [0, 0, 1])
                    or abs(matrix[0, 1]) > 1e-8 or abs(matrix[1, 0]) > 1e-8
                    or len(distortion) not in (4, 5, 8, 12, 14)
                    or not np.isfinite(distortion).all()):
                raise ValueError("Invalid camera intrinsics")
            source = "supplied_intrinsics"
        else:
            # h1,h2 are orthogonal equal-length rotation columns after K^-1.
            centered = np.array([[1., 0., -width/2], [0., 1., -height/2], [0., 0., 1.]]) @ court.H_inv
            a, b = centered[:, 0], centered[:, 1]
            lhs = np.array([a[2]*b[2], a[2]**2-b[2]**2])
            rhs = -np.array([a[:2] @ b[:2], a[:2] @ a[:2]-b[:2] @ b[:2]])
            if np.linalg.norm(lhs) < 1e-12:
                raise ValueError("Court view does not constrain focal length")
            focal_squared = float(lhs @ rhs / (lhs @ lhs))
            if not (.3*width)**2 < focal_squared < (5*width)**2:
                raise ValueError("Court view does not support a plausible focal length")
            focal = np.sqrt(focal_squared)
            matrix = np.array([[focal, 0., width/2], [0., focal, height/2], [0., 0., 1.]])
            source = "estimated_intrinsics"
        world = np.column_stack([court.court_corners_ft, np.zeros(4)]).astype(float)
        image = court.pixel_corners.astype(float)
        ok, rotation, translation = cv2.solvePnP(world, image, matrix, distortion,
                                                flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok:
            raise ValueError("Camera pose could not be recovered")
        rotation_matrix, _ = cv2.Rodrigues(rotation)
        center = (-rotation_matrix.T @ translation).ravel()
        projected, _ = cv2.projectPoints(world, rotation, translation, matrix, distortion)
        error = float(np.sqrt(np.mean(np.sum((projected.reshape(-1, 2)-image)**2, axis=1))))
        depth = (world @ rotation_matrix.T + translation.ravel())[:, 2]
        if (not np.isfinite(center).all() or not 3. < center[2] < 60.
                or np.any(depth <= 0) or error > max(2., width*.002)):
            raise ValueError("Court geometry is inconsistent with a usable camera pose")
        return cls(matrix, distortion, matrix @ np.column_stack([rotation_matrix, translation]),
                   center, source, error)

    def undistort(self, xy):
        return cv2.undistortPoints(np.asarray(xy, float).reshape(1, 1, 2), self.matrix,
                                   self.distortion, P=self.matrix).reshape(2)


class ObservationRecorder:
    """Attach raw detections and per-frame geometry without changing the 2D tracker."""

    def __init__(self, reference_court, fps):
        if not np.isfinite(fps) or fps <= 0:
            raise ValueError("Video FPS must be finite and positive")
        self.reference_court = reference_court
        self.fps = fps
        self._key = None
        self._camera = None
        self._reason = "Camera unavailable"

    def record(self, point, court, timestamp_s=None):
        if point is None:
            return
        point.timestamp_s = timestamp_s if timestamp_s is not None else point.frame_idx/self.fps
        point.timing_source = "video_timestamps" if timestamp_s is not None else "nominal_fps"
        if court is None:
            point.speed_reason = "Court tracking unavailable"
            return
        key = court.pixel_corners.tobytes()
        if key != self._key:
            self._key = key
            try:
                self._camera = CameraModel.from_court(court)
            except (ValueError, KeyError, TypeError, cv2.error, np.linalg.LinAlgError) as exc:
                self._camera = None
                self._reason = str(exc)
        if point.is_interpolated or point.measurement_pixel is None:
            point.speed_reason = "Predicted position is not a ball observation"
            return
        if self._camera is None:
            point.speed_reason = self._reason
            return
        point.observation_pixel = tuple(self._camera.undistort(point.measurement_pixel))
        point.camera_projection = self._camera.projection
        point.camera_source = self._camera.source
        # Only used for change-point detection, never for distance or speed.
        reference_xy = cv2.perspectiveTransform(
            np.asarray(point.measurement_pixel, float).reshape(1, 1, 2),
            self.reference_court.H_inv @ court.H).reshape(2)
        point.reference_pixel = tuple(reference_xy)
        point.speed_reason = "Awaiting flight reconstruction"


@dataclass
class FlightEstimate:
    status: str = "unavailable"
    reason: str = "Insufficient observations"
    average_mph: Optional[float] = None
    peak_mph: Optional[float] = None
    distance_ft: Optional[float] = None
    confidence: float = 0.
    reprojection_error_px: Optional[float] = None
    relative_uncertainty: Optional[float] = None
    positions: Optional[np.ndarray] = None
    velocities: Optional[np.ndarray] = None
    camera_source: str = "unavailable"
    drag_rate: Optional[float] = None
    boundary_start_s: Optional[float] = None
    boundary_end_s: Optional[float] = None


def _motion(initial, times, drag):
    """Linear-drag flight approximation; gravity fixed in world coordinates.

    A range of drag rates is tested rather than pretending drag is known from
    a few pixels. Spin and detailed quadratic aerodynamics are not modeled.
    """
    t = np.asarray(times)[:, None]
    if drag == 0:
        a, b = t, .5*t*t
    else:
        a = -np.expm1(-drag*t)/drag
        b = (t-a)/drag
    return initial[:3]+a*initial[3:]+b*GRAVITY, np.exp(-drag*t)*initial[3:]+a*GRAVITY


def _project(projections, positions):
    image = np.einsum("nij,nj->ni", projections, np.column_stack([positions, np.ones(len(positions))]))
    denominator = np.where(np.abs(image[:, 2:]) < 1e-6, 1e-6, image[:, 2:])
    return image[:, :2]/denominator, image[:, 2]


def fit_flight(points, fps, frame_stride=None):
    """Fit an observed flight; reject weak geometry, gaps and model ambiguity."""
    result = FlightEstimate()
    if not np.isfinite(fps) or fps <= 0:
        result.reason = "Video FPS must be finite and positive"
        return result
    if len(points) < 8:
        return result
    if any(p.is_interpolated or p.camera_projection is None or p.observation_pixel is None for p in points):
        result.reason = "Flight contains missing camera geometry or predicted observations"
        return result
    if len({p.timing_source for p in points}) != 1:
        result.reason = "Inconsistent video timestamps"
        return result
    times = np.array([p.timestamp_s if p.timestamp_s is not None else p.frame_idx/fps for p in points])
    times -= times[0]
    gaps = np.diff(times)
    if not np.isfinite(times).all() or np.any(gaps <= 0) or np.max(gaps) > .18:
        result.reason = "Invalid timestamps or a tracking gap"
        return result
    if not .35 <= times[-1] <= 3.:
        result.reason = "Flight is too short or contains unresolved contacts"
        return result
    stride = frame_stride or max(1, min(np.diff([p.frame_idx for p in points])))
    expected_count = (points[-1].frame_idx-points[0].frame_idx)/stride+1
    if len(points)/expected_count < .8:
        result.reason = "Too few actual ball observations across this flight"
        return result
    projections = np.array([p.camera_projection for p in points], float)
    observed = np.array([p.observation_pixel for p in points], float)
    if (projections.shape != (len(points), 3, 4) or not np.isfinite(projections).all()
            or not np.isfinite(observed).all()):
        result.reason = "Invalid camera observations"
        return result
    confidence = np.sqrt(np.clip([p.confidence for p in points], .1, 1.))[:, None]
    fits = []
    # Fixed alternatives make aerodynamic uncertainty visible and keep the
    # optimization small (six unknowns per flight, not one network per frame).
    for drag in (0., .4, .8, 1.2):
        t = times[:, None]
        a = t if drag == 0 else -np.expm1(-drag*t)/drag
        b = .5*t*t if drag == 0 else (t-a)/drag
        rays = projections[:, :2, :3]-observed[:, :, None]*projections[:, 2:3, :3]
        system = np.concatenate([rays, rays*a[:, :, None]], axis=2).reshape(-1, 6)
        rhs = -(projections[:, :2, 3]-observed*projections[:, 2:3, 3]
                + np.einsum("nij,nj->ni", rays, b*GRAVITY))
        initial, _, rank, _ = np.linalg.lstsq(system, rhs.ravel(), rcond=None)
        if rank < 6:
            continue

        def residual(values):
            positions, _ = _motion(values, times, drag)
            pixels, _ = _project(projections, positions)
            return ((pixels-observed)*confidence).ravel()

        try:
            fit = least_squares(residual, initial, loss="soft_l1", f_scale=1.5, max_nfev=60)
        except (ValueError, np.linalg.LinAlgError):
            continue
        positions, velocities = _motion(fit.x, times, drag)
        pixels, depths = _project(projections, positions)
        errors = np.linalg.norm(pixels-observed, axis=1)
        rms = float(np.sqrt(np.mean(errors**2)))
        speeds = np.linalg.norm(velocities, axis=1)*FPS_TO_MPH
        # Reject, never clip. A good pixel fit alone is not physical validation.
        if (not fit.success or not np.isfinite(positions).all() or rms > 2.5
                or np.quantile(errors, .95) > 4.5 or np.any(depths <= 0)
                or np.min(positions[:, 2]) < -.15 or np.max(positions[:, 2]) > 30.
                or np.any(positions[:, 0] < -15) or np.any(positions[:, 0] > 35)
                or np.any(positions[:, 1] < -20) or np.any(positions[:, 1] > 64)
                or np.max(speeds) > MAX_BALL_SPEED_MPH or np.max(speeds) < 1.):
            continue
        dense_t = np.linspace(0, times[-1], 160)
        dense_positions, dense_velocity = _motion(fit.x, dense_t, drag)
        distance = float(np.sum(np.linalg.norm(np.diff(dense_positions, axis=0), axis=1)))
        average = distance/times[-1]*FPS_TO_MPH
        # Local pixel-noise uncertainty, including the weak depth direction.
        _, singular, vt = np.linalg.svd(fit.jac, full_matrices=False)
        if singular[-1] < 1e-7:
            continue
        covariance = (vt.T / singular**2) @ vt * max(1.5, rms)**2
        gradient = np.zeros(6)
        for j in range(6):
            step = np.zeros(6)
            step[j] = .001
            _, plus = _motion(fit.x+step, dense_t, drag)
            _, minus = _motion(fit.x-step, dense_t, drag)
            gradient[j] = (np.mean(np.linalg.norm(plus, axis=1))-np.mean(np.linalg.norm(minus, axis=1)))/.002*FPS_TO_MPH
        relative = 1.96*np.sqrt(max(0., float(gradient @ covariance @ gradient)))/average
        fits.append((rms, relative, average, float(np.max(np.linalg.norm(dense_velocity, axis=1))*FPS_TO_MPH),
                     distance, positions, velocities, drag))
    if not fits:
        result.reason = "No physically plausible 3D flight fits the observations"
        return result
    fits.sort(key=lambda item: item[0])
    best = fits[0]
    alternatives = [f for f in fits if f[0] <= best[0]+.75]
    model_spread = (max(f[2] for f in alternatives)-min(f[2] for f in alternatives))/best[2]
    uncertainty = max(best[1], model_spread)
    result.relative_uncertainty = float(uncertainty)
    result.reprojection_error_px = best[0]
    if uncertainty > .35:
        result.reason = "Depth or aerodynamic uncertainty is too large"
        return result
    source = "estimated_intrinsics" if any(p.camera_source != "supplied_intrinsics" for p in points) else "supplied_intrinsics"
    return FlightEstimate(status="estimated", reason="Average over observed flight; endpoints may be partial",
                          average_mph=best[2], peak_mph=best[3], distance_ft=best[4],
                          confidence=float(min(.75 if source == "estimated_intrinsics" else .9,
                                               (1-uncertainty)*np.exp(-best[0]/5))),
                          reprojection_error_px=best[0], relative_uncertainty=float(uncertainty),
                          positions=best[5], velocities=best[6], camera_source=source, drag_rate=best[7])


def flight_segments(trajectory, fps, frame_stride=None):
    """Split raw observations at gaps and candidate paddle/bounce discontinuities.

    Boundaries are candidates, not verified contact classifications. A segment
    still has to pass the independent 3D reprojection/uncertainty checks.
    """
    runs, current = [], []
    stride = frame_stride or 1
    for index, point in enumerate(trajectory):
        valid = (not point.is_interpolated and point.observation_pixel is not None
                 and np.isfinite(point.observation_pixel).all()
                 and point.camera_projection is not None and point.confidence >= .25
                 and (point.timestamp_s is None or np.isfinite(point.timestamp_s))
                 and (point.reference_pixel is None or np.isfinite(point.reference_pixel).all()))
        if current:
            previous = trajectory[current[-1]]
            gap = point.frame_idx-previous.frame_idx
            elapsed = ((point.timestamp_s-previous.timestamp_s)
                       if point.timestamp_s is not None and previous.timestamp_s is not None else gap/fps)
            valid_gap = (0 < gap and 0 < elapsed <= max(stride/fps, .10)+1e-6
                         and gap % stride == 0 and point.track_generation == previous.track_generation
                         and point.timing_source == previous.timing_source)
            # A brief occlusion may be spanned using measured endpoints. Never
            # add predicted samples, cross a tracker reset, or bridge lost geometry.
            short_occlusion = (point.is_interpolated and
                               point.speed_reason == "Predicted position is not a ball observation")
            if not valid and short_occlusion and valid_gap:
                continue
            if not valid or not valid_gap:
                runs.append(current)
                current = []
        if valid:
            current.append(index)
    if current:
        runs.append(current)
    segments = []
    for run in runs:
        points = [trajectory[i] for i in run]
        times = np.array([p.timestamp_s if p.timestamp_s is not None else p.frame_idx/fps for p in points])
        if len(points) < 10 or np.any(np.diff(times) <= 0):
            segments.append((run, fit_flight(points, fps, frame_stride)))
            continue
        xy = np.array([p.reference_pixel if p.reference_pixel is not None else p.observation_pixel for p in points])
        changes = np.zeros(len(points))
        for i in range(4, len(points)-4):
            left_t, right_t = times[i-4:i+1]-times[i], times[i:i+5]-times[i]
            left = np.polynomial.polynomial.polyfit(left_t, xy[i-4:i+1], 2)
            right = np.polynomial.polynomial.polyfit(right_t, xy[i:i+5], 2)
            jump = np.linalg.norm(left[1]-right[1])
            noise_limit = 6./min(times[i]-times[i-4], times[i+4]-times[i])
            limit = max(noise_limit, .45*max(np.linalg.norm(left[1]), np.linalg.norm(right[1])))
            changes[i] = jump/limit
        peaks, _ = find_peaks(changes, height=1., distance=max(4, int(.18/np.median(np.diff(times)))))
        boundaries = [0, *map(int, peaks), len(points)-1]
        for start, end in zip(boundaries, boundaries[1:]):
            # The velocity discontinuity localizes an event only approximately.
            # Sharing that sample between fits can put a pre-impact observation
            # into the outgoing flight (especially across a paddle occlusion).
            # Exclude a short transition interval; do not relax fit quality.
            indices = [run[i] for i in range(start, end+1)
                       if (start == 0 or times[i] >= times[start]+.05)
                       and (end == len(points)-1 or times[i] <= times[end]-.05)]
            if not indices:
                continue
            estimate = fit_flight([trajectory[i] for i in indices], fps, frame_stride)
            estimate.boundary_start_s = float(times[start]) if start != 0 else None
            estimate.boundary_end_s = float(times[end]) if end != len(points)-1 else None
            segments.append((indices, estimate))
    # A shared image pixel can still produce different depths in independent
    # fits. Such flights cannot both describe a continuous physical ball path.
    inconsistent = set()
    for i in range(1, len(segments)):
        previous_indices, previous_fit = segments[i-1]
        indices, fit = segments[i]
        if previous_fit.status != "estimated" or fit.status != "estimated":
            continue
        shared_event = (previous_fit.boundary_end_s is not None and
                        previous_fit.boundary_end_s == fit.boundary_start_s)
        if shared_event:
            previous_end = trajectory[previous_indices[-1]]
            next_start = trajectory[indices[0]]
            t0 = previous_end.timestamp_s if previous_end.timestamp_s is not None else previous_end.frame_idx/fps
            t1 = next_start.timestamp_s if next_start.timestamp_s is not None else next_start.frame_idx/fps
            boundary = previous_fit.boundary_end_s
            incoming, _ = _motion(np.r_[previous_fit.positions[-1], previous_fit.velocities[-1]],
                                  [boundary-t0], previous_fit.drag_rate or 0.)
            outgoing, _ = _motion(np.r_[fit.positions[0], fit.velocities[0]],
                                  [boundary-t1], fit.drag_rate or 0.)
            if np.linalg.norm(incoming[0]-outgoing[0]) > 2.:
                inconsistent.update([i-1, i])
    for i in inconsistent:
        indices, _ = segments[i]
        segments[i] = (indices, FlightEstimate(reason="Adjacent flights disagree on depth at their shared boundary"))
    return segments


class LiveFlightSpeed:
    """Bounded live reconstruction with exactly the final-report quality gates.

    The display is the last qualifying observed flight average, not instantaneous
    speed. A stale estimate expires instead of following the ball indefinitely.
    """

    def __init__(self, fps, frame_stride=1):
        self.fps = fps
        self.frame_stride = frame_stride
        self._last_check_frame = None
        self._last_valid_end = None
        self._last_valid_generation = None
        self._last_estimate = None
        self.peak_mph = None
        self.status = "pending"
        self.reason = "Collecting actual ball observations for a 3D speed estimate"

    def update(self, trajectory, frame_idx, court_ok=True):
        if not court_ok:
            self._last_estimate = None
            self.status = "unavailable"
            self.reason = "MPH unavailable: court tracking is uncertain"
            return None
        generation = trajectory[-1].track_generation if trajectory else None
        if (self._last_valid_end is not None and
                (frame_idx-self._last_valid_end > 1.5*self.fps
                 or generation != self._last_valid_generation)):
            self._last_estimate = None
        # At most three small reconstructions per source second. Sampling and
        # timing remain source-frame based, independent of processing speed.
        interval = max(self.frame_stride, round(self.fps/3))
        if self._last_check_frame is None or frame_idx-self._last_check_frame >= interval:
            self._last_check_frame = frame_idx
            recent = []
            for point in reversed(trajectory):
                if point.frame_idx < frame_idx-3*self.fps or len(recent) >= 180:
                    break
                recent.append(point)
            recent.reverse()
            segments = flight_segments(recent, self.fps, self.frame_stride)
            accepted = [(indices, estimate) for indices, estimate in segments
                        if estimate.status == "estimated" and
                        recent[indices[-1]].track_generation == generation and
                        frame_idx-recent[indices[-1]].frame_idx <= 1.5*self.fps]
            if accepted:
                indices, estimate = accepted[-1]
                self._last_estimate = estimate
                self._last_valid_end = recent[indices[-1]].frame_idx
                self._last_valid_generation = generation
                self.peak_mph = max(self.peak_mph or 0., estimate.peak_mph)
            elif self._last_estimate is None:
                self.status = "unavailable" if len(recent) >= 8 else "pending"
                if segments and segments[-1][1].status == "estimated":
                    self.reason = "MPH unavailable: previous estimate expired or belongs to an earlier track; collecting a new flight"
                else:
                    self.reason = ("MPH unavailable: "+segments[-1][1].reason if segments else
                                   "MPH unavailable: waiting for continuous measured ball positions and usable camera geometry")
        if self._last_estimate is not None:
            self.status = "estimated"
            self.reason = "Last qualified 3D flight average (partial flight); final report rechecks complete segments"
            return self._last_estimate.average_mph
        return None
