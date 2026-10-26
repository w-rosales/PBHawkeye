"""Analytics Engine for Pickleball Vision.

Performs:
  - Trajectory smoothing via Savitzky-Golay / rolling median filters
  - Inflection-point hit detection (velocity vector sign changes and acceleration spikes)
  - Constrained monocular 3D flight speed estimates with uncertainty checks
  - Player kitchen (Non-Volley Zone) occupancy tracking
  - Shot classification (Dink, Drive, Drop, Speedup, Lob, Reset)
  - CSV and JSON report generation
"""

import csv
import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from scipy.signal import savgol_filter

from src.config import (
    DEFAULT_FPS,
    DINK_MAX_MPH,
    DRIVE_MIN_MPH,
    DROP_MAX_MPH,
    DROP_MIN_MPH,
    FPS_TO_MPH,
    KITCHEN_FAR_Y,
    KITCHEN_NEAR_Y,
    MAX_BALL_SPEED_MPH,
    NET_Y,
    SPEED_SMOOTHING_WINDOW,
    SPEEDUP_MIN_MPH,
)
from src.tracker import BallPoint, PlayerDetection
from src.ball_3d import flight_segments


@dataclass
class ShotEvent:
    """Represents a classified pickleball shot within a rally."""

    shot_id: int
    frame_start: int
    frame_end: int
    hitter_player_id: Optional[int]
    shot_type: str
    peak_mph: Optional[float]
    avg_mph: Optional[float]
    start_x_ft: Optional[float]
    start_y_ft: Optional[float]
    end_x_ft: Optional[float]
    end_y_ft: Optional[float]
    start_zone: str
    landing_zone: str
    duration_sec: float
    distance_ft: Optional[float]
    speed_status: str = "unavailable"
    speed_reason: str = "No calibrated 3D flight"
    speed_confidence: float = 0.0
    speed_method: str = "monocular_3d_flight_v1"
    camera_source: str = "unavailable"
    reprojection_error_px: Optional[float] = None
    relative_uncertainty: Optional[float] = None
    start_height_ft: Optional[float] = None
    end_height_ft: Optional[float] = None
    max_height_ft: Optional[float] = None
    boundary_start: str = "tracking_boundary"
    boundary_end: str = "tracking_boundary"


@dataclass
class PlayerOccupancyReport:
    """Kitchen and court zone occupancy statistics for a player."""

    player_id: int
    total_frames_tracked: int
    kitchen_frames: int
    kitchen_percentage: float
    transition_percentage: float
    baseline_percentage: float
    shots_hit_count: int = 0
    max_shot_mph: Optional[float] = None


class TrajectoryAnalytics:
    """Deterministic analytics engine for pickleball trajectories."""

    def __init__(self, fps: float = DEFAULT_FPS, frame_stride: Optional[int] = None):
        self.fps = fps
        self.dt = 1.0 / fps
        if frame_stride is not None and (frame_stride < 1 or int(frame_stride) != frame_stride):
            raise ValueError("frame_stride must be a positive integer")
        self.frame_stride = frame_stride

    @staticmethod
    def _continuous_court_runs(
        trajectory: List[BallPoint],
        max_frame_gap: int = 5,
        fps: float = 30.0,
        expected_frame_gap: Optional[int] = None,
    ) -> List[List[int]]:
        """Return valid continuous tracking runs, splitting on measurement gaps or unphysical jumps."""
        runs: List[List[int]] = []
        current: List[int] = []
        for index, point in enumerate(trajectory):
            valid = (point.court_x_ft is not None and point.court_y_ft is not None
                     and np.isfinite([point.court_x_ft, point.court_y_ft]).all())
            if not valid:
                if current:
                    runs.append(current)
                    current = []
                continue
            if current:
                prev = trajectory[current[-1]]
                gap = point.frame_idx - prev.frame_idx
                # Allow small frame sampling gaps (<= max_frame_gap), but split on larger gaps or backward frames
                if (gap <= 0 or gap > max_frame_gap
                        or (expected_frame_gap is not None and gap != expected_frame_gap)):
                    runs.append(current)
                    current = []
                else:
                    dt = gap / max(1.0, fps)
                    dist = float(np.hypot(point.court_x_ft - prev.court_x_ft, point.court_y_ft - prev.court_y_ft))
                    # Physical distance limit at 80 MPH max
                    max_allowed_dist = (MAX_BALL_SPEED_MPH / FPS_TO_MPH) * dt * 1.05
                    if dist > max_allowed_dist:
                        runs.append(current)
                        current = []
            current.append(index)
        if current:
            runs.append(current)
        return runs

    def calculate_velocities_and_speeds(
        self,
        court_coords: List[Tuple[float, float]],
        smoothing_window: int = SPEED_SMOOTHING_WINDOW,
        frame_indices: Optional[List[int]] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculates smoothed velocities (vx, vy in ft/s) and speed in MPH.

        Planar motion utility; not a physical airborne ball-speed measurement.
        Uses Savitzky-Golay filter to differentiate without quantization noise.
        Rejects unphysical jumps above MAX_BALL_SPEED_MPH rather than clipping.

        Args:
            court_coords: List of (x_ft, y_ft) court points
            smoothing_window: Filter window size (must be odd integer >= 3)
            frame_indices: Optional list of frame numbers for accurate dt with variable sampling

        Returns:
            (vx_ft_s, vy_ft_s, speed_mph) as numpy arrays
        """
        n = len(court_coords)
        if n == 0:
            return np.empty(0), np.empty(0), np.empty(0)
        if n == 1:
            return np.zeros(1), np.zeros(1), np.zeros(1)

        coords_arr = np.array(court_coords, dtype=np.float64)  # shape (n, 2)
        x = coords_arr[:, 0]
        y = coords_arr[:, 1]

        # Determine true time steps
        if frame_indices is not None and len(frame_indices) == n:
            times = np.array(frame_indices, dtype=np.float64) * self.dt
            diffs = np.diff(times)
            is_uniform = np.allclose(diffs, diffs[0], rtol=1e-3, atol=1e-5) if len(diffs) > 0 else True
            dt_step = diffs[0] if (len(diffs) > 0 and is_uniform and diffs[0] > 1e-6) else self.dt
        else:
            times = None
            is_uniform = True
            dt_step = self.dt

        # Ensure smoothing window is valid
        window = min(smoothing_window, n)
        if window % 2 == 0:
            window -= 1
        if window < 3 or not is_uniform:
            # Fallback to finite differences with true elapsed time
            if times is not None:
                vx = np.gradient(x, times)
                vy = np.gradient(y, times)
            else:
                vx = np.gradient(x, dt_step)
                vy = np.gradient(y, dt_step)
        else:
            # Polyorder 2 Savitzky-Golay differentiator
            polyorder = min(2, window - 1)
            vx = savgol_filter(x, window_length=window, polyorder=polyorder, deriv=1, delta=dt_step)
            vy = savgol_filter(y, window_length=window, polyorder=polyorder, deriv=1, delta=dt_step)

        speed_ft_s = np.hypot(vx, vy)
        raw_speed_mph = speed_ft_s * FPS_TO_MPH
        # Reject unphysical speed spikes rather than concealing them as plausible maximums
        speed_mph = np.where(raw_speed_mph <= MAX_BALL_SPEED_MPH, raw_speed_mph, 0.0)
        vx = np.where(raw_speed_mph <= MAX_BALL_SPEED_MPH, vx, 0.0)
        vy = np.where(raw_speed_mph <= MAX_BALL_SPEED_MPH, vy, 0.0)

        return vx, vy, speed_mph

    def detect_inflection_points(
        self,
        trajectory: List[BallPoint],
        min_shot_gap_frames: int = 8,
    ) -> List[int]:
        """Detects frame indices where paddle hits or major trajectory reversals occur.

        Identifies:
          1. Longitudinal velocity direction flips (sign changes in vy along court length)
          2. Significant acceleration deflection peaks
        """
        if len(trajectory) < 6:
            return []

        runs = self._continuous_court_runs(trajectory, fps=self.fps,
                                           expected_frame_gap=self.frame_stride)
        if len(runs) != 1 or len(runs[0]) != len(trajectory):
            return [run[index] for run in runs
                    for index in self.detect_inflection_points(
                        [trajectory[i] for i in run], min_shot_gap_frames)]

        # Extract court Y positions and valid frame indices
        frames = [pt.frame_idx for pt in trajectory]
        valid_indices = [
            i for i, pt in enumerate(trajectory) if pt.court_y_ft is not None
        ]

        if len(valid_indices) < 6:
            return []

        y_coords = np.array([trajectory[i].court_y_ft for i in valid_indices])
        coords = [
            (trajectory[i].court_x_ft, trajectory[i].court_y_ft)
            for i in valid_indices
        ]

        _, vy, _ = self.calculate_velocities_and_speeds(coords, smoothing_window=5,
                                                      frame_indices=[trajectory[i].frame_idx for i in valid_indices])

        inflection_indices: List[int] = [valid_indices[0]]  # Rally start is first hit

        # Look for zero-crossings in vy (direction reversal across the net)
        sign_vy = np.sign(vy)
        # Clean zeros
        for i in range(1, len(sign_vy)):
            if sign_vy[i] == 0:
                sign_vy[i] = sign_vy[i - 1]

        sign_diff = np.diff(sign_vy)
        reversals = np.where(sign_diff != 0)[0]

        last_hit_frame = frames[valid_indices[0]]

        for rev in reversals:
            hit_idx = valid_indices[rev]
            curr_frame = frames[hit_idx]
            # Ensure minimum frame interval between distinct shots
            if (curr_frame - last_hit_frame) >= min_shot_gap_frames:
                # Confirm significant travel before and after
                inflection_indices.append(hit_idx)
                last_hit_frame = curr_frame

        # Always include the last valid index as terminal frame of rally
        if (
            len(inflection_indices) > 0
            and inflection_indices[-1] != valid_indices[-1]
            and (frames[valid_indices[-1]] - frames[inflection_indices[-1]]) >= 4
        ):
            inflection_indices.append(valid_indices[-1])

        return inflection_indices

    def attribute_hitter(
        self,
        hit_court_pos: Tuple[float, float],
        frame_idx: int,
        player_history: Dict[int, List[PlayerDetection]],
        max_reach_ft: float = 12.0,
    ) -> Optional[int]:
        """Attributes a detected paddle hit to the nearest player at that frame."""
        hx, hy = hit_court_pos
        best_pid = None
        min_dist = float("inf")

        for pid, detections in player_history.items():
            # Find detection closest in time to hit frame
            closest_det = min(detections, key=lambda d: abs(getattr(d, "frame_idx", frame_idx) - frame_idx), default=None)
            if closest_det is not None and closest_det.court_feet is not None:
                px, py = closest_det.court_feet
                dist = np.hypot(px - hx, py - hy)
                if dist < min_dist and dist <= max_reach_ft:
                    min_dist = dist
                    best_pid = pid

        return best_pid

    def classify_shot(
        self,
        start_pt: Tuple[float, float],
        end_pt: Tuple[float, float],
        peak_mph: float,
        avg_mph: float,
        duration_sec: float,
        start_zone: str,
        landing_zone: str,
        incoming_speed_mph: float = 0.0,
    ) -> str:
        """Heuristic sports classifier mapping physics features to pickleball shots:

        - Dink: Soft, low-speed shot landing in or near the kitchen
        - Drive: Fast penetrating shot hit from backcourt deep into opponent court
        - Drop / Third Shot Drop: Controlled shot hit from deep backcourt dropping softly into NVZ
        - Speedup: Rapid acceleration attack shot initiated from kitchen or transition
        - Lob: High flight time / deep landing shot over opponent
        - Reset: Soft defensive block absorbing pace into NVZ
        """
        dist_ft = np.hypot(end_pt[0] - start_pt[0], end_pt[1] - start_pt[1])
        is_nvz_landing = landing_zone in ["near_kitchen", "far_kitchen"]
        is_deep_start = start_zone in ["near_baseline", "far_baseline"]
        is_deep_landing = landing_zone in ["near_baseline", "far_baseline"]
        is_kitchen_start = start_zone in ["near_kitchen", "far_kitchen", "near_transition", "far_transition"]

        # 1. Reset check: Absorbing high incoming speed into soft kitchen landing
        if incoming_speed_mph >= 28.0 and peak_mph <= 26.0 and is_nvz_landing:
            return "Reset"

        # 2. Speedup: Accelerated attack initiated from NVZ or transition zone
        if is_kitchen_start and peak_mph >= SPEEDUP_MIN_MPH:
            return "Speedup"

        # 3. Drive: High velocity penetrating shot from backcourt / midcourt
        if peak_mph >= DRIVE_MIN_MPH and (is_deep_start or dist_ft >= 15.0):
            return "Drive"

        # 4. Third Shot Drop / Drop: Hit from deep baseline softly into kitchen
        if is_deep_start and is_nvz_landing and (DROP_MIN_MPH <= peak_mph <= DROP_MAX_MPH or peak_mph <= 28.0):
            return "Drop"

        # 5. Dink: Low speed kitchen-to-kitchen exchange
        if peak_mph <= DINK_MAX_MPH and is_nvz_landing:
            return "Dink"

        # 6. Lob: High duration and deep landing
        if duration_sec >= 1.1 and is_deep_landing:
            return "Lob"

        # General fallbacks
        if peak_mph > 30.0:
            return "Drive"
        if is_nvz_landing:
            return "Dink"
        return "Standard Rally"

    def analyze_rally(
        self,
        trajectory: List[BallPoint],
        player_history: Dict[int, List[PlayerDetection]],
        homography=None,
    ) -> Tuple[List[ShotEvent], Dict[int, PlayerOccupancyReport], Dict]:
        """Comprehensive analysis of the entire rally:

        Detects all shots, classifies them, calculates speeds, and generates player reports.
        """
        # Ground-plane ball coordinates are not used to calculate MPH.
        for point in trajectory:
            point.speed_mph = None
            point.world_x_ft = point.world_y_ft = point.height_ft = None
            point.velocity_ft_x = point.velocity_ft_y = 0.0
            point.velocity_ft_z = None
            point.speed_confidence = 0.0
        segments = flight_segments(trajectory, self.fps, self.frame_stride)
        shots: List[ShotEvent] = []
        measured_distance = measured_duration = 0.0
        for indices, estimate in segments:
            if len(indices) < 2:
                for index in indices:
                    trajectory[index].speed_reason = "Isolated detection; insufficient flight observations"
                continue
            points = [trajectory[i] for i in indices]
            for point in points:
                point.speed_reason = estimate.reason
            valid = estimate.status == "estimated"
            if valid:
                for point, position, velocity in zip(points, estimate.positions, estimate.velocities):
                    point.world_x_ft, point.world_y_ft, point.height_ft = map(float, position)
                    point.velocity_ft_x, point.velocity_ft_y, point.velocity_ft_z = map(float, velocity)
                    point.speed_mph = float(np.linalg.norm(velocity)*FPS_TO_MPH)
                    point.speed_confidence = estimate.confidence
                start_x, start_y = map(float, estimate.positions[0, :2])
                end_x, end_y = map(float, estimate.positions[-1, :2])
            else:
                # No false court locations or speed-dependent shot classification.
                start_x = start_y = end_x = end_y = None
            first, last = points[0], points[-1]
            start_t = first.timestamp_s if first.timestamp_s is not None else first.frame_idx*self.dt
            end_t = last.timestamp_s if last.timestamp_s is not None else last.frame_idx*self.dt
            if valid:
                measured_distance += estimate.distance_ft
                measured_duration += end_t-start_t
            start_zone = homography.get_court_zone(start_x, start_y) if valid and homography else "unknown"
            landing_zone = homography.get_court_zone(end_x, end_y) if valid and homography else "unknown"
            # A change in the trajectory can be a bounce or paddle contact.
            # Do not claim verified shot types/hitters from these boundaries alone.
            shots.append(ShotEvent(
                shot_id=len(shots)+1, frame_start=first.frame_idx, frame_end=last.frame_idx,
                hitter_player_id=None, shot_type="Flight segment" if valid else "Unclassified",
                peak_mph=round(estimate.peak_mph, 1) if valid else None,
                avg_mph=round(estimate.average_mph, 1) if valid else None,
                start_x_ft=round(start_x, 2) if valid else None,
                start_y_ft=round(start_y, 2) if valid else None,
                end_x_ft=round(end_x, 2) if valid else None,
                end_y_ft=round(end_y, 2) if valid else None,
                start_zone=start_zone, landing_zone=landing_zone,
                duration_sec=round(end_t-start_t, 3),
                distance_ft=round(estimate.distance_ft, 2) if valid else None,
                speed_status=estimate.status, speed_reason=estimate.reason,
                speed_confidence=round(estimate.confidence, 3), camera_source=estimate.camera_source,
                reprojection_error_px=estimate.reprojection_error_px,
                relative_uncertainty=estimate.relative_uncertainty,
                start_height_ft=float(estimate.positions[0, 2]) if valid else None,
                end_height_ft=float(estimate.positions[-1, 2]) if valid else None,
                max_height_ft=float(np.max(estimate.positions[:, 2])) if valid else None,
                boundary_start="unverified", boundary_end="unverified",
            ))

        # 2. Player Kitchen Occupancy Analysis
        player_reports: Dict[int, PlayerOccupancyReport] = {}
        for pid, p_list in player_history.items():
            p_list = [point for point in p_list if point.court_feet is not None
                      and np.isfinite(point.court_feet).all()]
            total = len(p_list)
            if total == 0:
                continue

            kitchen_count = 0
            trans_count = 0
            base_count = 0

            for p_det in p_list:
                if p_det.court_feet is not None:
                    _, cy = p_det.court_feet
                    if KITCHEN_NEAR_Y <= cy <= KITCHEN_FAR_Y:
                        kitchen_count += 1
                    elif (8.0 <= cy < KITCHEN_NEAR_Y) or (KITCHEN_FAR_Y < cy <= 36.0):
                        trans_count += 1
                    else:
                        base_count += 1

            k_pct = (kitchen_count / total) * 100.0
            t_pct = (trans_count / total) * 100.0
            b_pct = (base_count / total) * 100.0

            # Calculate shots hit by this player
            p_shots = [s for s in shots if s.hitter_player_id == pid]
            max_mph = max([s.peak_mph for s in p_shots if s.peak_mph is not None], default=None)

            player_reports[pid] = PlayerOccupancyReport(
                player_id=pid,
                total_frames_tracked=total,
                kitchen_frames=kitchen_count,
                kitchen_percentage=round(k_pct, 1),
                transition_percentage=round(t_pct, 1),
                baseline_percentage=round(b_pct, 1),
                shots_hit_count=len(p_shots),
                max_shot_mph=round(max_mph, 1) if max_mph is not None else None,
            )

        # 3. Rally Summary Metrics
        rally_duration = (
            (trajectory[-1].frame_idx - trajectory[0].frame_idx) * self.dt
            if len(trajectory) > 1
            else 0.0
        )
        if (len(trajectory) > 1 and trajectory[0].timestamp_s is not None
                and trajectory[-1].timestamp_s is not None
                and trajectory[0].timing_source == trajectory[-1].timing_source
                and trajectory[-1].timestamp_s > trajectory[0].timestamp_s):
            rally_duration = trajectory[-1].timestamp_s-trajectory[0].timestamp_s
        all_peaks = [s.peak_mph for s in shots if s.peak_mph is not None]
        all_avgs = [s.avg_mph for s in shots if s.avg_mph is not None]
        rejections = Counter(s.speed_reason for s in shots if s.speed_status != "estimated")
        if all_avgs:
            display_reason = f"{len(all_avgs)} observed flight(s) passed the 3D speed checks"
        elif shots:
            explanations = {
                "Insufficient observations": "tracks have fewer than 8 actual ball observations",
                "Flight is too short or contains unresolved contacts": "flights are too short or contain unresolved contacts",
                "No physically plausible 3D flight fits the observations": "tracks do not fit a plausible 3D flight",
                "Depth or aerodynamic uncertainty is too large": "depth or aerodynamic estimates remain ambiguous",
            }
            reasons = "; ".join(explanations.get(reason, reason) for reason, _ in rejections.most_common(2))
            display_reason = f"MPH unavailable: none of {len(shots)} flight fragments passed the checks. Main reasons: {reasons}."
        else:
            display_reason = "MPH unavailable: no continuous measured ball flight with usable camera geometry"

        summary = {
            "total_shots": 0,  # Flight boundaries do not yet identify paddle contacts reliably.
            "total_flight_segments": len(shots),
            "speed_method": "monocular_3d_flight_v1",
            "speed_display_reason": display_reason,
            "speed_rejection_counts": dict(rejections),
            "speed_metric": "average_over_observed_flight",
            "estimated_speed_segments": len(all_avgs),
            "unavailable_speed_segments": sum(s.avg_mph is None for s in shots),
            "speed_unavailable_reasons": sorted({p.speed_reason for p in trajectory if p.speed_mph is None}),
            "speed_note": "Experimental monocular estimate; partial flights, inferred camera, drag and spin can affect accuracy. Validate against reference measurements.",
            "rally_duration_sec": round(rally_duration, 2),
            "max_rally_speed_mph": round(max(all_peaks), 1) if all_peaks else None,
            "avg_rally_speed_mph": round(measured_distance/measured_duration*FPS_TO_MPH, 1) if measured_duration > 0 else None,
            "measured_flight_duration_sec": round(measured_duration, 3),
            "shot_type_distribution": {
                st: sum(1 for s in shots if s.shot_type == st)
                for st in set(s.shot_type for s in shots)
            },
        }

        return shots, player_reports, summary

    @staticmethod
    def export_to_csv(shots: List[ShotEvent], csv_path: Union[str, Path]) -> None:
        """Exports shot list to CSV file."""
        path = Path(csv_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        fieldnames = list(ShotEvent.__dataclass_fields__)

        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for s in shots:
                writer.writerow(asdict(s))

    @staticmethod
    def export_to_json(
        summary: dict,
        shots: List[ShotEvent],
        player_reports: Dict[int, PlayerOccupancyReport],
        json_path: Union[str, Path],
        trajectory: Optional[List[BallPoint]] = None,
    ) -> None:
        """Exports complete analytics report to JSON file."""
        path = Path(json_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "summary": summary,
            "shots": [asdict(s) for s in shots],
            "players": {str(k): asdict(v) for k, v in player_reports.items()},
            "trajectory_3d": [{"frame_idx": p.frame_idx, "timestamp_s": p.timestamp_s,
                               "timing_source": p.timing_source, "x_ft": p.world_x_ft,
                               "y_ft": p.world_y_ft, "height_ft": p.height_ft,
                               "speed_mph": p.speed_mph, "fit_confidence": p.speed_confidence}
                              for p in (trajectory or []) if p.height_ft is not None],
            # Keep evidence for diagnosing rejected flights without decoding and
            # running detection again. These are not exposed as measured speeds.
            "tracking_observations": [{
                "frame_idx": p.frame_idx, "timestamp_s": p.timestamp_s,
                "timing_source": p.timing_source, "measurement_pixel": p.measurement_pixel,
                "observation_pixel": p.observation_pixel, "reference_pixel": p.reference_pixel,
                "camera_projection": np.asarray(p.camera_projection).tolist() if p.camera_projection is not None else None,
                "camera_source": p.camera_source, "confidence": p.confidence,
                "is_interpolated": p.is_interpolated, "track_generation": p.track_generation,
                "speed_reason": p.speed_reason,
            } for p in (trajectory or [])],
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
