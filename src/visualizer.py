"""Visualizer and broadcast HUD overlay engine for PickleTrack-Local.

Renders:
  - Player bounding boxes, IDs, and court zone badges
  - Pickleball detection circles and fading trajectory motion ribbon
  - Live broadcast speed HUD with peak MPH and shot classification
  - 2D Top-Down Pickleball Court Mini-Map (Radar) with real-time player and ball positions
"""

from typing import List, Optional, Tuple

import cv2
import numpy as np

from src.config import (
    BASELINE_FAR_Y,
    BASELINE_NEAR_Y,
    CENTERLINE_X,
    COLOR_BALL_DOT,
    COLOR_BALL_TRAIL,
    COLOR_COURT_LINES,
    COLOR_HUD_BG,
    COLOR_KITCHEN_ZONE,
    COLOR_PLAYER_BOX,
    COLOR_TEXT_ACCENT,
    COLOR_TEXT_PRIMARY,
    COLOR_WARNING_RED,
    COURT_LENGTH_FT,
    COURT_WIDTH_FT,
    KITCHEN_FAR_Y,
    KITCHEN_NEAR_Y,
    MAX_BALL_SPEED_MPH,
    MINI_MAP_HEIGHT,
    MINI_MAP_WIDTH,
    NET_Y,
    SIDELINE_LEFT_X,
    SIDELINE_RIGHT_X,
)
from src.homography import CourtHomography
from src.tracker import BallPoint, PlayerDetection


class PickleVisualizer:
    """Renders professional sports tracking overlays, HUD telemetry, and 2D radar mini-maps."""

    def __init__(
        self,
        homography: Optional[CourtHomography] = None,
        detection_mode: str = "A",
        mini_map_scale: float = 1.0,
    ):
        self.homography = homography
        self.detection_mode = detection_mode.upper()
        self.map_w = int(MINI_MAP_WIDTH * mini_map_scale)
        self.map_h = int(MINI_MAP_HEIGHT * mini_map_scale)

        # Scale factor from court feet to mini-map pixels
        self.scale_x = self.map_w / COURT_WIDTH_FT
        self.scale_y = self.map_h / COURT_LENGTH_FT

    def court_to_minimap(self, x_ft: float, y_ft: float) -> Tuple[int, int]:
        """Converts court feet coordinates (0..20, 0..44) into mini-map pixel coordinates."""
        # Top-down view: y=0 (near baseline) at bottom, y=44 (far baseline) at top
        px = int(np.clip(x_ft * self.scale_x, 0, self.map_w - 1))
        py = int(np.clip((COURT_LENGTH_FT - y_ft) * self.scale_y, 0, self.map_h - 1))
        return px, py

    def render_minimap(
        self,
        players: List[PlayerDetection],
        ball_pt: Optional[BallPoint],
        ball_trail_court: List[Tuple[float, float]],
    ) -> np.ndarray:
        """Generates a standalone 2D bird's-eye view pickleball court image with real-time assets."""
        # Base court background (dark green / deep court slate)
        court_img = np.full((self.map_h, self.map_w, 3), (40, 60, 45), dtype=np.uint8)

        # Draw Non-Volley Zone (Kitchen) shaded area (between y=15 and y=29)
        _, k_top_py = self.court_to_minimap(0.0, KITCHEN_FAR_Y)
        _, k_bot_py = self.court_to_minimap(0.0, KITCHEN_NEAR_Y)
        cv2.rectangle(
            court_img,
            (0, k_top_py),
            (self.map_w, k_bot_py),
            (60, 85, 95),  # Kitchen tint
            -1,
        )

        # Boundary lines
        cv2.rectangle(court_img, (0, 0), (self.map_w - 1, self.map_h - 1), COLOR_COURT_LINES, 2)

        # Kitchen lines
        cv2.line(court_img, (0, k_top_py), (self.map_w, k_top_py), COLOR_COURT_LINES, 2)
        cv2.line(court_img, (0, k_bot_py), (self.map_w, k_bot_py), COLOR_COURT_LINES, 2)

        # Net line (y = 22)
        _, net_py = self.court_to_minimap(0.0, NET_Y)
        cv2.line(court_img, (0, net_py), (self.map_w, net_py), (255, 255, 255), 2)
        cv2.putText(
            court_img,
            "NET",
            (self.map_w - 35, net_py - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (200, 200, 200),
            1,
            cv2.LINE_AA,
        )

        # Centerlines (from baseline to kitchen on both halves)
        cx_px, _ = self.court_to_minimap(CENTERLINE_X, 0.0)
        # Near half centerline: y=0 to y=15
        _, near_base_py = self.court_to_minimap(CENTERLINE_X, 0.0)
        cv2.line(court_img, (cx_px, near_base_py), (cx_px, k_bot_py), COLOR_COURT_LINES, 2)
        # Far half centerline: y=29 to y=44
        _, far_base_py = self.court_to_minimap(CENTERLINE_X, 44.0)
        cv2.line(court_img, (cx_px, far_base_py), (cx_px, k_top_py), COLOR_COURT_LINES, 2)

        # Kitchen label
        cv2.putText(
            court_img,
            "NVZ / KITCHEN",
            (int(self.map_w / 2) - 48, int((k_top_py + k_bot_py) / 2) + 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.36,
            (210, 210, 210),
            1,
            cv2.LINE_AA,
        )

        # Draw ball trail on mini-map
        if len(ball_trail_court) >= 2:
            trail_pts = [self.court_to_minimap(x, y) for x, y in ball_trail_court]
            for i in range(1, len(trail_pts)):
                cv2.line(
                    court_img,
                    trail_pts[i - 1],
                    trail_pts[i],
                    (0, 200, 255),
                    2,
                    cv2.LINE_AA,
                )

        # Draw ball on mini-map
        if ball_pt is not None and ball_pt.court_x_ft is not None and ball_pt.court_y_ft is not None:
            bx, by = self.court_to_minimap(ball_pt.court_x_ft, ball_pt.court_y_ft)
            cv2.circle(court_img, (bx, by), 5, COLOR_BALL_DOT, -1)
            cv2.circle(court_img, (bx, by), 7, (0, 0, 0), 1)

        # Draw players on mini-map
        for p in players:
            if p.court_feet is not None:
                px, py = self.court_to_minimap(p.court_feet[0], p.court_feet[1])
                color = COLOR_WARNING_RED if p.in_kitchen else (255, 180, 0)
                cv2.circle(court_img, (px, py), 7, color, -1)
                cv2.circle(court_img, (px, py), 8, (255, 255, 255), 1)
                cv2.putText(
                    court_img,
                    f"P{p.tracker_id}",
                    (px - 6, py - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.35,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

        return court_img

    def render_frame(
        self,
        frame: np.ndarray,
        ball_pt: Optional[BallPoint],
        ball_trail_pixels: List[Tuple[float, float]],
        ball_trail_court: List[Tuple[float, float]],
        players: List[PlayerDetection],
        current_speed_mph: Optional[float] = None,
        current_shot_type: str = "Rally",
        shot_count: int = 0,
        frame_idx: int = 0,
    ) -> np.ndarray:
        """Composites annotations, ball trails, HUD cards, and 2D radar overlay onto the frame."""
        annotated = frame.copy()
        fh, fw = annotated.shape[:2]
        if current_speed_mph is not None and (not np.isfinite(current_speed_mph) or
                                               not 0 <= current_speed_mph <= MAX_BALL_SPEED_MPH):
            current_speed_mph = None

        # 1. Draw Court Boundary Polygon if homography is available
        if self.homography is not None:
            corners_px = self.homography.pixel_corners.astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(annotated, [corners_px], isClosed=True, color=(100, 255, 100), thickness=2)

        # 2. Draw Players
        for p in players:
            x1, y1, x2, y2 = [int(v) for v in p.bbox_xyxy]
            # Clip
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(fw - 1, x2), min(fh - 1, y2)

            box_color = COLOR_WARNING_RED if p.in_kitchen else COLOR_PLAYER_BOX
            cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, 2)

            # Player feet coordinate marker
            if p.feet_pixel is not None:
                fx, fy = int(p.feet_pixel[0]), int(p.feet_pixel[1])
                cv2.circle(annotated, (fx, fy), 4, (0, 255, 255), -1)

            # Tag pill
            zone_tag = " [KITCHEN!]" if p.in_kitchen else ""
            tag = f"Player #{p.tracker_id}{zone_tag}"
            cv2.rectangle(annotated, (x1, max(0, y1 - 22)), (x1 + len(tag) * 9 + 10, y1), box_color, -1)
            cv2.putText(
                annotated,
                tag,
                (x1 + 4, max(14, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 0, 0) if not p.in_kitchen else (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        # 3. Draw Pickleball Motion Ribbon / Trail
        if len(ball_trail_pixels) >= 2:
            num_pts = len(ball_trail_pixels)
            for i in range(1, num_pts):
                pt1 = (int(ball_trail_pixels[i - 1][0]), int(ball_trail_pixels[i - 1][1]))
                pt2 = (int(ball_trail_pixels[i][0]), int(ball_trail_pixels[i][1]))
                alpha = float(i) / float(num_pts)

                # Outer glow line
                glow_thick = max(2, int(6 * alpha))
                cv2.line(annotated, pt1, pt2, (0, int(160 * alpha), int(255 * alpha)), glow_thick, cv2.LINE_AA)
                # Inner sharp core line
                core_thick = max(1, int(3 * alpha))
                cv2.line(annotated, pt1, pt2, (50, 255, 255), core_thick, cv2.LINE_AA)

        # 4. Draw Ball Indicator
        if ball_pt is not None:
            bx, by = int(ball_pt.pixel_x), int(ball_pt.pixel_y)
            if 0 <= bx < fw and 0 <= by < fh:
                if ball_pt.is_interpolated:
                    # Predicted/interpolated indicator
                    cv2.circle(annotated, (bx, by), 12, (0, 140, 255), 2, cv2.LINE_AA)
                    cv2.circle(annotated, (bx, by), 5, (0, 165, 255), -1, cv2.LINE_AA)
                    cv2.putText(annotated, "EST", (bx + 14, by - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 165, 255), 1, cv2.LINE_AA)
                else:
                    # High-visibility broadcast ball marker
                    cv2.circle(annotated, (bx, by), 14, (0, 255, 255), 2, cv2.LINE_AA)
                    cv2.circle(annotated, (bx, by), 15, (0, 0, 0), 1, cv2.LINE_AA)
                    cv2.circle(annotated, (bx, by), 7, (0, 240, 255), -1, cv2.LINE_AA)
                    cv2.circle(annotated, (bx, by), 3, (255, 255, 255), -1, cv2.LINE_AA)
                    # Speed tag attached to ball if moving
                    if ball_pt.speed_mph is not None and ball_pt.speed_mph > 3.0:
                        speed_tag = f"{ball_pt.speed_mph:.0f} MPH"
                        cv2.rectangle(annotated, (bx + 16, by - 14), (bx + 16 + len(speed_tag) * 9, by + 4), (20, 20, 20), -1)
                        cv2.putText(annotated, speed_tag, (bx + 18, by), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 255, 255), 1, cv2.LINE_AA)

        # 5. Broadcast HUD Card (Top-Left)
        hud_w, hud_h = 320, 115
        hud_overlay = annotated[15 : 15 + hud_h, 15 : 15 + hud_w].copy()
        cv2.rectangle(hud_overlay, (0, 0), (hud_w, hud_h), COLOR_HUD_BG, -1)
        cv2.addWeighted(hud_overlay, 0.85, annotated[15 : 15 + hud_h, 15 : 15 + hud_w], 0.15, 0, annotated[15 : 15 + hud_h, 15 : 15 + hud_w])
        cv2.rectangle(annotated, (15, 15), (15 + hud_w, 15 + hud_h), (80, 80, 90), 1)

        # Telemetry Texts
        mode_text = f"MODE: {'Classical CV' if self.detection_mode == 'A' else 'YOLOv8 + ByteTrack'}"
        cv2.putText(annotated, mode_text, (26, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1, cv2.LINE_AA)

        speed_str = f"LAST FLIGHT: {current_speed_mph:.1f} MPH" if current_speed_mph is not None else "3D MPH: UNAVAILABLE"
        cv2.putText(annotated, speed_str, (26, 68), cv2.FONT_HERSHEY_DUPLEX, 0.65, COLOR_TEXT_ACCENT, 2, cv2.LINE_AA)

        shot_str = f"SHOT: {current_shot_type.upper()}"
        shot_color = (0, 240, 255) if current_shot_type == "Dink" else (50, 180, 255)
        if current_shot_type == "Drive":
            shot_color = (80, 80, 255)
        elif current_shot_type == "Speedup":
            shot_color = (0, 140, 255)
        cv2.putText(annotated, shot_str, (26, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.55, shot_color, 2, cv2.LINE_AA)

        rally_str = f"RALLY #{max(1, shot_count)} | F:{frame_idx}"
        cv2.putText(annotated, rally_str, (26, 118), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (190, 190, 190), 1, cv2.LINE_AA)

        # 6. Top-Down Mini-Map Radar Overlay (Bottom-Right)
        radar_img = self.render_minimap(players, ball_pt, ball_trail_court)
        rm_h, rm_w = radar_img.shape[:2]
        margin = 15
        y_offset = fh - rm_h - margin
        x_offset = fw - rm_w - margin

        if y_offset >= 0 and x_offset >= 0:
            radar_roi = annotated[y_offset : y_offset + rm_h, x_offset : x_offset + rm_w]
            cv2.addWeighted(radar_img, 0.90, radar_roi, 0.10, 0, radar_roi)
            cv2.rectangle(
                annotated,
                (x_offset, y_offset),
                (x_offset + rm_w, y_offset + rm_h),
                (120, 120, 130),
                2,
            )
            # Radar title pill
            cv2.rectangle(
                annotated,
                (x_offset, y_offset - 20),
                (x_offset + 120, y_offset),
                (30, 30, 35),
                -1,
            )
            cv2.putText(
                annotated,
                "2D COURT RADAR",
                (x_offset + 6, y_offset - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.38,
                (220, 220, 220),
                1,
                cv2.LINE_AA,
            )

        return annotated

