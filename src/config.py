"""Configuration and constants for PickleTrack-Local.

Pickleball court dimensions, HSV detection bounds, model parameters,
and physics conversion constants.
"""

from pathlib import Path
from typing import Dict, Tuple

# Base Directories
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_VIDEOS_DIR = DATA_DIR / "raw_videos"
CALIBRATION_DIR = DATA_DIR / "calibration"
OUTPUTS_DIR = DATA_DIR / "outputs"

# Ensure runtime directories exist
RAW_VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
CALIBRATION_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Official USA Pickleball Court Dimensions (in feet)
# Standard court: 20 ft wide x 44 ft long
# Net is in the center at 22 ft.
# Non-Volley Zone (Kitchen) extends 7 ft from net on each side (15 ft to 29 ft).
# ---------------------------------------------------------------------------
COURT_WIDTH_FT: float = 20.0
COURT_LENGTH_FT: float = 44.0

# Longitudinal landmarks (Y-axis along length of court: 0 to 44 ft)
BASELINE_NEAR_Y: float = 0.0
KITCHEN_NEAR_Y: float = 15.0
NET_Y: float = 22.0
KITCHEN_FAR_Y: float = 29.0
BASELINE_FAR_Y: float = 44.0

# Lateral landmarks (X-axis along width of court: 0 to 20 ft)
SIDELINE_LEFT_X: float = 0.0
CENTERLINE_X: float = 10.0
SIDELINE_RIGHT_X: float = 20.0

# Non-Volley Zone boundaries
NVZ_Y_MIN: float = KITCHEN_NEAR_Y
NVZ_Y_MAX: float = KITCHEN_FAR_Y

# Reference 4-corner coordinates on 2D court in feet
# Order: [Top-Left (Far-Left), Top-Right (Far-Right), Bottom-Right (Near-Right), Bottom-Left (Near-Left)]
COURT_CORNERS_ORDER = ["top_left", "top_right", "bottom_right", "bottom_left"]
COURT_CORNERS_FEET = [
    (0.0, COURT_LENGTH_FT),   # Top-Left (Far-Left)
    (COURT_WIDTH_FT, COURT_LENGTH_FT),  # Top-Right (Far-Right)
    (COURT_WIDTH_FT, 0.0),    # Bottom-Right (Near-Right)
    (0.0, 0.0),               # Bottom-Left (Near-Left)
]

# ---------------------------------------------------------------------------
# Physics & Speed Calculation Constants
# ---------------------------------------------------------------------------
DEFAULT_FPS: float = 30.0
# 1 foot/second = 3600 / 5280 miles/hour = 15 / 22 mph ~= 0.68181818 mph
FPS_TO_MPH: float = 3600.0 / 5280.0
SPEED_SMOOTHING_WINDOW: int = 5
MAX_BALL_SPEED_MPH: float = 80.0

# Court physical envelope for trajectory validity (in feet)
COURT_MIN_X_FT: float = -10.0
COURT_MAX_X_FT: float = 30.0
COURT_MIN_Y_FT: float = -10.0
COURT_MAX_Y_FT: float = 54.0

# ---------------------------------------------------------------------------
# Mode A: Classical Computer Vision Parameters (HSV & MOG2)
# ---------------------------------------------------------------------------
# High-visibility optic yellow / neon green pickleballs
DEFAULT_HSV_LOWER: Tuple[int, int, int] = (25, 60, 80)
DEFAULT_HSV_UPPER: Tuple[int, int, int] = (65, 255, 255)

HSV_PRESETS: Dict[str, Tuple[Tuple[int, int, int], Tuple[int, int, int]]] = {
    "optic_yellow_default": ((25, 60, 80), (65, 255, 255)),
    "high_contrast": ((28, 80, 100), (60, 255, 255)),
    "outdoor_sun": ((20, 40, 90), (70, 255, 255)),
    "indoor_gym": ((25, 70, 70), (65, 255, 255)),
}

# MOG2 Background Subtractor parameters
MOG2_HISTORY: int = 500
MOG2_VAR_THRESHOLD: float = 25.0
MOG2_DETECT_SHADOWS: bool = False

# Contour filtering for pickleball candidate (allowing for high-speed motion blur streaks and distant 2px radius balls)
BALL_MIN_AREA: float = 5.0
BALL_MAX_AREA: float = 750.0
BALL_MIN_CIRCULARITY: float = 0.12
BALL_ASPECT_RATIO_MIN: float = 0.20
BALL_ASPECT_RATIO_MAX: float = 4.50

# Player contour filtering in Mode A
PLAYER_MIN_AREA: float = 1200.0
PLAYER_MAX_AREA: float = 80000.0

# ---------------------------------------------------------------------------
# Mode B: Deep Learning (YOLOv8 + ByteTrack) & Shoe Filter
# ---------------------------------------------------------------------------
YOLO_MODEL_NAME: str = "yolov8n.pt"
COCO_PERSON_CLASS_ID: int = 0
COCO_SPORTS_BALL_CLASS_ID: int = 32

YOLO_BALL_CONF_THRESHOLD: float = 0.15
YOLO_PERSON_CONF_THRESHOLD: float = 0.35

# Heuristic Shoe Filter:
# Neon shoes/socks in the bottom 25% of a player's bounding box are rejected
SHOE_FILTER_BOTTOM_RATIO: float = 0.25

# ---------------------------------------------------------------------------
# Kalman Filter Parameters
# ---------------------------------------------------------------------------
# Image acceleration in pixels/second^2, not pixels/frame. Ball contacts can
# reverse hundreds of pixels/second within a single video frame.
KALMAN_PROCESS_NOISE_STD: float = 1800.0
KALMAN_MEASUREMENT_NOISE_STD: float = 2.0
MAX_MISSED_BALL_FRAMES: int = 3
MAX_BALL_JUMP_PIXELS: float = 135.0
KALMAN_VELOCITY_DAMPING: float = 0.65

# ---------------------------------------------------------------------------
# Shot Classification Thresholds
# ---------------------------------------------------------------------------
DINK_MAX_MPH: float = 26.0
DRIVE_MIN_MPH: float = 34.0
SPEEDUP_MIN_MPH: float = 30.0
DROP_MIN_MPH: float = 18.0
DROP_MAX_MPH: float = 35.0

# ---------------------------------------------------------------------------
# Visualization & HUD Colors (BGR format for OpenCV)
# ---------------------------------------------------------------------------
COLOR_COURT_LINES = (255, 255, 255)
COLOR_KITCHEN_ZONE = (180, 105, 255)      # Light magenta / pink
COLOR_BALL_DOT = (0, 240, 255)           # Optic Yellow / Amber
COLOR_BALL_TRAIL = (0, 220, 255)
COLOR_PLAYER_BOX = (255, 165, 0)         # Cyan/Orange
COLOR_TEXT_PRIMARY = (255, 255, 255)
COLOR_TEXT_ACCENT = (50, 255, 120)       # Electric Green
COLOR_WARNING_RED = (0, 0, 255)
COLOR_HUD_BG = (25, 25, 30)

MINI_MAP_WIDTH = 220
MINI_MAP_HEIGHT = 484  # 44 / 20 = 2.2 aspect ratio: 220 x 484 is exact!

