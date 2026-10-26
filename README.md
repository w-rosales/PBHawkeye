# PickleTrack-Local: Local Pickleball Vision & Analytics Prototype

A local Python computer vision studio for pickleball footage. It tracks players and the ball, calibrates the court, reports player kitchen occupancy, and estimates ball speed with a constrained monocular 3D flight model. Both the Browser Web Studio and CLI run locally without paid API calls. Ball speed is experimental and can be unavailable when tracking or geometry is insufficient.

---

## Key Features

1. **Zero Paid Cloud APIs**: 100% local CPU/GPU execution with no external API calls, token billing, or telemetry leakage.
2. **Interactive Browser Web Studio**:
   - Built-in local web dashboard running at `http://localhost:8000`.
   - HTML5 `<canvas>` click-to-calibrate 4 court corners with real-time quadrilateral preview.
   - Live HSV & temporal motion ball detector tuner with side-by-side binary mask preview.
   - Embedded video player, shot breakdown tables, and player kitchen occupancy progress bars.
3. **High-Speed Ball Tracker with Temporal Differencing**:
   - Resolves small fast-moving balls (6-15 px) and motion blur using 3-frame temporal differencing + adaptive HSV filtering + elongated capsule contour modeling.
   - 4-state constant-velocity Kalman Filter (`filterpy.kalman.KalmanFilter`) with up to 3 sampled frames of occlusion prediction.
   - Broadcast-grade glowing ball markers and fading motion ribbons.
4. **Modular Hybrid Detection Pipeline**:
   - **Mode A (Pure Classical CV)**: Zero neural networks (`MOG2` background subtraction + HSV color masking + contour filtering).
   - **Mode B (Local Open-Source Weights)**: Local `ultralytics` (`yolov8n.pt`) with persistent `ByteTrack` tracking for players.
5. **Domain-Specific Heuristic Shoe Filter**:
   - Automatically rejects yellow objects in the bottom 25% of any player's bounding box so neon athletic shoes never steal the ball track.
6. **Accurate Perspective-to-Feet Homography**:
   - Maps 4 user-calibrated camera pixel corners to USA Pickleball regulation dimensions (20 ft wide × 44 ft long).
7. **Deterministic Sports Analytics Engine**:
   - Candidate flight boundaries from changes in the observed trajectory.
   - Robust 3D flight fitting using calibrated camera rays, gravity, and alternative linear-drag models.
   - Estimated average flight speed, modeled peak speed, height, fit quality, and rejection reasons.
   - Paddle/bounce identity and complete-shot attribution remain unverified; flight segments are not counted as confirmed shots. Legacy classification utilities remain available for development.
   - Player Non-Volley Zone (Kitchen) occupancy percentage reports.

---

## Directory Structure

```text
PBHawkeye/
├── data/
│   ├── raw_videos/          # Test match clips (e.g., sample_rally.mp4)
│   ├── calibration/         # Saved JSON files storing 4-corner court pixel coordinates
│   └── outputs/             # Annotated MP4s, 2D radar overlays, shot_log.csv, and analytics.json
├── src/
│   ├── __init__.py
│   ├── config.py            # Regulation dimensions (20x44 ft), HSV bounds, FPS constants
│   ├── calibration.py       # Interactive OpenCV mouse-click UI to select 4 court corners
│   ├── detector.py          # Mode A (Classical CV) & Mode B (YOLOv8 + ByteTrack) + Shoe Filter
│   ├── tracker.py           # 2D Kalman Filter (filterpy) to interpolate missing ball frames
│   ├── homography.py        # cv2.getPerspectiveTransform pixel-to-feet coordinate mapper
│   ├── analytics.py         # Inflection hit detector, MPH calculator, shot classifier, reports
│   └── visualizer.py        # Bounding boxes, ball motion trails, speed HUD, and 2D mini-map
├── tests/
│   ├── test_homography.py   # Unit tests verifying pixel-to-feet perspective math accuracy
│   ├── test_analytics.py    # Synthetic trajectory tests for MPH and shot classification
│   └── test_synthetic_cv.py # Synthetic rally test for Mode A and Mode B end-to-end
├── main.py                  # CLI entry point with flags (--video, --mode, --calibrate)
├── requirements.txt
└── README.md
```

---

## Installation & Setup

1. **Clone or navigate to the repository directory**:
   ```bash
   cd PBHawkeye
   ```

2. **Install local dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

*(All required packages are 100% open-source: `numpy`, `opencv-python`, `scipy`, `filterpy`, `ultralytics`, `supervision`, `pytest`.)*

---

## Quickstart & Usage

### 1. Launch the Interactive Web Studio (Recommended)
Launch the browser UI with zero setup:
```bash
python app.py
# or
python main.py --ui
```
Then open **`http://localhost:8000`** in any web browser. From the UI you can:
- Select match clips (`.MOV` or `.MP4`)
- Click the 4 court corners on an interactive canvas to calibrate
- Use real-time sliders with side-by-side mask preview to inspect ball detection
- Run tracking and watch the annotated video with live telemetry, shot logs, and kitchen stats

### Ball tracking observations and predictions

Accepted ball detections now determine the displayed marker position directly.
The Kalman state predicts the next search location, with velocity uncertainty and
acceleration noise expressed in pixels/second and pixels/second^2. This prevents
the marker from lagging well behind fast balls. Trails break on tracking resets
and predicted gaps rather than drawing a line between unrelated detections.

Motion evidence is still required. Local recovery preserves small, overlapping
ball images without falling back to static yellow objects. The detector learns
the observed ball hue/saturation to reject differently colored moving shirt
logos. This appearance model resets after camera tracking is lost or a long
absence. Smart AI mode also uses player body/shoe exclusions. Predictions remain
explicitly marked `EST` and are stopped after three missed sampled frames.

These changes reduce known failures in the Gold Medal footage, but do not make
occluded or ambiguous balls directly observable. A missing marker is preferable
to presenting a shirt logo as the ball. Use 1x sampling when reviewing tracking;
skipping frames can miss paddle contacts and fast changes in direction.

### Frame-by-frame Gold Medal audit

Open `http://127.0.0.1:8000/validation/gold_ball_audit/` for a frame slider,
reviewed-reference/before/after panes, magnified crops and per-frame errors.
The report covers 180 consecutive frames (150?329), approximately six seconds.
References use hand-selected regions, local RGB center snapping and visual review;
partial/ambiguous frames are excluded from strict localization scoring.

Processing from frame zero, the targeted fixes increased correct detections on
156 clearly visible frames from 2 to 134 within a 10-pixel tolerance. This is a
development failure-case audit, not full-match accuracy or held-out validation.
The cold-start-at-150 experiment and exact original source snapshots are also
preserved. See `data/validation/gold_ball_audit/REPORT.md` for methods and commands.

Fixes use the selected ball-color profile to prevent a false learned appearance
from permanently rejecting the ball, keep calibrated lob headroom to the top of
the image, and exclude uncertain contact transitions from flight fitting. Strict
physical and uncertainty thresholds remain unchanged. The normal automatic run
still produced no qualified speeds in this audit; manually guided positions and
event intervals produced two model fits, without independent speed ground truth.

### Experimental 3D ball speed

The former ball MPH calculation differentiated image positions projected onto
`Z=0`. An airborne ball does not lie on that plane, so these values were not
reliable physical speeds. The tracker now leaves MPH unavailable until a
3D flight reconstruction passes its checks.

The reconstruction uses raw accepted detections, not smoothed/predicted positions.
It estimates camera pose from the court dimensions, follows per-frame camera
geometry, and fits flight segments in feet with gravity fixed at 32.174 ft/s^2.
It compares several linear-drag approximations (0, 0.4, 0.8, 1.2 per second).
This is a first approximation to pickleball aerodynamics; quadratic drag and spin
are not modeled. Changes in trajectory are candidate boundaries, not verified
paddle contacts or bounces. Average MPH is integrated 3D path length divided by
observed duration; partial segments are explicitly labeled as observed flights.

Live reconstruction checks a bounded recent observation window at most three
times per source second. A flight must pass the same strict 3D checks as the final
report. The live readout/HUD shows the **last qualified observed flight average**,
not instantaneous speed. It displays `--` when no estimate qualifies; a stale
estimate expires after 1.5 source seconds or a tracker/court reset. The final
report rechecks complete segments, so a provisional partial-flight estimate can
be withdrawn. Results show average flight MPH or **Unavailable**, with a visible
reason for rejection. Uncertain MPH ranges are not displayed. CSV/JSON include the
modeled peak, heights, camera source, reprojection error, and fit quality. JSON
also includes `trajectory_3d` with accepted per-observation 3D positions and
`tracking_observations` with the camera/2D evidence needed to diagnose rejected flights. Unknown
speeds are JSON `null` and blank CSV cells, never zero or a clipped maximum.

Automatic camera intrinsics assume square pixels, no lens distortion, and a
principal point at the image center. This is not a replacement for lens
calibration. For a measured camera calibration, add the following entry inside
`calibration` in the video's saved court JSON (stop analysis before editing):

```json
"camera_intrinsics": {
  "matrix": [[1200, 0, 960], [0, 1200, 540], [0, 0, 1]],
  "distortion": [-0.10, 0.02, 0, 0, 0]
}
```

Those numbers are an example, not values for your camera. Supply OpenCV camera
intrinsics/distortion for the exact video resolution, crop and zoom. Remove or
update supplied intrinsics if zoom changes. Automatic intrinsics are inferred
per view; ambiguous or inconsistent poses are rejected. Reconfirming the court
in the editor can replace metadata, so reapply measured intrinsics afterward.
Use a stable camera and calibrated lens for the most useful results.

Video timestamps are used when the decoder exposes them; otherwise timing falls
back to source frame number / nominal FPS. Mixed or nonmonotonic timing is not
fitted across. For variable-frame-rate footage, unavailable timestamps limit
accuracy. Sampling at 1x preserves the most ball observations; faster sampling
can make short exchanges too sparse to reconstruct.

Flights need at least eight actual observations over 0.35 seconds. Predicted
positions are excluded. Brief occlusions up to 0.10 seconds may be spanned using
actual endpoints, with at least 80% observation coverage and no tracker reset.
Missing camera geometry, longer gaps, implausible heights/speeds, poor pixel
fits, and high depth/drag uncertainty result in unavailable MPH. These checks
favor withholding an estimate over inventing one. Fit quality is a heuristic,
not a probability or a validated accuracy percentage; it does not capture every
camera, lens, tracking or aerodynamic error.

Synthetic camera/flight tests cover physical scale, height, timestamps, zoom,
noise and rejection. Real MPH accuracy still needs validation against synchronized
reference footage or radar with the same speed metric. Radar launch speed should
not be compared directly with average speed over an entire flight.

### Court detection and camera movement

Selecting a video now fits a local OpenCV court template to the painted baselines,
sidelines, kitchen lines, and service centerlines. The displayed **line match** is
an image-alignment score, not a line-call accuracy percentage. The detector is
intended for a full court viewed from behind a baseline, including modest camera
roll and an off-center position. Low-confidence proposals require manual adjustment.

Court setup always uses the opening frame. Review the projected markings, drag
the four corner handles if necessary, and choose **Looks Good - Confirm Court**
or **Save & Lock Court** before starting analysis. The Start Frame control changes
the analysis range; it does not change the calibration reference image.

Calibration is saved per video with its image dimensions, source identity, fit
score, and explicit confirmation. Old calibration files are automatically
re-evaluated; an unrelated global default is never treated as a detected court.

During analysis, ORB features, optical flow, and robust camera transforms follow
pan and zoom, including when a previously visible baseline moves outside the
image. If the camera cannot be matched reliably, the court overlay and physical
measurements pause and the UI reports that tracking is uncertain. A clear,
stationary camera view still gives the strongest results. No LLM or cloud API is
used for either court registration or camera tracking.

Ball image history and the filter's position, velocity, and uncertainty follow
the estimated camera motion. The filter recalculates its uncertainty using the
elapsed source-frame time. Sampling factors 1 through 5 are supported by the API;
the reports retain the original timestamps and separate genuine tracking gaps
from deliberately skipped frames. Higher sampling factors can miss brief ball
contacts, so use all frames when evaluating ball-tracking accuracy.

Recorded video is drained through a bounded, lossless writer queue. Encoding or
report failures are surfaced in live status and always end the running state.

---

### 2. Instant Synthetic Rally Demo via CLI
Test the entire pipeline from terminal using a synthetically generated pickleball rally:
```bash
python main.py --generate-sample --mode A
```
This generates `data/raw_videos/sample_rally.mp4`, performs homography mapping, tracks the optic yellow ball, detects shots, and outputs all reports into `data/outputs/`.

### 2. Processing a Real Video with Mode A (Classical CV)
```bash
python main.py --video data/raw_videos/my_match.mp4 --mode A
```

### 3. Processing with Mode B (YOLOv8 + ByteTrack)
```bash
python main.py --video data/raw_videos/my_match.mp4 --mode B
```

### 4. Interactive 4-Corner Court Calibration
Before processing, click the 4 corners of the court in the interactive OpenCV window:
```bash
python main.py --video data/raw_videos/my_match.mp4 --calibrate
```
**Interactive Calibration Controls**:
- Click 4 corners in order:
  1. **Top-Left**: Far-Left court corner $(0, 44)$ ft
  2. **Top-Right**: Far-Right court corner $(20, 44)$ ft
  3. **Bottom-Right**: Near-Right court corner $(20, 0)$ ft
  4. **Bottom-Left**: Near-Left court corner $(0, 0)$ ft
- Press `R` to reset points
- Press `S` or `Enter` to save calibration to JSON
- Press `Q` or `Esc` to cancel

### 5. Live Preview During Processing
Display the annotated video playback live while processing:
```bash
python main.py --video data/raw_videos/sample_rally.mp4 --mode A --preview
```

---

## Output Files

For every processed match clip (e.g., `sample_rally.mp4`), PickleTrack-Local generates three outputs in `data/outputs/`:

1. **`annotated_<video_name>.mp4`**:
   - Player bounding boxes and IDs.
   - Ball detection indicator and smooth fading yellow trajectory ribbon.
   - Tracking HUD with the last qualified 3D flight average, or unavailable MPH.
   - 2D Top-Down Radar Mini-Map in the bottom-right corner showing real-time player and ball positions on the regulation court.

2. **`shot_log_<video_name>.csv`**:
   Flight segment table, retaining the historical filename and columns for compatibility.
   Speed, distance and height values are blank when reconstruction is unavailable.
   Additional columns include `speed_status`, `speed_reason`, `speed_confidence`,
   `camera_source`, `reprojection_error_px`, `relative_uncertainty`, and heights.

3. **`analytics_<video_name>.json`**:
   Machine-readable report summarizing:
   - Observed duration, flight segment counts, estimated peak and average flight speeds.
   - Accepted/rejected segment counts and reasons; confirmed shot count is not inferred from flight count.
   - Player kitchen occupancy percentages (% of time spent in Kitchen, Transition, and Baseline).
   - Flight segment records (under the historical `shots` key) and accepted `trajectory_3d` samples.

---

## Running Unit and Integration Tests

The **Ball Tracking Test** dashboard page compares the existing tracker with the experimental local ball model and saves frame corrections for later training. [docs/ball_learning.md](docs/ball_learning.md) covers review/export, training and evaluation. Match Analysis keeps its existing detector; reviewing/exporting never retrains or activates a model.

Run the full pytest test suite:
```bash
python -m pytest tests/ -v -p no:cacheprovider --basetemp=data/outputs/pytest
node tests/test_calibration_ui.cjs
```

The test suite covers:
- `tests/test_homography.py`: 4-corner perspective mappings, roundtrip inverse transforms, landmark locations, and zone queries.
- `tests/test_analytics.py`: MPH velocity math, inflection-point reversal detection, shot classification rules, and kitchen occupancy percentage math.
- `tests/test_synthetic_cv.py`: End-to-end computer vision testing for Mode A on synthetic video, verifying shoe-filter rejection logic and Mode B initialization.
- `tests/test_court_detection.py`: Independently labeled real footage, multiple resolutions, shadows, occlusions, neighboring courts, and rejected unsupported shapes.
- `tests/test_calibration_cache.py` and `tests/test_web_calibration.py`: Saved calibration provenance, validation, confirmation, and API behavior.
- `tests/test_court_motion.py`: Camera zoom/perspective motion, clipped baselines, failed matching, and recovery.
- `tests/test_calibration_ui.cjs`: Browser confirmation, failed saves, request races, preview coordinates, and camera-tracking warnings.

