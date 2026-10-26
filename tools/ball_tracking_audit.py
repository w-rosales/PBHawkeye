"""Reproducible raw-frame/independent-label ball tracking audit.

Run from the repository root. Labels are hand-entered after inspecting raw PNGs;
the software simulation never uses those labels for detection or tracking.
"""

import argparse
from dataclasses import asdict
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time
from collections import Counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np

ROOT = Path("data/validation/gold_ball_audit")


def archived_module(name):
    module_name = "audit_archived_"+name
    if module_name not in sys.modules:
        spec = importlib.util.spec_from_file_location(module_name, ROOT/"source_before"/(name+".py"))
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    return sys.modules[module_name]


def refine_labels():
    """Optional snap inside a hand-selected region, independent of production.

    Uses local RGB yellow contrast, no temporal mask, predictions or detector
    outputs. Every snapped reference is subsequently reviewed on raw crops.
    This is annotation assistance, not an independent automatic ground truth.
    """
    payload = json.loads((ROOT/"manual_labels.json").read_text())
    for label in payload["labels"]:
        if label["x"] is None:
            continue
        label["manual_click"] = [label["x"], label["y"]]
        if label["visibility"] != "visible":
            continue
        frame = cv2.imread(str(ROOT/"frames"/f'{label["frame"]:06d}.png'))
        x, y = round(label["x"]), round(label["y"])
        x0, y0 = max(0, x-24), max(0, y-24)
        patch = frame[y0:min(frame.shape[0], y+25), x0:min(frame.shape[1], x+25)].astype(float)
        contrast = np.minimum(patch[..., 2], patch[..., 1])-patch[..., 0]
        threshold = max(35., float(np.max(contrast))*.62)
        mask = np.uint8(contrast > threshold)*255
        count, components, stats, centers = cv2.connectedComponentsWithStats(mask)
        candidates = []
        for i in range(1, count):
            if 4 <= stats[i, cv2.CC_STAT_AREA] <= 800:
                center = centers[i]+[x0, y0]
                distance = np.linalg.norm(center-[x, y])
                if distance < 23:
                    candidates.append((distance, i))
        if candidates:
            _, index = min(candidates)
            yy, xx = np.where(components == index)
            weights = contrast[yy, xx]
            label["x"] = round(float(np.average(xx, weights=weights)+x0), 2)
            label["y"] = round(float(np.average(yy, weights=weights)+y0), 2)
            label["center_method"] = "manual_region_with_local_RGB_snap_pending_visual_review"
            label["uncertainty_px"] = 3
        else:
            label["center_method"] = "manual_click_only"
    payload["method"] = ("Raw images inspected individually; hand-selected ball regions with optional local RGB center snapping. "
                         "No production detector coordinates/predictions used as references. Raw-crop overlay review required. "
                         "Approximate reference centers are not subpixel ground truth; ambiguous frames excluded from strict scoring.")
    (ROOT/"manual_labels.json").write_text(json.dumps(payload, indent=2))


def review_sheets():
    labels = json.loads((ROOT/"manual_labels.json").read_text())["labels"]
    previous = (960, 540)
    tiles = []
    for label in labels:
        frame = cv2.imread(str(ROOT/"frames"/f'{label["frame"]:06d}.png'))
        if label["x"] is not None:
            previous = (label["x"], label["y"])
        x, y = map(round, previous)
        x0, y0 = max(0, min(1840, x-40)), max(0, min(1000, y-40))
        patch = cv2.resize(frame[y0:y0+80, x0:x0+80], (160, 160), interpolation=cv2.INTER_NEAREST)
        if label["x"] is not None:
            center = (round((label["x"]-x0)*2), round((label["y"]-y0)*2))
            cv2.circle(patch, center, 26, (255, 255, 255), 1)
        canvas = np.zeros((190, 160, 3), np.uint8)
        canvas[30:] = patch
        cv2.putText(canvas, f'F{label["frame"]} {label["visibility"][:7]}', (3, 13), 0, .4, (0, 255, 255), 1)
        if label["x"] is not None:
            cv2.putText(canvas, f'{label["x"]:.0f},{label["y"]:.0f}', (3, 26), 0, .35, (255, 255, 255), 1)
        tiles.append(canvas)
    for start in range(0, len(tiles), 24):
        batch = tiles[start:start+24]
        while len(batch) % 6:
            batch.append(np.zeros_like(batch[0]))
        result = cv2.vconcat([cv2.hconcat(batch[i:i+6]) for i in range(0, len(batch), 6)])
        cv2.imwrite(str(ROOT/"sheets"/f"review_{150+start:03d}.png"), result)


def sheet(start, end, crop, name):
    x, y, width, height = crop
    tiles = []
    for frame_idx in range(start, end+1):
        raw = cv2.imread(str(ROOT/"frames"/f"{frame_idx:06d}.png"))
        patch = raw[y:y+height, x:x+width].copy()
        canvas = np.full((height+35, width+35, 3), 25, np.uint8)
        canvas[35:, 35:] = patch
        cv2.putText(canvas, f"F{frame_idx}", (3, 15), 0, .42, (0, 255, 255), 1)
        for value in range(0, width, 50):
            cv2.putText(canvas, str(x+value), (35+value, 30), 0, .32, (255, 255, 255), 1)
            cv2.line(canvas, (35+value, 33), (35+value, 39), (255, 255, 255))
        for value in range(0, height, 50):
            cv2.putText(canvas, str(y+value), (0, 38+value), 0, .30, (255, 255, 255), 1)
            cv2.line(canvas, (33, 35+value), (39, 35+value), (255, 255, 255))
        tiles.append(canvas)
    columns = 4
    while len(tiles) % columns:
        tiles.append(np.zeros_like(tiles[0]))
    result = cv2.vconcat([cv2.hconcat(tiles[i:i+columns]) for i in range(0, len(tiles), columns)])
    output = ROOT/"sheets"/f"{name}.png"
    cv2.imwrite(str(output), result)
    print(output)


def simulate(output, hue_hint=None, warmup_start=None, legacy=False):
    from src.ball_3d import ObservationRecorder
    from src.court_motion import CourtMotionTracker
    from src.detector import PickleballDetector
    from src.homography import CourtHomography
    from src.tracker import BallKalmanTracker

    metadata = json.loads((ROOT/"metadata.json").read_text())
    court = CourtHomography.load_from_file("data/calibration/Gold_Medal_Match_Punta_Gorda_calibration.json")
    cap = cv2.VideoCapture(metadata["video"])
    ok, reference = cap.read()
    if not ok:
        raise RuntimeError("Could not decode reference frame")
    camera = CourtMotionTracker(reference, court)
    if legacy:
        PickleballDetector = archived_module("detector").PickleballDetector
        ObservationRecorder = archived_module("ball_3d").ObservationRecorder
        detector = PickleballDetector(mode="B", hsv_lower=(20, 35, 85), hsv_upper=(70, 255, 255))
    else:
        detector = PickleballDetector(mode="B", hsv_lower=(20, 35, 85), hsv_upper=(70, 255, 255), ball_hue_hint=hue_hint)
    tracker = BallKalmanTracker(metadata["fps"])
    recorder = ObservationRecorder(court, metadata["fps"])
    records = []
    previous_ok = True
    started = time.perf_counter()
    first = metadata["frame_start"] if warmup_start is None else warmup_start
    cap.set(cv2.CAP_PROP_POS_FRAMES, first)
    for frame_idx in range(first, metadata["frame_end_inclusive"]+1):
        if frame_idx < metadata["frame_start"]:
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError("Could not decode warmup frames")
        else:
            frame = cv2.imread(str(ROOT/"frames"/f"{frame_idx:06d}.png"))
        current_court = camera.update(frame)
        court_ok = camera.tracking_ok
        current_court = current_court if court_ok else None
        if not court_ok or not previous_ok:
            tracker.reset()
            detector.reset_ball_history()
        previous_ok = court_ok
        transform = camera.prev_to_curr_transform
        prediction = tracker.predict_position(frame_idx, camera_transform=transform)
        old_hue, old_saturation = detector._ball_hue, detector._ball_saturation
        measurement, confidence, players = detector.detect(frame, prediction, homography=current_court,
                                                          camera_transform=transform)
        point = tracker.update(frame_idx, measurement, confidence, homography=current_court,
                               camera_transform=transform)
        recorder.record(point, current_court, frame_idx/metadata["fps"])
        record = {"frame": frame_idx, "measurement": measurement, "confidence": confidence,
                        "prediction": prediction, "marker": [point.pixel_x, point.pixel_y] if point else None,
                        "interpolated": point.is_interpolated if point else None,
                        "track_generation": tracker.track_generation, "point": asdict(point) if point else None,
                        "court_ok": court_ok, "court_pixels": current_court.pixel_corners.tolist() if current_court else None,
                        "camera_step": transform.tolist(),
                        "ball_hue_before": old_hue, "ball_saturation_before": old_saturation,
                        "player_boxes": [list(p.bbox_xyxy) for p in players]}
        if frame_idx >= metadata["frame_start"]:
            records.append(record)
        if (frame_idx-metadata["frame_start"]) % 60 == 0:
            print("Simulated", frame_idx, flush=True)
    source_files = ["src/detector.py", "src/tracker.py", "src/config.py", "src/ball_3d.py", "src/court_motion.py"]
    cap.release()
    source_hashes = {p:hashlib.sha256((ROOT/"source_before"/Path(p).name if legacy else Path(p)).read_bytes()).hexdigest() for p in source_files}
    payload = {"metadata": metadata, "mode": "B", "warmup_start":first,"estimator_version":"source_before" if legacy else "current",
               "settings": {"hsv_lower": [20, 35, 85], "hsv_upper": [70, 255, 255], "hue_hint":hue_hint,"stride": 1},
               "calibration":court.to_dict(),
               "source_sha256":source_hashes,
               "elapsed_seconds": time.perf_counter()-started, "records": records}
    (ROOT/output).write_text(json.dumps(payload, default=lambda value: value.tolist(), indent=2))
    print("Saved", ROOT/output, flush=True)


def evaluate(simulation="baseline.json", output="baseline_metrics.json"):
    from src.ball_3d import ObservationRecorder, fit_flight, flight_segments, CameraModel
    from src.detector import PickleballDetector
    from src.homography import CourtHomography
    from src.tracker import BallPoint

    baseline = json.loads((ROOT/simulation).read_text())
    if baseline.get("estimator_version") == "source_before":
        archived = archived_module("ball_3d")
        fit_flight, flight_segments = archived.fit_flight, archived.flight_segments
        PickleballDetector = archived_module("detector").PickleballDetector
    labels = json.loads((ROOT/"manual_labels.json").read_text())["labels"]
    records = {r["frame"]: r for r in baseline["records"]}
    fps = baseline["metadata"]["fps"]
    court = CourtHomography.load_from_file("data/calibration/Gold_Medal_Match_Punta_Gorda_calibration.json")
    rows, reference_points, software_points, camera_diagnostics = [], [], [], []
    recorder = ObservationRecorder(court, fps)
    for label in labels:
        r = records[label["frame"]]
        point = np.array([label["x"], label["y"]]) if label["x"] is not None else None
        row = {"frame": label["frame"], "visibility": label["visibility"], "reference_xy": point.tolist() if point is not None else None,
               "measurement_error_px": None, "marker_error_px": None,
               "detector_hit": None, "marker_hit": None, "evidence": []}
        if label["visibility"] == "visible":
            for name, field in [("measurement", "measurement_error_px"), ("marker", "marker_error_px")]:
                if r[name] is not None:
                    row[field] = float(np.linalg.norm(np.asarray(r[name])-point))
            row["detector_hit"] = row["measurement_error_px"] is not None and row["measurement_error_px"] <= 10.
            row["marker_hit"] = row["marker_error_px"] is not None and row["marker_error_px"] <= 10.
            frame = cv2.imread(str(ROOT/"frames"/f'{label["frame"]:06d}.png'))
            current_court = CourtHomography(r["court_pixels"], image_size=(1920, 1080), metadata=court.metadata) if r["court_ok"] else None
            mask = PickleballDetector._get_court_mask(frame.shape[:2], current_court)
            x, y = np.rint(point).astype(int)
            row["inside_action_mask"] = bool(mask[y, x])
            boxes = r["player_boxes"]
            row["inside_player_body"] = PickleballDetector.is_inside_player_body(tuple(point), boxes)
            row["inside_shoe_zone"] = PickleballDetector.is_inside_shoe_zone(tuple(point), boxes)
            if not row["inside_action_mask"]:
                row["evidence"].append("reference_center_excluded_by_action_mask")
            if row["inside_player_body"]:
                row["evidence"].append("reference_center_inside_body_exclusion")
            if row["inside_shoe_zone"]:
                row["evidence"].append("reference_center_inside_shoe_exclusion")
        else:
            current_court = CourtHomography(r["court_pixels"], image_size=(1920, 1080), metadata=court.metadata) if r["court_ok"] else None
        if r["point"] is not None:
            software_points.append(BallPoint(**r["point"]))
        reference = BallPoint(label["frame"], *(point if point is not None else (0., 0.)),
                              measurement_pixel=tuple(point) if label["visibility"] == "visible" else None,
                              is_interpolated=label["visibility"] != "visible", confidence=1., track_generation=0)
        recorder.record(reference, current_court, label["frame"]/fps)
        reference_points.append(reference)
        if current_court is not None:
            try:
                camera = CameraModel.from_court(current_court)
                camera_diagnostics.append({"frame":label["frame"], "height_ft":float(camera.center[2]),
                                           "focal_px":float(camera.matrix[0, 0]), "corner_rms_px":camera.reprojection_error_px})
            except ValueError:
                pass
        rows.append(row)
    visible = [r for r in rows if r["visibility"] == "visible"]
    matched = [r["measurement_error_px"] for r in visible if r["detector_hit"]]
    def segments_report(points):
        segments = flight_segments(points, fps, 1)
        return [{"frame_start":points[indices[0]].frame_idx,"frame_end":points[indices[-1]].frame_idx,
                 "observations":len(indices), "status":f.status,"reason":f.reason,
                 "average_mph":f.average_mph,"relative_uncertainty":f.relative_uncertainty,
                 "reprojection_error_px":f.reprojection_error_px} for indices,f in segments]
    software_segments = segments_report(software_points)
    reference_segments = segments_report(reference_points)
    events = json.loads((ROOT/"manual_events.json").read_text())["events"]
    intervals = []
    start = labels[0]["frame"]
    for event in events:
        end = event["frame"]-event["uncertainty_frames"]
        intervals.append((start, end))
        start = event["frame"]+event["uncertainty_frames"]+1
    intervals.append((start, labels[-1]["frame"]))
    manual_flights = []
    for start,end in intervals:
        points = [p for p in reference_points if start <= p.frame_idx <= end and not p.is_interpolated]
        if not points:
            continue
        f = fit_flight(points, fps, 1)
        manual_flights.append({"frame_start":start,"frame_end":end,"observations":len(points),
                               "status":f.status,"reason":f.reason,"average_mph":f.average_mph,
                               "relative_uncertainty":f.relative_uncertainty,"reprojection_error_px":f.reprojection_error_px})
    result = {"method":"180-frame development audit. Independent hand-localized regions, local RGB snap and visual review. Approximate 3-5 px annotation uncertainty; 10 px hit tolerance. Occluded/partial/ambiguous frames excluded from strict pixel metrics. No independent 3D/radar speed ground truth.",
              "simulation":simulation,"source_sha256":baseline["source_sha256"],
              "replay_estimator_sha256":hashlib.sha256((ROOT/"source_before"/"ball_3d.py" if baseline.get("estimator_version") == "source_before" else Path("src/ball_3d.py")).read_bytes()).hexdigest(),
              "reference_labels_sha256":hashlib.sha256((ROOT/"manual_labels.json").read_bytes()).hexdigest(),
              "visibility_counts":dict(Counter(l["visibility"] for l in labels)),
              "visible_frames":len(visible),"correct_measured_detections":sum(r["detector_hit"] for r in visible),
              "visible_frames_with_no_detection":sum(records[r["frame"]]["measurement"] is None for r in visible),
              "visible_frames_with_wrong_detection":sum(records[r["frame"]]["measurement"] is not None and not r["detector_hit"] for r in visible),
              "correct_display_markers":sum(r["marker_hit"] for r in visible),
              "matched_detection_median_error_px":float(np.median(matched)) if matched else None,
              "matched_detection_p95_error_px":float(np.quantile(matched,.95)) if matched else None,
              "visible_action_mask_exclusions":[r["frame"] for r in visible if not r["inside_action_mask"]],
              "visible_body_exclusions":[r["frame"] for r in visible if r["inside_player_body"]],
              "visible_shoe_exclusions":[r["frame"] for r in visible if r["inside_shoe_zone"]],
              "software_segments":software_segments,"manual_positions_automatic_segments":reference_segments,
              "manual_positions_manual_event_intervals":manual_flights,
              "camera_diagnostics":camera_diagnostics,"frames":rows}
    (ROOT/output).write_text(json.dumps(result, indent=2))
    print(json.dumps({k:v for k,v in result.items() if k not in ["frames","camera_diagnostics","software_segments","manual_positions_automatic_segments","manual_positions_manual_event_intervals"]},indent=2))
    for name in ["software_segments","manual_positions_automatic_segments","manual_positions_manual_event_intervals"]:
        print(name, json.dumps(result[name]))


def render():
    """Produce an inspectable HTML viewer, MP4 and concise audit report."""
    labels = json.loads((ROOT/"manual_labels.json").read_text())["labels"]
    before = json.loads((ROOT/"baseline_warm.json").read_text())
    after = json.loads((ROOT/"candidate_warm.json").read_text())
    bm = json.loads((ROOT/"baseline_warm_metrics.json").read_text())
    am = json.loads((ROOT/"candidate_warm_metrics.json").read_text())
    cold_before = json.loads((ROOT/"baseline.json").read_text())
    cold_after = json.loads((ROOT/"candidate.json").read_text())
    cold_bm = json.loads((ROOT/"baseline_metrics.json").read_text())
    cold_am = json.loads((ROOT/"candidate_metrics.json").read_text())
    reference = {r["frame"]:r for r in labels}
    baseline = {r["frame"]:r for r in before["records"]}
    candidate = {r["frame"]:r for r in after["records"]}
    fps = before["metadata"]["fps"]
    outputs = Path("data/outputs")
    video_path = outputs/"gold_ball_frame_audit.mp4"
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (1920, 560))
    if not writer.isOpened():
        raise RuntimeError("Cannot create comparison video")
    previous = (960., 540.)
    key_tiles = []
    for frame_idx, label in reference.items():
        raw = cv2.imread(str(ROOT/"frames"/f"{frame_idx:06d}.png"))
        if label["x"] is not None:
            previous = (label["x"], label["y"])
        x0, y0 = max(0, min(1820, round(previous[0])-50)), max(0, min(980, round(previous[1])-50))
        panels = []
        for name, record in [("REVIEWED REFERENCE",None),("SOFTWARE BEFORE",baseline[frame_idx]),("SOFTWARE AFTER",candidate[frame_idx])]:
            frame = raw.copy()
            canvas = np.full((560, 640, 3), (22, 25, 30), np.uint8)
            color = (70, 230, 90)
            marker = None
            if record is None and label["x"] is not None:
                marker = (label["x"], label["y"])
                color = (200, 100, 220) if label["visibility"] != "visible" else color
            elif record is not None and record["marker"] is not None:
                marker = record["marker"]
                color = (0, 170, 255) if record["interpolated"] else (0, 220, 255)
            if marker is not None:
                cv2.circle(frame, tuple(map(round, marker)), 24, color, 3, cv2.LINE_AA)
            canvas[48:408] = cv2.resize(frame, (640, 360))
            canvas[414:554, 8:148] = cv2.resize(frame[y0:y0+100, x0:x0+100], (140, 140))
            cv2.putText(canvas, name, (12, 25), 0, .65, (240, 240, 240), 1, cv2.LINE_AA)
            cv2.putText(canvas, f"Frame {frame_idx} | {frame_idx/fps:.3f} s", (12, 43), 0, .4, (180, 180, 180), 1)
            if record is None:
                lines = ["Reference: "+label["visibility"], "Manually located region", "Local RGB snap + visual review", "Pixel scoring: visible frames only"]
            else:
                lines = ["Measurement: "+("missing" if record["measurement"] is None else "present"),
                         "Marker: "+("missing" if marker is None else "predicted" if record["interpolated"] else "observed")]
                if label["visibility"] == "visible" and record["measurement"] is not None:
                    error = np.linalg.norm(np.asarray(record["measurement"])-[label["x"],label["y"]])
                    lines += [f"Detection error: {error:.1f} px", "MATCH (<=10 px)" if error <= 10 else "WRONG LOCATION"]
                elif label["visibility"] == "visible":
                    lines += ["Visible ball, no detection", "MISS (counted as failure)"]
                else:
                    lines += ["No detection error scored", "Hidden/partial frames are excluded"]
            for line_idx, line in enumerate(lines):
                cv2.putText(canvas, line, (160, 438+line_idx*27), 0, .43, (220, 220, 220), 1, cv2.LINE_AA)
            panels.append(canvas)
        result = cv2.hconcat(panels)
        writer.write(result)
        if frame_idx in [160, 180, 240, 265, 282, 300]:
            key_tiles.append(cv2.resize(result, (1440, 420)))
    writer.release()
    cv2.imwrite(str(outputs/"gold_ball_audit_comparison.jpg"), cv2.vconcat(key_tiles))
    def count_qualified(records):
        return sum(r["status"] == "estimated" for r in records)
    text = f'''# Gold Medal frame audit

This is a **180-frame development audit**, source frames 150–329 ({180/fps:.2f} seconds), not whole-match accuracy validation.
Raw frames were inspected individually. Ball regions were hand-selected, centers optionally snapped using local RGB contrast,
and the overlays visually reviewed. Centers have approximately 3–5 px uncertainty. A detection counts as a match within 10 px.
Partial and ambiguous/hidden frames are excluded from strict pixel scoring. Neither simulator receives the labels.

| Result on {am['visible_frames']} clearly visible frames | Before | After |
|---|---:|---:|
| Correct measured detections | {bm['correct_measured_detections']} | {am['correct_measured_detections']} |
| No detection | {bm['visible_frames_with_no_detection']} | {am['visible_frames_with_no_detection']} |
| Wrong measured object/location | {bm['visible_frames_with_wrong_detection']} | {am['visible_frames_with_wrong_detection']} |
| Correct displayed marker (including predictions) | {bm['correct_display_markers']} | {am['correct_display_markers']} |

Both versions process from frame zero; scoring covers only the labeled frames 150–329. The before run acquires
the wrong appearance. This is a reproducible failure case, not a claim that the old software misses every ball
in every clip. The after run uses the same start and settings plus the selected yellow-ball colour hint.
A separate cold-start test beginning at frame 150 scores {cold_bm['correct_measured_detections']}/{cold_bm['visible_frames']}
before and {cold_am['correct_measured_detections']}/{cold_am['visible_frames']} after. These frames informed the fixes,
so a disjoint holdout is still needed. The viewer lets you switch between these two experiments.

There are {am['visibility_counts']['partial']} partial and {am['visibility_counts']['occluded_or_ambiguous']} ambiguous/hidden frames.
On matched detections, after median error is {am['matched_detection_median_error_px']:.2f} px and 95th percentile error
is {am['matched_detection_p95_error_px']:.2f} px. Those errors also include reference annotation uncertainty.

## Measured software limitations

1. Appearance acquisition depended on the first few frames. A false green-object acquisition could suppress the yellow ball.
2. The old action mask excluded the reference centers on frames {bm['visible_action_mask_exclusions']} at the lob apex.
3. Whole-player body/shoe exclusions overlap clearly visible balls on some frames. Overlap is evidence of a restrictive gate,
   not a complete causal attribution for every miss.
4. Automatic flight segmentation can split a continuous flight, or combine nearby bounce/contact events. In this sample,
   the reference bounce near frame 242 and paddle contact near frame 244 are much closer than the 0.18-second peak spacing.
5. The camera/physics model still rejects usable-looking tracks. Better 2D positions alone do not resolve every depth ambiguity.
6. Motion-mask centroids can drift toward the edge of a slowly moving ball, even when the correct object was selected.
   This is visible near the lob apex and affects the precision needed for speed estimation.

## Strict 3D replay

| Input to unchanged physical/uncertainty checks | Qualified flight fits |
|---|---:|
| Before software positions/boundaries | {count_qualified(bm['software_segments'])} |
| After software positions/boundaries | {count_qualified(am['software_segments'])} |
| Reviewed positions + automatic boundaries (before) | {count_qualified(bm['manual_positions_automatic_segments'])} |
| Reviewed positions + manually identified event intervals | {count_qualified(am['manual_positions_manual_event_intervals'])} |

The manually guided fits around 18.7 and 19.2 MPH are **model estimates**, not independently measured ball speeds.
There is no radar or synchronized second-camera ground truth. Passing these checks does not establish real MPH accuracy.
Automatic MPH remains unavailable where checks fail; no uncertainty thresholds were loosened.

## Files and reproduction

- `index.html`: frame slider, raw/reference/before/after views and per-frame errors.
- `manual_labels.json`, `manual_events.json`: reviewed references and approximate event intervals.
- `baseline_warm.json`, `candidate_warm.json`: simulation traces processing from frame zero.
- `baseline_warm_metrics.json`, `candidate_warm_metrics.json`: primary scoring and 3D replay results.
- `baseline.json`, `candidate.json` and associated metrics: separate cold-start experiment.
- `source_before/`: exact original source, verified against baseline SHA-256 hashes.
- Raw PNGs are under `frames/`; original/reference-review sheets under `sheets/`.
- `tools/ball_tracking_audit.py` has `sheet`, `simulate`, `evaluate`, `review`, and `render` commands.
- Reproduce before: `python -B tools/ball_tracking_audit.py simulate --output baseline_warm.json --warmup-start 0 --legacy`
- Simulate after: `python -B tools/ball_tracking_audit.py simulate --output candidate_warm.json --warmup-start 0 --hue-hint 32`
- Score: `python -B tools/ball_tracking_audit.py evaluate --simulation candidate_warm.json --output candidate_warm_metrics.json`
- Generate artifacts: `python -B tools/ball_tracking_audit.py render`
'''
    (ROOT/"REPORT.md").write_text(text)
    payload = {"labels":labels,"baseline":before["records"],"candidate":after["records"],"before":bm,"after":am,"fps":fps,
               "cold":{"baseline":cold_before["records"],"candidate":cold_after["records"],"before":cold_bm,"after":cold_am}}
    template = (Path(__file__).with_name("ball_audit_viewer.html")).read_text()
    (ROOT/"index.html").write_text(template.replace("__AUDIT_DATA__",json.dumps(payload).replace("</","<\\/")))
    print("Rendered", ROOT/"index.html", "and",video_path)


def main():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    sheets = commands.add_parser("sheet")
    sheets.add_argument("start", type=int)
    sheets.add_argument("end", type=int)
    sheets.add_argument("crop", type=int, nargs=4)
    sheets.add_argument("name")
    simulation = commands.add_parser("simulate")
    simulation.add_argument("--output", default="baseline.json")
    simulation.add_argument("--hue-hint", type=float)
    simulation.add_argument("--warmup-start", type=int)
    simulation.add_argument("--legacy", action="store_true")
    commands.add_parser("refine")
    commands.add_parser("review")
    commands.add_parser("render")
    evaluation = commands.add_parser("evaluate")
    evaluation.add_argument("--simulation", default="baseline.json")
    evaluation.add_argument("--output", default="baseline_metrics.json")
    args = parser.parse_args()
    if args.command == "sheet":
        sheet(args.start, args.end, args.crop, args.name)
    elif args.command == "simulate":
        simulate(args.output, args.hue_hint, args.warmup_start, args.legacy)
    elif args.command == "refine":
        refine_labels()
    elif args.command == "review":
        review_sheets()
    elif args.command == "evaluate":
        evaluate(args.simulation, args.output)
    elif args.command == "render":
        render()


if __name__ == "__main__":
    main()
