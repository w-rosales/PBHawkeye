"""Regression tests for Tasks 4 & 5:
- Task 4: End-to-end pipeline evaluation matching application inference.
- Task 5: Confirmed absent / outside-frame examples usable for training.
"""

import json
from pathlib import Path
import pytest
import numpy as np
import cv2
import torch

from src.ball_model import (
    MODEL_FORMAT, TemporalBallNet, NeuralBallLocator, frame_tensor,
    heatmap_focal_loss
)
from tools.train_ball_model import (
    prepare, PatchDataset, summarize, operating_threshold, evaluate_pipeline
)


def create_mock_audit(audit_dir: Path, labels_spec: list, image_size=(64, 48)):
    w, h = image_size
    (audit_dir / "frames").mkdir(parents=True, exist_ok=True)
    all_frames = set()
    for item in labels_spec:
        f = item["frame"]
        all_frames.update([f - 2, f - 1, f])

    for f in all_frames:
        img = np.full((h, w, 3), 40, np.uint8)
        # If any visible label at f, draw the ball
        for item in labels_spec:
            if item["frame"] == f and item.get("visibility") == "visible" and item.get("x") is not None:
                cv2.circle(img, (int(item["x"]), int(item["y"])), 3, (0, 240, 240), -1)
        cv2.imwrite(str(audit_dir / "frames" / f"{f:06d}.png"), img)

    labels = []
    for item in labels_spec:
        lbl = {
            "frame": item["frame"],
            "visibility": item["visibility"],
            "x": item.get("x"),
            "y": item.get("y"),
            "confirmed": True,
            "origin": "manual"
        }
        if "split" in item:
            lbl["split"] = item["split"]
        labels.append(lbl)

    (audit_dir / "manual_labels.json").write_text(json.dumps({"labels": labels, "revision": 1}))
    (audit_dir / "metadata.json").write_text(json.dumps({
        "video": "fixture.mp4",
        "width": w,
        "height": h,
        "status": "completed"
    }))


def test_absent_labels_prepare_as_negative_training_data_with_finite_loss(tmp_path):
    audit = tmp_path / "absent_audit"
    # Create an absent-only clip
    labels = [{"frame": f, "visibility": "absent", "split": "train"} for f in range(10, 16)]
    create_mock_audit(audit, labels, image_size=(64, 48))

    output = tmp_path / "prepared_absent"
    prepare(audit, output, width=64)

    manifest_path = output / "manifest.json"
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text())
    assert len(manifest["samples"]) == 6
    for sample in manifest["samples"]:
        assert sample["target"] is None
        assert sample["visibility"] == "absent"

    # PatchDataset must load absent samples as pure negative background crops
    dataset = PatchDataset(manifest_path, patch_size=32, samples_per_frame=4)
    assert len(dataset) == 6 * 4

    # Check that all crops produce zero-target heatmaps
    all_zero_count = 0
    for idx in range(len(dataset)):
        inp, target = dataset[idx]
        assert inp.shape == (9, 32, 32)
        assert target.shape == (1, 16, 16)
        if target.max().item() == 0.0:
            all_zero_count += 1
    assert all_zero_count == len(dataset)

    # Loss computation on absent batch must be finite
    loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=False)
    model = TemporalBallNet()
    inputs, targets = next(iter(loader))
    loss = heatmap_focal_loss(model(inputs), targets)
    assert torch.isfinite(loss)
    assert loss.item() > 0.0


def test_mixed_audit_distinguishes_label_types_and_skips_partial(tmp_path):
    audit = tmp_path / "mixed_audit"
    labels = [
        {"frame": 10, "visibility": "visible", "x": 20.0, "y": 20.0, "split": "train"},
        {"frame": 11, "visibility": "absent", "split": "train"},
        {"frame": 12, "visibility": "partial", "x": 22.0, "y": 22.0, "split": "train"},
        {"frame": 13, "visibility": "occluded", "split": "train"},
        {"frame": 14, "visibility": "uncertain", "split": "train"},
        {"frame": 15, "visibility": "visible", "x": 25.0, "y": 25.0, "split": "train"},
    ]
    create_mock_audit(audit, labels, image_size=(64, 48))

    output = tmp_path / "prepared_mixed"
    prepare(audit, output, width=64)

    manifest = json.loads((output / "manifest.json").read_text())
    # Only visible (2) and absent (1) are consumed; partial, occluded, uncertain are skipped
    assert len(manifest["samples"]) == 3
    vis_samples = [s for s in manifest["samples"] if s["visibility"] == "visible"]
    abs_samples = [s for s in manifest["samples"] if s["visibility"] == "absent"]
    assert len(vis_samples) == 2
    assert len(abs_samples) == 1
    assert vis_samples[0]["target"] is not None
    assert abs_samples[0]["target"] is None

    counts = manifest["sources"][0]["counts"]
    assert counts["visible"] == 2
    assert counts["absent"] == 1
    assert counts["skipped_partial"] == 1
    assert counts["skipped_occluded"] == 1
    assert counts["skipped_uncertain"] == 1


def test_summarize_accurately_reports_presence_and_absence_metrics():
    records = [
        # Visible frames
        {"visibility": "visible", "score": 0.9, "error_px": 2.0, "inference_ms": 1.0, "target": [10.0, 10.0]},
        {"visibility": "visible", "score": 0.8, "error_px": 5.0, "inference_ms": 1.0, "target": [20.0, 20.0]},
        {"visibility": "visible", "score": 0.4, "error_px": 1.0, "inference_ms": 1.0, "target": [30.0, 30.0]}, # below threshold 0.5
        {"visibility": "visible", "score": 0.9, "error_px": 25.0, "inference_ms": 1.0, "target": [40.0, 40.0]}, # wrong accepted
        # Absent frames
        {"visibility": "absent", "score": 0.2, "error_px": None, "inference_ms": 1.0, "target": None}, # true negative
        {"visibility": "absent", "score": 0.3, "error_px": None, "inference_ms": 1.0, "target": None}, # true negative
        {"visibility": "absent", "score": 0.85, "error_px": None, "inference_ms": 1.0, "target": None}, # false positive at threshold 0.5
    ]

    stats = summarize(records, threshold=0.5)
    assert stats["visible_frames"] == 4
    assert stats["accepted"] == 3
    assert stats["correct_within_10px"] == 2
    assert stats["wrong_accepted"] == 1
    assert stats["missing"] == 1
    assert stats["recall"] == 0.5
    assert stats["localization_precision_on_visible_frames"] == pytest.approx(2 / 3)

    assert stats["absent_frames"] == 3
    assert stats["absent_false_positives"] == 1
    assert stats["absent_true_negatives"] == 2
    assert stats["absent_accuracy"] == pytest.approx(2 / 3)


def test_end_to_end_pipeline_evaluation_suppresses_static_distractor_and_tracks_gaps(tmp_path):
    # Create a small synthetic sequence:
    # A moving ball on frames 10..16
    # A stationary distractor at (50, 40) present on all frames
    audit = tmp_path / "e2e_audit"
    (audit / "frames").mkdir(parents=True)

    labels = []
    for f in range(8, 20):
        img = np.full((48, 64, 3), 30, np.uint8)
        # Stationary distractor on every frame
        cv2.circle(img, (50, 40), 3, (0, 255, 255), -1)

        # Moving ball on frames 10..15
        if 10 <= f <= 15:
            bx = 10.0 + (f - 10) * 4.0
            by = 10.0 + (f - 10) * 2.0
            cv2.circle(img, (int(bx), int(by)), 3, (0, 240, 240), -1)
            labels.append({"frame": f, "visibility": "visible", "x": bx, "y": by, "confirmed": True})
        elif 16 <= f <= 18:
            labels.append({"frame": f, "visibility": "absent", "x": None, "y": None, "confirmed": True})

        cv2.imwrite(str(audit / "frames" / f"{f:06d}.png"), img)

    (audit / "manual_labels.json").write_text(json.dumps({"labels": labels, "revision": 1}))
    (audit / "metadata.json").write_text(json.dumps({"video": "fixture.mp4", "width": 64, "height": 48}))

    # Save a qualified checkpoint with mock weights
    model = TemporalBallNet()
    ckpt_path = tmp_path / "best_court.pt"
    torch.save({
        "format": MODEL_FORMAT,
        "state_dict": model.state_dict(),
        "input_width": 64,
        "acceptance_threshold": 0.40,
        "epoch": 25,
        "requires_court_mask": False,
        "manifest_sha256": "mock"
    }, ckpt_path)

    # Run evaluate_pipeline
    e2e_res = evaluate_pipeline(ckpt_path, audit)
    assert "visible_frames" in e2e_res
    assert "absent_frames" in e2e_res
    assert "longest_streak_frames" in e2e_res
    assert "mean_gap_frames" in e2e_res
    assert e2e_res["visible_frames"] == 6
    assert e2e_res["absent_frames"] == 3

