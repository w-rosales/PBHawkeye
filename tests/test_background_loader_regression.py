"""Regression tests for Task 3: BackgroundDataLoader exception propagation and cleanup."""

import time
import pytest
from tools.train_ball_model import BackgroundDataLoader


class FailingIterable:
    """Mock dataset loader that fails at a specified batch index."""
    def __init__(self, fail_at_batch=None, total_batches=10, exc_type=ValueError):
        self.fail_at_batch = fail_at_batch
        self.total_batches = total_batches
        self.exc_type = exc_type

    def __iter__(self):
        for i in range(self.total_batches):
            if self.fail_at_batch is not None and i == self.fail_at_batch:
                raise self.exc_type(f"Simulated failure at batch {i}")
            yield f"batch_{i}"


def test_normal_exhaustion_yields_all_batches():
    loader = FailingIterable(fail_at_batch=None, total_batches=8)
    bg = BackgroundDataLoader(loader, queue_size=2)
    results = list(bg)
    assert len(results) == 8
    assert results[0] == "batch_0"
    assert results[-1] == "batch_7"


def test_failure_before_first_batch_propagates_to_caller():
    loader = FailingIterable(fail_at_batch=0, total_batches=5)
    bg = BackgroundDataLoader(loader, queue_size=2)
    with pytest.raises(RuntimeError, match="Background DataLoader failed after 0 batches") as exc_info:
        list(bg)
    assert isinstance(exc_info.value.__cause__, ValueError)
    assert "Simulated failure at batch 0" in str(exc_info.value.__cause__)


def test_failure_after_some_batches_propagates_and_rejects_epoch():
    loader = FailingIterable(fail_at_batch=3, total_batches=10)
    bg = BackgroundDataLoader(loader, queue_size=2)
    yielded = []
    with pytest.raises(RuntimeError, match="Background DataLoader failed after 3 batches") as exc_info:
        for batch in bg:
            yielded.append(batch)
    assert len(yielded) == 3
    assert yielded == ["batch_0", "batch_1", "batch_2"]
    assert isinstance(exc_info.value.__cause__, ValueError)


def test_cancellation_with_full_queue_terminates_promptly():
    # Producer has 1000 batches, queue size 2. Consumer breaks after 2 batches.
    loader = FailingIterable(fail_at_batch=None, total_batches=1000)
    bg = BackgroundDataLoader(loader, queue_size=2)
    t0 = time.perf_counter()
    count = 0
    for batch in bg:
        count += 1
        if count >= 2:
            break
    elapsed = time.perf_counter() - t0
    # Must terminate promptly (under 0.5s) without hanging or thread leak
    assert elapsed < 0.5
    assert count == 2


def test_multiple_epochs_reusing_same_loader_yields_all_batches():
    """Regression test for Codex Finding A: loader must yield full batches across multiple epochs."""
    items = [f"item_{i}" for i in range(5)]
    bg = BackgroundDataLoader(items, queue_size=2)
    epoch_results = [list(bg) for _ in range(4)]
    for epoch_idx, batches in enumerate(epoch_results):
        assert len(batches) == 5, f"Epoch {epoch_idx} failed: got {len(batches)} batches, expected 5"
        assert batches == items


def test_fixture_training_multiple_epochs_produces_finite_losses(tmp_path):
    """Ensure train() function executes multiple epochs without empty epoch or NaN loss."""
    import json
    import numpy as np
    import cv2
    import torch
    from tools.train_ball_model import train, prepare

    audit = tmp_path / "train_audit"
    (audit / "frames").mkdir(parents=True)
    labels = []
    # 20 frames
    for f in range(20):
        img = np.full((96, 128, 3), 40, np.uint8)
        if f >= 5:
            cv2.circle(img, (20, 20), 3, (0, 240, 240), -1)
            labels.append({"frame": f, "visibility": "visible", "x": 20.0, "y": 20.0, "confirmed": True})
        cv2.imwrite(str(audit / "frames" / f"{f:06d}.png"), img)

    # Set train and validation splits with temporal gap to prevent causal leakage
    filtered_labels = []
    for l in labels:
        if l["frame"] <= 12:
            l["split"] = "train"
            filtered_labels.append(l)
        elif l["frame"] >= 16:
            l["split"] = "validation"
            filtered_labels.append(l)

    (audit / "manual_labels.json").write_text(json.dumps({"labels": filtered_labels, "revision": 1}))
    (audit / "metadata.json").write_text(json.dumps({"video": "fixture.mp4", "width": 128, "height": 96}))

    prepared = tmp_path / "prepared"
    prepare(audit, prepared, width=128)

    output = tmp_path / "model_out"
    train(prepared / "manifest.json", output, epochs=3, threads=2, batch_size=4, seed=42, patch_size=32)

    history = json.loads((output / "training_history.json").read_text())
    assert len(history) == 3
    for row in history:
        assert row["loss"] > 0.0
        assert np.isfinite(row["loss"])

