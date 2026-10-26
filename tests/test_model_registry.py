"""Regression tests for Task 6: Model registry, qualification state machine, and explicit promotion."""

import json
from pathlib import Path
import pytest
import torch

from src.ball_model import MODEL_FORMAT, TemporalBallNet
from src.model_registry import ModelRegistry, RegisteredModel, get_model_registry


def create_dummy_checkpoint(path: Path, filename="best_court.pt", threshold=0.45, epoch=5, corrupt=False):
    path.mkdir(parents=True, exist_ok=True)
    target = path / filename
    if corrupt:
        target.write_bytes(b"corrupt non-torch bytes")
        return target

    model = TemporalBallNet()
    data = {
        "format": MODEL_FORMAT,
        "state_dict": model.state_dict(),
        "input_width": 960,
        "epoch": epoch,
    }
    if threshold is not None:
        data["acceptance_threshold"] = float(threshold)
    torch.save(data, target)
    return target


def create_dummy_evaluation(path: Path, ckpt_path: Path, epoch=5, threshold=0.45, empty=False, missing_splits=False, mismatch_sha=False):
    import hashlib
    target = path / "evaluation.json"
    if empty:
        target.write_text("{}", encoding="utf-8")
        return target

    if missing_splits:
        target.write_text(json.dumps({"checkpoint": str(ckpt_path), "splits": {}}), encoding="utf-8")
        return target

    sha = hashlib.sha256(ckpt_path.read_bytes()).hexdigest()
    if mismatch_sha:
        sha = "0000000000000000000000000000000000000000000000000000000000000000"

    data = {
        "checkpoint": str(ckpt_path),
        "checkpoint_sha256": sha,
        "epoch": epoch,
        "threshold": threshold,
        "splits": {
            "validation": {
                "validation_threshold": {
                    "visible_frames": 50,
                    "accepted": 45,
                    "recall": 0.90,
                    "localization_precision_on_visible_frames": 0.95
                }
            }
        }
    }
    target.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return target


def test_empty_directory_and_no_checkpoint_is_unavailable_for_promotion(tmp_path):
    registry = ModelRegistry(tmp_path)
    (tmp_path / "ball_v1").mkdir()
    models = registry.scan_models()
    assert "ball_v1" in models
    assert models["ball_v1"].state == "uninitialized"
    assert models["ball_v1"].deployable is False

    with pytest.raises(ValueError, match="Cannot promote unqualified model"):
        registry.promote_model("ball_v1")


def test_active_training_checkpoint_is_unavailable_for_promotion(tmp_path):
    registry = ModelRegistry(tmp_path)
    v1_dir = tmp_path / "ball_v1"
    create_dummy_checkpoint(v1_dir)
    # Active training lock
    (v1_dir / "training.lock").write_text("lock")

    models = registry.scan_models()
    assert models["ball_v1"].state == "training"
    assert models["ball_v1"].deployable is False

    with pytest.raises(ValueError, match="Cannot promote unqualified model"):
        registry.promote_model("ball_v1")


def test_calibrated_but_unevaluated_checkpoint_is_not_deployable(tmp_path):
    """Codex Finding F: A best_court.pt with a threshold must NOT be deployable without holdout evaluation."""
    registry = ModelRegistry(tmp_path)
    v1_dir = tmp_path / "ball_v1"
    create_dummy_checkpoint(v1_dir, filename="best_court.pt", threshold=0.45)

    models = registry.scan_models()
    assert models["ball_v1"].state == "calibrated"
    assert models["ball_v1"].deployable is False
    assert "holdout evaluation unqualified" in models["ball_v1"].status_reason.lower()

    with pytest.raises(ValueError, match="Cannot promote unqualified model"):
        registry.promote_model("ball_v1")


def test_empty_evaluation_json_remains_calibrated_and_not_deployable(tmp_path):
    """Codex Finding F: An empty evaluation.json ({}) must NOT qualify a model."""
    registry = ModelRegistry(tmp_path)
    v1_dir = tmp_path / "ball_v1"
    ckpt = create_dummy_checkpoint(v1_dir, filename="best_court.pt", threshold=0.45)
    create_dummy_evaluation(v1_dir, ckpt, empty=True)

    models = registry.scan_models()
    assert models["ball_v1"].state == "calibrated"
    assert models["ball_v1"].deployable is False

    with pytest.raises(ValueError, match="Cannot promote unqualified model"):
        registry.promote_model("ball_v1")


def test_mismatched_evaluation_sha_is_rejected_from_qualification(tmp_path):
    """Evaluation evidence must match the checkpoint bytes exactly."""
    registry = ModelRegistry(tmp_path)
    v1_dir = tmp_path / "ball_v1"
    ckpt = create_dummy_checkpoint(v1_dir, filename="best_court.pt", threshold=0.45)
    create_dummy_evaluation(v1_dir, ckpt, mismatch_sha=True)

    models = registry.scan_models()
    assert models["ball_v1"].state == "calibrated"
    assert models["ball_v1"].deployable is False
    assert "does not match checkpoint" in models["ball_v1"].status_reason


def test_corrupt_checkpoint_is_marked_invalid(tmp_path):
    registry = ModelRegistry(tmp_path)
    v1_dir = tmp_path / "ball_v1"
    create_dummy_checkpoint(v1_dir, filename="best_court.pt", corrupt=True)

    models = registry.scan_models()
    assert models["ball_v1"].state == "invalid"
    assert models["ball_v1"].deployable is False


def test_qualified_model_promotion_and_persistence(tmp_path):
    registry = ModelRegistry(tmp_path)
    v1_dir = tmp_path / "ball_v1"
    ckpt = create_dummy_checkpoint(v1_dir, filename="best_court.pt", threshold=0.45, epoch=10)
    create_dummy_evaluation(v1_dir, ckpt, epoch=10, threshold=0.45)

    models = registry.scan_models()
    assert models["ball_v1"].state == "qualified"
    assert models["ball_v1"].deployable is True

    promoted = registry.promote_model("ball_v1")
    assert promoted.name == "ball_v1"
    assert (tmp_path / "active_model.json").is_file()

    active_rec = registry.get_active_model_record()
    assert active_rec["model_name"] == "ball_v1"
    assert active_rec["checkpoint_sha256"] == promoted.sha256


def test_adding_newer_directory_does_not_change_active_model(tmp_path):
    """Codex Finding F: Merely adding a newer directory does not change the active model."""
    registry = ModelRegistry(tmp_path)
    # v1 is promoted
    v1_dir = tmp_path / "ball_v1"
    ckpt1 = create_dummy_checkpoint(v1_dir, filename="best_court.pt", threshold=0.45, epoch=10)
    create_dummy_evaluation(v1_dir, ckpt1, epoch=10, threshold=0.45)
    registry.promote_model("ball_v1")

    # Add v2 (also qualified)
    v2_dir = tmp_path / "ball_v2"
    ckpt2 = create_dummy_checkpoint(v2_dir, filename="best_court.pt", threshold=0.50, epoch=20)
    create_dummy_evaluation(v2_dir, ckpt2, epoch=20, threshold=0.50)

    # get_deployable_model without argument MUST return the promoted v1, not v2!
    active = registry.get_deployable_model()
    assert active.name == "ball_v1"
    assert active.checkpoint_file == ckpt1

    # Explicit promotion changes active model to v2
    promoted2 = registry.promote_model("ball_v2")
    assert registry.get_deployable_model().name == "ball_v2"

