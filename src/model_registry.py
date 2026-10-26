"""Centralized model registry for local ball tracking neural networks.

Manages version discovery, qualification states, checkpoint provenance,
explicit promotion, and safe model deployment without hardcoded version ladders
or silent fallbacks.
"""

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from typing import Dict, List, Literal, Optional

import torch

from src.ball_model import MODEL_FORMAT, load_ball_model

ModelState = Literal["uninitialized", "training", "trained", "calibrated", "qualified", "invalid"]


@dataclass
class RegisteredModel:
    name: str
    version_int: int
    path: Path
    checkpoint_file: Optional[Path]
    state: ModelState
    acceptance_threshold: Optional[float]
    format: Optional[str]
    sha256: Optional[str]
    epoch: Optional[int]
    evaluation_summary: Optional[dict]
    deployable: bool
    status_reason: str


class ModelRegistry:
    def __init__(self, models_root: Optional[Path] = None):
        self.root = Path(models_root or (Path(__file__).resolve().parents[1] / "data" / "models"))
        self.root.mkdir(parents=True, exist_ok=True)
        self.active_file = self.root / "active_model.json"

    def scan_models(self) -> Dict[str, RegisteredModel]:
        """Discovers all models under models root and strictly validates qualification state."""
        models: Dict[str, RegisteredModel] = {}
        if not self.root.exists():
            return models

        for path in sorted(self.root.iterdir()):
            if not path.is_dir():
                continue
            name = path.name
            version_int = 0
            if name.startswith("ball_v"):
                try:
                    version_int = int(name.replace("ball_v", ""))
                except ValueError:
                    version_int = 0

            best_court = path / "best_court.pt"
            best_pt = path / "best.pt"
            eval_file = path / "evaluation.json"
            history_file = path / "training_history.json"
            lock_file = path / "training.lock"

            # Check if training is currently active
            is_active_training = lock_file.is_file()
            if history_file.is_file():
                try:
                    hist_data = json.loads(history_file.read_text(encoding="utf-8"))
                    if isinstance(hist_data, dict) and hist_data.get("status") == "running":
                        is_active_training = True
                except Exception:
                    pass

            if is_active_training:
                models[name] = RegisteredModel(
                    name=name, version_int=version_int, path=path, checkpoint_file=None,
                    state="training", acceptance_threshold=None, format=None, sha256=None,
                    epoch=None, evaluation_summary=None, deployable=False,
                    status_reason="Training is actively in progress."
                )
                continue

            # Deployment and qualification requires a calibrated court checkpoint
            ckpt_to_use = best_court if best_court.is_file() else (best_pt if best_pt.is_file() else None)

            if ckpt_to_use is None:
                if history_file.is_file():
                    models[name] = RegisteredModel(
                        name=name, version_int=version_int, path=path, checkpoint_file=None,
                        state="training", acceptance_threshold=None, format=None, sha256=None,
                        epoch=None, evaluation_summary=None, deployable=False,
                        status_reason="Training history exists but no saved checkpoint found."
                    )
                else:
                    models[name] = RegisteredModel(
                        name=name, version_int=version_int, path=path, checkpoint_file=None,
                        state="uninitialized", acceptance_threshold=None, format=None, sha256=None,
                        epoch=None, evaluation_summary=None, deployable=False,
                        status_reason="No checkpoint file found in model directory."
                    )
                continue

            # Inspect checkpoint binary safely
            try:
                raw_bytes = ckpt_to_use.read_bytes()
                sha = hashlib.sha256(raw_bytes).hexdigest()
                ckpt_data = torch.load(ckpt_to_use, map_location="cpu", weights_only=True)
                if not isinstance(ckpt_data, dict) or "state_dict" not in ckpt_data:
                    models[name] = RegisteredModel(
                        name=name, version_int=version_int, path=path, checkpoint_file=ckpt_to_use,
                        state="invalid", acceptance_threshold=None, format=None, sha256=sha,
                        epoch=None, evaluation_summary=None, deployable=False,
                        status_reason="Checkpoint is missing state_dict or has invalid dictionary format."
                    )
                    continue

                fmt = ckpt_data.get("format")
                threshold = ckpt_data.get("acceptance_threshold")
                epoch = ckpt_data.get("epoch")

                if fmt != MODEL_FORMAT:
                    models[name] = RegisteredModel(
                        name=name, version_int=version_int, path=path, checkpoint_file=ckpt_to_use,
                        state="invalid", acceptance_threshold=None, format=fmt, sha256=sha,
                        epoch=epoch, evaluation_summary=None, deployable=False,
                        status_reason=f"Unsupported format '{fmt}'"
                    )
                    continue

                # Uncalibrated best.pt (without court calibration) cannot be deployed or qualified
                if ckpt_to_use.name != "best_court.pt":
                    models[name] = RegisteredModel(
                        name=name, version_int=version_int, path=path, checkpoint_file=ckpt_to_use,
                        state="trained", acceptance_threshold=threshold, format=fmt, sha256=sha,
                        epoch=epoch, evaluation_summary=None, deployable=False,
                        status_reason="Checkpoint requires court mask calibration (best_court.pt) before deployment."
                    )
                    continue

                # Calibrated checkpoint check: requires valid threshold in (0, 1]
                if threshold is None or not (0.0 < float(threshold) <= 1.0):
                    models[name] = RegisteredModel(
                        name=name, version_int=version_int, path=path, checkpoint_file=ckpt_to_use,
                        state="trained", acceptance_threshold=None, format=fmt, sha256=sha,
                        epoch=epoch, evaluation_summary=None, deployable=False,
                        status_reason="Checkpoint has no qualifying validation threshold."
                    )
                    continue

                # Evaluation check: qualification requires non-empty, valid holdout evaluation matching checkpoint
                eval_summary = None
                eval_error = None
                if not eval_file.is_file():
                    eval_error = "Holdout evaluation is missing."
                else:
                    try:
                        eval_data = json.loads(eval_file.read_text(encoding="utf-8"))
                        if not isinstance(eval_data, dict) or not eval_data:
                            eval_error = "Evaluation file is empty or invalid."
                        else:
                            splits = eval_data.get("splits")
                            if not isinstance(splits, dict) or not splits:
                                eval_error = "Evaluation contains no holdout splits."
                            elif not any(s in splits for s in ("validation", "test", "external")):
                                eval_error = "Evaluation missing required validation/test/external splits."
                            else:
                                # Validate evaluation metrics are present
                                has_metrics = False
                                for split_name, split_metrics in splits.items():
                                    if isinstance(split_metrics, dict):
                                        vt = split_metrics.get("validation_threshold") or split_metrics.get("unthresholded")
                                        if isinstance(vt, dict) and any(k in vt for k in ("recall", "localization_precision_on_visible_frames", "visible_frames")):
                                            has_metrics = True
                                            break
                                if not has_metrics:
                                    eval_error = "Evaluation splits contain no valid verification metrics."

                                # Provenance matching: check sha256, path, epoch, and threshold consistency
                                if not eval_error:
                                    if "checkpoint_sha256" in eval_data:
                                        if eval_data["checkpoint_sha256"] != sha:
                                            eval_error = f"Evaluation checkpoint sha256 ({eval_data['checkpoint_sha256'][:8]}...) does not match checkpoint ({sha[:8]}...)."
                                    elif "checkpoint" in eval_data:
                                        if Path(eval_data["checkpoint"]).name != ckpt_to_use.name:
                                            eval_error = f"Evaluation was run on '{eval_data['checkpoint']}', not '{ckpt_to_use.name}'."

                                    if not eval_error and eval_data.get("epoch") is not None:
                                        if int(eval_data["epoch"]) != int(epoch):
                                            eval_error = f"Evaluation was run on epoch {eval_data['epoch']}, not checkpoint epoch {epoch}."

                                    if not eval_error and eval_data.get("threshold") is not None:
                                        if not math.isclose(float(eval_data["threshold"]), float(threshold), rel_tol=1e-3):
                                            eval_error = f"Evaluation threshold {eval_data['threshold']} differs from checkpoint threshold {threshold}."

                                if not eval_error:
                                    eval_summary = {
                                        "splits": splits,
                                        "threshold": eval_data.get("threshold"),
                                        "epoch": eval_data.get("epoch"),
                                    }
                    except Exception as exc:
                        eval_error = f"Could not parse evaluation file: {exc}"

                if eval_summary is not None:
                    models[name] = RegisteredModel(
                        name=name, version_int=version_int, path=path, checkpoint_file=ckpt_to_use,
                        state="qualified", acceptance_threshold=float(threshold), format=fmt, sha256=sha,
                        epoch=epoch, evaluation_summary=eval_summary, deployable=True,
                        status_reason="Fully calibrated and evaluated with holdout metrics."
                    )
                else:
                    models[name] = RegisteredModel(
                        name=name, version_int=version_int, path=path, checkpoint_file=ckpt_to_use,
                        state="calibrated", acceptance_threshold=float(threshold), format=fmt, sha256=sha,
                        epoch=epoch, evaluation_summary=None, deployable=False,
                        status_reason=f"Calibrated with acceptance threshold, but holdout evaluation unqualified: {eval_error}"
                    )
            except Exception as exc:
                models[name] = RegisteredModel(
                    name=name, version_int=version_int, path=path, checkpoint_file=ckpt_to_use,
                    state="invalid", acceptance_threshold=None, format=None, sha256=None,
                    epoch=None, evaluation_summary=None, deployable=False,
                    status_reason=f"Corrupt checkpoint: {exc}"
                )

        return models

    def promote_model(self, name: str) -> RegisteredModel:
        """Promotes a qualified model to be the explicit active model.

        Fails with ValueError if the model is not qualified, incomplete, or invalid.
        """
        scanned = self.scan_models()
        if name not in scanned:
            raise ValueError(f"Model '{name}' not found in registry.")

        m = scanned[name]
        if m.state != "qualified" or not m.deployable or m.checkpoint_file is None:
            raise ValueError(f"Cannot promote unqualified model '{name}': {m.status_reason}")

        payload = {
            "model_name": m.name,
            "checkpoint_file": m.checkpoint_file.name,
            "checkpoint_sha256": m.sha256,
            "acceptance_threshold": m.acceptance_threshold,
            "promoted_at": datetime.now(timezone.utc).isoformat(),
            "status_reason": m.status_reason
        }
        tmp = self.active_file.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        tmp.replace(self.active_file)
        return m

    def get_active_model_record(self) -> Optional[dict]:
        """Reads active_model.json if present."""
        if self.active_file.is_file():
            try:
                return json.loads(self.active_file.read_text(encoding="utf-8"))
            except Exception:
                return None
        return None

    def get_deployable_model(self, requested_name: Optional[str] = None) -> RegisteredModel:
        """Resolves the requested model or returns the explicit active model.

        If no active model has been recorded, automatically promotes the highest qualified model.
        Merely adding a newer directory does not change the active model.
        Raises ValueError if requested model is unqualified, incomplete, or not found.
        """
        scanned = self.scan_models()
        if requested_name:
            if requested_name not in scanned:
                raise ValueError(f"Requested model '{requested_name}' does not exist in registry.")
            m = scanned[requested_name]
            if not m.deployable or m.checkpoint_file is None:
                raise ValueError(f"Model '{requested_name}' is not deployable: {m.status_reason}")
            return m

        # Check explicit active model
        active_rec = self.get_active_model_record()
        if active_rec is not None:
            active_name = active_rec.get("model_name")
            active_sha = active_rec.get("checkpoint_sha256")
            if active_name in scanned:
                m = scanned[active_name]
                if m.deployable and m.checkpoint_file is not None and m.sha256 == active_sha:
                    return m
                raise ValueError(f"Explicit active model '{active_name}' is no longer qualified: {m.status_reason}")
            raise ValueError(f"Explicit active model '{active_name}' no longer exists in {self.root}.")

        # If no explicit promotion recorded yet, promote the highest qualified model
        qualified = [m for m in scanned.values() if m.deployable and m.state == "qualified" and m.checkpoint_file is not None]
        if not qualified:
            raise ValueError("No qualified deployable ball model found in registry.")

        qualified.sort(key=lambda m: (m.version_int, m.name), reverse=True)
        return self.promote_model(qualified[0].name)


_global_registry = ModelRegistry()


def get_model_registry() -> ModelRegistry:
    return _global_registry
