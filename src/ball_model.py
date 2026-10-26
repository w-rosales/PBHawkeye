"""Small causal temporal heatmap model for local ball localization.

This learns from labeled images. It neither contains LLM weights nor measures
3D ball speed. Heatmap scores are not calibrated accuracy probabilities.
"""

from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

MODEL_FORMAT = "pbhawkeye_temporal_ball_v1"


class TemporalBallNet(nn.Module):
    """Three RGB frames -> current-frame heatmap at half input resolution."""

    def __init__(self):
        super().__init__()

        def block(a, b, stride=1):
            return nn.Sequential(nn.Conv2d(a, b, 3, stride=stride, padding=1), nn.SiLU(),
                                 nn.Conv2d(b, b, 3, padding=1), nn.SiLU())

        self.first = block(9, 12, 2)
        self.second = block(12, 24, 2)
        self.third = block(24, 32, 2)
        self.up_second = block(56, 24)
        self.up_first = block(36, 12)
        self.head = nn.Conv2d(12, 1, 1)
        nn.init.constant_(self.head.bias, -4.6)

    def forward(self, frames):
        first = self.first(frames)
        second = self.second(first)
        third = self.third(second)
        up = F.interpolate(third, size=second.shape[-2:], mode="bilinear", align_corners=False)
        up = self.up_second(torch.cat([up, second], dim=1))
        up = F.interpolate(up, size=first.shape[-2:], mode="bilinear", align_corners=False)
        return self.head(self.up_first(torch.cat([up, first], dim=1)))


def heatmap_focal_loss(logits, target):
    """Sparse heatmap focal loss, including all-negative background crops."""
    probability = logits.sigmoid().clamp(1e-5, 1-1e-5)
    positive = target >= .99
    negative_weight = (1-target).pow(4)
    positive_loss = -(1-probability).pow(2)*probability.log()*positive
    negative_loss = -probability.pow(2)*(1-probability).log()*negative_weight*(~positive)
    return (positive_loss.sum()+negative_loss.sum())/positive.sum().clamp(min=1)


def frame_tensor(frames):
    """Three uint8 BGR images in time order, normalized consistently."""
    rgb = np.concatenate([cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames], axis=2)
    return torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1))).float()/127.5-1.


def peak_from_heatmap(probability, image_size, court_mask=None):
    """Return a model peak and score in original image coordinates."""
    heatmap = np.asarray(probability, dtype=float)
    if heatmap.ndim != 2 or not np.isfinite(heatmap).all() or not heatmap.size:
        raise ValueError("A finite two-dimensional heatmap is required")
    if court_mask is not None:
        width,height=image_size
        if court_mask.shape != (height,width):
            raise ValueError("Court mask dimensions must match the source image")
        allowed=cv2.resize(np.uint8(court_mask>0), (heatmap.shape[1],heatmap.shape[0]),interpolation=cv2.INTER_NEAREST)>0
        if not allowed.any():
            raise ValueError("Court mask excludes the entire image")
        heatmap=np.where(allowed,heatmap,-1.)
    y, x = np.unravel_index(np.argmax(heatmap), heatmap.shape)
    yy, xx = np.mgrid[max(0,y-1):min(heatmap.shape[0],y+2), max(0,x-1):min(heatmap.shape[1],x+2)]
    weights = np.maximum(heatmap[yy, xx],0.)
    if weights.sum() > 0:
        cx, cy = float(np.average(xx, weights=weights)), float(np.average(yy, weights=weights))
    else:
        cx, cy = float(x), float(y)
    width, height = image_size
    return (cx*width/heatmap.shape[1], cy*height/heatmap.shape[0]), float(heatmap[y,x])


def load_ball_model(path, device="cpu"):
    # Model files contain tensors and basic metadata, not serialized model code.
    checkpoint = torch.load(Path(path), map_location=device, weights_only=True)
    if checkpoint.get("format") != MODEL_FORMAT:
        raise ValueError("Unsupported ball model checkpoint")
    model = TemporalBallNet().to(device)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    return model, checkpoint


class NeuralBallLocator:
    """Optional local model runtime; no HSV/player-box exclusions or API calls.

    A checkpoint without a validation-selected operating threshold cannot be
    silently enabled. Experimental evaluation can explicitly supply a threshold.
    """

    def __init__(self, checkpoint_path, threshold: Optional[float] = None, threads=4):
        torch.set_num_threads(max(1, threads))
        self.model, self.metadata = load_ball_model(checkpoint_path)
        selected = self.metadata.get("acceptance_threshold") if threshold is None else threshold
        if selected is None or not 0 <= selected <= 1:
            raise ValueError("This model has no qualifying validation threshold; evaluate it before enabling inference")
        self.threshold = float(selected)
        self.width = int(self.metadata["input_width"])
        self.frames = []
        self.stationary_tracks = []

    def reset(self):
        self.frames.clear()
        self.stationary_tracks.clear()

    @torch.inference_mode()
    def update(self, frame, court_mask=None, homography=None):
        height, width = frame.shape[:2]
        small_height = max(8, round(height*self.width/width))
        small = cv2.resize(frame, (self.width, small_height), interpolation=cv2.INTER_AREA)
        self.frames.append(small)
        self.frames = self.frames[-3:]
        if len(self.frames) < 3:
            return None, 0.
        if self.metadata.get("requires_court_mask") and court_mask is None:
            raise ValueError("This model's validation threshold requires a calibrated court mask")
        probability = self.model(frame_tensor(self.frames).unsqueeze(0))[0, 0].sigmoid().cpu().numpy()

        if court_mask is not None:
            if court_mask.shape != (height, width):
                raise ValueError("Court mask dimensions must match the source image")
            allowed = cv2.resize(np.uint8(court_mask > 0), (probability.shape[1], probability.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
            if not allowed.any():
                raise ValueError("Court mask excludes the entire image")
            probability = np.where(allowed, probability, -1.)

        from scipy.ndimage import maximum_filter
        local_max = (maximum_filter(probability, size=5) == probability) & (probability >= self.threshold)
        ys, xs = np.where(local_max)

        candidates = []
        for y, x in zip(ys, xs):
            yy, xx = np.mgrid[max(0, y-1):min(probability.shape[0], y+2), max(0, x-1):min(probability.shape[1], x+2)]
            weights = np.maximum(probability[yy, xx], 0.)
            cx_sub = float(np.average(xx, weights=weights)) if weights.sum() > 0 else float(x)
            cy_sub = float(np.average(yy, weights=weights)) if weights.sum() > 0 else float(y)
            px = cx_sub * width / probability.shape[1]
            py = cy_sub * height / probability.shape[0]
            candidates.append((float(probability[y, x]), (px, py)))

        if not candidates:
            center, score = peak_from_heatmap(probability, (width, height), court_mask=None)
            if score >= self.threshold:
                candidates.append((score, center))

        candidates.sort(key=lambda c: c[0], reverse=True)

        # Update candidate tracks for stationary distractor detection
        for t in self.stationary_tracks:
            t["age"] += 1

        matched_tracks = set()
        for score, (px, py) in candidates:
            best_idx = None
            best_dist = 15.0
            for idx, t in enumerate(self.stationary_tracks):
                if idx in matched_tracks:
                    continue
                d = float(np.hypot(px - t["last"][0], py - t["last"][1]))
                if d < best_dist:
                    best_dist = d
                    best_idx = idx

            if best_idx is not None:
                matched_tracks.add(best_idx)
                t = self.stationary_tracks[best_idx]
                t["last"] = (float(px), float(py))
                t["history"].append((float(px), float(py)))
                t["age"] = 0
                drift = float(np.hypot(px - t["start"][0], py - t["start"][1]))
                t["max_drift"] = max(t["max_drift"], drift)
                # Displacement > 2.0 px confirms movement (e.g. 1 px/frame, subpixel lob apex drift)
                if t["max_drift"] > 2.0:
                    t["is_stationary"] = False
                elif len(t["history"]) >= 6 and t["max_drift"] <= 1.5:
                    t["is_stationary"] = True
            else:
                self.stationary_tracks.append({
                    "start": (float(px), float(py)),
                    "last": (float(px), float(py)),
                    "history": [(float(px), float(py))],
                    "max_drift": 0.0,
                    "is_stationary": False,
                    "age": 0
                })

        # Prune inactive tracks
        self.stationary_tracks = [t for t in self.stationary_tracks if t["age"] <= 30]

        valid_detection = None
        for score, (px, py) in candidates:
            # 1. Lateral playing corridor check if ground homography is provided
            if homography is not None:
                cx, cy = homography.pixel_to_court(px, py)
                # If on the ground surface near the court, check lateral bounds:
                # Do NOT reject in-flight balls where cy > 60 or cy < -10 because Z > 0 projects
                # far beyond the baseline or past the vanishing line.
                if -10.0 <= cy <= 60.0:
                    if not (-4.0 <= cx <= 24.0):
                        continue

            # 2. Stationary veto: reject only if candidate is near a confirmed stationary object
            # that has stayed within 1.5 px across 8+ frames.
            is_distractor = False
            for t in self.stationary_tracks:
                if t["is_stationary"]:
                    if np.hypot(px - t["start"][0], py - t["start"][1]) <= 6.0:
                        is_distractor = True
                        break
            if is_distractor:
                continue

            valid_detection = ((px, py), score)
            break

        if valid_detection is not None:
            return valid_detection
        return (None, candidates[0][0] if candidates else 0.)
