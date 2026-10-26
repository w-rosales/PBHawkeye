"""Prepare, train and evaluate a local temporal ball heatmap network.

The seed audit is small. This is an experimental learning pipeline, not a
replacement for measured camera calibration, contact detection or 3D speed.
"""

import argparse
import hashlib
import json
from pathlib import Path
import random
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from src.ball_model import (MODEL_FORMAT, NeuralBallLocator, TemporalBallNet, frame_tensor,
                            heatmap_focal_loss, load_ball_model, peak_from_heatmap)


def is_confirmed_label(label: dict) -> bool:
    if "confirmed" in label:
        return bool(label["confirmed"])
    origin = label.get("origin", "manual")
    if origin in ("prefilled", "interpolated"):
        return False
    notes = str(label.get("notes", ""))
    if notes.startswith("prefill:") or notes.startswith("interpolated:"):
        return False
    return bool(label.get("reviewed", True))


def prepare(audit, output, width=960, external_audit=None):
    audits=list(audit) if isinstance(audit,(list,tuple)) else [audit]
    output.mkdir(parents=True, exist_ok=True)
    (output/"cache").mkdir(exist_ok=True)
    samples, sources = [], []
    for source_index, root in enumerate(audits+([external_audit] if external_audit else [])):
        labels_path = root/"manual_labels.json"
        labels = json.loads(labels_path.read_text())["labels"]
        metadata = json.loads((root/"metadata.json").read_text())
        source_name = f"audit{source_index}"
        available = {int(p.stem):p for p in (root/"frames").glob("*.png")}
        hard = {}
        trace = root/"baseline_warm.json"
        if trace.exists():
            hard = {r["frame"]:r["measurement"] for r in json.loads(trace.read_text())["records"] if r["measurement"] is not None}
        counts = {
            "visible": 0, "absent": 0, "skipped_unconfirmed": 0,
            "skipped_partial": 0, "skipped_occluded": 0, "skipped_uncertain": 0,
            "skipped_missing_context": 0
        }
        for label in labels:
            frame = int(label["frame"])
            vis = label.get("visibility", "visible")

            # Must be confirmed to become training reference data
            if not is_confirmed_label(label):
                counts["skipped_unconfirmed"] += 1
                continue

            if vis == "partial":
                counts["skipped_partial"] += 1
                continue
            elif vis == "occluded":
                counts["skipped_occluded"] += 1
                continue
            elif vis == "uncertain":
                counts["skipped_uncertain"] += 1
                continue
            elif vis not in ("visible", "absent"):
                continue

            if vis == "visible" and (label.get("x") is None or label.get("y") is None):
                continue

            if any(frame-offset not in available for offset in (0,1,2)):
                counts["skipped_missing_context"] += 1
                continue

            if label.get("split") in ["train","validation","test","external"]:
                split=label["split"]
            elif source_index >= len(audits):
                split = "external"
            elif frame <= 239:
                split = "train"
            elif 245 <= frame <= 269:
                split = "validation"
            elif frame >= 275:
                split = "test"
            else:
                continue  # Purge temporal-context gaps between chronological blocks.
            paths = []
            for index in range(frame-2, frame+1):
                raw_bytes=available[index].read_bytes()
                digest=hashlib.sha256(raw_bytes).hexdigest()[:16]
                cached = output/"cache"/f"{source_name}_{index:06d}_{width}_{digest}.npy"
                raw = cv2.imdecode(np.frombuffer(raw_bytes,np.uint8),cv2.IMREAD_COLOR)
                if raw is None:
                    raise ValueError(f"Cannot read frame: {available[index]}")
                resized = cv2.resize(raw, (width, round(raw.shape[0]*width/raw.shape[1])), interpolation=cv2.INTER_AREA)
                if not cached.exists():
                    np.save(cached, resized)
                paths.append(str(cached.relative_to(output)))
            scale = width/metadata["width"]
            if vis == "visible":
                counts["visible"] += 1
                samples.append({"source":source_name,"frame":frame,"split":split,"inputs":paths,
                                "input_keys":[[str(Path(metadata['video']).resolve()),i] for i in range(frame-2,frame+1)],
                                "raw_frame":str(available[frame]), "image_size":[metadata["width"],metadata["height"]],
                                "visibility":"visible",
                                "target":[label["x"]*scale,label["y"]*scale],
                                "target_native":[label["x"],label["y"]],
                                "hard_negative":[v*scale for v in hard[frame]] if frame in hard else None})
            else:
                counts["absent"] += 1
                samples.append({"source":source_name,"frame":frame,"split":split,"inputs":paths,
                                "input_keys":[[str(Path(metadata['video']).resolve()),i] for i in range(frame-2,frame+1)],
                                "raw_frame":str(available[frame]), "image_size":[metadata["width"],metadata["height"]],
                                "visibility":"absent",
                                "target":None,
                                "target_native":None,
                                "hard_negative":[v*scale for v in hard[frame]] if frame in hard else None})
        sources.append({"audit":str(root),"labels_sha256":hashlib.sha256(labels_path.read_bytes()).hexdigest(),"metadata":metadata,"counts":counts})
    for a,b in [("train","validation"),("train","test"),("validation","test"),("train","external"),("validation","external"),("test","external")]:
        left = {tuple(p) for s in samples if s["split"]==a for p in s["input_keys"]}
        right = {tuple(p) for s in samples if s["split"]==b for p in s["input_keys"]}
        if left & right:
            raise ValueError(f"Temporal input leakage between {a} and {b}")
    payload = {"format":MODEL_FORMAT,"input_width":width,"sources":sources,"samples":samples,
               "split_policy":"Explicit per-label splits when present; otherwise chronological blocks: train through 239, validation 245-269, test 275 onward. Three-frame causal inputs, no shared input frames. External audit never used to train/select thresholds.",
               "limitation":"Seed annotations are approximate and from one court/short clip. These splits do not demonstrate cross-camera generalization."}
    (output/"manifest.json").write_text(json.dumps(payload,indent=2))
    print(json.dumps({name:sum(s["split"]==name for s in samples) for name in ["train","validation","test","external"]}))


class PatchDataset(Dataset):
    def __init__(self, manifest_path, patch_size=160, samples_per_frame=8, seed=7):
        self.root = manifest_path.parent
        self.manifest = json.loads(manifest_path.read_text())
        self.samples = [s for s in self.manifest["samples"] if s["split"] == "train"]
        self.patch_size = patch_size
        self.samples_per_frame = samples_per_frame
        self.seed, self.epoch = seed, 0
        # Only training images are cached here; no validation/test data are loaded.
        from concurrent.futures import ThreadPoolExecutor
        paths = list({p for s in self.samples for p in s["inputs"]})
        with ThreadPoolExecutor(max_workers=16) as pool:
            self.cache = dict(pool.map(lambda p: (p, np.load(self.root/p)), paths))

    def __len__(self):
        return len(self.samples)*self.samples_per_frame

    def __getitem__(self, index):
        sample = self.samples[index//self.samples_per_frame]
        rng = np.random.default_rng(self.seed+self.epoch*100000+index)
        frames = [self.cache[p] for p in sample["inputs"]]
        height,width = frames[0].shape[:2]
        size = self.patch_size
        if size > min(height,width):
            raise ValueError("Training patches must fit within the resized image")
        is_absent = (sample.get("target") is None)
        if is_absent:
            positive = False
            target = None
        else:
            target = np.asarray(sample["target"],float)
            positive = index % self.samples_per_frame < self.samples_per_frame//2
        if positive:
            center = target+rng.uniform(-size*.35,size*.35,2)
            origin = np.rint(center-size/2).astype(int)
            origin = np.clip(origin,[0,0],[width-size,height-size])
        else:
            for attempt in range(100):
                if attempt == 0 and sample["hard_negative"] is not None:
                    origin = np.asarray(sample["hard_negative"])-size/2
                    origin = np.clip(np.rint(origin).astype(int),[0,0],[width-size,height-size])
                else:
                    origin = np.array([rng.integers(0,width-size+1),rng.integers(0,height-size+1)])
                if target is None:
                    break
                # Background crops must not contain the labeled ball, even near an edge.
                if np.any(target < origin-12) or np.any(target > origin+size+12):
                    break
            else:
                raise ValueError("Could not sample a ball-free background crop")
        x0,y0 = map(int,origin)
        crops = [f[y0:y0+size,x0:x0+size].copy() for f in frames]
        local = (target-origin) if target is not None else None
        if rng.random() < .5:
            crops = [np.ascontiguousarray(f[:,::-1]) for f in crops]
            if local is not None:
                local[0] = size-1-local[0]
        # Same photometric transform for all frames: no artificial temporal flicker.
        gain = rng.uniform(.8,1.2)
        channels = rng.uniform(.94,1.06,(1,1,3))
        offset = rng.uniform(-8,8)
        crops = [np.clip(f.astype(float)*gain*channels+offset,0,255).astype(np.uint8) for f in crops]
        target_map = np.zeros((size//2,size//2),np.float32)
        if positive and local is not None:
            cx,cy = np.rint(local/2).astype(int)
            cx,cy = np.clip([cx,cy],[0,0],[size//2-1,size//2-1])
            yy,xx = np.mgrid[:size//2,:size//2]
            target_map = np.exp(-((xx-cx)**2+(yy-cy)**2)/(2*1.3**2)).astype(np.float32)
        return frame_tensor(crops), torch.from_numpy(target_map[None])


@torch.inference_mode()
def predict_split(model, manifest, root, split, court_mask=None):
    records = []
    for sample in manifest["samples"]:
        if sample["split"] != split:
            continue
        frames = [np.load(root/p) for p in sample["inputs"]]
        started = time.perf_counter()
        probability = model(frame_tensor(frames).unsqueeze(0))[0,0].sigmoid().cpu().numpy()
        predicted, score = peak_from_heatmap(probability, sample["image_size"], court_mask)
        vis = sample.get("visibility", "visible")
        tgt = sample.get("target_native")
        err_px = float(np.linalg.norm(np.asarray(predicted)-tgt)) if tgt is not None else None
        records.append({"source":sample["source"],"frame":sample["frame"],"visibility":vis,
                        "predicted":list(predicted),"score":score,"target":tgt,
                        "error_px":err_px,
                        "inference_ms":(time.perf_counter()-started)*1000,
                        "raw_frame":sample["raw_frame"]})
    return records


@torch.inference_mode()
def predict_external(model, audit, input_width, court_mask=None):
    """Evaluate new labels without modifying the frozen training manifest."""
    labels_path=audit/"manual_labels.json"
    labels=json.loads(labels_path.read_text())["labels"]
    metadata=json.loads((audit/"metadata.json").read_text())
    records=[]
    for label in labels:
        if not is_confirmed_label(label):
            continue
        vis = label.get("visibility", "visible")
        if vis not in ("visible", "absent"):
            continue
        if vis == "visible" and label.get("x") is None:
            continue
        paths=[audit/"frames"/f'{label["frame"]-offset:06d}.png' for offset in [2,1,0]]
        if not all(p.exists() for p in paths):
            continue
        frames=[]
        for p in paths:
            frame=cv2.imread(str(p))
            frames.append(cv2.resize(frame,(input_width,round(frame.shape[0]*input_width/frame.shape[1])),interpolation=cv2.INTER_AREA))
        started=time.perf_counter()
        probability=model(frame_tensor(frames).unsqueeze(0))[0,0].sigmoid().cpu().numpy()
        predicted,score=peak_from_heatmap(probability,(metadata["width"],metadata["height"]),court_mask)
        tgt = [label["x"], label["y"]] if vis == "visible" else None
        err_px = float(np.linalg.norm(np.asarray(predicted)-tgt)) if tgt is not None else None
        records.append({"source":"external","frame":label["frame"],"visibility":vis,
                        "predicted":list(predicted),"score":score,"target":tgt,
                        "error_px":err_px,
                        "inference_ms":(time.perf_counter()-started)*1000,"raw_frame":str(paths[-1])})
    return records,{"audit":str(audit),"labels_sha256":hashlib.sha256(labels_path.read_bytes()).hexdigest(),"metadata":metadata}


def summarize(records, threshold=None):
    visible = [r for r in records if r.get("visibility", "visible") == "visible" and (r.get("target") is not None or r.get("error_px") is not None)]
    absent = [r for r in records if r.get("visibility") == "absent" or (r.get("target") is None and r.get("error_px") is None and "visibility" in r)]

    accepted_visible = visible if threshold is None else [r for r in visible if r["score"] >= threshold]
    correct_visible = [r for r in accepted_visible if r.get("error_px") is not None and r["error_px"] <= 10.]

    accepted_absent = absent if threshold is None else [r for r in absent if r["score"] >= threshold]
    absent_fp = len(accepted_absent)
    absent_tn = len(absent) - absent_fp

    return {
        "visible_frames": len(visible),
        "accepted": len(accepted_visible),
        "correct_within_10px": len(correct_visible),
        "wrong_accepted": len(accepted_visible) - len(correct_visible),
        "missing": len(visible) - len(accepted_visible),
        "recall": len(correct_visible) / max(1, len(visible)) if visible else 0.0,
        "localization_precision_on_visible_frames": len(correct_visible) / max(1, len(accepted_visible)) if accepted_visible else 0.0,
        "median_error_all_predictions_px": float(np.median([r["error_px"] for r in visible if r.get("error_px") is not None])) if visible else None,
        "absent_frames": len(absent),
        "absent_false_positives": absent_fp,
        "absent_true_negatives": absent_tn,
        "absent_accuracy": absent_tn / max(1, len(absent)) if absent else 1.0,
        "median_inference_ms": float(np.median([r["inference_ms"] for r in records])) if records else None
    }


def operating_threshold(records, target_precision=.95):
    eligible = []
    for threshold in sorted({r["score"] for r in records}):
        result = summarize(records, threshold)
        total_wrong = result["wrong_accepted"] + result["absent_false_positives"]
        if result["accepted"] >= 5 and result["localization_precision_on_visible_frames"] >= target_precision:
            if result["absent_frames"] > 0 and result["absent_false_positives"] > 0.5 * result["absent_frames"]:
                continue
            eligible.append((result["correct_within_10px"], -total_wrong, threshold))
    return max(eligible)[2] if eligible else None


class BackgroundDataLoader:
    def __init__(self, dataloader, queue_size=4):
        self.dataloader = dataloader
        self.queue_size = queue_size
        import threading
        self._current_cancel = None
        self._lock = threading.Lock()

    def cancel(self):
        with self._lock:
            if self._current_cancel is not None:
                self._current_cancel.set()

    def __iter__(self):
        import queue
        import threading
        q = queue.Queue(maxsize=self.queue_size)
        sentinel = object()
        exc_holder = []
        cancel_event = threading.Event()
        with self._lock:
            self._current_cancel = cancel_event

        def producer():
            try:
                for batch in self.dataloader:
                    if cancel_event.is_set():
                        break
                    while not cancel_event.is_set():
                        try:
                            q.put(batch, timeout=0.05)
                            break
                        except queue.Full:
                            continue
            except BaseException as exc:
                exc_holder.append(exc)
            finally:
                while True:
                    try:
                        q.put((sentinel, exc_holder[0] if exc_holder else None), timeout=0.05)
                        break
                    except queue.Full:
                        if cancel_event.is_set():
                            break

        thread = threading.Thread(target=producer, daemon=True)
        thread.start()

        batches_yielded = 0
        try:
            while True:
                item = q.get()
                if isinstance(item, tuple) and len(item) == 2 and item[0] is sentinel:
                    err = item[1]
                    if err is not None:
                        raise RuntimeError(f"Background DataLoader failed after {batches_yielded} batches") from err
                    break
                batches_yielded += 1
                yield item
        finally:
            cancel_event.set()
            with self._lock:
                if self._current_cancel is cancel_event:
                    self._current_cancel = None


def train(manifest_path, output, epochs=40, threads=14, batch_size=32, seed=7, patch_size=160):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.set_num_threads(threads)
    output.mkdir(parents=True,exist_ok=True)
    manifest = json.loads(manifest_path.read_text())
    dataset = PatchDataset(manifest_path,patch_size=patch_size,seed=seed)
    if len(dataset) == 0 or not any(s["split"] == "validation" for s in manifest["samples"]):
        raise ValueError("Training and validation samples are required")
    raw_loader = DataLoader(dataset,batch_size=batch_size,shuffle=True,num_workers=0,
                            generator=torch.Generator().manual_seed(seed))
    loader = BackgroundDataLoader(raw_loader, queue_size=4)
    model = TemporalBallNet()
    optimizer = torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
    history = []
    best_score = (-1.,-float("inf"))
    started = time.perf_counter()
    for epoch in range(epochs):
        dataset.epoch = epoch
        model.train(); losses=[]
        for inputs,targets in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = heatmap_focal_loss(model(inputs),targets)
            if not torch.isfinite(loss):
                raise ValueError("Training loss became nonfinite")
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),10.)
            optimizer.step();losses.append(float(loss.detach()))
        if not losses:
            raise RuntimeError(f"Epoch {epoch+1} yielded 0 batches; training loader stopped unexpectedly")
        epoch_loss = float(np.mean(losses))
        if not np.isfinite(epoch_loss):
            raise ValueError(f"Epoch {epoch+1} loss became nonfinite: {epoch_loss}")
        row={"epoch":epoch+1,"loss":epoch_loss,"elapsed_sec":round(time.perf_counter()-started,2)}
        if epoch == 0 or (epoch+1) % 5 == 0 or epoch+1 == epochs:
            model.eval()
            validation = predict_split(model,manifest,manifest_path.parent,"validation")
            stats = summarize(validation)
            row["validation"] = stats
            score = (stats["recall"],-stats["median_error_all_predictions_px"])
            if score > best_score:
                best_score=score
                threshold = operating_threshold(validation)
                checkpoint={"format":MODEL_FORMAT,"state_dict":model.state_dict(),"input_width":manifest["input_width"],
                            "epoch":epoch+1,"seed":seed,"acceptance_threshold":threshold,
                            "manifest_sha256":hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
                            "validation":stats,"validation_at_threshold":summarize(validation,threshold) if threshold is not None else None,
                            "status":"experimental_single_court_seed_model","training_note":manifest["limitation"]}
                torch.save(checkpoint,output/"best.tmp.pt")
                (output/"best.tmp.pt").replace(output/"best.pt")
        history.append(row)
        (output/"training_history.json").write_text(json.dumps(history,indent=2))
        print(json.dumps(row),flush=True)
    print("Saved",output/"best.pt",flush=True)


def make_court_mask(calibration):
    from src.homography import CourtHomography
    from src.detector import PickleballDetector
    court=CourtHomography.load_from_file(calibration)
    if court.image_size is None:
        raise ValueError("Court calibration needs image dimensions")
    return PickleballDetector._get_court_mask(court.image_size[::-1],court)


def calibrate(manifest_path,checkpoint_path,calibration,output):
    """Select a court-constrained threshold using validation only, never tests."""
    torch.set_num_threads(4)
    model,checkpoint=load_ball_model(checkpoint_path)
    manifest=json.loads(manifest_path.read_text())
    if checkpoint["manifest_sha256"] != hashlib.sha256(manifest_path.read_bytes()).hexdigest():
        raise ValueError("Training manifest changed")
    validation=predict_split(model,manifest,manifest_path.parent,"validation",make_court_mask(calibration))
    threshold=operating_threshold(validation)
    checkpoint["unconstrained_validation"]=checkpoint["validation"]
    checkpoint["validation"]=summarize(validation)
    checkpoint["acceptance_threshold"]=threshold
    checkpoint["validation_at_threshold"]=summarize(validation,threshold) if threshold is not None else None
    checkpoint["requires_court_mask"]=True
    checkpoint["court_mask_note"]="Use the current calibrated playing-court region, including lob headroom; no player-box or HSV vetoes."
    checkpoint["threshold_calibration_sha256"]=hashlib.sha256(Path(calibration).read_bytes()).hexdigest()
    output.parent.mkdir(parents=True,exist_ok=True);torch.save(checkpoint,output)
    print(json.dumps({"epoch":checkpoint["epoch"],"threshold":threshold,"validation":checkpoint["validation"],
                      "validation_at_threshold":checkpoint["validation_at_threshold"]},indent=2))


@torch.inference_mode()
def evaluate_pipeline(checkpoint_path, audit_dir, court_calibration=None):
    """Evaluate full application inference pipeline (NeuralBallLocator) chronologically."""
    labels_path = audit_dir / "manual_labels.json"
    if not labels_path.exists():
        raise FileNotFoundError(f"Missing manual_labels.json in {audit_dir}")
    labels_data = json.loads(labels_path.read_text(encoding="utf-8"))
    labels = {l["frame"]: l for l in labels_data.get("labels", []) if is_confirmed_label(l)}

    locator = NeuralBallLocator(checkpoint_path)
    court_mask = make_court_mask(court_calibration) if court_calibration else None
    homography = None
    if court_calibration:
        from src.homography import CourtHomography
        court = CourtHomography.load_from_file(court_calibration)
        homography = court

    frames_dir = audit_dir / "frames"
    available_frames = sorted([int(p.stem) for p in frames_dir.glob("*.png")])
    if not available_frames:
        return {"error": "No frames found"}

    records = []
    tracked_streak = 0
    longest_streak = 0
    current_gap = 0
    gaps = []

    for f in available_frames:
        frame_img = cv2.imread(str(frames_dir / f"{f:06d}.png"))
        if frame_img is None:
            continue
        started = time.perf_counter()
        point, score = locator.update(frame_img, court_mask=court_mask, homography=homography)
        elapsed_ms = (time.perf_counter() - started) * 1000

        target_lbl = labels.get(f)
        if target_lbl is not None:
            vis = target_lbl.get("visibility", "visible")
            target_pt = [target_lbl["x"], target_lbl["y"]] if target_lbl.get("x") is not None else None
            err_px = float(np.linalg.norm(np.asarray(point) - target_pt)) if (point is not None and target_pt is not None) else None
            records.append({
                "frame": f,
                "visibility": vis,
                "detected": point is not None,
                "point": list(point) if point is not None else None,
                "score": float(score),
                "target": target_pt,
                "error_px": err_px,
                "inference_ms": elapsed_ms
            })
            if point is not None:
                tracked_streak += 1
                longest_streak = max(longest_streak, tracked_streak)
                if current_gap > 0:
                    gaps.append(current_gap)
                    current_gap = 0
            else:
                tracked_streak = 0
                current_gap += 1

    if current_gap > 0:
        gaps.append(current_gap)

    visible_records = [r for r in records if r["visibility"] == "visible" and r["target"] is not None]
    absent_records = [r for r in records if r["visibility"] == "absent"]
    correct_visible = [r for r in visible_records if r["detected"] and r["error_px"] is not None and r["error_px"] <= 10.0]
    accepted_visible = [r for r in visible_records if r["detected"]]
    absent_fp = sum(1 for r in absent_records if r["detected"])

    return {
        "visible_frames": len(visible_records),
        "accepted": len(accepted_visible),
        "correct_within_10px": len(correct_visible),
        "wrong_accepted": len(accepted_visible) - len(correct_visible),
        "missing": len(visible_records) - len(accepted_visible),
        "recall": len(correct_visible) / max(1, len(visible_records)) if visible_records else 0.0,
        "precision": len(correct_visible) / max(1, len(accepted_visible)) if accepted_visible else 0.0,
        "median_error_px": float(np.median([r["error_px"] for r in correct_visible])) if correct_visible else None,
        "absent_frames": len(absent_records),
        "absent_false_positives": absent_fp,
        "absent_true_negatives": len(absent_records) - absent_fp,
        "absent_accuracy": (len(absent_records) - absent_fp) / max(1, len(absent_records)) if absent_records else 1.0,
        "longest_streak_frames": longest_streak,
        "mean_gap_frames": float(np.mean(gaps)) if gaps else 0.0,
        "max_gap_frames": max(gaps) if gaps else 0,
        "records": records
    }


def evaluate(manifest_path,checkpoint_path,output,external_audit=None,calibration=None):
    torch.set_num_threads(4)
    manifest=json.loads(manifest_path.read_text());model,checkpoint=load_ball_model(checkpoint_path)
    if checkpoint["manifest_sha256"] != hashlib.sha256(manifest_path.read_bytes()).hexdigest():
        raise ValueError("Manifest differs from the checkpoint; use the original manifest for a comparable evaluation")
    if checkpoint.get("requires_court_mask") and calibration is None:
        raise ValueError("This checkpoint requires --court-calibration")
    court_mask=make_court_mask(calibration) if calibration is not None else None
    ckpt_sha = hashlib.sha256(Path(checkpoint_path).read_bytes()).hexdigest()
    result={"checkpoint":str(checkpoint_path),"checkpoint_sha256":ckpt_sha,"epoch":checkpoint["epoch"],"threshold":checkpoint.get("acceptance_threshold"),
            "note":"Threshold and checkpoint selected using validation only. Test/external frames were never used for optimization or threshold selection. All visible-frame labels are approximate; no 3D speed ground truth.","splits":{},"records":{}}
    for split in ["validation","test","external"]:
        if split=="external" and external_audit is not None:
            records,provenance=predict_external(model,external_audit,manifest["input_width"],court_mask)
            result["external_provenance"]=provenance
        else:
            records=predict_split(model,manifest,manifest_path.parent,split,court_mask)
        result["records"][split]=records
        threshold=checkpoint.get("acceptance_threshold")
        result["splits"][split]={"unthresholded":summarize(records),
                                  "validation_threshold":summarize(records,threshold) if threshold is not None else None}

    if external_audit is not None:
        try:
            result["pipeline_external"] = evaluate_pipeline(checkpoint_path, external_audit, calibration)
        except Exception as exc:
            result["pipeline_external_error"] = str(exc)

    output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k != "records"},indent=2))


def main():
    parser=argparse.ArgumentParser();commands=parser.add_subparsers(dest="command",required=True)
    p=commands.add_parser("prepare");p.add_argument("--audit",type=Path,action="append",help="Repeat for multiple annotated clips");p.add_argument("--output",type=Path,default=Path("data/learning/ball_v1"));p.add_argument("--width",type=int,default=960);p.add_argument("--external-audit",type=Path)
    p=commands.add_parser("train");p.add_argument("--manifest",type=Path,default=Path("data/learning/ball_v1/manifest.json"));p.add_argument("--output",type=Path,default=Path("data/models/ball_v1"));p.add_argument("--epochs",type=int,default=40);p.add_argument("--threads",type=int,default=14);p.add_argument("--batch-size",type=int,default=32);p.add_argument("--seed",type=int,default=7)
    p=commands.add_parser("calibrate");p.add_argument("--manifest",type=Path,default=Path("data/learning/ball_v1/manifest.json"));p.add_argument("--checkpoint",type=Path,default=Path("data/models/ball_v1/best.pt"));p.add_argument("--court-calibration",type=Path,required=True);p.add_argument("--output",type=Path,default=Path("data/models/ball_v1/best_court.pt"))
    p=commands.add_parser("evaluate");p.add_argument("--manifest",type=Path,default=Path("data/learning/ball_v1/manifest.json"));p.add_argument("--checkpoint",type=Path,default=Path("data/models/ball_v1/best.pt"));p.add_argument("--output",type=Path,default=Path("data/models/ball_v1/evaluation.json"));p.add_argument("--external-audit",type=Path);p.add_argument("--court-calibration",type=Path)
    args=parser.parse_args()
    if args.command=="prepare":prepare(args.audit or Path("data/validation/gold_ball_audit"),args.output,args.width,args.external_audit)
    elif args.command=="train":train(args.manifest,args.output,args.epochs,args.threads,args.batch_size,args.seed)
    elif args.command=="calibrate":calibrate(args.manifest,args.checkpoint,args.court_calibration,args.output)
    else:evaluate(args.manifest,args.checkpoint,args.output,args.external_audit,args.court_calibration)


if __name__=="__main__":main()
