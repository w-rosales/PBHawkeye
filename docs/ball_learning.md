# Local temporal ball model

This is a small, trained convolutional network, not a copy of an LLM or its reasoning. It consumes the current frame and two previous frames and predicts a ball-center heatmap. Inference and training run locally without an API. It estimates 2D image position, not height or speed.

The seed checkpoint is `data/models/ball_v1/best_court.pt`. It has 79 training frames, 25 validation frames, 41 test frames, and a separate 26-visible-frame evaluation clip from the same match. Three-frame input windows do not overlap across splits. The checkpoint and score threshold were selected using validation only. Frame labels were created through assistant visual review with local RGB center snapping. They are approximate centers requiring independent human validation, not independently measured subpixel ground truth.

At the frozen validation-selected threshold, test results were 23 correct / 0 wrong / 18 withheld and later-clip results were 16 correct / 0 wrong / 10 withheld. Without thresholding, the network's top candidate localized 66 of those 67 visible frames within 10 pixels. The existing classical detector produced 34 correct / 0 wrong / 7 missing on the test block and 0 correct / 22 wrong / 4 missing on the later clip. The network is therefore promising but not uniformly better at its current operating threshold. These small, correlated samples are not production accuracy estimates.

## Dashboard review workflow

Open <http://127.0.0.1:8000/ball-testing> or use **Ball Tracking Test** in the main dashboard. Confirm the video's court in Match Analysis first. Select a short range (start with 3 seconds), its purpose, and ball color, then prepare the comparison. Both trackers run locally on every source frame. Preparing continues if the browser closes; interrupted jobs are marked for retry after a server restart.

Review the original frame first. Click the ball center once to save an initial visible reference and position the persistent magnified view. That position carries into the next frame as an **unsaved preview**. The red dot stays at the exact center of the magnified view, including near image edges. Use arrow keys to move the view one source pixel, or Shift + arrow for five pixels. Magnification can be changed to 3×, 5× or 8× without changing the label's source coordinates.

Press **Enter** to save the centered ball as visible and advance, or **P** to save it as partly visible and advance. **H** saves hidden and advances; **U** saves unsure and advances. Use **[ / ]**, Page Up / Down, the frame slider, or Previous/Next to change frames without confirming a preview. Escape resets adjustments to an existing reference (or the latest earlier reference). Existing saved references override carried positions when reopening a reviewed frame. Holding Enter does not automatically confirm successive frames.

Mouse clicks and visibility buttons still save immediately. Keyboard nudges alone do not save labels or mark a frame as reviewed. Corrections save to `data/ball_tests/<review-id>/manual_labels.json` with an edit history and revision checks. Failed saves block navigation until retried or discarded. Reopen the same review from **Saved reviews** to continue later.

Every review stores lossless source PNGs (including two prior context frames), predictions, the source video's identity, calibration and a frozen model checkpoint. Training/validation/final-test purpose cannot be changed. The API checks causal-frame overlap against saved reviews and the known seed experiment; the dataset preparer checks overlap again when combining exported datasets. Duplicated videos under different names are not content-fingerprinted and need manual split discipline.

**Export reviewed dataset** writes a versioned snapshot to `data/learning/reviews/<review-id>_r<revision>`. Later corrections do not alter an old snapshot. Use only the latest intended revision of each review when preparing the next training round. The existing `prepare --audit <export-directory>` command consumes these exports directly; repeat `--audit` for separate clips. Labels for absent, hidden, uncertain and partial balls are preserved, but the current trainer consumes visible-center labels only. Known-absent labels already support separate false-positive scoring in the dashboard.

Saving or exporting **never retrains or activates a model**. Training remains a separate explicit operation with new version directories and independent evaluation. Keep different clips for training, validation and final testing. Review metrics use saved references only; zero reviewed frames means no accuracy measurement yet. The classical comparison has up to 62 warmup frames before the requested clip rather than replaying the whole match, so its state may differ from a full-match run.

## Review the experiment

With Studio running, open <http://127.0.0.1:8000/validation/ball_model_experiment/>. Its comparison video shows reference, current classical detection and accepted neural detections on identical frames. The video is an inspection montage, not real-time playback.

Frozen machine-readable evaluation: `data/models/ball_v1/evaluation_court.json`. Run a fresh evaluation without changing weights or threshold:

```powershell
python -B tools/train_ball_model.py evaluate --checkpoint data/models/ball_v1/best_court.pt --court-calibration data/calibration/Gold_Medal_Match_Punta_Gorda_calibration.json --external-audit data/validation/gold_ball_holdout --output data/models/ball_v1/evaluation_repeat.json
python -B tools/ball_model_report.py
```

The report compares raw detector measurements, before Kalman filtering. Hidden/ambiguous and partial frames are excluded from strict scoring. These results cannot establish false-positive rates on ball-absent footage or accuracy on other courts. Heatmap scores are not calibrated probabilities. A validation-selected threshold deliberately withholds many otherwise well-localized candidates.

## Try the optional backend

Match Analysis still uses the existing detector. Ball Tracking Test compares that detector with the neural model, without changing Match Analysis. Use this explicit CLI option for an experimental analysis run. Supply distinct output paths to preserve existing analysis:

```powershell
python -B main.py --video data/raw_videos/Gold_Medal_Match_Punta_Gorda.mp4 --mode B --calibration-file data/calibration/Gold_Medal_Match_Punta_Gorda_calibration.json --ball-model data/models/ball_v1/best_court.pt --start-frame 418 --max-frames 32 --output-video data/outputs/neural_trial.mp4 --output-json data/outputs/neural_trial.json --output-csv data/outputs/neural_trial.csv
```

It needs two initial context frames. Loss of camera/court tracking resets that history. The network uses the calibrated court region, including lob headroom, to exclude spectators and adjacent courts. It bypasses the classical HSV and shoe/body rejection rules. The existing tracking and strict 3D speed qualification still apply; enabling the network does not guarantee a displayed MPH.

The current unoptimized CPU implementation is an accuracy experiment. Full-frame inference was measured in hundreds of milliseconds per frame during this run. It does not yet solve full-match processing speed. Benchmark separately on the target hardware before choosing accelerated inference, batching or a smaller input size. Changing input size requires validation again because the ball is small.

## Add labeled clips and train a new version

Extract short batches from varied rallies, near/far court, lobs, motion blur, paddle overlaps and lighting. Reserve separate clips, ideally separate matches/cameras, for validation and final testing before training. Do not tune repeatedly against the final test set.

```powershell
python -B tools/label_ball_frames.py --video data/raw_videos/Gold_Medal_Match_Punta_Gorda.mp4 --output data/validation/new_ball_clip --start 900 --count 180
```

Open `data/validation/new_ball_clip/label.html` locally, or its Studio `/validation/new_ball_clip/label.html` URL. Click the visible ball center, mark partial/uncertain frames honestly, choose the clip's split, and export `manual_labels.json` beside the page. Extraction includes the preceding two context frames. Repeat for other clips. Train and evaluate into **new version directories** to preserve the seed experiment:

```powershell
python -B tools/train_ball_model.py prepare --audit data/validation/training_clip --audit data/validation/validation_clip --audit data/validation/test_clip --output data/learning/ball_v2
python -B tools/train_ball_model.py train --manifest data/learning/ball_v2/manifest.json --output data/models/ball_v2 --epochs 40 --threads 4
python -B tools/train_ball_model.py calibrate --manifest data/learning/ball_v2/manifest.json --checkpoint data/models/ball_v2/best.pt --court-calibration data/calibration/Gold_Medal_Match_Punta_Gorda_calibration.json --output data/models/ball_v2/best_court.pt
python -B tools/train_ball_model.py evaluate --manifest data/learning/ball_v2/manifest.json --checkpoint data/models/ball_v2/best_court.pt --court-calibration data/calibration/Gold_Medal_Match_Punta_Gorda_calibration.json --output data/models/ball_v2/evaluation.json
```

The example calibration assumes every clip has the same fixed camera/court. Mixed-camera training is supported by labeled crops, but mixed-camera court-constrained evaluation needs per-clip calibration support first. Do not apply this one calibration to different cameras. The preparer rejects temporal input overlap between splits, caches resized inputs by content and records annotation hashes. Checkpoints are tensor state dictionaries loaded with `weights_only=True`.

For reliable MPH, the next stages remain contact/bounce detection, measured camera intrinsics and distortion, physically plausible 3D flights, and validation against independent speed measurements. Accurate 2D ball centers are necessary evidence for that work, not a direct speed measurement.
