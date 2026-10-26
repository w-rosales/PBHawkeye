# PBHawkeye implementation handoff

This document is the implementation checklist from the October 25, 2026 audit. Implement in stages and return the evidence requested below so a separate reviewer can audit completion. Check current source before changing it: the workspace is being actively developed. This is a specification, not a claim that the changes are already implemented.

## Current state and constraints

- Project: `C:\Users\willi\Documents\PBHawkeye`; local FastAPI Studio, OpenCV tracking, PyTorch temporal ball heatmap model.
- The user is running `ball_v4` training. Preserve the running process, its manifest/cache, checkpoints and output directory. Changes to training code apply to a subsequent versioned run; do not overwrite v4 or silently restart it.
- v4's audited manifest contains 944 training, 63 validation, 41 test and 26 external visible-ball frames. Of its training frames, 865 come from the user's review exports and 79 from the original seed audit. No shared three-frame input windows were found between these splits. Verify this again after dataset changes.
- Epoch 10 localized the strongest candidate within 10 native pixels on 52/63 validation frames. This is raw network localization, not end-to-end tracking accuracy or validated MPH.
- The user wants local inference without per-frame LLM API charges. Preserve the working court/player pipeline.
- Corrections save automatically, but saving/exporting must not silently train or activate a model. The user wants MPH displayed only when a single estimate passes the existing qualification checks.
- Preserve annotations, histories, model provenance and older dataset exports. Use new output/version directories. Do not treat old generated labels as independently verified human references.

## Stage 1: confirmed correctness issues

### 1. Fix rejection of legitimate slow ball movement — critical

**Locations:** `src/ball_model.py`, `NeuralBallLocator.update`, especially stationary history/blacklist logic.

**Problem:** A candidate within six native pixels of a previous location is treated as stationary after two observations. Blacklisted positions then reject nearby detections indefinitely until reset. A deterministic reproduction with high-confidence peaks moving two pixels per frame accepted the first two positions and rejected the following four. Far-court movement, a ball near a lob apex and a held ball can all be legitimate.

**Required changes:** Remove the permanent location veto or replace it with a tested, time-bounded stationary-object decision that distinguishes the active ball from static distractors. Account for camera motion, source timing, resolution and recovery. Do not classify a candidate as a spare ball merely because it moved a few pixels.

**Acceptance:** Tests cover slow motion, apex motion, held-then-hit ball, a true static distractor, camera movement, recovery after temporary rejection and the two-pixels-per-frame reproduction. Legitimate motion remains trackable. No weakening of MPH qualification.

### 2. Keep generated annotations separate from confirmed references — critical

**Locations:** `src/web/ball_testing.py`: `prefill_predictions`, `interpolate_labels`, `metrics`, `export`; `tools/train_ball_model.py`: `prepare`; review UI suggestion/confirmation handlers.

**Problem:** Prefilled and interpolated labels receive `reviewed=True`. Metrics and dataset preparation can then treat model guesses as truth. Interpolation unconditionally replaces intermediate annotations, including manual visible, hidden or outside-frame labels.

**Required changes:** Store annotation origin and explicit confirmation state. Prefill/interpolation should create suggestions, not confirmed references. Exclude unconfirmed suggestions from accuracy metrics, completed-review counts and default supervised training exports. Explicit human confirmation should record the transition and preserve the suggestion's origin. Preserve existing manual labels by default. Make any deliberate replacement explicit and recoverable. Keep inference predictions separate from reference annotations. Handle legacy annotations with a documented migration policy without modifying frozen training snapshots.

**Acceptance:** Prefill alone cannot increase verified accuracy or verified review completion. Interpolation cannot replace an existing manual/outside/hidden label by default. Confirmation makes the correct frame eligible. Revisions, origins, prior values and undo/recovery remain available. Test training, validation and final-test sessions separately.

### 3. Propagate background loader failures — critical

**Location:** `tools/train_ball_model.py`: `BackgroundDataLoader` and `train`.

**Problem:** The producer sends an end sentinel in `finally`, even when dataset iteration raises. The consumer interprets this as a normal end of epoch. A reproduction yielded one batch, raised a dataset error in the thread and returned successfully to the training loop.

**Required changes:** Propagate the original exception and context to the consumer. Distinguish successful exhaustion, cancellation and failure. Ensure producer threads and bounded queues clean up without hanging. Reject incomplete/empty/nonfinite epochs rather than reporting them as successful training. Preserve earlier valid checkpoints and mark the run failed when appropriate.

**Acceptance:** Test failure before the first batch, after some batches, cancellation with a full queue and normal exhaustion. Failures must reach the caller, produce a failed run status and never save a checkpoint as the result of a successful partial epoch. Tests must terminate promptly.

### 4. Evaluate the same pipeline that the application uses — high

**Locations:** `tools/train_ball_model.py`: `predict_split`, `predict_external`, `calibrate`, `evaluate`; `src/ball_model.py`; detector and review runners.

**Problem:** Offline evaluation uses a raw heatmap peak; the live application uses multiple candidates, rejection rules, temporal state and court geometry. Training validation scores therefore do not describe complete application behavior.

**Required changes:** Keep raw localization metrics for diagnosis, and add end-to-end evaluation through the actual inference/tracking path with identical thresholds and preprocessing. Replay source frames chronologically with causal context; never supply reference positions to inference. Score only the reserved target frames while providing valid preceding context. Report detections, accepted tracker observations and predictions separately. Match court/camera tracking and reset rules used in production.

**Acceptance:** A fixed clip replay produces matching accepted detections in the dashboard and evaluation runner. Reports include correct/wrong/missing visible detections, false detections on known-absent frames, continuity/gap lengths, and timing. Runtime-veto regressions are detected even if raw network localization remains good.

### 5. Make confirmed outside-frame examples usable for training — high

**Locations:** `tools/train_ball_model.py`: `prepare`, `PatchDataset`, validation/threshold selection; `src/ball_model.py`: loss/runtime; export readiness messages.

**Problem:** Preparation skips every label except `visible`. Outside-frame examples are saved and can measure false positives, but do not teach absence. Partial, hidden and unsure examples are also excluded, which the UI must state accurately.

**Required changes:** Support explicitly confirmed absent frames as negative training examples with no ball-center target. Preserve causal inputs and prevent false-negative crops that contain the labeled active ball. Keep hidden/unsure labels distinct from confirmed absence. Decide and document whether partial labels remain excluded or receive a separate, tested training policy. Balance positives/negatives so large absent ranges do not overwhelm useful visible examples. Calibrate acceptance using both localization quality and absent-frame false alarms on validation data.

**Acceptance:** An absent-only export prepares successfully as negative data and produces finite loss. Unknown/hidden frames are not silently relabeled absent. Dataset counts distinguish all label types and consumed/skipped examples. Evaluation reports presence/absence errors as well as visible-frame localization. Do not claim improvement until measured on held-out footage.

### 6. Replace hardcoded model selection with explicit version management — high

**Locations:** `src/web/app.py` and `src/web/ball_testing.py`: version selection; model/runtime metadata; main/review UIs.

**Problem:** Both applications check v3, then v2, then v1. Even a completed/calibrated v4 will not be selected by this code.

**Required changes:** Provide a shared model registry and explicit model selection/promotion. Distinguish training, trained, calibrated, evaluated and active states. An in-progress `best.pt` is not a deployable model merely because it exists. Preserve fixed checkpoint copies/hashes in review sessions. Show the model actually used in each result; selecting a newer model should create a new comparison rather than rewriting old predictions. Avoid silent fallbacks and automatic activation based only on a directory/version number.

**Acceptance:** An incomplete v4 is unavailable for serving. A completed, qualified version becomes selectable. Old reviews still reproduce their original checkpoint. Main analysis and review use the selected version consistently, and unsupported/unqualified models produce a clear error.

## Stage 2: trustworthy data and a repeatable learning workflow

### 7. Strengthen split protection and dataset provenance

- Extend protection beyond the hardcoded v1 manifest. Track reserved source-frame ranges and their two causal context frames across dataset/model versions and saved reviews.
- Check consistency between a review's purpose, export metadata and manifest sample splits. Record intentional split decisions; do not silently reassign frames during preparation.
- Detect duplicate source frames and conflicting labels when combining overlapping reviews or multiple export revisions. Choose revisions explicitly and retain old exports for audit.
- Identify copies/renames of the same source video sufficiently to prevent accidental cross-split reuse; path strings alone are not reliable identity.
- Keep a final set of unseen clips/matches reserved. Repeatedly using the same tiny test set to decide model changes reduces its value as independent evidence.

**Acceptance:** Automated overlap tests include renamed/copied footage, temporal boundaries, repeated exports and newer manifests. No source context crosses train/validation/test boundaries in the prepared result. Reports enumerate sources and exact hashes/revisions.

### 8. Complete the explicit export → train → evaluate workflow

- Show counts for saved, human-confirmed, suggested, exportable and actually consumed examples. Review completion is not training progress.
- Connect model run provenance to exact export revisions and manifests so the UI can accurately say whether a correction was used by a model.
- Provide an explicit training action or clearly documented command that creates a new model/run directory. Training and activation remain separate actions.
- Write atomic status/history files. Save dataset hash, code/config version, seed, optimizer settings, batch size, thread/device settings, checkpoint choice and errors.
- If resume is supported, restore optimizer/run state and verify dataset compatibility. Distinguish resuming a run from starting a fine-tune or training from scratch.
- Never rerun or interrupt the currently active v4 process to implement this workflow.

**Acceptance:** One small isolated fixture can be reviewed, exported, trained, evaluated and traced back to its exact annotations. Starting a new version never modifies old weights or snapshots. Progress/error state survives reopening the UI.

### 9. Finish review UX reliability

- Preserve the persistent magnifier, red target, source-pixel nudges and carried-position previews. Carried points do not become references until confirmed.
- Preserve working numpad shortcuts and clearly display them. Physical numpad codes must work with Num Lock on/off and must not intercept text entry or browser modifier shortcuts.
- Support save + next, save + next unreviewed, and save + stay; navigate only after successful saves. Ignore repeated confirmation keys so holding a key cannot label many frames accidentally.
- Keep bulk suggestions distinct from reviewed frames in status badges, progress and navigation. Expose export/training status clearly and keep focus mode usable on smaller displays.
- Avoid overwriting newer edits from another window. Preserve retry/discard behavior and handle stale image loads/session changes.

**Acceptance:** Browser tests cover each shortcut/mode, asynchronous saves, failed saves, stale revisions, image races, suggestion confirmation, completed clips and keyboard focus. Existing manual annotations survive all tests. Use isolated fixtures for automated mutation tests, not the user's real labels.

## Stage 3: reliable tracking, speed and processing performance

### 10. Improve temporal association and ball recovery

Connect plausible detections across frames without jumping to shirts, shoes, spare balls or neighboring courts. Handle brief occlusions and abrupt direction changes at contacts. Reacquire without permanently blocking legitimate locations. Keep observed positions distinguishable from interpolated/predicted positions. Audit geometry filters: applying a ground-plane homography to an airborne ball can reject valid positions.

**Acceptance:** Evaluate near/far court, lobs, fast exchanges, occlusions, static distractors and camera movement. Report track continuity and wrong-object switches, not only average pixel error. No smoothing/interpolation is presented as a measured observation.

### 11. Detect paddle contacts and bounces

Replace direction-change guesses with qualified event boundaries. Separate flights at contacts and bounces, including closely spaced events. Interpolated annotation suggestions must not silently bridge these events. Expose uncertain events rather than inventing shot/hitter labels.

**Acceptance:** Independently annotated event clips show timing error, missed events and false events. Flight fits do not span a known bounce/hit. Closely spaced bounce/contact sequences are tested.

### 12. Improve camera calibration and source timing

Validate camera intrinsics, lens distortion, resolution/zoom changes and court-pose recovery. Each camera setup needs its own calibration. Verify actual timestamps for variable-rate, slow-motion and skipped-frame footage. Preserve the correct calibration reference frame.

**Acceptance:** Tests cover changed resolution, zoom/pan, lost/recovered court tracking, variable timing and unsupported camera poses. The system reports unavailable geometry/timing instead of producing misleading world coordinates.

### 13. Validate qualified 3D speed

Use reliable measured observations and event-separated flights to improve the existing 3D fit. Preserve ambiguity, physical-plausibility and uncertainty checks. Clearly distinguish average observed-flight speed from speed at paddle contact or peak speed. Accurate-looking 2D tracking does not establish MPH accuracy. Some single-camera flights can remain unidentifiable even with good detections.

**Acceptance:** Compare against independent reference measurements in controlled recordings. Report error, coverage and rejection reasons. Do not loosen gates just to populate the speed display, substitute court-plane speed for an airborne ball, or display a guessed range against the user's preference.

### 14. Profile and optimize the complete analysis pipeline

Measure decoding, preprocessing, player detection, ball inference, camera tracking, flight fitting, preview and output encoding separately. Benchmark thread counts, memory/cache behavior, batching and available hardware acceleration on the actual machine. Preserve quality when testing lower resolutions or accelerated inference. Do not assume more CPU threads are faster.

Preserve the model's temporal assumptions: its three inputs were trained as consecutive source frames. A dashboard speed/stride option must not silently feed nonconsecutive frames and claim equivalent tracking or timing. Prefer reducing preview/player work while retaining the ball observations needed for tracking and speed.

**Acceptance:** Before/after end-to-end timing and memory measurements on the same fixed clips and hardware, with accuracy/continuity comparisons. Report actual time for longer footage before extrapolating to a ten-minute match. No dropped observations, timing corruption, repeated-frame shortcuts or unreported accuracy tradeoffs.

### 15. Run a final application-level acceptance pass

Use multiple reserved rallies and at least one unseen recording/camera setup. Include visible and known-absent sections, difficult contacts, lobs and interruptions. Verify court/player tracking, selected model identity, full ball tracks, qualified speed/rejection reasons, saved review recovery and exported analytics. Separate prototype results from performance demonstrated on independent data.

**Acceptance:** A reproducible report names source clips, annotation origins, model/data/calibration hashes, split roles, hardware, localization/presence/continuity metrics, speed-reference results and processing time. Compare with the previous active version before promotion.

## Required completion report from the implementing LLM

For each numbered task, provide:

1. Status: completed, partially completed or deferred; identify any remaining limitation.
2. Files/functions changed and the behavior before/after.
3. Tests run and their actual results, including regression reproductions.
4. Dataset/model versions created or used and any provenance/migration decisions.
5. Evaluation artifacts and before/after metrics where the task changes model or runtime behavior.

Do not mark a task complete merely because a button exists or an existing test suite passes. The acceptance conditions above must be demonstrated. Do not claim trained/improved behavior from collected annotations alone.

Existing checks include `tests/test_ball_testing_ui.cjs`, `tests/test_calibration_ui.cjs`, `tests/test_ball_model.py` and `tests/test_ball_testing.py`. At this audit, 24 targeted Python tests and both UI suites passed, but the stationary-filter and background-loader reproductions still failed their intended behavior. Add meaningful regressions for those failures before changing their implementations.
