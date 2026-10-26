# Stage 1 audit: not complete

Audited against `docs/IMPLEMENTATION_HANDOFF.md`, Stage 1 tasks 1–6. This review did not change application code, user annotations, checkpoints or training runs. Reproductions used isolated fixtures under `data/outputs`.

## Verdict by task

| Task | Result | Evidence |
| --- | --- | --- |
| 1. Legitimate slow motion | Partially implemented | The original two-pixels/frame regression passes, but a one-pixel/frame moving peak is repeatedly rejected. |
| 2. Confirmed versus generated annotations | Partially implemented | Backend suggestions, confirmation and filtered exports work in the new tests. Browser metrics/navigation and direct dataset preparation still treat unconfirmed suggestions as references. |
| 3. Background loader failures | Partially implemented; introduces a training blocker | Exceptions propagate now, but normal exhaustion permanently sets cancellation on the reused loader. Subsequent epochs yield no batches. |
| 4. Real pipeline evaluation | Not implemented | `evaluate_pipeline` is absent. Its new test module fails at import, and evaluation still scores raw heatmap peaks. |
| 5. Outside-frame training | Not implemented | The preparer still discards every non-visible frame. An absent-only fixture produces zero samples. |
| 6. Explicit model registry/promotion | Partially implemented | A registry module exists but neither application uses it. Both still hardcode v4/v3/v2/v1. The registry also permits deployment before meaningful evaluation or explicit promotion. |

## Blocking findings and required follow-up

### A. The reused background loader stops training after epoch one

**Code:** `tools/train_ball_model.py`, `BackgroundDataLoader.__iter__` and `train` (approximately lines 270 and 283).

Iterator cleanup always calls `self._cancel_event.set()`. The event is never cleared/replaced for a new iterator, while `train` constructs one loader outside its epoch loop.

**Reproduction using the actual class definition:**

```python
loader = BackgroundDataLoader([1, 2, 3], queue_size=2)
[list(loader) for _ in range(3)]
# Actual: [[1, 2, 3], [], []]
```

The training loop can subsequently compute an empty epoch's mean loss as NaN instead of performing another epoch of learning. Fix the loader lifecycle without losing explicit cancellation or exception propagation. Reject empty/incomplete/nonfinite epochs and record the correct run status.

**Required regressions:** Multiple complete epochs using the same loader; two or more actual fixture training epochs with expected batch counts and finite losses; producer failure before/after batches; cancellation with a full queue; no hanging/leaked producer. Existing loader tests exercise only a single iterator each and do not detect this failure.

### B. Suggested labels still inflate browser accuracy and completion

**Code:** `src/web/ball_testing.html`, `showMetrics` (approximately line 221), `nextUnreviewed` (approximately line 279), resume logic and label status; `tools/train_ball_model.py`, `prepare` (approximately line 42).

The backend now correctly stores suggestions with `confirmed=False` and filters them from its metrics/exports. The browser still counts all annotations by array length, scores all visible labels and treats every annotated frame as reviewed. The preparer also ignores confirmation state when given an audit directly.

**Reproductions:**

- One unconfirmed prefilled label makes the browser display `1/1 frames reviewed` and `1 correct` for matching predictions.
- `nextUnreviewed()` then reports that every frame was reviewed and saved.
- A direct audit with one visible, unconfirmed prefilled label produces one training sample.

Apply consistent confirmation rules to UI counts, scoring, navigation, resume, status labels, readiness messages and dataset preparation. Preserve suggestion origin when a user confirms/adjusts it. The backend confirmation route needs a usable UI path. Document legacy handling; old prefilled/interpolated annotations must not become verified merely because their older schema lacks an `origin` field.

**Required regressions:** Prefill/interpolation alone do not change verified UI metrics or completion; next-unreviewed visits suggestions; a confirmation makes the frame eligible and retains provenance; direct preparation skips unconfirmed data; legacy and train/validation/test cases are covered. Do not rely exclusively on the backend tests.

### C. Slow movement is still vetoed

**Code:** `src/ball_model.py`, stationary history update and rejection (approximately lines 178 and 209).

The change raises count thresholds and ages blacklist entries, but candidates remain grouped within six native pixels and are rejected once the group count reaches four. Continuing detections refresh blacklist entries, so expiry alone does not fix the decision.

**Deterministic reproduction:** A high-confidence Gaussian peak moving approximately one source pixel per frame produced:

```text
Expected x: 20, 21,    22, 23,   24,   25,   26,   27,   28, 29,   30,   31,   32,   33
Accepted x: 20, 20.82, 22, None, None, None, None, None, 28, None, None, None, None, None
```

Use a decision that distinguishes legitimate slow motion from distractors, or disable this veto until that distinction is demonstrated. Do not just raise the frame count again. Include source timing, resolution, camera movement and recovery in the policy/tests.

**Required regressions:** One-pixel and subpixel movement as well as the original two-pixel case; sustained low projected movement near an apex; held-then-hit and returning ball; true static distractor; camera motion; multiple resolutions/cadences. Assert correct ball positions and continuity, not merely that some detections occur.

### D. Actual pipeline evaluation is missing

**Code:** `tools/train_ball_model.py`, `predict_split`, `predict_external`, `evaluate`; `tests/test_training_pipeline_regression.py`, import near line 17.

The new regression module imports `evaluate_pipeline`, which does not exist. Current evaluation still calls the network and `peak_from_heatmap` directly. It does not run runtime candidate rejection, court/camera resets or tracker observation acceptance.

Implement the real chronological evaluation path and expose it through the evaluation workflow. Retain raw localization metrics separately. Report observed detections, accepted tracker observations and predictions distinctly, along with visible/absent errors, continuity and timing.

**Required regressions:** An identical source clip/checkpoint/config produces matching accepted detections in application and evaluation. References never influence inference. A runtime-filter failure is detected even when raw localization is correct. Tests must assert actual inference behavior rather than only the presence of result dictionary keys. Resolve the collection error by implementing the required behavior, not by stubbing or removing the test.

### E. Negative training data is still discarded

**Code:** `tools/train_ball_model.py`, `prepare`, `PatchDataset`, summarization and threshold calibration.

The filter remains `label['visibility'] != 'visible'`, and targets/dataset logic still assume a center exists. A confirmed absent-only fixture prepared successfully but contained zero samples. The newly written negative-training tests therefore do not represent current implementation; that entire module cannot collect because of finding D.

Implement confirmed-absent samples with no center and valid negative targets, a documented sampling balance, per-type counts, absence-aware validation and threshold selection. Do not substitute hidden/unsure annotations for confirmed absence. Keep UI/export readiness descriptions aligned with what the trainer actually consumes.

**Required regressions:** Absent-only and mixed datasets produce the expected sample types and finite loss; unknown/hidden are not relabeled absent; unconfirmed examples are excluded; positive/negative balancing is tested; absence metrics and false-alarm-aware calibration work. Complete the existing regression module and add the missing behavior tests.

### F. Registry exists but does not control deployment

**Code:** `src/web/app.py` around line 329; `src/web/ball_testing.py` around line 220; `src/model_registry.py` around lines 114 and 143.

Both application paths still choose the newest existing hardcoded court checkpoint. No application integration with the new registry or explicit active-model selection was found.

The standalone registry is also too permissive:

- A `best_court.pt` with a threshold is `deployable=True` without evaluation.
- An empty `evaluation.json` (`{}`) is enough for `state='qualified'`.
- Default selection returns the highest available version without an explicit promotion.
- Evaluation evidence is not tied to the checkpoint hash/config, and active training/completion cannot be established from the current artifact checks.

Wire both applications and their selectors to one explicit registry/promotion mechanism. Validate the contents and provenance of qualification evidence. Keep a running/incomplete model unavailable, preserve review checkpoint snapshots, and reject unqualified/unknown selections clearly.

**Required regressions:** No checkpoint, training checkpoint, calibrated-but-unevaluated checkpoint, empty/stale/mismatched evaluation and corrupt state dictionary are unavailable for promotion. Valid qualified models are selectable; merely adding a newer directory does not change the active model. Main analysis/review agree on the selection, and older sessions retain their original weights.

## Verification results

- Stage 1 targeted run: **12 passed, 1 test-module collection error**.
- Full Python run with `--continue-on-collection-errors`: **180 passed, 1 test-module collection error**, exit status 1. Without that option, normal pytest collection is blocked.
- Collection error: `ImportError: cannot import name 'evaluate_pipeline' from 'tools.train_ball_model'`.
- Both existing Node UI test suites passed. They do not currently cover the confirmation-state mismatch reproduced above.
- Additional isolated reproductions confirmed findings A, B, C, E and the registry qualification/selection behavior in F.
- Existing annotations, checkpoints and active training were not changed by this audit. Test durations are not performance benchmarks.

## Suggested implementation order

1. Fix A before starting another training run.
2. Finish B so references/metrics cannot silently include guesses.
3. Finish C and D together so the runtime fix is evaluated by the same pipeline.
4. Finish E and its validation/threshold behavior.
5. Finish F before promoting a new version.
6. Re-run the complete suite without ignoring collection errors and provide actual runtime comparison artifacts. Return a task-by-task completion report against the original Stage 1 acceptance criteria.

**Stage 1 is not ready for sign-off.** Successful backend tests and a new registry module are useful progress, but do not satisfy the complete application-level acceptance requirements.
