# What is backed up to GitHub

The private repository contains the application source, tests and small test fixtures, tools, dependencies, implementation handoff and audit documents. It also preserves small trained model checkpoints, model evaluation/history metadata, court calibration, saved ball-review annotations and histories, frozen review model checkpoints, prediction records, exported annotation snapshots and dataset manifests.

Large source recordings, extracted frame collections, NumPy training caches, generated analysis videos/reports, local logs, Python environments and credentials are excluded by `.gitignore`. Uploading this repository does not remove or modify those files on the original computer.

This is a source/model/annotation backup, not a complete recording archive. Keep a separate backup of the original videos and extracted frames. A fresh clone does not contain the image collections needed to reopen historical frame reviews, export their original images or immediately reuse cached training manifests. Restore those local collections from a separate backup or regenerate the corresponding source frames/caches from the original recordings before those operations. Annotation files retain exact source-frame indices and video metadata.

On another computer:

1. Install the packages in `requirements.txt` using a compatible Python environment.
2. Restore recordings into `data/raw_videos/` and confirm the court for each recording.
3. Restore historical review frames into their matching `data/ball_tests/<review-id>/frames/` directories to reopen saved image reviews. Restore exported/audit frame collections as needed when preparing a new training dataset.
4. Regenerate prepared datasets with `tools/train_ball_model.py prepare` into a new version directory if caches are unavailable. Preserve older manifests and model provenance.
5. Start Studio with `python app.py`.

Windows file paths and video modification times in metadata describe the original computer. Review those identities when restoring on another machine; do not assume a changed recording or calibration is equivalent solely because its filename matches.
