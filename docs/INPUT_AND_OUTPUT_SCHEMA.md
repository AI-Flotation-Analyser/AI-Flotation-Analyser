# Reproducible data interface

## Source video input

Provide a `video_manifest.csv` with the fields:

| Field | Required | Example |
|---|---|---|
| `video_id` | yes, unique | `P001_top_session1` |
| `participant_id` | yes | `P001` |
| `camera_view` | yes: `top` or `side_front` | `top` |
| `video_path` | yes, absolute or relative to manifest folder | `videos/P001_top_session1.mp4` |

Place source videos anywhere accessible to the run. The application reads the video path, not a pre-specified Drive or Windows directory. One participant may have multiple recordings, but the evaluation splits on participant IDs.

## 33-landmark input

To skip extraction, provide:

```text
<study_root>/
  data/
    metadata/video_inventory.csv
    landmarks_raw/mediapipe_all33_v2/
      <video_id>_raw.csv
```

`video_inventory.csv` must have at least `video_id,participant_id,camera_view,width,height`; source dimensions must match the original footage. `examples/video_inventory_template.csv` is a template. The extractor writes these files automatically when given videos and a manifest.

Each `_raw.csv` has **172 columns** in a fixed order: seven base columns (`video_id`, `participant_id`, `camera_view`, `video_name`, `frame`, `time_sec`, `pose_detected`) and 33 MediaPipe pose landmarks with `x`, `y`, `z`, `visibility`, `presence`. Normalised `x/y` are multiplied by the recorded width/height for image-plane measurements. All frames, including misses, receive timestamps.

Raw33 files already extracted by another implementation are accepted if they match the same schema. The pipeline does not require the user's input filenames to be identical to the research team's original human names.

## Existing feature batch

The fastest path is to supply a root containing an `outputs/final_delivery/run_*/` directory with:

```text
final_video_features.csv
final_window_features.csv
source_inventory_failures.csv      # optional
FINAL_RUN_EVIDENCE.json            # optional
```

The most recently modified complete run is imported. Supply `--rebuild` to ignore existing batch tables and recompute features from raw33 observations.

## Human-scored labels

Expert ratings are supplied via `--expert-labels path/to/ratings.csv`, using a minimum schema of:

```csv
participant_id,interval_start_sec,interval_end_sec,floating_outcome,overall_score
P001,30,60,2,10
```

One independently scored participant per row, where floating outcome is 0–3 and overall score 0–13. The existing release's four original expert examples are stored locally in `private_inputs/chris_expert_labels.csv` and never published by Git.

Verified new assessments go in `private_inputs/reviewed_labels.csv` with an additional `review_status` field set to `verified`. The chosen labelled interval must overlap a 30-second feature window from a matching top/side-front video trial.

## Generated output

The `outputs/latest/` folder includes: offline dashboard, report summary, experiment manifest, camera-coverage summaries and figures, unsupervised selection tables and stability metrics, participant-level train/test predictions, pseudo-label provenance, review queue and saved models. Generated private files and original inputs are excluded from Git.

Use `python run.py --help` for runtime options and `python run.py extract --help` for extraction options.
