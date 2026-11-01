"""Video ingestion and MediaPipe Pose landmark extraction.

This module writes the 33-landmark CSV schema consumed by features.py.
Only the extraction command requires OpenCV and MediaPipe. Input footage is
read without altering it, and one video is processed at a time.
"""
from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
import re
from typing import Callable

import pandas as pd

from .features import BASE, EXPECTED, LANDMARKS

MANIFEST_COLUMNS = ('video_id', 'participant_id', 'camera_view', 'video_path')
_ID = re.compile(r'^[A-Za-z0-9][A-Za-z0-9_.-]*$')


def read_video_manifest(path: Path) -> pd.DataFrame:
    """Validate the user-supplied mapping between video files and participants."""
    manifest = pd.read_csv(path, dtype=str, keep_default_na=False)
    missing = set(MANIFEST_COLUMNS).difference(manifest.columns)
    if missing:
        raise ValueError(f'Video manifest missing columns: {sorted(missing)}')
    if manifest.empty:
        raise ValueError('Video manifest is empty')
    for col in MANIFEST_COLUMNS:
        manifest[col] = manifest[col].astype(str).str.strip()
        if manifest[col].eq('').any():
            raise ValueError(f'Empty {col} in video manifest')
    if manifest.video_id.duplicated().any():
        raise ValueError('Duplicate video_id in video manifest')
    if not manifest.video_id.map(lambda s: bool(_ID.fullmatch(s))).all():
        raise ValueError('video_id must contain letters, numbers, underscores, dots or hyphens')
    if not manifest.camera_view.isin(['top', 'side_front']).all():
        raise ValueError('camera_view must be top or side_front')
    return manifest


def _video_path(video_path: str, root: Path) -> Path:
    p = Path(video_path).expanduser()
    return (p if p.is_absolute() else root / p).resolve()


def _imports():
    try:
        import cv2
        import mediapipe as mp
    except ImportError as exc:
        raise RuntimeError('Video extraction requires requirements-video.txt (OpenCV and MediaPipe).') from exc
    if not hasattr(mp, 'solutions') or not hasattr(mp.solutions, 'pose'):
        raise RuntimeError('MediaPipe Solutions Pose not found. Install the version specified in requirements-video.txt.')
    return cv2, mp


def _value(landmark, field: str) -> float:
    """Presence is optional in some MediaPipe Pose model versions."""
    if field == 'presence':
        try:
            if landmark.HasField('presence'):
                return float(landmark.presence)
        except (AttributeError, ValueError):
            # Mock objects and alternate protobuf bindings may omit HasField.
            return float(getattr(landmark, 'presence', 1.0))
        return 1.0
    return float(getattr(landmark, field))


def extract_video(video: Path, target: Path, identity: dict, *,
                  model_complexity: int = 1, frame_step: int = 1,
                  max_frames: int | None = None,
                  dependency_factory: Callable = _imports) -> dict:
    """Stream one source video into an atomic raw33 CSV.

    Frames that fail pose detection still produce timestamped rows with empty
    coordinates, so downstream coverage estimates preserve missing detections.
    """
    cv2, mp = dependency_factory()
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        cap.release()
        raise OSError(f'Unable to open video: {video}')
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if not (fps > 0 and width > 1 and height > 1):
        cap.release()
        raise ValueError(f'Video has invalid FPS or dimensions: {video}')
    if frame_step < 1 or model_complexity not in (0, 1, 2):
        cap.release()
        raise ValueError('frame_step must be >=1 and model_complexity must be 0, 1 or 2')

    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_name(target.name + '.part')
    selected = 0
    detected = 0
    frame_index = 0
    try:
        with temp.open('w', newline='', encoding='utf-8') as stream, mp.solutions.pose.Pose(
            static_image_mode=False, model_complexity=model_complexity,
            enable_segmentation=False, smooth_landmarks=True,
            min_detection_confidence=0.5, min_tracking_confidence=0.5,
        ) as pose:
            writer = csv.writer(stream)
            writer.writerow(EXPECTED)
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if frame_index % frame_step:
                    frame_index += 1
                    continue
                row = [identity['video_id'], identity['participant_id'],
                       identity['camera_view'], video.name, frame_index,
                       round(frame_index / fps, 9)]
                result = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                landmarks = result.pose_landmarks.landmark if result.pose_landmarks else None
                row.append(bool(landmarks))
                if landmarks is None:
                    row.extend([''] * (len(EXPECTED) - len(BASE)))
                else:
                    if len(landmarks) != len(LANDMARKS):
                        raise ValueError(f'Expected 33 pose landmarks, received {len(landmarks)}')
                    detected += 1
                    for lm in landmarks:
                        row.extend(_value(lm, field) for field in
                                   ('x', 'y', 'z', 'visibility', 'presence'))
                writer.writerow(row)
                selected += 1
                frame_index += 1
                if max_frames is not None and selected >= max_frames:
                    break
        if selected < 10:
            raise ValueError(f'Fewer than 10 sampled frames in {video.name}')
        os.replace(temp, target)
        return {'video_id': identity['video_id'], 'participant_id': identity['participant_id'],
                'camera_view': identity['camera_view'], 'video_path': str(video),
                'width': width, 'height': height, 'fps': fps,
                'video_frame_count': total_frames, 'sampled_frames': selected,
                'pose_detected_frames': detected, 'raw33_csv': str(target),
                'extraction_status': 'created'}
    finally:
        cap.release()
        temp.unlink(missing_ok=True)


def extract_dataset(source_root: Path, manifest_path: Path, *,
                    video_root: Path | None = None, model_complexity: int = 1,
                    frame_step: int = 1, max_frames: int | None = None,
                    overwrite: bool = False,
                    extract_fn: Callable = extract_video) -> pd.DataFrame:
    """Create a standard input inventory and raw33 files for any user dataset.

    Missing source footage remains documented in the inventory as a failure;
    the analysis pipeline will process all successfully extracted recordings.
    """
    source_root = Path(source_root).expanduser().resolve()
    manifest_path = Path(manifest_path).expanduser().resolve()
    video_root = Path(video_root).expanduser().resolve() if video_root else manifest_path.parent
    manifest = read_video_manifest(manifest_path)
    raw_dir = source_root / 'data/landmarks_raw/mediapipe_all33_v2'
    inventory_dir = source_root / 'data/metadata'
    raw_dir.mkdir(parents=True, exist_ok=True)
    inventory_dir.mkdir(parents=True, exist_ok=True)
    inventory_path = inventory_dir / 'video_inventory.csv'
    old = {}
    if inventory_path.is_file():
        old = {str(r['video_id']): r for r in pd.read_csv(inventory_path).to_dict('records')}
    records = []
    for idx, r in enumerate(manifest.to_dict('records'), 1):
        video = _video_path(r['video_path'], video_root)
        csv_path = raw_dir / f"{r['video_id']}_raw.csv"
        item = dict(r)
        item['video_path'] = str(video)
        if csv_path.is_file() and not overwrite:
            item.update(old.get(r['video_id'], {}))
            item.update({k:r[k] for k in MANIFEST_COLUMNS if k != 'video_path'})
            if not (pd.notna(item.get('width')) and pd.notna(item.get('height'))):
                raise ValueError(f'Cached CSV {csv_path.name} lacks width/height metadata in inventory')
            item['extraction_status'] = 'reused'
        elif not video.is_file():
            item.update({'extraction_status':'missing_video', 'width': None, 'height': None})
        else:
            try:
                item.update(extract_fn(video, csv_path, r,
                    model_complexity=model_complexity, frame_step=frame_step,
                    max_frames=max_frames))
            except (OSError, ValueError, RuntimeError) as exc:
                item.update({'extraction_status':'failed', 'extraction_detail':f'{type(exc).__name__}: {exc}'})
        records.append(item)
        print(f'Extracted {idx}/{len(manifest)} | {r["video_id"]}: {item["extraction_status"]}',flush=True)
    # An incremental manifest adds/replaces video IDs while preserving existing
    # inventory entries that were not included in this extraction request.
    requested = set(manifest.video_id)
    records.extend(row for vid, row in old.items() if vid not in requested)
    pd.DataFrame(records).to_csv(inventory_path, index=False)
    return pd.DataFrame(records)


def main(argv=None):
    parser = argparse.ArgumentParser(description='Extract MediaPipe Pose 33 landmarks from participant videos')
    parser.add_argument('--source-root', type=Path, required=True, help='Dataset root to receive data/metadata and data/landmarks_raw')
    parser.add_argument('--manifest', type=Path, required=True, help='CSV with video_id,participant_id,camera_view,video_path')
    parser.add_argument('--video-root', type=Path, help='Base directory for manifest relative video_path values')
    parser.add_argument('--frame-step', type=int, default=1)
    parser.add_argument('--model-complexity', type=int, choices=(0,1,2), default=1)
    parser.add_argument('--max-frames', type=int, default=None)
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args(argv)
    extract_dataset(args.source_root, args.manifest, video_root=args.video_root,
                    frame_step=args.frame_step, model_complexity=args.model_complexity,
                    max_frames=args.max_frames, overwrite=args.overwrite)
    return 0