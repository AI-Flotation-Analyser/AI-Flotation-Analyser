"""Load a completed feature batch or rebuild directly from raw33 observations.

The default import path is intentionally non-destructive. Research videos and
raw landmarks remain in the original project; this repository holds code and
small, local, generated outputs only.
"""
from __future__ import annotations

import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .features import VERSION, analyse_file, window_metrics

REQUIRED_VIDEO = {
    'video_id', 'participant_id', 'camera_view', 'torso_valid_fraction',
    'arm_measurement_basis', 'leg_measurement_basis',
    'arm_movement_speed_median_torso_s', 'leg_movement_speed_median_torso_s',
    'arm_speed_usable_fraction', 'arm_screened_frame_fraction',
    'body_path_rate_median_torso_s', 'body_max_excursion_torso',
    'duration_sec', 'qc_status',
}
REQUIRED_WINDOWS = {'video_id', 'participant_id', 'camera_view', 'window_start_sec', 'window_end_sec'}


def save_csv(df: pd.DataFrame, path: Path) -> None:
    """Write small derived tables atomically; never overwrite source datasets."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    try:
        df.to_csv(temp, index=False)
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def save_json(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    try:
        temp.write_text(json.dumps(obj, indent=2, allow_nan=False), encoding='utf-8')
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def validate(video: pd.DataFrame, windows: pd.DataFrame) -> None:
    absent = REQUIRED_VIDEO.difference(video.columns)
    if absent:
        raise ValueError(f'Video feature table missing columns: {sorted(absent)}')
    absent = REQUIRED_WINDOWS.difference(windows.columns)
    if absent:
        raise ValueError(f'Window feature table missing columns: {sorted(absent)}')
    if video.empty:
        raise ValueError('No video features supplied')
    if video['video_id'].isna().any() or video['video_id'].duplicated().any():
        raise ValueError('Missing/duplicate video IDs in feature table')
    if video['participant_id'].isna().any():
        raise ValueError('Missing participant IDs')
    if not video['camera_view'].isin(['top', 'side_front']).all():
        raise ValueError('Unsupported camera view in feature table')
    identities = video.set_index('video_id')[['participant_id', 'camera_view']]
    unknown = set(windows['video_id']) - set(identities.index)
    if unknown:
        raise ValueError(f'Windows reference video IDs absent from features ({len(unknown)} videos)')
    if not windows.empty:
        joined = windows.join(identities, on='video_id', rsuffix='_video')
        if (joined['participant_id'] != joined['participant_id_video']).any():
            raise ValueError('Window/video participant ID mismatch')
        if (joined['camera_view'] != joined['camera_view_video']).any():
            raise ValueError('Window/video camera view mismatch')
        if (pd.to_numeric(windows['window_end_sec'], errors='coerce') <=
                pd.to_numeric(windows['window_start_sec'], errors='coerce')).any():
            raise ValueError('Invalid window time bounds')
        if windows.duplicated(['video_id', 'window_start_sec', 'window_end_sec']).any():
            raise ValueError('Duplicate temporal windows')


def find_existing_batch(source_root: Path) -> Path | None:
    parent = source_root / 'outputs' / 'final_delivery'
    runs = [p for p in parent.glob('run_*') if
            (p / 'final_video_features.csv').is_file() and
            (p / 'final_window_features.csv').is_file()]
    return max(runs, key=lambda p: p.stat().st_mtime) if runs else None


def load_completed_batch(source_root: Path, batch: Path | None = None):
    """Import existing 383-video outputs without loading or copying raw videos."""
    batch = batch or find_existing_batch(source_root)
    if batch is None:
        raise FileNotFoundError('Completed final_delivery batch not found; use --rebuild')
    video = pd.read_csv(batch / 'final_video_features.csv', low_memory=False)
    windows = pd.read_csv(batch / 'final_window_features.csv', low_memory=False)
    validate(video, windows)
    fail_path = batch / 'source_inventory_failures.csv'
    failures = pd.read_csv(fail_path) if fail_path.is_file() else pd.DataFrame(columns=['video_id', 'reason'])
    evidence_path = batch / 'FINAL_RUN_EVIDENCE.json'
    evidence = json.loads(evidence_path.read_text(encoding='utf-8')) if evidence_path.is_file() else {}
    declared = evidence.get('n_output_videos')
    if declared is not None and int(declared) != len(video):
        raise ValueError(f'Evidence record claims {declared} videos, but CSV has {len(video)}')
    return video, windows, failures, {'mode': 'import_completed_feature_batch',
            'batch': str(batch.resolve()), 'existing_evidence': evidence}


def _valid_cache(obj, row: dict) -> bool:
    return (isinstance(obj, dict) and obj.get('schema_version') == VERSION and
            str(obj.get('video_id')) == str(row['video_id']) and
            str(obj.get('participant_id')) == str(row['participant_id']) and
            str(obj.get('camera_view')) == str(row['camera_view']) and
            obj.get('source_csv') == str(row['video_id']) + '_raw.csv')


def _load_cached_json(paths: list[Path], row: dict):
    for path in paths:
        if path.is_file():
            try:
                obj = json.loads(path.read_text(encoding='utf-8'))
                if _valid_cache(obj, row):
                    return obj
            except (ValueError, OSError):
                pass
    return None


def _load_cached_windows(paths: list[Path], vid: str):
    for path in paths:
        if path.is_file():
            try:
                df = pd.read_csv(path)
                if REQUIRED_WINDOWS.issubset(df.columns) and not df.empty and df.video_id.astype(str).eq(vid).all():
                    return df
            except (OSError, ValueError, pd.errors.ParserError):
                pass
    return None


def rebuild_from_raw33(source_root: Path, output: Path):
    """Compact, resumable raw33 processing; no PNG/HTML/per-frame CSV exports.

    Previously completed v1.1 features and windows can be reused, but the raw
    inputs remain authoritative for any missing or invalid cached result.
    """
    inventory = source_root / 'data/metadata/video_inventory.csv'
    raw_dir = source_root / 'data/landmarks_raw/mediapipe_all33_v2'
    if not inventory.is_file() or not raw_dir.is_dir():
        raise FileNotFoundError('Need source/data/metadata/video_inventory.csv and raw33 directory')
    entries = pd.read_csv(inventory)
    need = {'video_id', 'participant_id', 'camera_view', 'width', 'height'}
    if not need.issubset(entries.columns):
        raise ValueError(f'Inventory missing {sorted(need.difference(entries.columns))}')
    if entries.video_id.duplicated().any():
        raise ValueError('Inventory contains duplicate video IDs')
    records, all_windows, failures = [], [], []
    counts = {'reused': 0, 'computed': 0}
    for i, (_, rec) in enumerate(entries.iterrows(), 1):
        row = rec.to_dict()
        vid = str(row['video_id'])
        src = raw_dir / f'{vid}_raw.csv'
        local_json = output / 'cache/records' / f'{vid}.json'
        local_window = output / 'cache/windows' / f'{vid}_windows.csv'
        previous_json = source_root / 'outputs/final_v1/features' / f'{vid}.json'
        previous_win = source_root / 'outputs/final_v1/windows' / f'{vid}_windows.csv'
        if not src.is_file():
            failures.append({'video_id': vid, 'reason': 'missing_raw33_csv'})
            continue
        try:
            result = _load_cached_json([local_json, previous_json], row)
            w = _load_cached_windows([local_window, previous_win], vid)
            if result is None or w is None:
                result, ts = analyse_file(src, row)
                w = window_metrics(ts, result)
                if w.empty:
                    raise ValueError('No usable 10+ second temporal windows')
                save_json(result, local_json)
                save_csv(w, local_window)
                counts['computed'] += 1
            else:
                counts['reused'] += 1
            records.append(result)
            all_windows.append(w)
        except (OSError, ValueError, KeyError, TypeError, pd.errors.ParserError) as exc:
            failures.append({'video_id': vid, 'reason': f'{type(exc).__name__}: {str(exc)[:180]}'})
        if i % 25 == 0 or i == len(entries):
            print(f'  raw33 {i}/{len(entries)} | success={len(records)} | reused={counts["reused"]} | new={counts["computed"]} | missing/failed={len(failures)}', flush=True)
    if not records:
        raise RuntimeError('No valid raw33 features created')
    video = pd.DataFrame(records)
    windows = pd.concat(all_windows, ignore_index=True)
    validate(video, windows)
    return video, windows, pd.DataFrame(failures, columns=['video_id', 'reason']), {
        'mode': 'rebuild_raw33_resumable', 'inventory_count': int(len(entries)),
        'reused_cached': counts['reused'], 'newly_processed': counts['computed'],
        'schema_version': VERSION,
    }


def summarize_views(video: pd.DataFrame):
    rows = []
    for view, subset in video.groupby('camera_view', sort=True):
        torso = pd.to_numeric(subset.torso_valid_fraction, errors='coerce')
        rows.append({
            'camera_view': view, 'video_count': int(len(subset)),
            'distinct_participants': int(subset.participant_id.nunique()),
            'median_torso_valid_fraction': float(torso.median()),
            'arm_feature_available': int(subset.arm_movement_speed_median_torso_s.notna().sum()),
            'leg_feature_available': int(subset.leg_movement_speed_median_torso_s.notna().sum()),
            'unilateral_arm_records': int(subset.arm_measurement_basis.isin(['left_only', 'right_only']).sum()),
            'review_required': int(subset.qc_status.eq('review_required').sum()),
        })
    return pd.DataFrame(rows)


def snapshot_provenance(source: Path, video: pd.DataFrame, windows: pd.DataFrame,
                        failures: pd.DataFrame, origin: dict) -> dict:
    return {'created_utc': datetime.now(timezone.utc).isoformat(),
            'code_version': '3.0.0', 'feature_engine_version': VERSION,
            'source_root': str(source.resolve()), 'source': origin,
            'processed_videos': int(len(video)), 'participants': int(video.participant_id.nunique()),
            'temporal_windows': int(len(windows)), 'missing_or_failed_entries': int(len(failures)),
            'note': 'Feature coverage and exploratory clusters do not demonstrate flotation safety or expert-rated quality.'}