"""Feature engineering from 33-landmark MediaPipe CSVs.

The feature schema and QC thresholds are maintained at v1.1 to remain compatible
with the completed 383-video run. All measurements are image-space proxies.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

LANDMARKS = [
    'nose', 'left_eye_inner', 'left_eye', 'left_eye_outer', 'right_eye_inner',
    'right_eye', 'right_eye_outer', 'left_ear', 'right_ear', 'mouth_left',
    'mouth_right', 'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist', 'left_pinky', 'right_pinky', 'left_index',
    'right_index', 'left_thumb', 'right_thumb', 'left_hip', 'right_hip',
    'left_knee', 'right_knee', 'left_ankle', 'right_ankle', 'left_heel',
    'right_heel', 'left_foot_index', 'right_foot_index'
]
BASE = ['video_id', 'participant_id', 'camera_view', 'video_name', 'frame', 'time_sec', 'pose_detected']
EXPECTED = BASE + [f'{name}_{suffix}' for name in LANDMARKS
                   for suffix in ('x', 'y', 'z', 'visibility', 'presence')]
REQUIRED_FOR_MEASUREMENT = ['left_shoulder', 'right_shoulder', 'left_hip',
                            'right_hip', 'nose', 'left_wrist', 'right_wrist',
                            'left_ankle', 'right_ankle']
VIS_THRESHOLD = 0.5
PRES_THRESHOLD = 0.5
SMOOTH_WINDOW = 5  # median smoothing, 5 frames; does not interpolate missing points
# Engineering SCREEN only. Not a validated physiological ceiling: retain counts and raw values.
SPEED_SCREEN_TORSO_PER_SEC = 10.0
MIN_SPEED_FRAME_COVERAGE = 0.60  # provisional measurement completeness criterion
MIN_SIDE_COVERAGE = 0.60
WINDOW_SEC = 30.0
VERSION = 'final33_features_v1.1.0' 


def number(value, default=np.nan):
    try:
        x = float(value)
        return x if math.isfinite(x) else default
    except (ValueError, TypeError):
        return default


def optional(x):
    x = number(x)
    return None if not np.isfinite(x) else round(x, 5)


def median(x):
    a = np.asarray(x, dtype=float)
    a = a[np.isfinite(a)]
    return float(np.median(a)) if len(a) else np.nan


def iqr(x):
    a = np.asarray(x, dtype=float)
    a = a[np.isfinite(a)]
    return float(np.percentile(a, 75) - np.percentile(a, 25)) if len(a) else np.nan


def fraction(mask):
    arr = np.asarray(mask, dtype=bool)
    return float(np.mean(arr)) if len(arr) else np.nan


def series_number(df, name):
    return pd.to_numeric(df[name], errors='coerce').to_numpy(dtype=float)


def as_bool(s):
    return s.astype(str).str.lower().str.strip().isin(('true', '1', '1.0')).to_numpy()


def xy_qualified(df, name, width, height, detected):
    x = series_number(df, f'{name}_x') * width
    y = series_number(df, f'{name}_y') * height
    v = series_number(df, f'{name}_visibility')
    p = series_number(df, f'{name}_presence')
    ok = (detected & np.isfinite(x) & np.isfinite(y) &
          np.isfinite(v) & np.isfinite(p) &
          (v >= VIS_THRESHOLD) & (p >= PRES_THRESHOLD))
    x[~ok] = np.nan
    y[~ok] = np.nan
    return x, y, ok


def average_available(a, b):
    """Mean when both available; use one side when the other is occluded."""
    out = np.where(np.isfinite(a) & np.isfinite(b), (np.nan_to_num(a) + np.nan_to_num(b))/2,
                   np.where(np.isfinite(a), a, b))
    return out


def smoothed(x):
    s = pd.Series(x)
    z = s.rolling(SMOOTH_WINDOW, center=True, min_periods=2).median().to_numpy(dtype=float, copy=True)
    z[~np.isfinite(x)] = np.nan  # never fill missing/low-confidence frames
    return z


def adjacent_speed(x, y, times, scale):
    """Torso-lengths/sec for adjacent *valid* sampled frames; no gap bridging."""
    speed = np.full(len(times), np.nan)
    dt = np.diff(times)
    dx = np.diff(x)
    dy = np.diff(y)
    valid = (np.isfinite(dx) & np.isfinite(dy) & np.isfinite(dt) &
             (dt > 0) & (dt <= 0.5) & np.isfinite(scale) & (scale > 0))
    speed[1:][valid] = np.sqrt(dx[valid]**2 + dy[valid]**2) / dt[valid] / scale
    return speed


def analyse_file(path, record):
    df = pd.read_csv(path, low_memory=False)
    missing = sorted(set(EXPECTED) - set(df.columns))
    extra = sorted(set(df.columns) - set(EXPECTED))
    if missing or extra or len(df.columns) != 172:
        raise ValueError(f'33-landmark schema mismatch; missing={missing[:5]}, extra={extra[:5]}, columns={len(df.columns)}')
    if df.empty:
        raise ValueError('Empty CSV')
    if df['video_id'].astype(str).nunique() != 1 or str(df['video_id'].iloc[0]) != str(record['video_id']):
        raise ValueError('CSV video_id inconsistent with inventory')
    if df['participant_id'].astype(str).nunique() != 1 or str(df['participant_id'].iloc[0]) != str(record['participant_id']):
        raise ValueError('CSV participant_id inconsistent with inventory')
    if df['camera_view'].astype(str).nunique() != 1 or str(df['camera_view'].iloc[0]) != str(record['camera_view']):
        raise ValueError('CSV camera view inconsistent with inventory')
    t = series_number(df, 'time_sec')
    fr = series_number(df, 'frame')
    if len(t) < 10 or not np.isfinite(t).all() or not np.isfinite(fr).all():
        raise ValueError('Insufficient rows or invalid timestamps/frames')
    if np.any(np.diff(t) <= 0) or np.any(np.diff(fr) <= 0):
        raise ValueError('Time/frame values are not strictly increasing')
    width = number(record.get('width'))
    height = number(record.get('height'))
    if not (width > 1 and height > 1):
        raise ValueError('Inventory lacks valid video width/height; cannot measure pixel geometry')
    view = str(record['camera_view'])
    if view not in ('side_front', 'top'):
        raise ValueError(f'Unknown camera view: {view}')
    detected = as_bool(df['pose_detected'])
    pts = {n: xy_qualified(df, n, width, height, detected) for n in REQUIRED_FOR_MEASUREMENT}
    shoulder_x = average_available(pts['left_shoulder'][0], pts['right_shoulder'][0])
    shoulder_y = average_available(pts['left_shoulder'][1], pts['right_shoulder'][1])
    hip_x = average_available(pts['left_hip'][0], pts['right_hip'][0])
    hip_y = average_available(pts['left_hip'][1], pts['right_hip'][1])
    torso_len = np.hypot(shoulder_x - hip_x, shoulder_y - hip_y)
    valid_torso = (np.isfinite(torso_len) & (torso_len > min(width, height)*0.015))
    scale = median(torso_len[valid_torso])
    if not np.isfinite(scale):
        raise ValueError('No valid torso geometry in this video')
    center_x = (shoulder_x + hip_x)/2
    center_y = (shoulder_y + hip_y)/2
    center_x[~valid_torso] = np.nan
    center_y[~valid_torso] = np.nan
    bilateral = (pts['left_shoulder'][2] & pts['right_shoulder'][2] &
                 pts['left_hip'][2] & pts['right_hip'][2] & valid_torso)

    # Angle relative to image horizontal, folded into 0..90 degrees. Validated only for side view.
    angle = np.degrees(np.arctan2(np.abs(shoulder_y - hip_y), np.abs(shoulder_x - hip_x)))
    angle[~valid_torso] = np.nan
    angle = smoothed(angle) if view == 'side_front' else np.full(len(t), np.nan)

    # Head-position proxy only, not waterline/airway state and not head rotation.
    nose_x, nose_y, nose_ok = pts['nose']
    nose_offset = (nose_y - shoulder_y) / scale
    nose_offset[~(nose_ok & valid_torso)] = np.nan
    nose_offset = smoothed(nose_offset) if view == 'side_front' else np.full(len(t), np.nan)

    # Remove camera-relative body translation from wrist/ankle activity.
    limb_speed = {}
    limb_coverage = {}
    for limb in ('left_wrist', 'right_wrist', 'left_ankle', 'right_ankle'):
        x, y, ok = pts[limb]
        rx = smoothed(x - center_x)
        ry = smoothed(y - center_y)
        rx[~(valid_torso & ok)] = np.nan
        ry[~(valid_torso & ok)] = np.nan
        limb_speed[limb] = adjacent_speed(rx, ry, t, scale)
        limb_coverage[limb] = fraction(np.isfinite(rx))

    # Screen the left/right speed streams independently. Screening only the
    # average can hide a grossly incorrect wrist when the other wrist is steady.
    limb_speed_raw = {name: speed.copy() for name, speed in limb_speed.items()}
    limb_spike = {name: np.isfinite(speed) & (speed > SPEED_SCREEN_TORSO_PER_SEC)
                  for name, speed in limb_speed_raw.items()}
    for name in limb_speed:
        limb_speed[name][limb_spike[name]] = np.nan

    def combine_speed(names, streams):
        frame_average = np.full(len(t), np.nan)
        a = np.vstack([streams[n] for n in names])
        counts = np.isfinite(a).sum(axis=0)
        frame_average[counts > 0] = np.nansum(a[:, counts > 0], axis=0) / counts[counts > 0]
        return frame_average

    arm_names = ('left_wrist', 'right_wrist')
    leg_names = ('left_ankle', 'right_ankle')
    arm_speed_raw = combine_speed(arm_names, limb_speed_raw)
    leg_speed_raw = combine_speed(leg_names, limb_speed_raw)
    arm_speed = combine_speed(arm_names, limb_speed)
    leg_speed = combine_speed(leg_names, limb_speed)
    arm_spike = np.logical_or.reduce([limb_spike[n] for n in arm_names])
    leg_spike = np.logical_or.reduce([limb_spike[n] for n in leg_names])

    arm_speed_coverage = fraction(np.isfinite(arm_speed))
    leg_speed_coverage = fraction(np.isfinite(leg_speed))

    def basis(prefix):
        l = limb_coverage['left_' + prefix]
        r = limb_coverage['right_' + prefix]
        if l >= MIN_SIDE_COVERAGE and r >= MIN_SIDE_COVERAGE:
            return 'bilateral'
        if l >= MIN_SIDE_COVERAGE:
            return 'left_only'
        if r >= MIN_SIDE_COVERAGE:
            return 'right_only'
        return 'insufficient_joint_coverage'

    arm_basis = basis('wrist')
    leg_basis = basis('ankle')
    arm_usable = arm_speed_coverage >= MIN_SPEED_FRAME_COVERAGE and arm_basis != 'insufficient_joint_coverage'
    leg_usable = leg_speed_coverage >= MIN_SPEED_FRAME_COVERAGE and leg_basis != 'insufficient_joint_coverage'

    # Overhead only: camera-relative displacement, not physical water drift.
    excursion = np.full(len(t), np.nan)
    body_path_rate = np.full(len(t), np.nan)
    if view == 'top':
        cx = smoothed(center_x)
        cy = smoothed(center_y)
        good = np.flatnonzero(np.isfinite(cx) & np.isfinite(cy))
        if len(good):
            # Robust initial reference over first 2 seconds of observed body centre.
            ref = good[t[good] <= t[good[0]] + 2.0]
            ox, oy = median(cx[ref]), median(cy[ref])
            excursion = np.hypot(cx - ox, cy - oy) / scale
            body_path_rate = adjacent_speed(cx, cy, t, scale)

    ts = pd.DataFrame({
        'frame': fr.astype(int), 'time_sec': t,
        'pose_detected': detected.astype(int),
        'torso_geometry_valid': valid_torso.astype(int),
        'trunk_angle_image_deg': angle,
        'nose_shoulder_vertical_offset_torso': nose_offset,
        'arm_speed_torso_s': arm_speed,
        'leg_speed_torso_s': leg_speed,
        'arm_speed_raw_torso_s': arm_speed_raw,
        'leg_speed_raw_torso_s': leg_speed_raw,
        'arm_speed_screen_flag': arm_spike.astype(int),
        'leg_speed_screen_flag': leg_spike.astype(int),
        'body_excursion_torso': excursion,
        'body_path_rate_torso_s': body_path_rate,
    })

    out = {
        'schema_version': VERSION,
        'video_id': str(record['video_id']),
        'participant_id': str(record['participant_id']),
        'camera_view': view,
        'source_csv': path.name,
        'video_frame_count_in_csv': int(len(t)),
        'duration_sec': optional(t[-1] - t[0]),
        'video_width_px': int(width),
        'video_height_px': int(height),
        'pose_detection_fraction': optional(fraction(detected)),
        'torso_valid_fraction': optional(fraction(valid_torso)),
        'bilateral_torso_fraction': optional(fraction(bilateral)),
        'left_wrist_valid_fraction': optional(limb_coverage['left_wrist']),
        'right_wrist_valid_fraction': optional(limb_coverage['right_wrist']),
        'left_ankle_valid_fraction': optional(limb_coverage['left_ankle']),
        'right_ankle_valid_fraction': optional(limb_coverage['right_ankle']),
        'trunk_angle_median_deg': optional(median(angle)) if view == 'side_front' else None,
        'trunk_angle_iqr_deg': optional(iqr(angle)) if view == 'side_front' else None,
        'head_nose_vertical_offset_median_torso': optional(median(nose_offset)) if view == 'side_front' else None,
        'arm_measurement_basis': arm_basis,
        'leg_measurement_basis': leg_basis,
        'arm_speed_usable_fraction': optional(arm_speed_coverage),
        'leg_speed_usable_fraction': optional(leg_speed_coverage),
        'arm_screened_frame_count': int(arm_spike.sum()),
        'leg_screened_frame_count': int(leg_spike.sum()),
        'arm_screened_frame_fraction': optional(fraction(arm_spike)),
        'leg_screened_frame_fraction': optional(fraction(leg_spike)),
        'speed_screen_threshold_torso_per_sec': SPEED_SCREEN_TORSO_PER_SEC,
        'arm_speed_raw_p99_torso_s': optional(np.nanpercentile(arm_speed_raw, 99)) if np.isfinite(arm_speed_raw).any() else None,
        'leg_speed_raw_p99_torso_s': optional(np.nanpercentile(leg_speed_raw, 99)) if np.isfinite(leg_speed_raw).any() else None,
        'arm_movement_speed_median_torso_s': optional(median(arm_speed)) if arm_usable else None,
        'leg_movement_speed_median_torso_s': optional(median(leg_speed)) if leg_usable else None,
        'body_max_excursion_torso': optional(np.nanmax(excursion)) if np.isfinite(excursion).any() else None,
        'body_path_rate_median_torso_s': optional(median(body_path_rate)),
        'airway_submersion': 'not_measurable_without_validated_waterline',
        'floating_quality_label': 'unassigned_requires_domain_expert_ground_truth',
        'qc_status': ('review_required' if (fraction(valid_torso) < 0.6 or
                                             not arm_usable or not leg_usable or
                                             arm_spike.any() or leg_spike.any() or
                                             arm_basis != 'bilateral' or leg_basis != 'bilateral')
                      else 'preliminary_measurements'),
        'metric_notes': {
            'trunk_angle_median_deg': 'Side/front projection relative to IMAGE horizontal, not the water surface',
            'head_nose_vertical_offset_median_torso': 'Image-coordinate head-position proxy; NOT a head-orientation or airway score',
            'arm_movement_speed_median_torso_s': 'Mean of available left/right wrist body-relative speeds; not a force/tension measurement',
            'leg_movement_speed_median_torso_s': 'Mean of available left/right ankle body-relative speeds; not a force/tension measurement',
            'body_max_excursion_torso': 'Top-camera body-centre displacement relative to initial reference; may include camera motion',
            'body_path_rate_median_torso_s': 'Top-camera body-centre speed; not calibrated metres/sec',
            'speed_screen': 'Engineering diagnostic >10 torso lengths/sec; screened values withheld, not proven erroneous. Manual frame validation required.',
            'measurement_basis': 'Unilateral when one limb is reliably visible; not comparable with bilateral activity without stratification.',
            'missing_metric': 'NULL when fewer than 60% of frames support screened limb speed or no joint has sufficient coverage.',
            'qc_status': 'Technical flag only; independent manual ground truth validation still needed',
        },
    }
    if out['camera_view'] == 'side_front' and out['trunk_angle_median_deg'] is None:
        out['qc_status'] = 'review_required'
    return out, ts


def window_metrics(ts, result, window_sec=WINDOW_SEC):
    """Non-overlapping measurement windows; these are NOT expert-labelled periods.

    Report only complete >= 10 second observed windows, and only where >= 60%
    of sampled frames have valid screened speed and measurement basis is adequate.
    """
    t = ts['time_sec'].to_numpy(dtype=float)
    starts = np.arange(float(t[0]), float(t[-1]), window_sec)
    rows = []
    for start in starts:
        end = min(start + window_sec, float(t[-1]))
        if end - start < 10:
            continue
        idx = (t >= start) & (t < end if end < t[-1] else t <= end)
        n = int(idx.sum())
        if n < 10:
            continue
        entry = {'video_id': result['video_id'], 'participant_id': result['participant_id'],
                 'camera_view': result['camera_view'], 'window_start_sec': round(start, 3),
                 'window_end_sec': round(end, 3), 'n_frames': n,
                 'expert_aligned': False, 'floating_outcome': None}
        for source, target, basis in (
            ('arm_speed_torso_s','arm_speed_median_torso_s',result['arm_measurement_basis']),
            ('leg_speed_torso_s','leg_speed_median_torso_s',result['leg_measurement_basis'])):
            arr = ts.loc[idx, source].to_numpy(dtype=float)
            coverage = fraction(np.isfinite(arr))
            entry[target + '_coverage'] = optional(coverage)
            entry[target] = (optional(median(arr)) if coverage >= MIN_SPEED_FRAME_COVERAGE and
                             basis != 'insufficient_joint_coverage' else None)
        if result['camera_view'] == 'side_front':
            arr = ts.loc[idx,'trunk_angle_image_deg'].to_numpy(dtype=float)
            entry['trunk_angle_median_deg'] = optional(median(arr))
            entry['trunk_angle_iqr_deg'] = optional(iqr(arr))
        else:
            arr = ts.loc[idx,'body_path_rate_torso_s'].to_numpy(dtype=float)
            entry['body_path_rate_median_torso_s'] = optional(median(arr))
        entry['needs_review'] = bool(result['qc_status'] == 'review_required')
        rows.append(entry)
    return pd.DataFrame(rows)

