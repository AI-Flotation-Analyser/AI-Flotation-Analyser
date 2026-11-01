"""Data-integrity and end-to-end orchestration regression checks."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from flotation.pipeline import load_completed_batch, validate, summarize_views, rebuild_from_raw33
from flotation.cli import complete
from flotation.ml import eligible_cohort
from argparse import Namespace


def synthetic_batch(n=60):
    """An artificial dataset for software tests; not participant research evidence."""
    rng = np.random.default_rng(7)
    rows, windows = [], []
    for i in range(n):
        # Artificial groups with different coordinates, purely for test coverage.
        c = i % 3
        for view in ('side_front', 'top'):
            vid = f'P{i}_{view}_synthetic'
            rows.append(dict(video_id=vid, participant_id=f'P{i}', camera_view=view,
                qc_status='review_required', duration_sec=180, torso_valid_fraction=.99,
                arm_measurement_basis='bilateral' if view == 'top' else 'left_only',
                leg_measurement_basis='bilateral' if view == 'side_front' else 'insufficient_joint_coverage',
                arm_speed_usable_fraction=.97, arm_screened_frame_fraction=.01,
                arm_movement_speed_median_torso_s=.25+c*.9+abs(rng.normal(0,.06)),
                leg_movement_speed_median_torso_s=.8 if view == 'side_front' else None,
                body_path_rate_median_torso_s=.1+c*.15 if view == 'top' else None,
                body_max_excursion_torso=.3+c*.5 if view == 'top' else None))
            for k in range(6):
                windows.append(dict(video_id=vid, participant_id=f'P{i}', camera_view=view,
                                    window_start_sec=k*30, window_end_sec=(k+1)*30, n_frames=750))
    return pd.DataFrame(rows), pd.DataFrame(windows)


class ProductTests(unittest.TestCase):
    def test_reject_duplicate_video_ids(self):
        v, w = synthetic_batch(6)
        v.loc[1, 'video_id'] = v.loc[0, 'video_id']
        with self.assertRaisesRegex(ValueError, 'duplicate video IDs'):
            validate(v, w)

    def test_reject_window_participant_leak(self):
        v, w = synthetic_batch(6)
        w.loc[0, 'participant_id'] = 'NOT_SAME_PERSON'
        with self.assertRaisesRegex(ValueError, 'participant ID mismatch'):
            validate(v, w)

    def test_camera_summary(self):
        v, w = synthetic_batch(6)
        validate(v, w)
        q = summarize_views(v)
        top = q.loc[q.camera_view.eq('top')].iloc[0]
        self.assertEqual(top.video_count, 6)
        self.assertEqual(top.leg_feature_available, 0)

    def test_raw33_rebuild_and_cache_reuse(self):
        # A minimal synthetic 33-landmark recording verifies the rebuild path.
        from test_features import fixture
        with tempfile.TemporaryDirectory() as temp:
            project = Path(temp) / 'source'
            raw = project / 'data/landmarks_raw/mediapipe_all33_v2'
            inv = project / 'data/metadata'
            raw.mkdir(parents=True)
            inv.mkdir(parents=True)
            record = {'video_id':'P2_top_test', 'participant_id':'P2',
                      'camera_view':'top','width':1920,'height':1080}
            pd.DataFrame([record]).to_csv(inv/'video_inventory.csv', index=False)
            fixture(frame_count=900, view='top').to_csv(raw/'P2_top_test_raw.csv',index=False)
            output = Path(temp)/'new_outputs'
            one, window_one, missing_one, first = rebuild_from_raw33(project, output)
            self.assertEqual(len(one), 1)
            self.assertEqual(first['newly_processed'], 1)
            self.assertEqual(len(missing_one), 0)
            two, window_two, missing_two, second = rebuild_from_raw33(project, output)
            self.assertEqual(len(two), 1)
            self.assertEqual(len(window_one), len(window_two))
            self.assertEqual(second['reused_cached'], 1)

    def test_complete_product_from_prior_batch(self):
        with tempfile.TemporaryDirectory() as tmp:
            project = Path(tmp) / 'old_project'
            run = project / 'outputs/final_delivery/run_1'
            run.mkdir(parents=True)
            v, w = synthetic_batch(36)
            v.to_csv(run / 'final_video_features.csv', index=False)
            w.to_csv(run / 'final_window_features.csv', index=False)
            (run / 'source_inventory_failures.csv').write_text('video_id,reason\n', encoding='utf-8')
            (run / 'FINAL_RUN_EVIDENCE.json').write_text(json.dumps({'n_output_videos':len(v)}))
            out = Path(tmp) / 'generated'
            complete(Namespace(source_root=project, output=out, rebuild=False, seed=42, bootstrap=2))
            self.assertTrue((out / 'CLIENT_DASHBOARD.html').is_file())
            self.assertTrue((out / 'REPORT_RESULTS.md').is_file())
            self.assertTrue((out / 'ml/heldout_metrics.json').is_file())
            a = pd.read_csv(out / 'private/participant_model_assignments.csv')
            train = set(a.loc[a.split.eq('train'), 'participant_id'])
            test = set(a.loc[a.split.eq('test'), 'participant_id'])
            self.assertFalse(train & test)
            self.assertEqual(len(train|test), 36)


if __name__ == '__main__':
    unittest.main()