"""Regression tests for the final 33-landmark features, using synthetic geometry only.

Run from project root: python -m unittest discover -s tests -p 'test_final_flotation_features_v1.py' -v
"""
import unittest
from pathlib import Path
import sys
import tempfile

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from flotation.features import LANDMARKS, EXPECTED, analyse_file


def fixture(frame_count=110, view='side_front', vertical=False, move_body=False):
    points = []
    for i in range(frame_count):
        dx = (i * .0001) if move_body else 0
        shoulder = (.40 + dx, .40)
        hip = (.40 + dx, .70) if vertical else (.70 + dx, .40)
        landmarks = {p:(.5+dx,.5) for p in LANDMARKS}
        landmarks.update({
            'left_shoulder':shoulder, 'right_shoulder':shoulder,
            'left_hip':hip, 'right_hip':hip,
            'nose':(.30+dx,.35),
            'left_wrist':(.25+dx,.30), 'right_wrist':(.27+dx,.30),
            'left_ankle':(.80+dx,.5), 'right_ankle':(.81+dx,.5),
        })
        row = {'video_id':'P2_'+view+'_test','participant_id':'P2','camera_view':view,
               'video_name':'test.mp4','frame':i,'time_sec':i/25.,'pose_detected':True}
        for name in LANDMARKS:
            x,y = landmarks[name]
            for suffix,value in [('x',x),('y',y),('z',0),('visibility',.95),('presence',.98)]:
                row[f'{name}_{suffix}']=value
        points.append(row)
    df=pd.DataFrame(points)
    assert list(df.columns)==EXPECTED
    return df


class TestFinalFeatures(unittest.TestCase):
    def test_side_view_horizontal_trunk_and_unlabelled_outcome(self):
        df=fixture()
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'input.csv'
            df.to_csv(path,index=False)
            result,ts=analyse_file(path,{'video_id':'P2_side_front_test','participant_id':'P2',
                                         'camera_view':'side_front','width':1920,'height':1080})
        self.assertAlmostEqual(result['trunk_angle_median_deg'], 0, places=5)
        self.assertEqual(result['torso_valid_fraction'],1)
        self.assertIsNone(result['body_max_excursion_torso'])
        self.assertIn('unassigned',result['floating_quality_label'])
        self.assertEqual(len(ts),110)

    def test_vertical_trunk_is_90_degrees_but_not_a_quality_label(self):
        df=fixture(vertical=True)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'input.csv'
            df.to_csv(path,index=False)
            result,_=analyse_file(path,{'video_id':'P2_side_front_test','participant_id':'P2',
                                        'camera_view':'side_front','width':1920,'height':1080})
        self.assertAlmostEqual(result['trunk_angle_median_deg'],90,places=5)
        self.assertEqual(result['floating_quality_label'],'unassigned_requires_domain_expert_ground_truth')

    def test_pose_confidence_gates_missing_wrist_activity(self):
        df=fixture()
        df['left_wrist_visibility']=0.1
        df['right_wrist_visibility']=0.1
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'input.csv'
            df.to_csv(path,index=False)
            result,_=analyse_file(path,{'video_id':'P2_side_front_test','participant_id':'P2',
                                        'camera_view':'side_front','width':1920,'height':1080})
        self.assertEqual(result['left_wrist_valid_fraction'],0)
        self.assertEqual(result['right_wrist_valid_fraction'],0)
        self.assertIsNone(result['arm_movement_speed_median_torso_s'])

    def test_top_translation_without_arm_relative_motion(self):
        df=fixture(view='top',move_body=True)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'input.csv'
            df.to_csv(path,index=False)
            result,_=analyse_file(path,{'video_id':'P2_top_test','participant_id':'P2',
                                        'camera_view':'top','width':1920,'height':1080})
        self.assertIsNone(result['trunk_angle_median_deg'])
        self.assertIsNotNone(result['body_max_excursion_torso'])
        self.assertGreater(result['body_max_excursion_torso'], 0)
        self.assertAlmostEqual(result['arm_movement_speed_median_torso_s'],0,places=5)

    def test_reject_invalid_time_axis(self):
        df=fixture()
        df.loc[10,'time_sec']=df.loc[9,'time_sec']
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'input.csv'
            df.to_csv(path,index=False)
            with self.assertRaisesRegex(ValueError,'Time/frame'):
                analyse_file(path,{'video_id':'P2_side_front_test','participant_id':'P2',
                                   'camera_view':'side_front','width':1920,'height':1080})



# Additional v1.1 quality-control tests; retained below the original test suite
# to ensure no change to v1 behaviour is silently accepted.
class TestQualityGatingV11(unittest.TestCase):
    def _analyse(self, df, view='side_front'):
        from flotation.features import analyse_file, window_metrics
        with tempfile.TemporaryDirectory() as temp:
            fp = Path(temp) / 'sample.csv'
            df.to_csv(fp, index=False)
            return analyse_file(fp, {'video_id':'P2_'+view+'_test',
                                     'participant_id':'P2','camera_view':view,
                                     'width':1920,'height':1080})

    def test_top_ankle_occlusion_withholds_leg_metric(self):
        df = fixture(view='top')
        df['left_ankle_visibility'] = 0.05
        df['right_ankle_visibility'] = 0.04
        result, ts = self._analyse(df, view='top')
        self.assertIsNone(result['leg_movement_speed_median_torso_s'])
        self.assertEqual(result['leg_measurement_basis'], 'insufficient_joint_coverage')
        self.assertEqual(result['qc_status'], 'review_required')
        self.assertEqual(len(ts), len(df))

    def test_unilateral_side_is_not_claimed_bilateral(self):
        df = fixture()
        df['right_wrist_visibility'] = 0.0
        result, ts = self._analyse(df)
        self.assertEqual(result['arm_measurement_basis'], 'left_only')
        self.assertIsNotNone(result['arm_movement_speed_median_torso_s'])
        self.assertEqual(result['qc_status'], 'review_required')

    def test_extreme_single_limb_jump_is_screened_and_logged(self):
        df = fixture()
        # Long-enough displacement to survive the 5-frame median, then return.
        df.loc[40:47, 'left_wrist_x'] += 0.60
        result, ts = self._analyse(df)
        self.assertGreater(result['arm_screened_frame_count'], 0)
        self.assertGreater(result['arm_speed_raw_p99_torso_s'], 10)
        self.assertGreater(int(ts['arm_speed_screen_flag'].sum()), 0)
        spike_rows = ts['arm_speed_screen_flag'] == 1
        self.assertTrue((ts.loc[spike_rows, 'arm_speed_torso_s'].fillna(0) <= 10).all())
        self.assertEqual(result['qc_status'], 'review_required')

    def test_30s_windows_not_expert_labels(self):
        from flotation.features import window_metrics
        df = fixture(frame_count=1600, view='side_front')
        result, ts = self._analyse(df)
        windows = window_metrics(ts, result)
        self.assertEqual(len(windows), 2)  # 0-30, 30-60; short 60-64 excluded
        self.assertTrue((windows['expert_aligned'] == False).all())
        self.assertTrue(windows['floating_outcome'].isna().all())

if __name__ == '__main__':
    unittest.main()