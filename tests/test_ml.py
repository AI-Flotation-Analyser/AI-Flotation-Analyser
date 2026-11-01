import sys
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from flotation.ml import eligible_cohort, train_and_evaluate, finite_silhouette

class HoldoutTests(unittest.TestCase):
    def demo(self):
        rng=np.random.default_rng(8)
        items=[]
        for i in range(66):
            group=i//22
            vals=np.maximum(.01,np.array([.3 + group*1.4, .05+group*.3, .3+group*.8])+rng.normal(0,.1,3))
            row=dict(video_id=f'v{i}_long',participant_id=f'P{i}',camera_view='top',
                     arm_measurement_basis='bilateral',torso_valid_fraction=.98,
                     arm_speed_usable_fraction=.96,arm_screened_frame_fraction=.01,
                     duration_sec=170,
                     arm_movement_speed_median_torso_s=vals[0],
                     body_path_rate_median_torso_s=vals[1],body_max_excursion_torso=vals[2])
            items.append(row)
            dup=dict(row,video_id=f'v{i}_short',duration_sec=80)
            items.append(dup)
            side=dict(row,video_id=f'v{i}_side',camera_view='side_front')
            items.append(side)
        return pd.DataFrame(items)
    def test_one_video_one_participant(self):
        co=eligible_cohort(self.demo())
        self.assertEqual(len(co),66)
        self.assertFalse(co.participant_id.duplicated().any())
        self.assertTrue(co.video_id.str.endswith('_long').all())
    def test_strict_qc(self):
        d=self.demo()
        d.loc[0,'arm_screened_frame_fraction']=.80
        d.loc[1,'arm_screened_frame_fraction']=.80
        co=eligible_cohort(d)
        self.assertEqual(len(co),65)
    def test_participant_holdout_and_repeatability(self):
        co=eligible_cohort(self.demo())
        a=train_and_evaluate(co,seed=42,n_bootstrap=4)
        b=train_and_evaluate(co,seed=42,n_bootstrap=4)
        ra=a['report'];rb=b['report']
        self.assertEqual(ra['participant_overlap'],0)
        self.assertEqual(ra['train_participants']+ra['test_participants'],66)
        self.assertEqual(ra['kmeans']['selected_k'],rb['kmeans']['selected_k'])
        self.assertEqual(ra['kmeans']['test_assigned_silhouette'],rb['kmeans']['test_assigned_silhouette'])
        self.assertEqual(set(a['assignments'].split),{'train','test'})
        self.assertEqual(len(a['assignments']),66)
    def test_holdout_silhouette_with_singleton_cluster(self):
        from sklearn.metrics import silhouette_score
        X = np.array([[0.], [.2], [3.], [3.1], [10.]])
        labels = [0, 0, 1, 1, 2]
        self.assertAlmostEqual(finite_silhouette(X, labels), silhouette_score(X, labels))
        self.assertIsNone(finite_silhouette(X, [0, 0, 0, 0, 0]))

    def test_reject_insufficient_cohort(self):
        co=eligible_cohort(self.demo()).head(12)
        with self.assertRaisesRegex(ValueError,'At least 30'):
            train_and_evaluate(co)

if __name__=='__main__':unittest.main()