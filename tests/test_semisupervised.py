"""Regression tests for held-out evaluation and provisional prediction tables."""
import unittest
from pathlib import Path
import sys

import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from flotation.semisupervised import participant_matrix,run
from test_pipeline import synthetic_batch


class SemiSupervisedProductTests(unittest.TestCase):
    @staticmethod
    def fixture(n=187):
        video,windows=synthetic_batch(n)
        idx=np.arange(len(windows))
        windows['arm_speed_median_torso_s']=np.where(windows.camera_view.eq('top'),
                                      0.25+idx%17*.035,0.1)
        windows['leg_speed_median_torso_s']=np.where(windows.camera_view.eq('side_front'),
                                      0.28+idx%11*.045,0.1)
        windows['trunk_angle_median_deg']=np.where(windows.camera_view.eq('side_front'),
                                      23+idx%19, np.nan)
        labels=pd.DataFrame({'participant_id':['P2','P68','P81','P118'],
            'interval_start_sec':[30]*4,'interval_end_sec':[60]*4,
            'floating_outcome':[3,2,0,1], 'overall_score':[12,11,4,6],
            'label_provenance':['expert_chris']*4})
        return video,windows,labels

    def test_187_participant_prediction_table_and_folds(self):
        video,windows,labels=self.fixture()
        m=participant_matrix(video,windows)
        self.assertEqual(len(m),187)
        result=run(m,labels)
        p=result['predictions']
        self.assertEqual(len(p),187)
        self.assertEqual(p.participant_id.nunique(),187)
        self.assertEqual(int(p.evaluation_kind.eq('leave_one_person_out_test').sum()),4)
        self.assertEqual(int(p.evaluation_kind.eq('unlabelled_inference').sum()),183)
        self.assertEqual(len(result['folds']),8) # 4 people x 2 targets
        self.assertTrue(result['folds'].n_test_label_in_training.eq(0).all())
        self.assertTrue(result['folds'].n_train_verified.eq(3).all())
        self.assertFalse(result['predictions'].is_pseudo_ground_truth.any())
        self.assertEqual(len(result['review_queue']),30)
        for target,(lo,hi) in [('floating_outcome',(0,3)),('overall_score',(0,13))]:
            estimates=p.loc[p.evaluation_kind.eq('unlabelled_inference'),
                   'semisupervised_prediction_'+target]
            self.assertTrue(estimates.between(lo,hi).all())

    def test_withheld_label_does_not_change_own_predicted_value(self):
        v,w,labels=self.fixture(125)
        matrix=participant_matrix(v,w)
        base=run(matrix,labels)
        changed=labels.copy()
        changed.loc[changed.participant_id.eq('P2'),'floating_outcome']=1
        altered=run(matrix,changed)
        a=base['folds'].query("target=='floating_outcome' and test_id=='P2'").iloc[0]
        b=altered['folds'].query("target=='floating_outcome' and test_id=='P2'").iloc[0]
        for method in ('ridge','baseline','graph','selftrain'):
            self.assertAlmostEqual(a[method+'_prediction'],b[method+'_prediction'],places=5)
        self.assertNotEqual(a.actual,b.actual)

    def test_nonpaired_participant_is_kept_in_prediction_table(self):
        v,w,labels=self.fixture(125)
        remove=w.participant_id.eq('P124') & w.camera_view.eq('top')
        w=w.loc[~remove]
        matrix=participant_matrix(v,w)
        self.assertEqual(matrix.loc[matrix.participant_id.eq('P124'),'alignment_status'].iloc[0], 'unmatched_trial')
        result=run(matrix,labels)
        row=result['predictions'].set_index('participant_id').loc['P124']
        self.assertEqual(row.evaluation_kind,'no_paired_features')
        self.assertTrue(np.isnan(row['semisupervised_prediction_floating_outcome']))

if __name__=='__main__':
    unittest.main()