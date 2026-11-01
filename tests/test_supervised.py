"""Tests for expert-rated data alignment, independence and real model fitting."""
import json
from argparse import Namespace
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from flotation.supervised import (trial_key,read_labels,align_expert_intervals,
                                   evaluate, FEATURES)
from flotation.cli import complete
from test_pipeline import synthetic_batch


def four_expert_fixture():
    ids=['P2','P3','P4','P5']
    labels=pd.DataFrame({'participant_id':ids,'interval_start_sec':[30]*4,
             'interval_end_sec':[60]*4,'floating_outcome':[3,0,2,1],
             'overall_score':[12,4,11,6]})
    v,w=synthetic_batch(36)
    w['arm_speed_median_torso_s']=np.where(w.camera_view.eq('top'),0.8,0.2)
    w['leg_speed_median_torso_s']=np.where(w.camera_view.eq('side_front'),0.9,0.3)
    w['trunk_angle_median_deg']=np.where(w.camera_view.eq('side_front'),35.0,np.nan)
    return v,w,labels


class ExpertPilotTests(unittest.TestCase):
    def test_expert_labels_constraints(self):
        _,_,labels=four_expert_fixture()
        with tempfile.TemporaryDirectory() as temp:
            f=Path(temp)/'labels.csv'
            labels.to_csv(f,index=False)
            self.assertEqual(len(read_labels(f)),4)
            labels.loc[1,'participant_id']='P2'
            labels.to_csv(f,index=False)
            with self.assertRaisesRegex(ValueError,'distinct'):
                read_labels(f)

    def test_trial_pair_and_30s_window(self):
        v,w,labels=four_expert_fixture()
        samples,alignment=align_expert_intervals(v,w,labels)
        self.assertEqual(len(samples),4)
        self.assertTrue(alignment.trial_pair_status.eq('matched_camera_pair').all())
        self.assertTrue(np.isfinite(samples.loc[:,FEATURES].to_numpy(float)).all())
        self.assertEqual(trial_key('P2_side_front_p2_josh_3min_tread_1_2_side_front'),
                         trial_key('P2_top_p2_josh_3min_tread_1_3_top'))
        self.assertTrue(samples.interval_start_sec.eq(30).all())

    def test_no_cross_video_trial_matching(self):
        _,w,labels=four_expert_fixture()
        w.loc[w.camera_view.eq('side_front'),'video_id'] = (
            w.loc[w.camera_view.eq('side_front'),'video_id'].str.replace('_side_front_','_side_front_different_',regex=False))
        samples,alignment=align_expert_intervals(pd.DataFrame(),w,labels)
        self.assertEqual(len(samples),0)
        self.assertEqual(len(alignment),4)

    def test_lopo_folds_and_models(self):
        v,w,labels=four_expert_fixture()
        samples,_=align_expert_intervals(v,w,labels)
        result=evaluate(samples)
        self.assertEqual(result['report']['fold_count'],4)
        self.assertEqual(result['report']['participant_leakage'],0)
        self.assertEqual(len(result['predictions']),8)
        self.assertTrue(result['predictions'].fold_train_participants.eq(3).all())
        self.assertSetEqual(set(result['trained_models']),{'floating_outcome','overall_score'})
        self.assertTrue(np.isfinite(result['predictions'].held_out_prediction).all())
        with self.assertRaisesRegex(ValueError,'duplicate participants'):
            evaluate(pd.concat([samples,samples.iloc[:1]],ignore_index=True))

    def test_full_product_with_expert_pilot(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            source=root/'research'
            batch=source/'outputs/final_delivery/run_1'
            batch.mkdir(parents=True)
            v,w,labels=four_expert_fixture()
            v.to_csv(batch/'final_video_features.csv',index=False)
            w.to_csv(batch/'final_window_features.csv',index=False)
            (batch/'source_inventory_failures.csv').write_text('video_id,reason\n')
            (batch/'FINAL_RUN_EVIDENCE.json').write_text(json.dumps({'n_output_videos':len(v)}))
            label_file=root/'labels.csv'
            labels.to_csv(label_file,index=False)
            out=root/'output'
            complete(Namespace(source_root=source,output=out,rebuild=False,seed=42,
                               bootstrap=2,expert_labels=label_file))
            self.assertTrue((out/'supervised/supervised_metrics.json').is_file())
            self.assertTrue((out/'supervised/trained_expert_pilot_models.joblib').is_file())
            self.assertTrue((out/'figures/supervised_model_vs_baseline.png').is_file())
            report=(out/'REPORT_RESULTS.md').read_text()
            self.assertIn('Expert-anchored supervised pilot',report)
            dash=(out/'CLIENT_DASHBOARD.html').read_text()
            self.assertIn('Expert-supervised pilot',dash)
            metrics=json.loads((out/'supervised/supervised_metrics.json').read_text())
            self.assertEqual(metrics['participant_count'],4)
            self.assertEqual(metrics['targets']['floating_outcome']['n_held_out_predictions'],4)

if __name__=='__main__':
    unittest.main()
