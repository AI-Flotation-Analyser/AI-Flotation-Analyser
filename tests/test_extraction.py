"""Video-ingestion tests use a mock video decoder and pose provider."""
import csv
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from flotation.extract import (extract_video, extract_dataset, read_video_manifest, _value)
from flotation.features import EXPECTED, analyse_file, window_metrics, LANDMARKS


class FakeLandmark:
    def __init__(self, index, t):
        self.x = .15 + (index % 3) * .15 + 0.001 * np.sin(t / 6)
        self.y = .20 + (index % 5) * .07 + 0.001 * np.cos(t / 6)
        self.z = .01
        self.visibility = .99
        self.presence = .99

    def HasField(self, name):
        return True


class FakePose:
    def __init__(self, **_):
        self.t = 0

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def process(self, _frame):
        self.t += 1
        pts = [FakeLandmark(i, self.t) for i in range(33)]
        # Adequate torso lengths, with different shoulder and hip centres.
        for j in (11,12):
            pts[j].y = .36
        for j in (23,24):
            pts[j].y = .56
        for j in (15,16):
            pts[j].y = .25
        for j in (27,28):
            pts[j].y = .77
        for j in (0,):
            pts[j].y = .30
        return SimpleNamespace(pose_landmarks=SimpleNamespace(landmark=pts))


class FakeCapture:
    def __init__(self, _path):
        self.i = 0

    def isOpened(self):
        return True

    def get(self, code):
        return {1:4.,2:640.,3:480.,4:120.}[code]

    def read(self):
        if self.i >= 120:
            return False, None
        self.i += 1
        return True, np.zeros((1,1,3),dtype=np.uint8)

    def release(self):
        pass


class FakeCV2:
    CAP_PROP_FPS = 1
    CAP_PROP_FRAME_WIDTH = 2
    CAP_PROP_FRAME_HEIGHT = 3
    CAP_PROP_FRAME_COUNT = 4
    COLOR_BGR2RGB = 1
    VideoCapture = FakeCapture

    @staticmethod
    def cvtColor(frame, _code):
        return frame


class ExtractorTests(unittest.TestCase):
    def fake(self):
        return FakeCV2, SimpleNamespace(solutions=SimpleNamespace(pose=SimpleNamespace(Pose=FakePose)))

    def test_manifest_rejects_bad_ids(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'manifest.csv'
            p.write_text('video_id,participant_id,camera_view,video_path\n../evil,P1,top,video.mp4\n')
            with self.assertRaisesRegex(ValueError,'video_id'):
                read_video_manifest(p)

    def test_video_schema_and_downstream_compatibility(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)
            video=p/'demo.mp4'; video.write_bytes(b'mock')
            dest=p/'P1_top_session1_raw.csv'
            info=extract_video(video,dest,{'video_id':'P1_top_session1','participant_id':'P1','camera_view':'top'},dependency_factory=self.fake)
            self.assertEqual(info['sampled_frames'],120)
            df=pd.read_csv(dest)
            self.assertEqual(df.columns.tolist(),EXPECTED)
            self.assertEqual(len(df.columns),172)
            self.assertEqual(len(df),120)
            self.assertTrue(df.pose_detected.all())
            rec={'video_id':'P1_top_session1','participant_id':'P1','camera_view':'top','width':640,'height':480}
            out,ts=analyse_file(dest,rec)
            self.assertGreater(out['torso_valid_fraction'],0.9)
            self.assertFalse(window_metrics(ts,out).empty)

    def test_dataset_and_cache_reuse(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)
            (p/'videos').mkdir()
            for view in ('top','side_front'):
                (p/'videos'/f'P1_{view}_session1.mp4').write_bytes(b'fake')
            manifest=p/'manifest.csv'
            manifest.write_text('video_id,participant_id,camera_view,video_path\n'
                                'P1_top_session1,P1,top,videos/P1_top_session1.mp4\n'
                                'P1_side_front_session1,P1,side_front,videos/P1_side_front_session1.mp4\n')
            def produce(video,target,identity,**kwargs):
                return extract_video(video,target,identity,dependency_factory=self.fake)
            source=p/'dataset'
            a=extract_dataset(source,manifest,extract_fn=produce)
            self.assertEqual(set(a.extraction_status),{'created'})
            b=extract_dataset(source,manifest,extract_fn=lambda *a,**k:self.fail('cache must be reused'))
            self.assertEqual(set(b.extraction_status),{'reused'})
            inv=pd.read_csv(source/'data/metadata/video_inventory.csv')
            self.assertEqual(len(inv),2)
            self.assertEqual(set(inv.camera_view),{'top','side_front'})
            from flotation.pipeline import rebuild_from_raw33
            videos,windows,failures,_=rebuild_from_raw33(source,p/'results')
            self.assertEqual(len(videos),2)
            self.assertEqual(len(failures),0)
            self.assertGreaterEqual(len(windows),2)

    def test_optional_presence(self):
        class WithoutPresence:
            def HasField(self, name):
                return False
        self.assertEqual(_value(WithoutPresence(),'presence'),1.)


if __name__=='__main__':
    unittest.main()