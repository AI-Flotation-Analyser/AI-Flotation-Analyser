import hashlib
import json
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[1]

class UpstreamManifestTests(unittest.TestCase):
    def test_manifest_files_and_shas(self):
        m=json.loads((ROOT/'stages/03_labelled_supervised_final/UPSTREAM_SOURCE_MANIFEST.json').read_text())
        self.assertEqual(len(m['files']),38)
        self.assertEqual(len(set(x['file'] for x in m['files'])),38)
        for x in m['files']:
            self.assertEqual(len(x['git_blob_sha1']),40)
            self.assertGreater(x['bytes'],0)
            self.assertNotIn('/',x['file'])
    def test_frozen_classifier_files_named(self):
        m=json.loads((ROOT/'stages/03_labelled_supervised_final/UPSTREAM_SOURCE_MANIFEST.json').read_text())
        names={x['file'] for x in m['files']}
        for f in ('swim_train.py','swim_test.py','swim_predict.py','mp_features.py','make_splits.py','CODEBOOK.md'):
            self.assertIn(f,names)
    def test_no_embedded_sensitive_rows(self):
        self.assertFalse((ROOT/'private_inputs').exists())
        self.assertFalse((ROOT/'data'/'real_study').exists())

if __name__=='__main__': unittest.main()