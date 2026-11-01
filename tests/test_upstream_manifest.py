"""Repository structure and public-data checks for the consolidated project.

The project stores historical classifier source in research/three_class rather
than in the temporary stages/ folder used during initial packaging.
"""
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
CLASSIFIER = ROOT / "research" / "three_class"


class RepositoryLayoutTests(unittest.TestCase):
    def test_core_application_files_are_present(self):
        required = [
            "run.py",
            "src/flotation/__init__.py",
            "src/flotation/cli.py",
            "src/flotation/features.py",
            "src/flotation/ml.py",
            "src/flotation/supervised.py",
            "src/flotation/semisupervised.py",
        ]
        for name in required:
            with self.subTest(file=name):
                self.assertTrue((ROOT / name).is_file(), f"Missing core source: {name}")

    def test_historical_classifier_sources_are_present(self):
        required = [
            "swim_train.py",
            "swim_test.py",
            "mp_features.py",
            "make_splits.py",
            "CODEBOOK.md",
        ]
        for name in required:
            with self.subTest(file=name):
                self.assertTrue(
                    (CLASSIFIER / name).is_file(),
                    f"Missing historical classifier source: {name}",
                )

    def test_no_embedded_sensitive_rows(self):
        for path in ("private_inputs", "data/real_study", "outputs/latest/private"):
            with self.subTest(path=path):
                self.assertFalse((ROOT / path).exists())


if __name__ == "__main__":
    unittest.main()
