import sys
import unittest
import importlib
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

data_prep = importlib.import_module("data_prep")
clean_numeric_features = data_prep.clean_numeric_features
prepare_features = data_prep.prepare_features
remove_collinear_features = data_prep.remove_collinear_features


class DataPrepTests(unittest.TestCase):
    def test_clean_numeric_features_converts_and_imputes(self):
        df = pd.DataFrame(
            {
                "a": [1, "2", None, "bad"],
                "b": ["3.5", "4.1", None, 2],
            }
        )

        cleaned = clean_numeric_features(df)

        self.assertTrue(np.issubdtype(cleaned["a"].dtype, np.number))
        self.assertTrue(np.issubdtype(cleaned["b"].dtype, np.number))
        self.assertFalse(cleaned.isna().any().any())

    def test_remove_collinear_features_drops_high_corr_columns(self):
        df = pd.DataFrame(
            {
                "x": [1.0, 2.0, 3.0, 4.0, 5.0],
                "x_dup": [2.0, 4.0, 6.0, 8.0, 10.0],
                "y": [5.0, 1.0, 3.0, 4.0, 2.0],
            }
        )

        reduced, dropped = remove_collinear_features(df, threshold=0.95)

        self.assertIn("x_dup", dropped)
        self.assertEqual(reduced.shape[1], 2)
        self.assertNotIn("x_dup", reduced.columns)

    def test_prepare_features_runs_full_pipeline(self):
        df = pd.DataFrame(
            {
                "f1": [1, 2, 3, 4, 5],
                "f2": [2, 4, 6, 8, 10],
                "f3": [1, None, 1, None, 1],
            }
        )

        prepared, dropped = prepare_features(df, corr_threshold=0.9)

        self.assertFalse(prepared.isna().any().any())
        self.assertGreaterEqual(len(dropped), 1)


if __name__ == "__main__":
    unittest.main()
