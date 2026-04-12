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

missingness = importlib.import_module("missingness")
MissingDataGenerator = missingness.MissingDataGenerator


class MissingnessTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(42)
        self.X = rng.normal(size=(200, 8))
        self.y = rng.integers(0, 2, size=200)

    def _assert_common_properties(self, X_out, y_obs):
        y_array = np.asarray(y_obs)
        self.assertEqual(X_out.shape, self.X.shape)
        self.assertEqual(y_array.shape, self.y.shape)
        self.assertTrue(np.isin(y_array[y_array != -1], [0, 1]).all())
        self.assertGreater(np.mean(y_array == -1), 0.01)

    def test_apply_mcar_has_expected_missing_ratio(self):
        X_out, y_obs = MissingDataGenerator.apply_mcar(
            self.X, self.y, c=0.3, random_state=7
        )

        self._assert_common_properties(X_out, y_obs)
        ratio = np.mean(np.asarray(y_obs) == -1)
        self.assertGreater(ratio, 0.15)
        self.assertLess(ratio, 0.45)

    def test_apply_mar1_mar2_mnar_output_shape_and_values(self):
        for fn in (
            MissingDataGenerator.apply_mar1,
            MissingDataGenerator.apply_mar2,
            MissingDataGenerator.apply_mnar,
        ):
            X_out, y_obs = fn(self.X, self.y, random_state=11)
            self._assert_common_properties(X_out, y_obs)

    def test_output_type_is_series_when_input_is_series(self):
        y_series = pd.Series(self.y)
        _, y_obs = MissingDataGenerator.apply_mar1(self.X, y_series, random_state=17)
        self.assertIsInstance(y_obs, pd.Series)

    def test_random_state_reproducibility(self):
        _, y_obs_1 = MissingDataGenerator.apply_mnar(self.X, self.y, random_state=99)
        _, y_obs_2 = MissingDataGenerator.apply_mnar(self.X, self.y, random_state=99)

        np.testing.assert_array_equal(np.asarray(y_obs_1), np.asarray(y_obs_2))


if __name__ == "__main__":
    unittest.main()
