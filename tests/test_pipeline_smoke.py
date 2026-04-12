import sys
import unittest
import importlib
from pathlib import Path

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

benchmarks = importlib.import_module("benchmarks")
fista = importlib.import_module("fista")
missingness = importlib.import_module("missingness")
unlabeled = importlib.import_module("unlabeled")

train_naive_model = benchmarks.train_naive_model
train_oracle_model = benchmarks.train_oracle_model
FISTA = fista.FISTA
MissingDataGenerator = missingness.MissingDataGenerator
UnlabeledLogReg = unlabeled.UnlabeledLogReg


class PipelineSmokeTests(unittest.TestCase):
    def test_semi_supervised_methods_fit_and_predict(self):
        X, y = make_classification(
            n_samples=220,
            n_features=16,
            n_informative=8,
            n_redundant=4,
            random_state=5,
        )
        y = y.astype(int)

        X_train, X_test, y_train_true, _ = train_test_split(
            X, y, test_size=0.2, random_state=5, stratify=y
        )
        _, y_train_obs = MissingDataGenerator.apply_mcar(
            X_train, y_train_true, c=0.35, random_state=5
        )
        y_train_obs = np.asarray(y_train_obs)

        for method in ("pseudo_labeling", "em"):
            model = UnlabeledLogReg(
                model=FISTA(lambdas=[0.01], max_iter=120), method=method
            )
            model.fit(X_train, y_train_obs)
            preds = model.predict(X_test)
            self.assertEqual(preds.shape[0], X_test.shape[0])

    def test_baselines_fit_and_predict(self):
        X, y = make_classification(
            n_samples=200,
            n_features=12,
            n_informative=6,
            n_redundant=2,
            random_state=9,
        )
        y = y.astype(int)

        X_train, X_test, y_train_true, _ = train_test_split(
            X, y, test_size=0.25, random_state=9, stratify=y
        )
        _, y_train_obs = MissingDataGenerator.apply_mcar(
            X_train, y_train_true, c=0.3, random_state=9
        )

        naive = train_naive_model(FISTA(lambdas=[0.01], max_iter=120), X_train, y_train_obs)
        oracle = train_oracle_model(FISTA(lambdas=[0.01], max_iter=120), X_train, y_train_true)

        naive_preds = naive.predict(X_test)
        oracle_preds = oracle.predict(X_test)

        self.assertEqual(naive_preds.shape[0], X_test.shape[0])
        self.assertEqual(oracle_preds.shape[0], X_test.shape[0])


if __name__ == "__main__":
    unittest.main()
