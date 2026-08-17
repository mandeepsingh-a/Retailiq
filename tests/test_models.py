"""Smoke tests for the modeling pipelines. These don't re-check exact
metric values (those depend on the seeded synthetic data and would be
brittle); instead they confirm the pipelines run end-to-end, produce
correctly-shaped outputs, and -- critically -- that there is no target
leakage in the churn feature set."""

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import churn_model


class TestChurnModelSetup(unittest.TestCase):
    def test_recency_excluded_from_features_to_prevent_leakage(self):
        """recency_days was used to determine ELIGIBILITY (must be <=90
        at T0 to enter the panel), so it's entangled with row sampling
        and must never appear in the model's feature list."""
        self.assertNotIn("recency_days", churn_model.NUMERIC_FEATURES)
        self.assertNotIn("recency_days", churn_model.CATEGORICAL_FEATURES)

    def test_preprocessor_builds_without_error(self):
        pre = churn_model._build_preprocessor()
        self.assertEqual(len(pre.transformers), 2)

    def test_evaluate_returns_expected_keys(self):
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        rng = np.random.default_rng(0)
        X = pd.DataFrame({"x1": rng.normal(size=200), "x2": rng.normal(size=200)})
        y = (X["x1"] + rng.normal(scale=0.1, size=200) > 0).astype(int)

        pipe = Pipeline([("scale", StandardScaler()), ("clf", LogisticRegression())])
        pipe.fit(X, y)

        metrics = churn_model._evaluate("test_model", pipe, X, y)
        for key in ("model", "roc_auc", "pr_auc", "precision", "recall", "f1"):
            self.assertIn(key, metrics)
        self.assertGreater(metrics["roc_auc"], 0.5)  # better than random on separable data


if __name__ == "__main__":
    unittest.main()
