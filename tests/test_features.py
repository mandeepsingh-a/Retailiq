"""Unit tests for RFM / behavioral feature engineering."""

import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import features


class TestFeatureEngineering(unittest.TestCase):
    def setUp(self):
        self.tmp_clean = Path(__file__).parent / "_tmp_clean_txn.csv"
        self.tmp_cust = Path(__file__).parent / "_tmp_customers.csv"
        self.tmp_out = features.FEATURES_CSV

        txn = pd.DataFrame({
            "order_id": [1, 2, 3, 4],
            "customer_id": ["C1", "C1", "C2", "C3"],
            "order_date": ["2024-01-01", "2024-03-01", "2024-01-15", "2024-01-01"],
            "product_id": ["P1", "P2", "P1", "P1"],
            "category": ["Books", "Toys", "Books", "Books"],
            "quantity": [1, 2, 1, 1],
            "unit_price": [10.0, 5.0, 20.0, 15.0],
            "region": ["North", "North", "South", "East"],
            "is_return": [False, False, False, False],
            "line_revenue": [10.0, 10.0, 20.0, 15.0],
            "order_year_month": ["2024-01", "2024-03", "2024-01", "2024-01"],
        })
        txn.to_csv(self.tmp_clean, index=False)

        customers = pd.DataFrame({
            "customer_id": ["C1", "C2", "C3"],
            "value_tier": ["high_value", "mid_value", "low_value"],
            "signup_date": ["2023-12-01", "2023-12-01", "2023-12-01"],
            "region": ["North", "South", "East"],
        })
        customers.to_csv(self.tmp_cust, index=False)

    def tearDown(self):
        for p in (self.tmp_clean, self.tmp_cust, self.tmp_out):
            if p.exists():
                p.unlink()

    def test_every_customer_present(self):
        result = features.build_customer_features(self.tmp_clean, self.tmp_cust)
        self.assertEqual(set(result["customer_id"]), {"C1", "C2", "C3"})

    def test_frequency_counts_distinct_orders(self):
        result = features.build_customer_features(self.tmp_clean, self.tmp_cust)
        c1 = result.loc[result["customer_id"] == "C1"].iloc[0]
        self.assertEqual(c1["frequency"], 2)  # two distinct orders
        self.assertEqual(c1["monetary"], 20.0)

    def test_recency_uses_latest_dataset_date_not_today(self):
        # Max order_date in fixture is 2024-03-01, so C3 (last purchase
        # 2024-01-01) should have recency of 60 days, not "days since
        # actual today's date" -- a common, subtle bug.
        result = features.build_customer_features(self.tmp_clean, self.tmp_cust)
        c3 = result.loc[result["customer_id"] == "C3"].iloc[0]
        self.assertEqual(c3["recency_days"], 60)

    def test_churn_flag_respects_window(self):
        result = features.build_customer_features(self.tmp_clean, self.tmp_cust)
        c3 = result.loc[result["customer_id"] == "C3"].iloc[0]  # recency 60d < 90d window
        c2 = result.loc[result["customer_id"] == "C2"].iloc[0]  # recency 45d < 90d
        self.assertEqual(c3["is_churned"], 0)
        self.assertEqual(c2["is_churned"], 0)

    def test_rfm_scores_are_within_expected_range(self):
        result = features.build_customer_features(self.tmp_clean, self.tmp_cust)
        for col in ["R_score", "F_score", "M_score"]:
            self.assertTrue(result[col].between(1, 5).all())


if __name__ == "__main__":
    unittest.main()
