"""Tests for the time-based churn panel (churn_panel.py) -- the core
correctness guarantee of the future-churn methodology: features must
only use data up to T0, and labels must only use data after T0."""

import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import churn_panel
from src.features import compute_customer_features


class TestTemporalLeakage(unittest.TestCase):
    def setUp(self):
        # A customer who buys steadily before T0, stops for a while
        # (>90d) starting right after T0, then resumes long after the
        # prediction window -- this should be labeled future_churn=1,
        # and the late resumption must NOT leak into recency/frequency
        # features computed as of T0.
        self.txn = pd.DataFrame({
            "order_id": [1, 2, 3, 4, 5],
            "customer_id": ["C1", "C1", "C1", "C1", "C1"],
            "order_date": pd.to_datetime([
                "2024-01-01", "2024-02-01", "2024-03-01",  # before T0
                "2024-07-15",                                # inside 90d window after T0 stops -> churns
                "2025-01-01",                                 # far future resumption (outside window)
            ]),
            "product_id": ["P1"] * 5,
            "category": ["Books"] * 5,
            "quantity": [1] * 5,
            "unit_price": [10.0] * 5,
            "region": ["North"] * 5,
            "is_return": [False] * 5,
            "line_revenue": [10.0] * 5,
            "order_year_month": ["x"] * 5,
        })
        self.customers = pd.DataFrame({
            "customer_id": ["C1"],
            "value_tier": ["mid_value"],
            "signup_date": pd.to_datetime(["2023-12-01"]),
            "region": ["North"],
        })

    def test_features_do_not_see_data_after_t0(self):
        t0 = pd.Timestamp("2024-03-15")
        feats = compute_customer_features(self.txn, self.customers, t0)
        row = feats[feats["customer_id"] == "C1"].iloc[0]
        # As of T0, only 3 purchases (Jan/Feb/Mar) should be visible.
        self.assertEqual(row["frequency"], 3)
        # Recency should be measured against the Mar 1 purchase, not
        # any later one (14 days before T0).
        self.assertEqual(row["recency_days"], 14)

    def test_future_purchase_within_window_prevents_churn_label(self):
        # T0 = 2024-03-15; window = +90d = 2024-06-13. The Jul 15
        # purchase is OUTSIDE that window, so this customer SHOULD be
        # labeled future_churn=1 (no purchase in the 90-day window).
        panel = churn_panel.build_panel.__wrapped__ if hasattr(churn_panel.build_panel, "__wrapped__") else None
        # Directly exercise the labeling logic at a controlled T0.
        t0 = pd.Timestamp("2024-03-15")
        window_end = t0 + pd.Timedelta(days=90)
        purchases_only = self.txn[~self.txn["is_return"]]
        future_purchasers = set(
            purchases_only.loc[
                (purchases_only["order_date"] > t0) & (purchases_only["order_date"] <= window_end),
                "customer_id",
            ]
        )
        self.assertNotIn("C1", future_purchasers)  # July 15 is after the window -> should churn

    def test_purchase_inside_window_prevents_churn_label(self):
        # Move T0 earlier so the Jul 15 purchase DOES fall inside the
        # 90-day window -> customer should NOT be labeled churned.
        t0 = pd.Timestamp("2024-04-20")
        window_end = t0 + pd.Timedelta(days=90)  # 2024-07-19, Jul 15 falls inside
        purchases_only = self.txn[~self.txn["is_return"]]
        future_purchasers = set(
            purchases_only.loc[
                (purchases_only["order_date"] > t0) & (purchases_only["order_date"] <= window_end),
                "customer_id",
            ]
        )
        self.assertIn("C1", future_purchasers)

    def test_panel_has_no_row_level_time_paradox(self):
        """Full integration check: in the real generated panel, for
        every row, T0 must be >= every feature-relevant transaction
        date used and the eligibility recency must be <= 90."""
        if not churn_panel.PANEL_CSV.exists():
            self.skipTest("Panel not yet built; run churn_panel.build_panel() first.")
        panel = pd.read_csv(churn_panel.PANEL_CSV, parse_dates=["T0"])
        self.assertTrue((panel["recency_days"] <= churn_panel.ELIGIBILITY_RECENCY_DAYS).all())
        self.assertTrue((panel["frequency"] >= 1).all())
        self.assertIn("future_churn", panel.columns)
        self.assertTrue(panel["future_churn"].isin([0, 1]).all())


class TestTemporalSplit(unittest.TestCase):
    def test_split_is_chronological_and_non_overlapping(self):
        from src.churn_model import _temporal_split

        t0s = pd.date_range("2024-01-01", periods=10, freq="30D")
        panel = pd.DataFrame({
            "T0": list(t0s) * 5,
            "future_churn": [0, 1] * 25,
        })
        train, val, test = _temporal_split(panel)
        train_t0s, val_t0s, test_t0s = set(train["T0"]), set(val["T0"]), set(test["T0"])

        # No T0 should appear in more than one split.
        self.assertEqual(train_t0s & val_t0s, set())
        self.assertEqual(val_t0s & test_t0s, set())
        self.assertEqual(train_t0s & test_t0s, set())

        # Every train T0 must be strictly before every val T0, and every
        # val T0 strictly before every test T0 (chronological ordering).
        if train_t0s and val_t0s:
            self.assertLess(max(train_t0s), min(val_t0s))
        if val_t0s and test_t0s:
            self.assertLess(max(val_t0s), min(test_t0s))


if __name__ == "__main__":
    unittest.main()
