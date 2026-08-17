"""Unit tests for the ETL pipeline. Uses small in-memory fixtures rather
than the full generated dataset so tests run in milliseconds and pin
down exact expected behavior."""

import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import etl


class TestETLCleaning(unittest.TestCase):
    def setUp(self):
        self.tmp_raw = Path(__file__).parent / "_tmp_raw.csv"
        self.tmp_out = Path(__file__).parent / "_tmp_clean.csv"
        df = pd.DataFrame({
            "order_id": [1, 1, 2, 3, 4],
            "customer_id": ["C1", "C1", "C1", "C2", "C2"],
            "order_date": ["2024-01-01", "2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"],
            "product_id": ["P1", "P1", "P2", "P3", "P1"],
            "category": ["Books", "Books", " ELECTRONICS ", "Toys", "Books"],
            "quantity": [1, 1, -2, 3, 1],
            "unit_price": [10.0, 10.0, 25.0, 8.0, 99999.0],  # last one is an outlier
            "region": ["North", "North", None, "South", "South"],
        })
        df.to_csv(self.tmp_raw, index=False)

    def tearDown(self):
        for p in (self.tmp_raw, self.tmp_out):
            if p.exists():
                p.unlink()

    def test_duplicates_are_removed(self):
        cleaned, report = etl.clean(self.tmp_raw, self.tmp_out)
        # rows 0 and 1 are exact duplicates -> exactly one should remain
        self.assertEqual(report.duplicates_removed, 1)

    def test_negative_quantity_is_flagged_not_dropped(self):
        cleaned, report = etl.clean(self.tmp_raw, self.tmp_out)
        self.assertEqual(report.negative_quantity_rows_fixed, 1)
        self.assertTrue((cleaned["quantity"] >= 0).all())
        self.assertTrue(cleaned["is_return"].any())

    def test_category_is_standardized(self):
        cleaned, _ = etl.clean(self.tmp_raw, self.tmp_out)
        self.assertIn("Electronics", cleaned["category"].values)
        self.assertNotIn(" ELECTRONICS ", cleaned["category"].values)

    def test_null_region_filled(self):
        cleaned, report = etl.clean(self.tmp_raw, self.tmp_out)
        self.assertEqual(cleaned["region"].isna().sum(), 0)
        self.assertIn("Unknown", cleaned["region"].values)
        self.assertEqual(report.nulls_filled["region"], 1)

    def test_no_rows_silently_lost_beyond_duplicates(self):
        cleaned, report = etl.clean(self.tmp_raw, self.tmp_out)
        # 5 rows in, 1 exact duplicate removed -> 4 rows out
        self.assertEqual(report.rows_in, 5)
        self.assertEqual(report.rows_out, 4)


if __name__ == "__main__":
    unittest.main()
