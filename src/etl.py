"""
etl.py
======
Cleaning and transformation pipeline for RetailIQ.

Takes the messy raw transaction export and produces an analysis-ready
table. Every cleaning decision is logged so the pipeline is auditable —
a requirement in any real data team (you must be able to explain exactly
what you changed and why).
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("retailiq.etl")

RAW_PATH = Path(__file__).resolve().parent.parent / "data" / "raw" / "transactions_raw.csv"
PROCESSED_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "transactions_clean.csv"


@dataclass
class CleaningReport:
    rows_in: int = 0
    rows_out: int = 0
    duplicates_removed: int = 0
    negative_quantity_rows_fixed: int = 0
    nulls_filled: dict = field(default_factory=dict)
    outliers_capped: int = 0

    def summary(self) -> str:
        pct_removed = 100 * (1 - self.rows_out / self.rows_in) if self.rows_in else 0
        lines = [
            "=" * 60,
            "ETL CLEANING REPORT",
            "=" * 60,
            f"Rows in:                  {self.rows_in:,}",
            f"Rows out:                 {self.rows_out:,}  ({pct_removed:.2f}% removed)",
            f"Exact duplicates removed: {self.duplicates_removed:,}",
            f"Negative-quantity fixed:  {self.negative_quantity_rows_fixed:,} (treated as returns, flagged)",
            f"Nulls filled:             {self.nulls_filled}",
            f"Outlier prices capped:    {self.outliers_capped:,} (IQR method)",
            "=" * 60,
        ]
        return "\n".join(lines)


def _standardize_categories(df: pd.DataFrame) -> pd.DataFrame:
    df["category"] = df["category"].str.strip().str.title()
    return df


def _cap_outliers_iqr(df: pd.DataFrame, col: str, report: CleaningReport) -> pd.DataFrame:
    q1, q3 = df[col].quantile([0.25, 0.75])
    iqr = q3 - q1
    upper = q3 + 3 * iqr  # wide (3x) multiplier: only kill genuine data-entry errors, not premium items
    lower = max(q1 - 3 * iqr, 0)
    mask = (df[col] > upper) | (df[col] < lower)
    report.outliers_capped = int(mask.sum())
    df.loc[df[col] > upper, col] = upper
    df.loc[df[col] < lower, col] = lower
    return df


def clean(raw_path: Path = RAW_PATH, processed_path: Path = PROCESSED_PATH) -> tuple[pd.DataFrame, CleaningReport]:
    report = CleaningReport()
    df = pd.read_csv(raw_path, parse_dates=["order_date"])
    report.rows_in = len(df)

    # 1. Drop exact duplicate line items.
    before = len(df)
    df = df.drop_duplicates()
    report.duplicates_removed = before - len(df)

    # 2. Returns logged as negative quantity: keep them, but flag explicitly
    #    rather than silently dropping revenue-relevant rows.
    df["is_return"] = df["quantity"] < 0
    report.negative_quantity_rows_fixed = int(df["is_return"].sum())
    df["quantity"] = df["quantity"].abs()

    # 3. Standardize category text.
    df = _standardize_categories(df)

    # 4. Fill missing region with "Unknown" rather than dropping ~1.5% of
    #    otherwise-valid revenue rows.
    n_null_region = int(df["region"].isna().sum())
    df["region"] = df["region"].fillna("Unknown")
    report.nulls_filled["region"] = n_null_region

    # 5. Cap extreme price outliers (data-entry errors) using IQR, rather
    #    than dropping — preserves order/customer continuity for the
    #    downstream RFM and churn features.
    df = _cap_outliers_iqr(df, "unit_price", report)

    # 6. Derived columns used throughout the rest of the pipeline.
    df["line_revenue"] = np.where(df["is_return"], -1, 1) * df["quantity"] * df["unit_price"]
    df["order_date"] = pd.to_datetime(df["order_date"])
    df["order_year_month"] = df["order_date"].dt.to_period("M").astype(str)

    df = df.reset_index(drop=True)
    report.rows_out = len(df)

    processed_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(processed_path, index=False)
    logger.info("Wrote cleaned dataset to %s", processed_path)
    logger.info("\n%s", report.summary())

    return df, report


if __name__ == "__main__":
    clean()
