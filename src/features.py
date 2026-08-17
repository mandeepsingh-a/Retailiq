"""
features.py
===========
Feature engineering for segmentation and churn modeling.

Builds an RFM (Recency, Frequency, Monetary) table plus behavioral
features (tenure, category diversity, return rate, trend) at the
customer grain.

The core logic (`compute_customer_features`) is parameterized by an
`obs_date`, so it can be reused two ways:

  1. `build_customer_features()` -- obs_date = the dataset's max date.
     Used for segmentation and the "current state" view in the
     dashboard. This is descriptive analytics: "what does my customer
     base look like right now?"

  2. `churn_panel.py` -- obs_date = each rolling T0 in the past.
     Used to build leakage-safe training snapshots for the *predictive*
     future-churn model: "what did this customer look like before T0,
     and did they churn in the 90 days after?"

Only data with order_date <= obs_date is ever used inside
compute_customer_features -- this is the leakage boundary the temporal
churn model depends on, and it's covered by tests/test_temporal_leakage.py.
"""

from pathlib import Path

import numpy as np
import pandas as pd

CLEAN_CSV = Path(__file__).resolve().parent.parent / "data" / "processed" / "transactions_clean.csv"
CUSTOMERS_CSV = Path(__file__).resolve().parent.parent / "data" / "raw" / "customers_raw.csv"
FEATURES_CSV = Path(__file__).resolve().parent.parent / "data" / "processed" / "customer_features.csv"

# A customer is labeled "churned" if their most recent purchase is more
# than CHURN_WINDOW_DAYS before the observation date. This mirrors how
# churn is typically defined in subscription-less retail (no explicit
# cancellation event, so recency is the ground truth proxy).
CHURN_WINDOW_DAYS = 90

BEHAVIORAL_COLS = [
    "avg_basket_value", "n_categories", "n_line_items", "tenure_days",
    "purchase_rate", "return_rate", "spend_last_30d", "spend_trend_ratio", "n_returns",
]


def compute_customer_features(txn: pd.DataFrame, customers: pd.DataFrame, obs_date: pd.Timestamp) -> pd.DataFrame:
    """Compute RFM + behavioral features for every customer, as of obs_date.

    CRITICAL: only rows with order_date <= obs_date are used anywhere in
    this function. Callers building a historical snapshot (T0 in the
    past) must pre-filter `txn` or rely on this internal filter -- both
    are done here defensively so the function is safe to call with a
    full, un-filtered transaction log.
    """
    txn = txn[txn["order_date"] <= obs_date]
    purchases = txn[~txn["is_return"]]

    if len(purchases) == 0:
        # Degenerate case (obs_date before any purchases): return an
        # all-zero feature frame rather than crashing.
        base = customers.copy()
        base["recency_days"] = (obs_date - base["signup_date"]).dt.days
        for col in ["frequency", "monetary"] + BEHAVIORAL_COLS:
            base[col] = 0
        return base

    agg = purchases.groupby("customer_id").agg(
        last_purchase=("order_date", "max"),
        first_purchase=("order_date", "min"),
        frequency=("order_id", "nunique"),
        monetary=("line_revenue", "sum"),
        avg_basket_value=("line_revenue", lambda s: s.sum() / purchases.loc[s.index, "order_id"].nunique()),
        n_categories=("category", "nunique"),
        n_line_items=("line_revenue", "count"),
    ).reset_index()

    agg["recency_days"] = (obs_date - agg["last_purchase"]).dt.days
    agg["tenure_days"] = (agg["last_purchase"] - agg["first_purchase"]).dt.days
    agg["purchase_span_days"] = (obs_date - agg["first_purchase"]).dt.days
    agg["purchase_rate"] = agg["frequency"] / agg["purchase_span_days"].clip(lower=1) * 30  # orders/month

    returns = txn[txn["is_return"]].groupby("customer_id").size().rename("n_returns")
    agg = agg.merge(returns, on="customer_id", how="left")
    agg["n_returns"] = agg["n_returns"].fillna(0)
    agg["return_rate"] = agg["n_returns"] / agg["frequency"].clip(lower=1)

    # Recent vs. historical spend trend: ratio of last-30-day spend
    # (relative to obs_date) to prior monthly average.
    last_30 = purchases[purchases["order_date"] > obs_date - pd.Timedelta(days=30)]
    recent_spend = last_30.groupby("customer_id")["line_revenue"].sum().rename("spend_last_30d")
    agg = agg.merge(recent_spend, on="customer_id", how="left")
    agg["spend_last_30d"] = agg["spend_last_30d"].fillna(0)
    monthly_avg = (agg["monetary"] / (agg["purchase_span_days"].clip(lower=30) / 30))
    agg["spend_trend_ratio"] = agg["spend_last_30d"] / monthly_avg.replace(0, np.nan)
    agg["spend_trend_ratio"] = agg["spend_trend_ratio"].fillna(0)

    # Only keep customers who had signed up by obs_date.
    eligible_customers = customers[customers["signup_date"] <= obs_date]
    features = eligible_customers.merge(agg, on="customer_id", how="left")

    features["frequency"] = features["frequency"].fillna(0)
    features["monetary"] = features["monetary"].fillna(0)
    features["recency_days"] = features["recency_days"].fillna(
        (obs_date - features["signup_date"]).dt.days
    )
    for col in BEHAVIORAL_COLS:
        features[col] = features[col].fillna(0)

    return features


def build_customer_features(clean_csv: Path = CLEAN_CSV, customers_csv: Path = CUSTOMERS_CSV) -> pd.DataFrame:
    """'Current state' feature table (obs_date = latest date in the
    data). Used for segmentation and the dashboard's live customer
    view -- NOT used to train the future-churn model (see churn_panel.py)."""
    txn = pd.read_csv(clean_csv, parse_dates=["order_date"])
    customers = pd.read_csv(customers_csv, parse_dates=["signup_date"])
    obs_date = txn["order_date"].max()

    features = compute_customer_features(txn, customers, obs_date)
    features["is_churned"] = (features["recency_days"] > CHURN_WINDOW_DAYS).astype(int)

    # RFM quintile scores (1=worst, 5=best) -- classic marketing-analytics construct.
    features["R_score"] = pd.qcut(features["recency_days"].rank(method="first"), 5, labels=[5, 4, 3, 2, 1]).astype(int)
    features["F_score"] = pd.qcut(features["frequency"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(int)
    features["M_score"] = pd.qcut(features["monetary"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(int)
    features["RFM_score"] = features["R_score"] + features["F_score"] + features["M_score"]

    FEATURES_CSV.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(FEATURES_CSV, index=False)
    print(f"Built customer feature table (current state, as of {obs_date.date()}): "
          f"{features.shape[0]:,} customers x {features.shape[1]} columns -> {FEATURES_CSV}")
    print(f"Currently churned (recency > {CHURN_WINDOW_DAYS}d): {features['is_churned'].mean():.1%}")
    return features


if __name__ == "__main__":
    build_customer_features()
