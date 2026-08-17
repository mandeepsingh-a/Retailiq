"""
churn_panel.py
==============
Builds a leakage-safe, time-aware training panel for the *future*
churn prediction problem, replacing the earlier "classify current
status" approach.

Why this exists
----------------
The original churn model set `y = (recency_days > 90)` and trained
directly on features computed as of the same observation date. That
target is a description of the customer's CURRENT state, not a
prediction of a FUTURE event -- the model was really answering "can I
tell who is currently inactive from who is currently active?", which
is a much easier and less useful question than "will this currently
active customer go quiet in the next 90 days?"

This module builds the latter, correctly:

  For a rolling set of observation dates T0:
    1. Compute customer features using ONLY transactions with
       order_date <= T0 (see features.compute_customer_features).
    2. Keep only customers who are ELIGIBLE at T0: they have made at
       least one purchase before T0, and are not already churned as of
       T0 (recency_days_at_T0 <= 90). Predicting "future churn" for a
       customer who is already churned, or who has never purchased,
       isn't a meaningful prediction target.
    3. Look FORWARD from T0 to T0 + 90 days (data the model never
       sees as input) and label future_churn = 1 if the customer makes
       zero purchases in that window, else 0.

Multiple T0s (rolling every SNAPSHOT_STRIDE_DAYS) are used rather than
one single snapshot for two reasons: (a) more training rows, and (b)
it enables a genuine time-based train/validation/test split later in
churn_model.py, where the model is tuned and evaluated on T0s it has
never been trained on -- not just random rows held out from a single
snapshot.
"""

from pathlib import Path

import pandas as pd

from src.features import compute_customer_features, BEHAVIORAL_COLS

CLEAN_CSV = Path(__file__).resolve().parent.parent / "data" / "processed" / "transactions_clean.csv"
CUSTOMERS_CSV = Path(__file__).resolve().parent.parent / "data" / "raw" / "customers_raw.csv"
PANEL_CSV = Path(__file__).resolve().parent.parent / "data" / "processed" / "churn_panel.csv"

MIN_HISTORY_DAYS = 180     # require at least 6 months of history before T0 for stable RFM
PREDICTION_WINDOW_DAYS = 90  # look this far forward from T0 to determine future_churn
SNAPSHOT_STRIDE_DAYS = 45   # spacing between rolling T0 snapshots
ELIGIBILITY_RECENCY_DAYS = 90  # must be "currently active" at T0 to be a valid prediction target


def _eligible_t0_range(txn: pd.DataFrame) -> tuple[pd.Timestamp, pd.Timestamp]:
    min_date = txn["order_date"].min()
    max_date = txn["order_date"].max()
    earliest_t0 = min_date + pd.Timedelta(days=MIN_HISTORY_DAYS)
    latest_t0 = max_date - pd.Timedelta(days=PREDICTION_WINDOW_DAYS)
    return earliest_t0, latest_t0


def build_panel(clean_csv: Path = CLEAN_CSV, customers_csv: Path = CUSTOMERS_CSV) -> pd.DataFrame:
    txn = pd.read_csv(clean_csv, parse_dates=["order_date"])
    customers = pd.read_csv(customers_csv, parse_dates=["signup_date"])
    purchases_only = txn[~txn["is_return"]]

    earliest_t0, latest_t0 = _eligible_t0_range(txn)
    if earliest_t0 > latest_t0:
        raise ValueError(
            f"No valid T0 range: earliest possible T0 ({earliest_t0.date()}) is after the "
            f"latest possible T0 ({latest_t0.date()}). Need more history or a smaller "
            f"MIN_HISTORY_DAYS / PREDICTION_WINDOW_DAYS."
        )

    t0_list = pd.date_range(earliest_t0, latest_t0, freq=f"{SNAPSHOT_STRIDE_DAYS}D")
    print(f"Building rolling churn panel: {len(t0_list)} snapshots from "
          f"{t0_list[0].date()} to {t0_list[-1].date()} (stride={SNAPSHOT_STRIDE_DAYS}d)")

    snapshots = []
    for t0 in t0_list:
        feats = compute_customer_features(txn, customers, t0)

        # Eligibility: at least one purchase before T0, and active (not
        # already churned) as of T0.
        eligible = feats[(feats["frequency"] >= 1) & (feats["recency_days"] <= ELIGIBILITY_RECENCY_DAYS)].copy()
        if eligible.empty:
            continue

        # Forward-looking label: did they purchase again within the
        # next PREDICTION_WINDOW_DAYS? This is the ONLY place future
        # data (order_date > t0) is used, and it only touches the
        # label, never a feature column.
        window_end = t0 + pd.Timedelta(days=PREDICTION_WINDOW_DAYS)
        future_purchasers = set(
            purchases_only.loc[
                (purchases_only["order_date"] > t0) & (purchases_only["order_date"] <= window_end),
                "customer_id",
            ]
        )
        eligible["future_churn"] = (~eligible["customer_id"].isin(future_purchasers)).astype(int)
        eligible["T0"] = t0

        snapshots.append(eligible)

    panel = pd.concat(snapshots, ignore_index=True)
    panel = panel.sort_values(["T0", "customer_id"]).reset_index(drop=True)

    PANEL_CSV.parent.mkdir(parents=True, exist_ok=True)
    panel.to_csv(PANEL_CSV, index=False)

    print(f"Panel built: {len(panel):,} (customer, T0) rows across {panel['T0'].nunique()} snapshots, "
          f"{panel['customer_id'].nunique():,} unique customers -> {PANEL_CSV}")
    print(f"Overall future_churn rate: {panel['future_churn'].mean():.1%}")
    return panel


if __name__ == "__main__":
    build_panel()
