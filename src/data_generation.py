"""
data_generation.py
===================
Generates a realistic synthetic e-commerce retail dataset for RetailIQ.

Why synthetic data?
--------------------
Public retail transaction datasets with real customer-level detail are
either tiny, outdated, or under restrictive licenses. To build a project
with full control over data volume, seasonality, and *known* ground-truth
churn/segment patterns (useful for validating the modeling pipeline),
we generate a synthetic dataset that mimics the statistical properties
of real e-commerce transaction logs:

  - Multiplicative yearly seasonality (Nov/Dec holiday spike)
  - Day-of-week effects (weekend lift)
  - Customer heterogeneity (high/medium/low value segments)
  - Realistic churn dynamics (recency-driven dropout)
  - Missing values, duplicate rows, and outliers injected deliberately
    so the ETL stage has real cleaning work to do.

This script is fully deterministic (seeded) so results are reproducible.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

RNG_SEED = 42
N_CUSTOMERS = 6000
N_PRODUCTS = 250
START_DATE = datetime(2023, 1, 1)
END_DATE = datetime(2025, 12, 31)

CATEGORIES = [
    "Electronics", "Home & Kitchen", "Apparel", "Beauty",
    "Sports & Outdoors", "Books", "Toys", "Grocery",
]

OUTPUT_PATH = Path(__file__).resolve().parent.parent / "data" / "raw" / "transactions_raw.csv"


def _make_customers(rng: np.random.Generator) -> pd.DataFrame:
    """Create a customer base with heterogeneous value tiers.

    Value tier drives purchase frequency and basket size, which is what
    creates realistic, learnable segmentation/churn structure downstream.
    """
    tiers = rng.choice(
        ["high_value", "mid_value", "low_value"],
        size=N_CUSTOMERS,
        p=[0.12, 0.38, 0.50],
    )
    signup_offset_days = rng.integers(0, (END_DATE - START_DATE).days - 30, size=N_CUSTOMERS)
    signup_dates = [START_DATE + timedelta(days=int(d)) for d in signup_offset_days]

    regions = rng.choice(
        ["North", "South", "East", "West", "Central"], size=N_CUSTOMERS
    )

    return pd.DataFrame({
        "customer_id": [f"C{100000+i}" for i in range(N_CUSTOMERS)],
        "value_tier": tiers,
        "signup_date": signup_dates,
        "region": regions,
    })


def _make_products(rng: np.random.Generator) -> pd.DataFrame:
    category = rng.choice(CATEGORIES, size=N_PRODUCTS)
    base_price = np.round(rng.gamma(shape=2.2, scale=18, size=N_PRODUCTS) + 4, 2)
    return pd.DataFrame({
        "product_id": [f"P{2000+i}" for i in range(N_PRODUCTS)],
        "category": category,
        "unit_price": base_price,
    })


def _tier_purchase_rate(tier: str) -> float:
    """Expected purchases per active month, by value tier."""
    return {"high_value": 3.2, "mid_value": 1.3, "low_value": 0.45}[tier]


def _simulate_transactions(customers: pd.DataFrame, products: pd.DataFrame,
                            rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    order_id_counter = 500000

    days_span = (END_DATE - START_DATE).days
    day_index = np.arange(days_span)
    dates = np.array([START_DATE + timedelta(days=int(d)) for d in day_index])

    # Yearly seasonality: holiday bump in Nov/Dec, summer dip in Jul.
    month_of_day = np.array([d.month for d in dates])
    seasonal_multiplier = np.ones(days_span)
    seasonal_multiplier[np.isin(month_of_day, [11, 12])] = 1.9
    seasonal_multiplier[np.isin(month_of_day, [7, 8])] = 0.75

    # Day-of-week effect: weekend lift.
    dow = np.array([d.weekday() for d in dates])
    dow_multiplier = np.where(dow >= 5, 1.35, 1.0)

    daily_multiplier = seasonal_multiplier * dow_multiplier

    for _, cust in customers.iterrows():
        tier = cust["value_tier"]
        base_rate_per_month = _tier_purchase_rate(tier)
        signup_day_idx = (cust["signup_date"] - START_DATE).days

        # Simulate a churn point for a fraction of customers: after this
        # day index, purchase probability collapses toward zero. This is
        # what gives the churn model a real, learnable signal.
        will_churn = rng.random() < (0.42 if tier == "low_value" else 0.22 if tier == "mid_value" else 0.10)
        churn_day_idx = None
        if will_churn:
            churn_day_idx = signup_day_idx + rng.integers(60, max(days_span - signup_day_idx - 1, 61))

        active_days = day_index[day_index >= signup_day_idx]
        for d in active_days:
            if churn_day_idx is not None and d > churn_day_idx:
                # 3% chance of a "win-back" one-off purchase after churn
                if rng.random() > 0.03:
                    continue
            daily_p = (base_rate_per_month / 30.0) * daily_multiplier[d]
            if rng.random() < daily_p:
                n_items = rng.choice([1, 1, 2, 2, 3, 4], p=[0.35, 0.25, 0.18, 0.12, 0.06, 0.04])
                chosen_products = products.sample(n=n_items, replace=True, random_state=rng.integers(0, 1_000_000))
                order_id_counter += 1
                order_date = dates[d]
                for _, prod in chosen_products.iterrows():
                    qty = rng.choice([1, 1, 1, 2, 3], p=[0.55, 0.2, 0.15, 0.06, 0.04])
                    price = prod["unit_price"] * rng.uniform(0.9, 1.15)  # minor price variance/promo
                    rows.append((
                        order_id_counter, cust["customer_id"], order_date.strftime("%Y-%m-%d"),
                        prod["product_id"], prod["category"], qty, round(price, 2),
                        cust["region"],
                    ))

    df = pd.DataFrame(rows, columns=[
        "order_id", "customer_id", "order_date", "product_id",
        "category", "quantity", "unit_price", "region",
    ])
    return df


def _inject_data_quality_issues(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Inject the kind of mess real transactional exports actually have."""
    df = df.copy()

    # 1. Duplicate ~0.8% of rows (double-scanned line items).
    dup_frac = 0.008
    dup_rows = df.sample(frac=dup_frac, random_state=1)
    df = pd.concat([df, dup_rows], ignore_index=True)

    # 2. Null out ~1.5% of region values (optional field at checkout).
    null_idx = df.sample(frac=0.015, random_state=2).index
    df.loc[null_idx, "region"] = np.nan

    # 3. Corrupt ~0.5% of quantities to negative (return/refund logging bug).
    neg_idx = df.sample(frac=0.005, random_state=3).index
    df.loc[neg_idx, "quantity"] = -df.loc[neg_idx, "quantity"]

    # 4. Inject a few extreme outlier prices (data entry errors, e.g. missing decimal).
    outlier_idx = df.sample(frac=0.002, random_state=4).index
    df.loc[outlier_idx, "unit_price"] = df.loc[outlier_idx, "unit_price"] * rng.uniform(50, 100)

    # 5. Inconsistent category casing/whitespace (system migration artifact).
    messy_idx = df.sample(frac=0.01, random_state=5).index
    df.loc[messy_idx, "category"] = df.loc[messy_idx, "category"].str.upper() + "  "

    return df.sample(frac=1.0, random_state=6).reset_index(drop=True)  # shuffle


def generate(output_path: Path = OUTPUT_PATH) -> pd.DataFrame:
    rng = np.random.default_rng(RNG_SEED)
    customers = _make_customers(rng)
    products = _make_products(rng)
    transactions = _simulate_transactions(customers, products, rng)
    transactions = _inject_data_quality_issues(transactions, rng)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    transactions.to_csv(output_path, index=False)

    customers.to_csv(output_path.parent / "customers_raw.csv", index=False)

    print(f"Generated {len(transactions):,} raw transaction line items "
          f"for {N_CUSTOMERS:,} customers -> {output_path}")
    return transactions


if __name__ == "__main__":
    generate()
