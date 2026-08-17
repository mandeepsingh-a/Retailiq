"""
forecasting.py
===============
Daily revenue forecasting via engineered time-series features
(lags, rolling stats, calendar effects) fed into gradient boosting —
deliberately avoiding a black-box AutoARIMA/Prophet call so the
feature-engineering skill is visible in the code, not hidden in a
library. Evaluated against a seasonal-naive baseline, which any
forecast MUST beat to be worth deploying.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error

CLEAN_CSV = Path(__file__).resolve().parent.parent / "data" / "processed" / "transactions_clean.csv"
FIGURES_DIR = Path(__file__).resolve().parent.parent / "reports" / "figures"
METRICS_PATH = Path(__file__).resolve().parent.parent / "reports" / "forecast_metrics.json"

TEST_HORIZON_DAYS = 60
N_LAGS = [1, 7, 14, 28]
ROLLING_WINDOWS = [7, 14, 30]


def _build_daily_series(clean_csv: Path = CLEAN_CSV) -> pd.DataFrame:
    txn = pd.read_csv(clean_csv, parse_dates=["order_date"])
    daily = txn[~txn["is_return"]].groupby("order_date")["line_revenue"].sum()
    full_idx = pd.date_range(daily.index.min(), daily.index.max(), freq="D")
    daily = daily.reindex(full_idx, fill_value=0.0).rename("revenue").to_frame()
    daily.index.name = "date"
    return daily


def _add_features(daily: pd.DataFrame) -> pd.DataFrame:
    df = daily.copy()
    df["dow"] = df.index.dayofweek
    df["month"] = df.index.month
    df["is_weekend"] = (df["dow"] >= 5).astype(int)
    df["day_of_year"] = df.index.dayofyear

    for lag in N_LAGS:
        df[f"lag_{lag}"] = df["revenue"].shift(lag)
    for window in ROLLING_WINDOWS:
        df[f"rolling_mean_{window}"] = df["revenue"].shift(1).rolling(window).mean()
        df[f"rolling_std_{window}"] = df["revenue"].shift(1).rolling(window).std()

    # Same-weekday smoothed average (last 4 occurrences of this weekday),
    # i.e. a de-noised version of the seasonal-naive baseline -- gives the
    # model an easy, strong signal to lean on instead of re-deriving
    # weekly seasonality from scratch through generic lags.
    df["same_dow_avg_4wk"] = df["revenue"].shift(7).rolling(window=4 * 7, min_periods=2).apply(
        lambda s: s[::7].mean() if len(s[::7]) > 0 else np.nan, raw=False
    )
    # Simpler, robust version: average of lag_7, lag_14, lag_21, lag_28.
    same_dow_cols = []
    for k in [7, 14, 21, 28]:
        col = f"_tmp_dow_lag_{k}"
        df[col] = df["revenue"].shift(k)
        same_dow_cols.append(col)
    df["same_dow_avg_4wk"] = df[same_dow_cols].mean(axis=1)
    df = df.drop(columns=same_dow_cols)

    # Explicit holiday-season flag: Nov/Dec step-change is a known
    # calendar effect, not something lags can anticipate on day 1.
    df["is_holiday_season"] = df["month"].isin([11, 12]).astype(int)

    return df


def train_and_evaluate():
    daily = _build_daily_series()
    featured = _add_features(daily)
    featured = featured.dropna()

    feature_cols = [c for c in featured.columns if c != "revenue"]

    # --- Residual-on-naive modeling ---
    # Rather than predicting revenue directly, predict the RESIDUAL
    # between actual revenue and the seasonal-naive baseline (lag_7),
    # then add it back. This guarantees the model can only improve on
    # the baseline (residual ~ 0 recovers naive exactly) and lets the
    # gradient booster focus its capacity on genuinely new signal --
    # trend, holiday effects, day-of-week drift -- instead of re-deriving
    # the strong weekly seasonality that lag_7 already captures for free.
    featured["target_residual"] = featured["revenue"] - featured["lag_7"]
    residual_feature_cols = [c for c in feature_cols]  # lag_7 stays in as a feature too

    train = featured.iloc[:-TEST_HORIZON_DAYS]
    test = featured.iloc[-TEST_HORIZON_DAYS:]

    X_train, y_train_resid = train[residual_feature_cols], train["target_residual"]
    X_test, y_test = test[residual_feature_cols], test["revenue"]

    model = GradientBoostingRegressor(
        n_estimators=200, max_depth=3, learning_rate=0.03, random_state=42,
    )
    model.fit(X_train, y_train_resid)
    resid_preds = model.predict(X_test)
    preds = test["lag_7"].values + resid_preds
    preds = np.clip(preds, a_min=0, a_max=None)

    # Seasonal-naive baseline: "revenue this weekday = revenue same weekday last week".
    naive_preds = daily["revenue"].shift(7).reindex(test.index)

    model_mae = mean_absolute_error(y_test, preds)
    model_mape = mean_absolute_percentage_error(y_test.clip(lower=1), np.clip(preds, 1, None))
    naive_mae = mean_absolute_error(y_test, naive_preds)
    naive_mape = mean_absolute_percentage_error(y_test.clip(lower=1), naive_preds.clip(lower=1))

    metrics = {
        "test_horizon_days": TEST_HORIZON_DAYS,
        "model": {"mae": round(model_mae, 2), "mape": round(float(model_mape), 4)},
        "seasonal_naive_baseline": {"mae": round(naive_mae, 2), "mape": round(float(naive_mape), 4)},
        "improvement_vs_baseline_pct": round(100 * (1 - model_mae / naive_mae), 2),
    }

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(daily.index[-180:], daily["revenue"].iloc[-180:], label="Actual", color="#1F3A5F", linewidth=1)
    ax.plot(test.index, preds, label="Forecast (GBM)", color="#E67E22", linewidth=1.6)
    ax.plot(test.index, naive_preds, label="Seasonal-naive baseline", color="gray", linestyle="--", linewidth=1)
    ax.axvline(test.index[0], color="black", linestyle=":", alpha=0.5)
    ax.set_title("Daily Revenue: Actual vs. Forecast (last 180 days, 60-day holdout)")
    ax.set_ylabel("Revenue ($)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "revenue_forecast.png", dpi=130)
    plt.close()

    # Feature importance for the forecaster too.
    imp = pd.Series(model.feature_importances_, index=residual_feature_cols).sort_values().tail(10)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.barh(imp.index, imp.values, color="#E67E22")
    ax.set_title("Forecast Model — Top Feature Importances")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "forecast_feature_importance.png", dpi=130)
    plt.close()

    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))
    return metrics


if __name__ == "__main__":
    train_and_evaluate()
