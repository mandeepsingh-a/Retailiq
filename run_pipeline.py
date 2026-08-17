"""
run_pipeline.py
================
Runs the full RetailIQ pipeline end to end, in order:

  1. Generate synthetic raw data          -> data/raw/
  2. Clean it (ETL)                       -> data/processed/transactions_clean.csv
  3. Load into SQLite + run SQL analyses  -> data/processed/retailiq.db
  4. Build RFM/behavioral features        -> data/processed/customer_features.csv
  5. Customer segmentation (K-Means)      -> data/processed/customer_segments.csv, reports/figures/
  6. Churn prediction model               -> models/churn_model.joblib, reports/
  7. Revenue forecasting model            -> reports/figures/, reports/forecast_metrics.json

Usage:
    python run_pipeline.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src import data_generation, etl, database, features, segmentation, churn_panel, churn_model, forecasting


def step(name, fn, *args):
    print(f"\n{'#' * 70}\n# STEP: {name}\n{'#' * 70}")
    t0 = time.time()
    result = fn(*args)
    print(f"-> done in {time.time() - t0:.1f}s")
    return result


def main():
    step("1/8 Generate synthetic raw data", data_generation.generate)
    step("2/8 ETL cleaning", etl.clean)

    print(f"\n{'#' * 70}\n# STEP: 3/8 Load SQLite + run SQL analyses\n{'#' * 70}")
    conn = database.build_database()
    sql_path = Path(__file__).resolve().parent / "sql" / "analysis_queries.sql"
    database.run_query_file(conn, sql_path)
    conn.close()
    print("-> done")

    step("4/8 Build customer features (current-state RFM)", features.build_customer_features)
    step("5/8 Customer segmentation", segmentation.run_segmentation)
    step("6/8 Build rolling churn panel (time-based snapshots)", churn_panel.build_panel)
    step("7/8 Future-churn prediction model (temporal train/val/test)", churn_model.train)
    step("8/8 Revenue forecasting model", forecasting.train_and_evaluate)

    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)
    print("Next steps:")
    print("  - View generated charts in reports/figures/")
    print("  - Launch the dashboard:  streamlit run dashboard/app.py")


if __name__ == "__main__":
    main()
