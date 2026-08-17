"""
RetailIQ Dashboard
===================
Interactive Streamlit app for exploring the RetailIQ analytics outputs:
KPIs, customer segments, churn risk list, and the revenue forecast.

Run with:
    streamlit run dashboard/app.py

Requires the pipeline to have been run first (`python run_pipeline.py`)
so that data/processed/customer_segments.csv, models/churn_model.joblib,
and reports/*.json exist.
"""

import json
from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
SEGMENTS_CSV = ROOT / "data" / "processed" / "customer_segments.csv"
CHURN_MODEL_PATH = ROOT / "models" / "churn_model.joblib"
CHURN_METRICS_PATH = ROOT / "reports" / "churn_model_metrics.json"
FORECAST_METRICS_PATH = ROOT / "reports" / "forecast_metrics.json"
FORECAST_FIG = ROOT / "reports" / "figures" / "revenue_forecast.png"

st.set_page_config(page_title="RetailIQ Dashboard", layout="wide", page_icon="📊")


@st.cache_data
def load_segments() -> pd.DataFrame:
    return pd.read_csv(SEGMENTS_CSV)


@st.cache_resource
def load_churn_model():
    return joblib.load(CHURN_MODEL_PATH)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


def main():
    st.title("📊 RetailIQ — Retail Analytics & Forecasting Platform")
    st.caption(
        "End-to-end customer segmentation, churn prediction, and revenue "
        "forecasting built on a synthetic e-commerce transaction dataset."
    )

    if not SEGMENTS_CSV.exists():
        st.error(
            "No processed data found. Run `python run_pipeline.py` from the "
            "project root first to generate data, train models, and populate "
            "reports/ before launching the dashboard."
        )
        st.stop()

    df = load_segments()
    churn_metrics = load_json(CHURN_METRICS_PATH)
    forecast_metrics = load_json(FORECAST_METRICS_PATH)

    # ---------------- KPI row ----------------
    total_revenue = df["monetary"].sum()
    active_customers = (df["is_churned"] == 0).sum()
    churn_rate = df["is_churned"].mean()
    champion = churn_metrics.get("champion_model", "—")
    champion_auc = next(
        (r["roc_auc"] for r in churn_metrics.get("results", []) if r["model"] == champion), None
    )

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Customer Revenue", f"${total_revenue:,.0f}")
    k2.metric("Active Customers", f"{active_customers:,}", delta=f"-{churn_rate:.1%} churned", delta_color="inverse")
    k3.metric("Future-Churn Model (ROC-AUC)", f"{champion_auc:.3f}" if champion_auc else "—",
              help=f"Champion: {champion}. Evaluated out-of-time on the most recent held-out snapshots — "
                   f"see the Churn Risk tab for methodology.")
    if forecast_metrics:
        improvement = forecast_metrics.get("improvement_vs_baseline_pct")
        k4.metric("Forecast vs. Naive Baseline", f"{improvement:+.1f}%" if improvement is not None else "—")

    st.divider()

    tab1, tab2, tab3 = st.tabs(["🧩 Customer Segments", "⚠️ Future Churn Risk", "📈 Revenue Forecast"])

    # ---------------- Tab 1: Segmentation ----------------
    with tab1:
        left, right = st.columns([2, 1])
        with left:
            fig = px.scatter(
                df, x="monetary", y="frequency", color="segment",
                size="recency_days", hover_data=["customer_id", "value_tier", "region"],
                title="Customers by Monetary Value vs. Purchase Frequency",
                labels={"monetary": "Total Spend ($)", "frequency": "Number of Orders"},
            )
            st.plotly_chart(fig, use_container_width=True)
        with right:
            seg_counts = df["segment"].value_counts().reset_index()
            seg_counts.columns = ["segment", "count"]
            fig2 = px.pie(seg_counts, names="segment", values="count", title="Segment Mix", hole=0.4)
            st.plotly_chart(fig2, use_container_width=True)

        st.subheader("Segment Profile")
        profile = df.groupby("segment").agg(
            customers=("customer_id", "count"),
            avg_recency_days=("recency_days", "mean"),
            avg_frequency=("frequency", "mean"),
            avg_monetary=("monetary", "mean"),
            churn_rate=("is_churned", "mean"),
        ).round(2).sort_values("avg_monetary", ascending=False)
        st.dataframe(profile, use_container_width=True)

    # ---------------- Tab 2: Churn risk ----------------
    with tab2:
        st.subheader("Future 90-Day Churn Risk")
        st.caption(
            "The model predicts whether a customer, given only what was known about them at "
            "an observation date, goes quiet for the following 90 days — not whether they are "
            "already inactive today. It's evaluated on customer snapshots from time periods it "
            "was never trained on (an out-of-time test set), which is what makes the ROC-AUC "
            "below a realistic estimate of live performance rather than an inflated backward-"
            "looking number."
        )
        if churn_metrics.get("test_period"):
            st.caption(f"Out-of-time test period: {churn_metrics['test_period'][0]} to {churn_metrics['test_period'][1]}")

        st.subheader("Highest Churn-Risk Active Customers (current RFM ranking)")
        st.caption(
            "Ranked by RFM score (lower = higher risk) among customers not yet flagged "
            "churned today — the actionable retention target list."
        )
        at_risk = (
            df[df["is_churned"] == 0]
            .sort_values("RFM_score")
            .loc[:, ["customer_id", "segment", "value_tier", "region", "recency_days",
                     "frequency", "monetary", "RFM_score"]]
            .head(25)
        )
        st.dataframe(at_risk, use_container_width=True)

        if champion_auc:
            st.subheader("Future-Churn Model Performance (out-of-time test)")
            results_df = pd.DataFrame(churn_metrics.get("results", []))
            st.dataframe(results_df, use_container_width=True)

    # ---------------- Tab 3: Forecast ----------------
    with tab3:
        st.subheader("Daily Revenue Forecast (60-day holdout)")
        if FORECAST_FIG.exists():
            st.image(str(FORECAST_FIG), use_container_width=True)
        if forecast_metrics:
            c1, c2, c3 = st.columns(3)
            c1.metric("Model MAE", f"${forecast_metrics['model']['mae']:,.0f}")
            c2.metric("Seasonal-Naive MAE", f"${forecast_metrics['seasonal_naive_baseline']['mae']:,.0f}")
            c3.metric("Improvement", f"{forecast_metrics['improvement_vs_baseline_pct']:+.1f}%")


if __name__ == "__main__":
    main()
