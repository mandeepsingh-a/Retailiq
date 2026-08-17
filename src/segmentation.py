"""
segmentation.py
================
Unsupervised customer segmentation via K-Means on RFM + behavioral
features. Chooses K rigorously (elbow + silhouette score, not a guess),
then labels each cluster with a human-readable business name based on
its centroid characteristics.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

FEATURES_CSV = Path(__file__).resolve().parent.parent / "data" / "processed" / "customer_features.csv"
FIGURES_DIR = Path(__file__).resolve().parent.parent / "reports" / "figures"
SEGMENTED_CSV = Path(__file__).resolve().parent.parent / "data" / "processed" / "customer_segments.csv"

CLUSTER_FEATURES = [
    "recency_days", "frequency", "monetary", "avg_basket_value",
    "purchase_rate", "n_categories", "return_rate", "spend_trend_ratio",
]

K_RANGE = range(2, 9)


def _choose_k(X_scaled: np.ndarray) -> tuple[int, dict]:
    inertias, silhouettes = {}, {}
    for k in K_RANGE:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        inertias[k] = km.inertia_
        silhouettes[k] = silhouette_score(X_scaled, labels)
    best_k = max(silhouettes, key=silhouettes.get)
    return best_k, {"inertias": inertias, "silhouettes": silhouettes}


def _label_segment(row: pd.Series, medians: pd.Series) -> str:
    """Turn a cluster centroid into a human-readable, business-usable name."""
    if row["recency_days"] > medians["recency_days"] * 1.5 and row["monetary"] < medians["monetary"]:
        return "At Risk / Lapsed"
    if row["monetary"] >= medians["monetary"] * 2 and row["frequency"] >= medians["frequency"]:
        return "VIP / Champions"
    if row["frequency"] >= medians["frequency"] and row["recency_days"] <= medians["recency_days"]:
        return "Loyal Regulars"
    if row["frequency"] < medians["frequency"] * 0.5 and row["monetary"] < medians["monetary"] * 0.5:
        return "Low-Engagement / New"
    return "Steady / Average"


def run_segmentation(features_csv: Path = FEATURES_CSV) -> pd.DataFrame:
    df = pd.read_csv(features_csv)
    X = df[CLUSTER_FEATURES].copy()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    best_k, diagnostics = _choose_k(X_scaled)

    km = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    df["cluster"] = km.fit_predict(X_scaled)

    centroids = df.groupby("cluster")[CLUSTER_FEATURES].mean()
    medians = df[CLUSTER_FEATURES].median()
    cluster_names = {c: _label_segment(centroids.loc[c], medians) for c in centroids.index}
    df["segment"] = df["cluster"].map(cluster_names)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # --- Elbow + silhouette diagnostic plot ---
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(list(diagnostics["inertias"].keys()), list(diagnostics["inertias"].values()), marker="o")
    axes[0].set_title("Elbow Method (Inertia vs. K)")
    axes[0].set_xlabel("K"); axes[0].set_ylabel("Inertia")
    axes[1].plot(list(diagnostics["silhouettes"].keys()), list(diagnostics["silhouettes"].values()),
                 marker="o", color="darkorange")
    axes[1].axvline(best_k, color="gray", linestyle="--", label=f"chosen K={best_k}")
    axes[1].set_title("Silhouette Score vs. K")
    axes[1].set_xlabel("K"); axes[1].set_ylabel("Silhouette Score"); axes[1].legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "segmentation_k_selection.png", dpi=130)
    plt.close()

    # --- PCA 2D projection of segments ---
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X_scaled)
    fig, ax = plt.subplots(figsize=(7, 6))
    for seg in df["segment"].unique():
        mask = df["segment"] == seg
        ax.scatter(coords[mask, 0], coords[mask, 1], label=seg, alpha=0.5, s=12)
    ax.set_title(f"Customer Segments (PCA projection, K={best_k})")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} var)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} var)")
    ax.legend(markerscale=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "customer_segments_pca.png", dpi=130)
    plt.close()

    df.to_csv(SEGMENTED_CSV, index=False)

    print(f"Chosen K = {best_k} (silhouette = {diagnostics['silhouettes'][best_k]:.3f})")
    print("\nSegment sizes and profile:")
    profile = df.groupby("segment").agg(
        n_customers=("customer_id", "count"),
        avg_recency=("recency_days", "mean"),
        avg_frequency=("frequency", "mean"),
        avg_monetary=("monetary", "mean"),
        churn_rate=("is_churned", "mean"),
    ).sort_values("avg_monetary", ascending=False)
    print(profile.round(2).to_string())

    return df


if __name__ == "__main__":
    run_segmentation()
