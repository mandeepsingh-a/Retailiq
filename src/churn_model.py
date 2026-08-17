"""
churn_model.py
==============
Future-churn prediction, trained on the rolling panel from
churn_panel.py with a genuine TIME-BASED train/validation/test split.

This replaces an earlier version of this module that trained on a
single "current state" snapshot with `y = (recency_days > 90)` -- which
was really a current-status classifier, not a future prediction model.
See README.md "Churn Methodology" section for the full writeup of why
that mattered and what changed.

Split strategy
--------------
Panel rows are grouped by their T0 (observation date). Unique T0s are
sorted chronologically and split ~60/20/20:

    train T0s  (earliest ~60%) -> model fitting + hyperparameter grid
    val T0s    (next ~20%)     -> hyperparameter / model selection
    test T0s   (most recent ~20%) -> ONE-TIME final evaluation

This means the model is always validated and tested on time periods
strictly AFTER the periods it was trained on -- mirroring how the
model would actually be used in production (trained on history, scored
on the present/future). A random row-level split would let the model
"see the future" via other snapshots of the same customer close in
time to a test row, which is exactly the kind of leakage a temporal
split is meant to prevent.

Model selection uses the validation split directly (not k-fold CV)
because k-fold CV on panel rows would shuffle T0s together, re-creating
the same near-future leakage the temporal split exists to avoid.
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    RocCurveDisplay, average_precision_score, classification_report,
    confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.churn_panel import PANEL_CSV, build_panel

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
FIGURES_DIR = Path(__file__).resolve().parent.parent / "reports" / "figures"
METRICS_PATH = Path(__file__).resolve().parent.parent / "reports" / "churn_model_metrics.json"

NUMERIC_FEATURES = [
    "frequency", "monetary", "avg_basket_value", "n_categories",
    "n_line_items", "tenure_days", "purchase_rate", "return_rate",
    "spend_last_30d", "spend_trend_ratio", "n_returns",
]
CATEGORICAL_FEATURES = ["value_tier", "region"]
TARGET = "future_churn"

# recency_days is excluded for the same leakage reason as before: it's
# how ELIGIBILITY at T0 was defined (recency <= 90 to enter the panel
# at all), so it's entangled with the sampling of rows, not a clean
# predictor of the future outcome.

TRAIN_FRAC = 0.60
VAL_FRAC = 0.20  # remainder (~0.20) is TEST


def _build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer([
        ("num", StandardScaler(), NUMERIC_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
    ])


def _temporal_split(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    t0_sorted = sorted(panel["T0"].unique())
    n = len(t0_sorted)
    n_train = max(1, int(n * TRAIN_FRAC))
    n_val = max(1, int(n * VAL_FRAC))
    train_t0s = t0_sorted[:n_train]
    val_t0s = t0_sorted[n_train:n_train + n_val]
    test_t0s = t0_sorted[n_train + n_val:]

    train = panel[panel["T0"].isin(train_t0s)]
    val = panel[panel["T0"].isin(val_t0s)]
    test = panel[panel["T0"].isin(test_t0s)]

    def _d(ts):
        return pd.Timestamp(ts).date()

    print(f"Temporal split: train={len(train_t0s)} T0s ({_d(train_t0s[0])} to {_d(train_t0s[-1])}, "
          f"{len(train):,} rows) | val={len(val_t0s)} T0s ({_d(val_t0s[0])} to {_d(val_t0s[-1])}, "
          f"{len(val):,} rows) | test={len(test_t0s)} T0s ({_d(test_t0s[0])} to {_d(test_t0s[-1])}, "
          f"{len(test):,} rows)")
    return train, val, test


def _evaluate(name: str, model: Pipeline, X, y) -> dict:
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]
    return {
        "model": name,
        "roc_auc": round(roc_auc_score(y, y_proba), 4),
        "pr_auc": round(average_precision_score(y, y_proba), 4),
        "precision": round(precision_score(y, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y, y_pred, zero_division=0), 4),
        "f1": round(f1_score(y, y_pred, zero_division=0), 4),
    }


CANDIDATE_GRIDS = {
    "logistic_regression": (
        lambda: LogisticRegression(max_iter=1000, class_weight="balanced"),
        [{"C": c} for c in [0.01, 0.1, 1, 10]],
    ),
    "random_forest": (
        lambda: RandomForestClassifier(random_state=42, class_weight="balanced"),
        [{"n_estimators": n, "max_depth": d} for n in [200, 400] for d in [6, 10, None]],
    ),
    "gradient_boosting": (
        lambda: GradientBoostingClassifier(random_state=42),
        [{"n_estimators": n, "learning_rate": lr, "max_depth": d}
         for n in [150, 300] for lr in [0.05, 0.1] for d in [2, 3]],
    ),
}


def _fit_with_params(estimator_fn, params: dict, X_train, y_train) -> Pipeline:
    estimator = estimator_fn()
    estimator.set_params(**params)
    pipe = Pipeline([("preprocess", _build_preprocessor()), ("clf", estimator)])
    pipe.fit(X_train, y_train)
    return pipe


def train(panel_csv: Path = PANEL_CSV):
    if not panel_csv.exists():
        print("No churn panel found -- building it first...")
        build_panel()

    panel = pd.read_csv(panel_csv, parse_dates=["T0"])
    train_df, val_df, test_df = _temporal_split(panel)

    feature_cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    X_train, y_train = train_df[feature_cols], train_df[TARGET]
    X_val, y_val = val_df[feature_cols], val_df[TARGET]
    X_test, y_test = test_df[feature_cols], test_df[TARGET]

    # --- Baseline: majority-class dummy classifier, evaluated on test ---
    baseline = DummyClassifier(strategy="most_frequent")
    baseline.fit(X_train, y_train)
    baseline_pred = baseline.predict(X_test)
    baseline_metrics = {
        "model": "baseline_majority_class",
        "roc_auc": 0.5,
        "pr_auc": round(float(y_test.mean()), 4),  # PR-AUC of a random/majority classifier ~= positive rate
        "precision": round(precision_score(y_test, baseline_pred, zero_division=0), 4),
        "recall": round(recall_score(y_test, baseline_pred, zero_division=0), 4),
        "f1": round(f1_score(y_test, baseline_pred, zero_division=0), 4),
    }

    # --- Model selection on VAL (never touch TEST until the very end) ---
    results = [baseline_metrics]
    selected_models = {}

    for name, (estimator_fn, grid) in CANDIDATE_GRIDS.items():
        best_val_auc, best_params, best_model = -1, None, None
        for params in grid:
            model = _fit_with_params(estimator_fn, params, X_train, y_train)
            val_auc = roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])
            if val_auc > best_val_auc:
                best_val_auc, best_params, best_model = val_auc, params, model
        selected_models[name] = (best_model, best_params, best_val_auc)
        print(f"[{name}] selected on VAL: params={best_params} val_roc_auc={best_val_auc:.4f}")

    # --- Refit each selected model on TRAIN+VAL, then evaluate ONCE on TEST ---
    X_trainval = pd.concat([X_train, X_val])
    y_trainval = pd.concat([y_train, y_val])

    fitted_models = {}
    for name, (_, best_params, best_val_auc) in selected_models.items():
        estimator_fn, _ = CANDIDATE_GRIDS[name]
        final_model = _fit_with_params(estimator_fn, best_params, X_trainval, y_trainval)
        fitted_models[name] = final_model

        metrics = _evaluate(name, final_model, X_test, y_test)
        metrics["best_params"] = best_params
        metrics["val_roc_auc"] = round(best_val_auc, 4)
        results.append(metrics)
        print(f"[{name}] TEST roc_auc={metrics['roc_auc']:.4f} pr_auc={metrics['pr_auc']:.4f} "
              f"precision={metrics['precision']:.4f} recall={metrics['recall']:.4f}")

    champion_name = max(
        (r for r in results if r["model"] != "baseline_majority_class"),
        key=lambda r: r["roc_auc"],
    )["model"]
    champion = fitted_models[champion_name]

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(champion, MODELS_DIR / "churn_model.joblib")
    print(f"\nChampion model: {champion_name} (fit on train+val, evaluated once on held-out future T0s)")

    # --- Confusion matrix + classification report for champion, on TEST ---
    y_pred = champion.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.imshow(cm, cmap="Blues")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Stays Active", "Churns"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["Stays Active", "Churns"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix (out-of-time test) — {champion_name}")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "churn_confusion_matrix.png", dpi=130)
    plt.close()

    fig, ax = plt.subplots(figsize=(6, 5))
    for name, model in fitted_models.items():
        RocCurveDisplay.from_estimator(model, X_test, y_test, ax=ax, name=name)
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random")
    ax.set_title("ROC Curves — Out-of-Time Test Set")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "churn_roc_comparison.png", dpi=130)
    plt.close()

    feature_names = (
        NUMERIC_FEATURES +
        list(champion.named_steps["preprocess"].named_transformers_["cat"].get_feature_names_out(CATEGORICAL_FEATURES))
    )
    clf = champion.named_steps["clf"]
    importances = clf.feature_importances_ if hasattr(clf, "feature_importances_") else np.abs(clf.coef_[0])
    imp_df = pd.DataFrame({"feature": feature_names, "importance": importances}).sort_values(
        "importance", ascending=True).tail(12)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(imp_df["feature"], imp_df["importance"], color="#1F3A5F")
    ax.set_title(f"Top Feature Importances — {champion_name} (future-churn model)")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "churn_feature_importance.png", dpi=130)
    plt.close()

    output = {
        "methodology": "time_based_future_churn",
        "prediction_window_days": 90,
        "min_history_days": 180,
        "snapshot_stride_days": 45,
        "n_snapshots": int(panel["T0"].nunique()),
        "champion_model": champion_name,
        "train_rows": len(X_train),
        "val_rows": len(X_val),
        "test_rows": len(X_test),
        "test_period": [str(pd.Timestamp(test_df["T0"].min()).date()), str(pd.Timestamp(test_df["T0"].max()).date())],
        "future_churn_rate_overall": round(float(panel[TARGET].mean()), 4),
        "results": results,
        "champion_classification_report_test": report,
    }
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(json.dumps(output, indent=2))
    print(f"\nMetrics written to {METRICS_PATH}")

    return output


if __name__ == "__main__":
    train()
