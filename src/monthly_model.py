"""Optional bonus: a monthly course-demand model for next-period forecasting.

Builds a course x month panel (60 courses x 12 months of 2025 = 720 rows),
engineers **lagged** demand features and calendar seasonality (no target
leakage — only information available before month *t* is used), and evaluates
with an **expanding-window, forward-chaining** protocol that mimics real
next-month forecasting. Run with::

    python -m src.monthly_model
"""

from __future__ import annotations

import json

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from . import config
from .data_loader import load_data
from .feature_engineering import build_master_table

# Lag/rolling configuration
LAGS: list[int] = [1, 2, 3]
ROLL_WINDOW: int = 3
FIRST_TEST_MONTH: int = 6  # forecast months 6..12 (expanding window)

STATIC_NUMERIC: list[str] = [
    "CoursePrice",
    "CourseDuration",
    "CourseRating",
    "CourseLevel_ordinal",
    "is_free",
    "inst_rating_wmean",
    "inst_years_mean",
    "expertise_match_score",
]


def build_panel() -> pd.DataFrame:
    """Construct the course x month demand panel with lag & seasonal features."""
    sheets = load_data()
    tx = sheets["Transactions"].copy()
    tx["month"] = tx["TransactionDate"].dt.month

    # Full 60 x 12 grid so months with zero demand are explicit zeros.
    courses = sheets["Courses"]["CourseID"].unique()
    months = range(1, 13)
    grid = pd.MultiIndex.from_product([courses, months],
                                      names=["CourseID", "month"]).to_frame(index=False)

    demand = (
        tx.groupby(["CourseID", "month"])
        .agg(enrollments=("TransactionID", "count"),
             revenue=("Amount", "sum"))
        .reset_index()
    )
    panel = grid.merge(demand, on=["CourseID", "month"], how="left").fillna(
        {"enrollments": 0, "revenue": 0.0}
    )
    panel = panel.sort_values(["CourseID", "month"]).reset_index(drop=True)

    # Lag features (per course) — strictly past information.
    g = panel.groupby("CourseID")["enrollments"]
    for lag in LAGS:
        panel[f"lag{lag}"] = g.shift(lag)
    # Rolling mean of the prior ROLL_WINDOW months (shifted to exclude current).
    panel["roll_mean"] = (
        g.shift(1).rolling(ROLL_WINDOW, min_periods=1).mean()
        .reset_index(level=0, drop=True)
    )

    # Seasonality: cyclical encoding of calendar month (known a priori).
    panel["month_sin"] = np.sin(2 * np.pi * panel["month"] / 12)
    panel["month_cos"] = np.cos(2 * np.pi * panel["month"] / 12)

    # Static course/instructor attributes (known before the month begins).
    master = build_master_table()[["CourseID"] + STATIC_NUMERIC]
    panel = panel.merge(master, on="CourseID", how="left")

    # Drop rows without a full lag history (first max(LAGS) months per course).
    panel = panel.dropna(subset=[f"lag{max(LAGS)}"]).reset_index(drop=True)
    return panel


def feature_columns() -> list[str]:
    return (
        [f"lag{l}" for l in LAGS]
        + ["roll_mean", "month_sin", "month_cos"]
        + STATIC_NUMERIC
    )


def expanding_window_eval(panel: pd.DataFrame, model_factory) -> dict:
    """Forward-chaining evaluation: train on months < m, test on month m."""
    feats = feature_columns()
    preds, actuals = [], []
    per_month = []
    for m in range(FIRST_TEST_MONTH, 13):
        train = panel[panel["month"] < m]
        test = panel[panel["month"] == m]
        if len(test) == 0 or len(train) == 0:
            continue
        model = model_factory()
        model.fit(train[feats], train["enrollments"])
        p = model.predict(test[feats])
        preds.extend(p)
        actuals.extend(test["enrollments"].to_numpy())
        per_month.append(
            {
                "test_month": m,
                "rmse": float(np.sqrt(mean_squared_error(test["enrollments"], p))),
                "mae": float(mean_absolute_error(test["enrollments"], p)),
            }
        )
    preds, actuals = np.array(preds), np.array(actuals)
    return {
        "rmse": float(np.sqrt(mean_squared_error(actuals, preds))),
        "mae": float(mean_absolute_error(actuals, preds)),
        "r2": float(r2_score(actuals, preds)),
        "per_month": per_month,
        "preds": preds,
        "actuals": actuals,
    }


def main() -> None:
    print("=" * 70)
    print("EduPro monthly demand model (bonus, next-period forecasting)")
    print("=" * 70)
    panel = build_panel()
    print(f"Panel shape: {panel.shape}  "
          f"(courses x months with full lag history)")
    print(f"Features ({len(feature_columns())}): {feature_columns()}")

    factories = {
        "Ridge": lambda: Ridge(alpha=1.0, random_state=config.RANDOM_STATE),
        "RandomForest": lambda: RandomForestRegressor(
            n_estimators=300, max_depth=4, min_samples_leaf=5,
            random_state=config.RANDOM_STATE, n_jobs=-1,
        ),
    }

    results = {}
    for name, fac in factories.items():
        res = expanding_window_eval(panel, fac)
        results[name] = res
        print(f"\n  {name}: forward-chaining RMSE={res['rmse']:.3f}  "
              f"MAE={res['mae']:.3f}  R2={res['r2']:.3f}")

    # Baseline: predict next month = previous month (lag1).
    base_mask = panel["month"] >= FIRST_TEST_MONTH
    base_rmse = float(np.sqrt(mean_squared_error(
        panel.loc[base_mask, "enrollments"], panel.loc[base_mask, "lag1"])))
    print(f"\n  Naive (last-month) baseline RMSE={base_rmse:.3f}")

    winner = min(results, key=lambda k: results[k]["rmse"])
    print(f"\n  -> Winner: {winner}")

    # Refit winner on all panel rows and persist.
    feats = feature_columns()
    final = factories[winner]()
    final.fit(panel[feats], panel["enrollments"])
    model_path = config.MODELS_DIR / "monthly_demand_best.joblib"
    joblib.dump({"model": final, "features": feats}, model_path)

    meta = {
        "winner": winner,
        "features": feats,
        "first_test_month": FIRST_TEST_MONTH,
        "panel_rows": int(len(panel)),
        "forward_chaining": {k: {"rmse": v["rmse"], "mae": v["mae"],
                                 "r2": v["r2"]} for k, v in results.items()},
        "naive_baseline_rmse": base_rmse,
        "random_state": config.RANDOM_STATE,
    }
    (config.MODELS_DIR / "monthly_demand_metadata.json").write_text(
        json.dumps(meta, indent=2))

    # Figure: aggregate monthly actual vs predicted (winner).
    res = results[winner]
    months_axis = range(FIRST_TEST_MONTH, 13)
    test_rows = panel[panel["month"] >= FIRST_TEST_MONTH].copy()
    test_rows["pred"] = res["preds"]
    agg = test_rows.groupby("month").agg(actual=("enrollments", "sum"),
                                         pred=("pred", "sum"))
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(agg.index, agg["actual"], marker="o", label="Actual", color="#3b6ea5")
    ax.plot(agg.index, agg["pred"], marker="s", label="Predicted",
            color="#a53b5b", ls="--")
    ax.set_title(f"Monthly demand — forward-chained forecast vs actual ({winner})")
    ax.set_xlabel("Month (2025)")
    ax.set_ylabel("Total enrollments")
    ax.legend()
    fig.tight_layout()
    path = config.FIGURES_DIR / "monthly_forecast.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] {path.name}")
    print(f"  Saved model -> {model_path.name}")


if __name__ == "__main__":
    main()
