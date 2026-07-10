"""EduPro demand & revenue forecasting dashboard.

A four-tab Streamlit app wired to the CV-selected models in ``models/``:

  1. Demand Prediction      — what-if predictions for a hypothetical course
  2. Revenue Forecast       — revenue visuals incl. the monthly trend
  3. Feature Importance     — driver explorer per target
  4. Category Comparison    — enrollment vs revenue and category roll-up

Launch with::

    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import os

# Limit native thread pools BEFORE numpy/scipy/sklearn import. On Streamlit
# Community Cloud's constrained container, OpenBLAS/OpenMP spawning many threads
# can segfault the process at startup. Single-threaded is plenty for n=60.
for _var in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(_var, "1")

import json
import sys
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")  # headless backend; avoids any GUI-backend crash on Cloud
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

# Make the project importable when run via `streamlit run app/streamlit_app.py`.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import config  # noqa: E402
from src.data_loader import load_data  # noqa: E402
from src.feature_engineering import (  # noqa: E402
    NOMINAL_FEATURES,
    NUMERIC_FEATURES,
    build_master_table,
)

st.set_page_config(page_title="EduPro Demand & Revenue Forecasting",
                   page_icon="🎓", layout="wide")


# --------------------------------------------------------------------------- #
# Cached loaders
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner=False)
def load_models() -> dict:
    """Load the trained course-level models + metadata once per session."""
    out = {}
    for target in config.TARGETS:
        model = joblib.load(config.MODELS_DIR / f"{target}_best.joblib")
        meta = json.loads(
            (config.MODELS_DIR / f"{target}_metadata.json").read_text()
        )
        out[target] = {"model": model, "meta": meta}
    return out


@st.cache_data(show_spinner=False)
def load_master() -> pd.DataFrame:
    """Build (and cache) the 60-row course master table."""
    return build_master_table()


@st.cache_data(show_spinner=False)
def load_transactions() -> pd.DataFrame:
    """Load the raw transactions sheet (for monthly visuals)."""
    return load_data()["Transactions"]


@st.cache_data(show_spinner=False)
def feature_ranges() -> dict[str, dict[str, float]]:
    """Min/median/max of each numeric feature — drives inputs & extrapolation."""
    m = load_master()
    return {
        c: {
            "min": float(m[c].min()),
            "med": float(m[c].median()),
            "mean": float(m[c].mean()),
            "max": float(m[c].max()),
        }
        for c in NUMERIC_FEATURES
    }


@st.cache_data(show_spinner=False)
def category_defaults() -> pd.DataFrame:
    """Per-category averages used to default the non-exposed inputs."""
    m = load_master()
    return m.groupby("CourseCategory")[NUMERIC_FEATURES].mean()


# --------------------------------------------------------------------------- #
# Prediction helper
# --------------------------------------------------------------------------- #
def build_input_row(values: dict) -> pd.DataFrame:
    """Assemble a single-row frame with every column the pipelines expect."""
    row = {c: values[c] for c in NUMERIC_FEATURES}
    row["CourseCategory"] = values["CourseCategory"]
    return pd.DataFrame([row])[NUMERIC_FEATURES + NOMINAL_FEATURES]


def extrapolation_flags(values: dict, ranges: dict, exposed: list[str]) -> list[str]:
    """Return human-readable warnings for inputs outside the training range."""
    warns = []
    pretty = {
        "CoursePrice": "Price",
        "CourseDuration": "Duration",
        "CourseRating": "Course rating",
        "inst_rating_wmean": "Instructor rating",
        "inst_years_mean": "Instructor experience",
        "expertise_match_score": "Category-match share",
    }
    for f in exposed:
        r = ranges[f]
        v = values[f]
        if v < r["min"] or v > r["max"]:
            warns.append(
                f"**{pretty.get(f, f)}** = {v:g} is outside the training range "
                f"[{r['min']:g}, {r['max']:g}] — prediction is an extrapolation."
            )
    return warns


# --------------------------------------------------------------------------- #
# Header
# --------------------------------------------------------------------------- #
st.title("🎓 EduPro — Course Demand & Revenue Forecasting")
st.caption(
    "Models selected by cross-validated RMSE on 60 courses. "
    "Revenue is highly predictable (CV R² ≈ 0.99, price-driven); course-level "
    "demand is near-uniform and only weakly predictable — read predictions with "
    "that caveat."
)

models = load_models()
master = load_master()
tx = load_transactions()
ranges = feature_ranges()
cat_def = category_defaults()
CATEGORIES = sorted(master["CourseCategory"].unique())

tab1, tab2, tab3, tab4 = st.tabs(
    ["🔮 Demand Prediction", "💰 Revenue Forecast",
     "📊 Feature Importance", "🏷️ Category Comparison"]
)

# --------------------------------------------------------------------------- #
# TAB 1 — Demand Prediction
# --------------------------------------------------------------------------- #
with tab1:
    st.subheader("Predict a hypothetical course")
    st.write(
        "Set the course's attributes; the app predicts **enrollment**, "
        "**revenue**, and **revenue per enrollment** from the trained models."
    )

    left, right = st.columns([1, 1])
    with left:
        category = st.selectbox("Category", CATEGORIES, index=0)
        course_type = st.radio("Course type", ["Paid", "Free"], horizontal=True)
        is_free = 1 if course_type == "Free" else 0

        default_price = 0.0 if is_free else float(round(ranges["CoursePrice"]["mean"], 2))
        price = st.number_input(
            "Price ($)", min_value=0.0, max_value=1000.0,
            value=default_price, step=10.0, disabled=is_free,
            help="Forced to $0 for free courses.",
        )
        if is_free:
            price = 0.0

        level = st.select_slider("Level", options=config.LEVEL_ORDER,
                                 value="Intermediate")
        duration = st.slider(
            "Duration (hours)", 1.0, 60.0,
            float(round(ranges["CourseDuration"]["mean"], 1)), 0.5,
        )
    with right:
        course_rating = st.slider("Expected course rating", 1.0, 5.0,
                                  float(round(ranges["CourseRating"]["mean"], 2)), 0.05)
        inst_rating = st.slider(
            "Instructor rating (avg)", 1.0, 5.0,
            float(round(ranges["inst_rating_wmean"]["mean"], 2)), 0.05,
        )
        inst_years = st.slider(
            "Instructor experience (avg years)", 0.0, 30.0,
            float(round(ranges["inst_years_mean"]["mean"], 1)), 0.5,
        )
        match_share = st.slider(
            "Share taught by category-matched instructors", 0.0, 1.0,
            float(round(ranges["expertise_match_score"]["mean"], 2)), 0.05,
            help="Fraction of enrollments delivered by an instructor whose "
                 "expertise equals the course category.",
        )
        with st.expander("Advanced instructor settings"):
            distinct_teachers = st.slider(
                "Distinct instructors", 1, 40,
                int(round(cat_def.loc[category, "inst_distinct_teachers"])),
            )
            years_max = st.slider(
                "Max instructor experience (years)", 0.0, 30.0,
                float(round(cat_def.loc[category, "inst_years_max"], 1)), 0.5,
            )

    values = {
        "CoursePrice": price,
        "CourseDuration": duration,
        "CourseRating": course_rating,
        "CourseLevel_ordinal": config.LEVEL_TO_ORDINAL[level],
        "is_free": is_free,
        "inst_rating_wmean": inst_rating,
        "inst_years_mean": inst_years,
        "inst_years_max": years_max,
        "inst_distinct_teachers": distinct_teachers,
        "expertise_match_score": match_share,
        "CourseCategory": category,
    }

    X_in = build_input_row(values)
    pred_enroll = float(models["enrollment_count"]["model"].predict(X_in)[0])
    pred_rev = float(models["course_revenue"]["model"].predict(X_in)[0])
    pred_rev = max(pred_rev, 0.0)
    pred_enroll = max(pred_enroll, 0.0)
    if is_free:
        pred_rev = 0.0
    rev_per_enroll = pred_rev / pred_enroll if pred_enroll > 0 else 0.0

    st.markdown("### Predictions")
    c1, c2, c3 = st.columns(3)
    c1.metric("Predicted enrollment", f"{pred_enroll:,.0f}")
    c2.metric("Predicted revenue", f"${pred_rev:,.0f}")
    c3.metric("Revenue / enrollment", f"${rev_per_enroll:,.2f}")

    exposed = ["CoursePrice", "CourseDuration", "CourseRating",
               "inst_rating_wmean", "inst_years_mean", "expertise_match_score"]
    warns = extrapolation_flags(values, ranges, exposed if not is_free
                                else [f for f in exposed if f != "CoursePrice"])
    if warns:
        st.warning("⚠️ Out-of-range inputs:\n\n" + "\n\n".join(f"- {w}" for w in warns))
    else:
        st.success("All inputs are within the training range.")

    st.info(
        f"Context: enrollment across the catalog averages "
        f"{master['enrollment_count'].mean():.0f} (range "
        f"{master['enrollment_count'].min():.0f}–{master['enrollment_count'].max():.0f}). "
        "Because demand is near-uniform, treat the enrollment number as a band, "
        "not a point estimate."
    )

# --------------------------------------------------------------------------- #
# TAB 2 — Revenue Forecast
# --------------------------------------------------------------------------- #
with tab2:
    st.subheader("Revenue overview")

    total_rev = master["course_revenue"].sum()
    paid = master[master["is_free"] == 0]
    k1, k2, k3 = st.columns(3)
    k1.metric("Total revenue (2025)", f"${total_rev:,.0f}")
    k2.metric("Paid courses", f"{len(paid)} / {len(master)}")
    k3.metric("Avg revenue / paid course", f"${paid['course_revenue'].mean():,.0f}")

    txm = tx.copy()
    txm["month"] = txm["TransactionDate"].dt.to_period("M").dt.to_timestamp()
    monthly = txm.groupby("month").agg(
        enrollments=("TransactionID", "count"),
        revenue=("Amount", "sum"),
    )

    st.markdown("#### Monthly trend (2025)")
    fig, ax1 = plt.subplots(figsize=(11, 4.5))
    ax1.bar(monthly.index, monthly["revenue"], width=20, color="#a53b5b",
            alpha=0.7, label="Revenue")
    ax1.set_ylabel("Revenue ($)", color="#a53b5b")
    ax1.set_xlabel("Month")
    ax2 = ax1.twinx()
    ax2.plot(monthly.index, monthly["enrollments"], marker="o",
             color="#3b6ea5", label="Enrollments")
    ax2.set_ylabel("Enrollments", color="#3b6ea5")
    ax1.set_title("Monthly revenue (bars) and enrollments (line)")
    st.pyplot(fig)
    plt.close(fig)

    st.markdown("#### Top courses by revenue")
    top = (master.sort_values("course_revenue", ascending=False)
           [["CourseName", "CourseCategory", "CoursePrice",
             "enrollment_count", "course_revenue"]].head(10)
           .reset_index(drop=True))
    top["course_revenue"] = top["course_revenue"].round(0)
    st.dataframe(top, use_container_width=True)

# --------------------------------------------------------------------------- #
# TAB 3 — Feature Importance Explorer
# --------------------------------------------------------------------------- #
with tab3:
    st.subheader("What drives each target?")
    target = st.selectbox(
        "Target", list(config.TARGETS.keys()),
        format_func=lambda t: config.TARGETS[t],
    )
    meta = models[target]["meta"]
    best = meta["cv_results"][0]
    m1, m2, m3 = st.columns(3)
    m1.metric("Winning model", meta["winner"])
    m2.metric("CV R²", f"{best['CV_R2']}")
    m3.metric("CV RMSE", f"{best['CV_RMSE']}")

    @st.cache_data(show_spinner="Computing permutation importance…")
    def perm_importance(target_name: str) -> pd.DataFrame:
        from sklearn.inspection import permutation_importance

        mdl = joblib.load(config.MODELS_DIR / f"{target_name}_best.joblib")
        X = master[NUMERIC_FEATURES + NOMINAL_FEATURES]
        y = master[target_name]
        # n_jobs=1: never spawn subprocesses (loky/multiprocessing segfaults in
        # the Streamlit Cloud sandbox). Fast enough for 60 rows x 20 repeats.
        r = permutation_importance(
            mdl, X, y, n_repeats=20, random_state=config.RANDOM_STATE,
            scoring="neg_root_mean_squared_error", n_jobs=1,
        )
        return (pd.DataFrame({"feature": NUMERIC_FEATURES + NOMINAL_FEATURES,
                              "importance": r.importances_mean,
                              "std": r.importances_std})
                .sort_values("importance", ascending=False)
                .reset_index(drop=True))

    # Compute lazily on demand: Streamlit runs every tab body on each rerun, so
    # gating behind a button keeps this off the startup path entirely.
    if st.button("Compute permutation importance", key="perm_btn"):
        imp = perm_importance(target)
        fig, ax = plt.subplots(figsize=(9, 5.5))
        top = imp.head(10).iloc[::-1]
        ax.barh(top["feature"], top["importance"], xerr=top["std"],
                color="#3b6ea5")
        ax.set_xlabel("Increase in RMSE when feature is shuffled")
        ax.set_title(f"Permutation importance — {config.TARGETS[target]}")
        st.pyplot(fig)
        plt.close(fig)
    else:
        st.info("Click the button above to compute permutation importance for "
                "the selected target.")

    st.caption(
        "Permutation importance measures how much CV error grows when a feature "
        "is randomly shuffled — bigger means more influential. For revenue, price "
        "dominates; for enrollment, no feature is strongly informative."
    )

# --------------------------------------------------------------------------- #
# TAB 4 — Category Comparison
# --------------------------------------------------------------------------- #
with tab4:
    st.subheader("Category leaderboard")

    cat = master.groupby("CourseCategory").agg(
        courses=("CourseID", "count"),
        enrollment=("enrollment_count", "sum"),
        actual_revenue=("course_revenue", "sum"),
        avg_price=("CoursePrice", "mean"),
    )
    # Category revenue rolled up from course-level predictions (per the brief).
    X_all = master[NUMERIC_FEATURES + NOMINAL_FEATURES]
    master_pred = master[["CourseCategory"]].copy()
    master_pred["pred_revenue"] = models["course_revenue"]["model"].predict(X_all)
    master_pred["pred_revenue"] = master_pred["pred_revenue"].clip(lower=0)
    cat["predicted_revenue"] = (master_pred.groupby("CourseCategory")["pred_revenue"]
                                .sum())
    cat = cat.sort_values("actual_revenue", ascending=False)

    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(7, 6))
        cat["enrollment"].sort_values().plot.barh(ax=ax, color="#3b6ea5")
        ax.set_xlabel("Total enrollments")
        ax.set_title("Enrollment by category")
        st.pyplot(fig)
        plt.close(fig)
    with col2:
        fig, ax = plt.subplots(figsize=(7, 6))
        comp = cat[["actual_revenue", "predicted_revenue"]].sort_values(
            "actual_revenue")
        comp.plot.barh(ax=ax, color=["#a53b5b", "#e0a3b3"])
        ax.set_xlabel("Revenue ($)")
        ax.set_title("Revenue by category — actual vs predicted roll-up")
        st.pyplot(fig)
        plt.close(fig)

    st.markdown("#### Category table")
    show = cat.copy()
    for c in ["enrollment", "actual_revenue", "predicted_revenue", "avg_price"]:
        show[c] = show[c].round(0)
    st.dataframe(show, use_container_width=True)
    st.caption(
        "Enrollment and revenue rank categories differently — popular (often "
        "free) categories need not be the top earners. Predicted revenue is the "
        "sum of course-level model predictions within each category."
    )
