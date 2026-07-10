"""Course-level feature engineering for the EduPro forecasting models.

Builds a single 60-row *master table* (one row per course) holding the two
modelling targets and all engineered predictors. Three design points matter:

1. **No clean course->teacher link.** ``Courses`` carries no ``TeacherID`` and a
   course is taught by many teachers across ``Transactions`` (many-to-many).
   Instructor features are therefore *aggregated over each course's
   transactions* (enrollment-weighted mean rating, mean/max experience,
   distinct-teacher count, dominant expertise, expertise-category match).

2. **Enrollment != revenue.** Free courses (38/60) earn $0 regardless of demand,
   so both targets are produced and ``revenue_per_enrollment`` is engineered.

3. **No target leakage.** ``course_revenue``, ``enrollment_count`` and
   ``revenue_per_enrollment`` are outcomes; none of them is used as a predictor.
   Highly collinear numeric predictors (|r| > 0.9) are dropped with a logged
   justification.

The expertise->category mapping is the identity over the 12 shared category
labels (verified in the data; see :data:`config.EXPERTISE_TO_CATEGORY`).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import config
from .data_loader import load_data

# Feature groups (populated/validated in build_master_table) ----------------- #
NUMERIC_FEATURES: list[str] = [
    "CoursePrice",
    "CourseDuration",
    "CourseRating",
    "CourseLevel_ordinal",
    "is_free",
    "inst_rating_wmean",
    "inst_years_mean",
    "inst_years_max",
    "inst_distinct_teachers",
    "expertise_match_score",
]
NOMINAL_FEATURES: list[str] = ["CourseCategory"]

# Outcome columns that must never be used as predictors (leakage guard).
LEAKAGE_COLUMNS: list[str] = [
    "enrollment_count",
    "course_revenue",
    "revenue_per_enrollment",
]

HIGH_CORR_THRESHOLD: float = 0.9


def _instructor_features(
    tx: pd.DataFrame, teachers: pd.DataFrame, courses: pd.DataFrame
) -> pd.DataFrame:
    """Aggregate teacher attributes over each course's transactions.

    Every transaction names the delivering ``TeacherID``; we join teacher
    attributes onto transactions and reduce per course. Because each transaction
    is one enrollment, a plain mean over transactions is already
    enrollment-weighted.
    """
    merged = tx.merge(
        teachers[["TeacherID", "Expertise", "YearsOfExperience", "TeacherRating"]],
        on="TeacherID",
        how="left",
    ).merge(courses[["CourseID", "CourseCategory"]], on="CourseID", how="left")

    # Expertise<->category match per transaction (1 if teacher's expertise maps
    # to the course's category), then averaged per course.
    mapped = merged["Expertise"].map(config.EXPERTISE_TO_CATEGORY)
    merged["expertise_match"] = (mapped == merged["CourseCategory"]).astype(int)

    def _dominant_expertise(s: pd.Series) -> str:
        return s.mode().iat[0]

    agg = merged.groupby("CourseID").agg(
        inst_rating_wmean=("TeacherRating", "mean"),
        inst_years_mean=("YearsOfExperience", "mean"),
        inst_years_max=("YearsOfExperience", "max"),
        inst_distinct_teachers=("TeacherID", "nunique"),
        inst_dominant_expertise=("Expertise", _dominant_expertise),
        expertise_match_score=("expertise_match", "mean"),
    )
    return agg.reset_index()


def build_master_table(sheets: dict[str, pd.DataFrame] | None = None) -> pd.DataFrame:
    """Construct the 60-row course-level master table (targets + features)."""
    sheets = sheets or load_data()
    courses = sheets["Courses"].copy()
    teachers = sheets["Teachers"]
    tx = sheets["Transactions"]

    # --- Targets ---------------------------------------------------------- #
    target_agg = tx.groupby("CourseID").agg(
        enrollment_count=("TransactionID", "count"),
        course_revenue=("Amount", "sum"),
    )
    df = courses.merge(target_agg, on="CourseID", how="left")
    df["revenue_per_enrollment"] = df["course_revenue"] / df["enrollment_count"]

    # --- Course features -------------------------------------------------- #
    df["is_free"] = (df["CourseType"] == "Free").astype(int)
    df["CourseLevel_ordinal"] = df["CourseLevel"].map(config.LEVEL_TO_ORDINAL)

    # Interpretable engineered bands/buckets/tiers (used in dashboard/EDA, not
    # fed to the model because they are monotone transforms of the numerics and
    # would be redundant for n=60).
    df["price_band"] = pd.cut(
        df["CoursePrice"],
        bins=[-0.01, 0.01, 150, 300, np.inf],
        labels=["Free", "Low", "Mid", "High"],
    )
    df["duration_bucket"] = pd.qcut(
        df["CourseDuration"], q=3, labels=["Short", "Medium", "Long"]
    )
    df["rating_tier"] = pd.cut(
        df["CourseRating"],
        bins=[0, 2.5, 4.0, 5.0],
        labels=["Low", "Mid", "High"],
    )

    # --- Instructor features (many-to-many aggregation) ------------------- #
    inst = _instructor_features(tx, teachers, courses)
    df = df.merge(inst, on="CourseID", how="left")

    return df


def drop_high_corr(
    df: pd.DataFrame, features: list[str], threshold: float = HIGH_CORR_THRESHOLD
) -> tuple[list[str], list[tuple[str, str, float]]]:
    """Drop one of every numeric feature pair with |corr| > threshold.

    Returns the surviving feature list and a log of dropped pairs for
    justification.
    """
    corr = df[features].corr().abs()
    dropped: list[tuple[str, str, float]] = []
    keep = list(features)
    for i, a in enumerate(features):
        for b in features[i + 1:]:
            if a in keep and b in keep and corr.loc[a, b] > threshold:
                keep.remove(b)  # drop the later one
                dropped.append((a, b, float(corr.loc[a, b])))
    return keep, dropped


def build_preprocessor(numeric: list[str], nominal: list[str]) -> ColumnTransformer:
    """Standardize numerics and one-hot encode nominals inside a transformer."""
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", drop=None, sparse_output=False),
                nominal,
            ),
        ],
        remainder="drop",
    )


def get_modeling_frame() -> tuple[pd.DataFrame, list[str], list[str]]:
    """Return the master table plus the *clean* numeric and nominal feature lists.

    The numeric list has had |r|>0.9 redundancies removed (logged).
    """
    master = build_master_table()
    numeric_keep, dropped = drop_high_corr(master, NUMERIC_FEATURES)
    if dropped:
        for a, b, r in dropped:
            print(f"  [drop |r|>{HIGH_CORR_THRESHOLD}] '{b}' (r={r:.2f} with '{a}')")
    return master, numeric_keep, NOMINAL_FEATURES


def main() -> None:
    """Build, summarise and persist the master table."""
    print("=" * 70)
    print("EduPro feature engineering — master table")
    print("=" * 70)
    master, numeric, nominal = get_modeling_frame()

    out = config.DATA_DIR / "course_master.csv"
    master.to_csv(out, index=False)

    print(f"\nMaster table shape: {master.shape}")
    print(f"Null cells: {int(master.isnull().sum().sum())}")
    print(f"\nNumeric features ({len(numeric)}): {numeric}")
    print(f"Nominal features ({len(nominal)}): {nominal}")
    print(f"Leakage columns excluded from features: {LEAKAGE_COLUMNS}")

    print("\nInstructor feature sample:")
    print(
        master[
            [
                "CourseID",
                "CourseCategory",
                "inst_rating_wmean",
                "inst_years_mean",
                "inst_distinct_teachers",
                "expertise_match_score",
            ]
        ].head(6).to_string(index=False)
    )

    print("\nTargets summary:")
    print(master[["enrollment_count", "course_revenue", "revenue_per_enrollment"]]
          .describe().round(2).to_string())

    # Sanity: preprocessor fits and produces a finite matrix.
    pre = build_preprocessor(numeric, nominal)
    X = pre.fit_transform(master)
    print(f"\nDesign matrix after preprocessing: {X.shape} "
          f"(finite={np.isfinite(X).all()})")
    print(f"Saved master table -> {out}")


if __name__ == "__main__":
    main()
