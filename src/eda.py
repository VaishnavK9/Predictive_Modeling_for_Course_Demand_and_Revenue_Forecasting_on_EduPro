"""Exploratory data analysis for the EduPro dataset.

Generates a suite of labelled figures into ``reports/figures`` and prints a
one-line insight for each. Run with::

    python -m src.eda
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless, file-only rendering
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from . import config
from .data_loader import load_data

sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = config.FIG_DPI


def _save(fig: plt.Figure, name: str, insight: str) -> None:
    """Save a figure to the figures dir and print its one-line insight."""
    path = config.FIGURES_DIR / name
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] {name:<34} -> {insight}")


def course_level_frame(sheets: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Build a lightweight course-level frame (targets + course attrs) for EDA."""
    courses = sheets["Courses"].copy()
    tx = sheets["Transactions"]
    agg = tx.groupby("CourseID").agg(
        enrollment_count=("TransactionID", "count"),
        course_revenue=("Amount", "sum"),
    )
    df = courses.merge(agg, on="CourseID", how="left")
    df["revenue_per_enrollment"] = df["course_revenue"] / df["enrollment_count"]
    return df


def run() -> None:
    """Execute the full EDA, saving every figure with an insight."""
    print("=" * 70)
    print("EduPro EDA — generating figures")
    print("=" * 70)
    sheets = load_data()
    users, teachers = sheets["Users"], sheets["Teachers"]
    courses, tx = sheets["Courses"], sheets["Transactions"]
    cl = course_level_frame(sheets)

    # 1. Distributions of numeric course attributes + targets ---------------- #
    num_cols = [
        ("CoursePrice", "Course price ($)"),
        ("CourseDuration", "Duration (hours)"),
        ("CourseRating", "Course rating"),
        ("enrollment_count", "Enrollment count"),
        ("course_revenue", "Course revenue ($)"),
        ("revenue_per_enrollment", "Revenue / enrollment ($)"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, (col, label) in zip(axes.ravel(), num_cols):
        sns.histplot(cl[col], kde=True, ax=ax, color="#3b6ea5")
        ax.set_title(f"{label}\nskew={cl[col].skew():.2f}")
        ax.set_xlabel(label)
    fig.suptitle("Distributions of course attributes and targets", fontsize=14)
    _save(
        fig,
        "01_distributions.png",
        "Revenue & revenue/enrollment are right-skewed (free courses pile at 0).",
    )

    # 2. Free vs Paid -------------------------------------------------------- #
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    sns.countplot(data=courses, x="CourseType", ax=axes[0],
                  order=["Free", "Paid"], palette="Set2", hue="CourseType",
                  legend=False)
    axes[0].set_title("Course count by type")
    sns.boxplot(data=cl, x="CourseType", y="enrollment_count", ax=axes[1],
                order=["Free", "Paid"], palette="Set2", hue="CourseType",
                legend=False)
    axes[1].set_title("Enrollment by type")
    sns.boxplot(data=cl, x="CourseType", y="course_revenue", ax=axes[2],
                order=["Free", "Paid"], palette="Set2", hue="CourseType",
                legend=False)
    axes[2].set_title("Revenue by type")
    fig.suptitle("Free vs Paid: demand vs revenue", fontsize=14)
    _save(
        fig,
        "02_free_vs_paid.png",
        "Free courses (38/60) drive enrollments but earn $0 — demand != revenue.",
    )

    # 3. Category leaderboard ------------------------------------------------ #
    cat = cl.groupby("CourseCategory").agg(
        enrollment=("enrollment_count", "sum"),
        revenue=("course_revenue", "sum"),
    ).sort_values("revenue", ascending=False)
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    cat["enrollment"].sort_values().plot.barh(ax=axes[0], color="#3b6ea5")
    axes[0].set_title("Total enrollment by category")
    axes[0].set_xlabel("Enrollments")
    cat["revenue"].sort_values().plot.barh(ax=axes[1], color="#a53b5b")
    axes[1].set_title("Total revenue by category")
    axes[1].set_xlabel("Revenue ($)")
    fig.suptitle("Category leaderboard", fontsize=14)
    _save(
        fig,
        "03_category_leaderboard.png",
        f"Top revenue category: {cat.index[0]}; enrollment and revenue rank differently.",
    )

    # 4. Level effects ------------------------------------------------------- #
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    sns.boxplot(data=cl, x="CourseLevel", y="enrollment_count",
                order=config.LEVEL_ORDER, ax=axes[0], palette="Blues",
                hue="CourseLevel", legend=False)
    axes[0].set_title("Enrollment by level")
    sns.boxplot(data=cl, x="CourseLevel", y="course_revenue",
                order=config.LEVEL_ORDER, ax=axes[1], palette="Reds",
                hue="CourseLevel", legend=False)
    axes[1].set_title("Revenue by level")
    fig.suptitle("Course level effects", fontsize=14)
    _save(fig, "04_level_effects.png",
          "Level has a modest effect on demand relative to price/type.")

    # 5. Correlation heatmap ------------------------------------------------- #
    corr_cols = ["CoursePrice", "CourseDuration", "CourseRating",
                 "enrollment_count", "course_revenue", "revenue_per_enrollment"]
    fig, ax = plt.subplots(figsize=(8, 6.5))
    sns.heatmap(cl[corr_cols].corr(), annot=True, fmt=".2f", cmap="coolwarm",
                center=0, ax=ax, square=True)
    ax.set_title("Correlation heatmap (course level)")
    _save(fig, "05_correlation_heatmap.png",
          "Revenue tracks price strongly; enrollment weakly negative with price.")

    # 6. Monthly time trend & seasonality ------------------------------------ #
    txm = tx.copy()
    txm["month"] = txm["TransactionDate"].dt.to_period("M").dt.to_timestamp()
    monthly = txm.groupby("month").agg(
        enrollments=("TransactionID", "count"),
        revenue=("Amount", "sum"),
    )
    fig, ax1 = plt.subplots(figsize=(13, 5))
    ax1.plot(monthly.index, monthly["enrollments"], marker="o",
             color="#3b6ea5", label="Enrollments")
    ax1.set_ylabel("Enrollments", color="#3b6ea5")
    ax1.set_xlabel("Month (2025)")
    ax2 = ax1.twinx()
    ax2.plot(monthly.index, monthly["revenue"], marker="s",
             color="#a53b5b", label="Revenue")
    ax2.set_ylabel("Revenue ($)", color="#a53b5b")
    ax1.set_title("Monthly enrollments and revenue (2025)")
    _save(fig, "06_monthly_trend.png",
          "Enrollments are broadly flat month-to-month — limited seasonality.")

    # 7. Seasonality by day-of-week / month bar ------------------------------ #
    txm["month_num"] = txm["TransactionDate"].dt.month
    fig, ax = plt.subplots(figsize=(12, 5))
    monthly_counts = txm.groupby("month_num").size()
    sns.barplot(x=monthly_counts.index, y=monthly_counts.values, ax=ax,
                color="#3b6ea5")
    ax.set_title("Enrollments by calendar month")
    ax.set_xlabel("Month")
    ax.set_ylabel("Enrollments")
    _save(fig, "07_seasonality_month.png",
          "No strong monthly seasonal spikes; volume is fairly uniform.")

    # 8. User demographics --------------------------------------------------- #
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    sns.histplot(users["Age"], bins=20, ax=axes[0], color="#3b6ea5")
    axes[0].set_title("User age distribution")
    sns.countplot(data=users, x="Gender", ax=axes[1], palette="Set2",
                  hue="Gender", legend=False)
    axes[1].set_title("User gender mix")
    fig.suptitle("User demographics", fontsize=14)
    _save(fig, "08_user_demographics.png",
          "Users are young (15-35) with a near 50/50 gender split.")

    # 9. Payment method mix -------------------------------------------------- #
    fig, ax = plt.subplots(figsize=(7, 5))
    pm = tx["PaymentMethod"].value_counts()
    ax.pie(pm.values, labels=pm.index, autopct="%1.1f%%",
           colors=sns.color_palette("Set2"))
    ax.set_title("Payment method mix (all transactions)")
    _save(fig, "09_payment_mix.png",
          "PayPal / Credit Card / Bank Transfer are evenly used (~1/3 each).")

    # 10. Price vs enrollment scatter (paid only) ---------------------------- #
    fig, ax = plt.subplots(figsize=(8, 6))
    paid = cl[cl["CourseType"] == "Paid"]
    sns.regplot(data=paid, x="CoursePrice", y="enrollment_count", ax=ax,
                scatter_kws={"color": "#a53b5b"}, line_kws={"color": "black"})
    ax.set_title("Price vs enrollment (paid courses)")
    ax.set_xlabel("Course price ($)")
    ax.set_ylabel("Enrollment count")
    _save(fig, "10_price_vs_enrollment.png",
          "Among paid courses, higher price associates with slightly lower demand.")

    print("\nEDA complete. Figures written to:", config.FIGURES_DIR)


if __name__ == "__main__":
    run()
