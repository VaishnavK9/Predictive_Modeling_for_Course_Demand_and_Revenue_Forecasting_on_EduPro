# Predictive Modeling for Course Demand and Revenue Forecasting on EduPro

*A reproducible study of course-level demand and revenue drivers on an online
learning platform.*

---

## Abstract

EduPro, an online learning platform, plans course launches, pricing, and
instructor onboarding largely by intuition. We build a reproducible predictive
suite to forecast **course demand (enrollment count)** and **course revenue**,
and — equally important — to surface the **drivers** of each. Working from a
four-table operational dataset (3,000 users, 60 teachers, 60 courses, 10,000
transactions for calendar year 2025), we aggregate to a 60-row course-level
table and evaluate models with **5-fold cross-validation and Leave-One-Out CV**,
reporting metrics as mean ± standard deviation and explicitly measuring the
**train-vs-CV gap**. Our central, honest finding is a contrast: **revenue is
highly predictable** (Lasso, cross-validated R² = 0.99) because it is mechanically
driven by price, while **enrollment demand is essentially unpredictable from
course attributes** (best cross-validated R² ≈ 0, negative) because demand is
near-uniform across courses. A bonus monthly forecasting model beats a naive
last-month baseline but confirms that period-to-period demand is close to
stationary noise. We translate these results into pricing and portfolio
recommendations and document the limitations a decision-maker must keep in mind.

---

## 1. Problem statement

EduPro cannot currently predict which courses will attract enrollments or
generate revenue. The business needs **predictive intelligence** to answer:

1. **What to launch** — which course/category/level configurations earn revenue?
2. **How to price** — how sensitive is demand and revenue to price?
3. **Who to onboard** — do instructor experience and rating move outcomes?

We therefore model two distinct targets — **enrollment count** and **course
revenue** — separately, because on this platform they diverge sharply (see §3).

## 2. Data

The workbook `EduPro_Online_Platform.xlsx` has four sheets. All were validated
programmatically against the expected schema (`src/data_loader.py`): correct row
counts, columns, **zero missing values**, and all transaction dates within
2025-01-01…2025-12-30.

| Sheet | Rows | Key fields |
|---|---|---|
| Users | 3,000 | UserID, Age (15–35), Gender (≈50/50) |
| Teachers | 60 | TeacherID, Expertise (12 categories), YearsOfExperience, TeacherRating |
| Courses | 60 | CourseID, CourseCategory (12), CourseType (38 Free / 22 Paid), CourseLevel, CoursePrice, CourseDuration, CourseRating |
| Transactions | 10,000 | TransactionID, UserID, CourseID, TransactionDate, Amount, PaymentMethod, TeacherID |

**Three structural facts shaped the entire design:**

1. **Small course table (n = 60).** Aggregating transactions per course yields
   only 60 modelling rows. A single train/test split would be unreliable, so all
   course-level evaluation uses 5-fold + LOOCV with mean ± std.
2. **No clean course→teacher link.** `Courses` carries no `TeacherID`; each
   course appears with *many* teachers across `Transactions` (many-to-many,
   15–29 distinct teachers per course). Instructor features are therefore
   *aggregated over each course's transactions*, not joined 1:1.
3. **Enrollment ≠ revenue.** Free courses (38 of 60) earn $0 regardless of how
   popular they are. ~64% of all transactions have `Amount = 0`. A course can
   top the enrollment chart and earn nothing.

A fourth fact emerged during EDA and is critical to interpretation: for paid
courses, every transaction's `Amount` **exactly equals** the `CoursePrice`
(verified: 100% match). Revenue is thus mechanically `price × paid-enrollments`,
which makes `CoursePrice`, `course_revenue`, and `revenue_per_enrollment`
perfectly collinear (|r| = 1.00).

## 3. Exploratory data analysis

Figures are in `reports/figures/` (regenerate with `python -m src.eda`). Key
observations:

- **Distributions (`01_distributions.png`).** Revenue and revenue-per-enrollment
  are strongly right-skewed and zero-inflated (the free-course spike at $0).
  Enrollment count is tight and roughly symmetric (mean 166.7, **std 12.5**,
  range 140–196).
- **Free vs Paid (`02_free_vs_paid.png`).** Free and paid courses enroll
  similarly, but only paid courses produce revenue — the demand/revenue split.
- **Category leaderboard (`03_category_leaderboard.png`).** Enrollment and revenue
  rank categories *differently*; the top revenue category is **Artificial
  Intelligence**, driven by paid pricing rather than raw demand.
- **Level effects (`04_level_effects.png`).** Course level has only a modest
  relationship with demand relative to type and price.
- **Correlation (`05_correlation_heatmap.png`).** `CoursePrice` ↔ `course_revenue`
  = 1.00; `CourseRating` ↔ `enrollment_count` = +0.29 (the strongest demand
  signal, still weak); `CoursePrice` ↔ `enrollment_count` = −0.16.
- **Seasonality (`06_monthly_trend.png`, `07_seasonality_month.png`).** Monthly
  enrollment volume is broadly flat across 2025 — no strong seasonal peaks.
- **Demographics & payments (`08`, `09`).** Users are 15–35, ~50/50 gender;
  PayPal/Credit Card/Bank Transfer split roughly evenly (~⅓ each).

## 4. Feature engineering

Built in `src/feature_engineering.py` into a single 60-row master table
(`data/course_master.csv`).

**Course features:** `CoursePrice`, `CourseDuration`, `CourseRating`,
`CourseLevel_ordinal` (Beginner<Intermediate<Advanced → 0/1/2), `is_free`. Also
interpretable bands (`price_band`, `duration_bucket`, `rating_tier`) used in the
dashboard/EDA but **kept out of the model** because they are monotone transforms
of the numerics (redundant for n = 60).

**Instructor features (aggregated per caveat #2, enrollment-weighted because each
transaction is one enrollment):** `inst_rating_wmean` (mean TeacherRating over a
course's transactions), `inst_years_mean`, `inst_years_max`,
`inst_distinct_teachers`, dominant `Expertise`, and an **expertise–category match
score** = the share of a course's transactions whose teacher's `Expertise`
matches the course's `CourseCategory`. The expertise→category mapping is the
**identity over the 12 shared labels** (the `Expertise` and `CourseCategory`
columns contain the same 12 values — verified in data; see
`config.EXPERTISE_TO_CATEGORY`).

**Leakage control.** `enrollment_count`, `course_revenue`, and
`revenue_per_enrollment` are outcomes and are **never used as predictors**. We
programmatically drop any numeric feature pair with |r| > 0.9 (none triggered
among the retained features). Numerics are standardized and the single nominal
(`CourseCategory`) is one-hot encoded **inside a scikit-learn `Pipeline`**, so all
preprocessing is fit within each CV fold (no leakage across folds).

## 5. Methodology

For each course-level target we cross-validate a model zoo and pick the winner on
**cross-validated RMSE**, never the training score:

- **Baselines:** `LinearRegression`, `Ridge`, `Lasso` (α tuned by `GridSearchCV`).
- **Advanced (constrained for n = 60):** `RandomForestRegressor` (max_depth 2–4,
  min_samples_leaf ≥ 2) and `GradientBoostingRegressor` (max_depth 1–3, small
  learning rate). Hyperparameter search runs in an **inner** 5-fold loop, so the
  reported scores come from **nested** cross-validation.
- **Protocol:** 5-fold `KFold(shuffle=True, random_state=42)` for the headline
  table plus Leave-One-Out CV for a second, fold-free view. We report MAE, RMSE,
  and R² as mean ± std and the **train-vs-CV R² gap** as an overfitting gauge.

All randomness is seeded (`random_state = 42`); dependencies are pinned.

## 6. Results

### 6.1 Course revenue — highly predictable

| Model | CV RMSE | CV R² | train R² | R² gap |
|---|---|---|---|---|
| **Lasso (winner)** | **2136.6 ± 1072.5** | **0.991 ± 0.004** | 0.995 | **0.004** |
| GradientBoosting | 2961.7 | 0.983 | 0.999 | 0.015 |
| Ridge | 2872.4 | 0.966 | 0.996 | 0.030 |
| LinearRegression | 3067.0 | 0.964 | 0.996 | 0.032 |
| RandomForest | 4492.7 | 0.961 | 0.989 | 0.028 |

LOOCV confirms it: R² = 0.992, RMSE = 2268.9. The negligible train-vs-CV gap
means this is genuine signal, not overfitting. The driver is overwhelming:
standardized Lasso coefficient on `CoursePrice` ≈ 25,000 versus < 700 for every
other feature. **Revenue is price × paid-demand, and price is the lever.**

### 6.2 Enrollment count — essentially unpredictable

| Model | CV RMSE | CV R² | train R² | R² gap |
|---|---|---|---|---|
| **Ridge (winner)** | **12.24 ± 1.98** | **−0.130 ± 0.160** | 0.153 | **0.283** |
| Lasso | 12.50 | −0.161 | 0.111 | 0.272 |
| RandomForest | 13.25 | −0.348 | 0.486 | 0.834 |
| LinearRegression | 13.49 | −0.427 | 0.559 | 0.986 |
| GradientBoosting | 13.73 | −0.520 | 0.537 | 1.057 |

Two things stand out. First, **every model has a negative cross-validated R²** —
none beats simply predicting the mean enrollment. The CV RMSE (~12.2) is about
the same size as the target's standard deviation (12.5), confirming no real
predictive power. Second, the **train-vs-CV gap is enormous for the flexible
models** (LinearRegression 0.99, GBR 1.06): they memorise the 60 points and
generalise worse than the regularised Ridge, which is exactly why we report CV
rather than training scores and why Ridge — the most regularised — "wins". The
honest conclusion is that **course-level demand is near-uniform (140–196
enrollments) and is not explained by the available attributes.** The strongest
(still weak) association is `CourseRating` (r = +0.29).

### 6.3 Category revenue (roll-up)

Per the brief, category revenue is **rolled up from course-level predictions**
(sum of predicted course revenue within each of the 12 categories) rather than
modelled directly on 12 rows. Because the course revenue model is accurate, the
category roll-up inherits its reliability; the leaderboard is dominated by
categories with higher-priced paid courses (Artificial Intelligence leads).

### 6.4 Bonus — monthly demand model

A course × month panel (60 × 12; 540 rows after requiring a 3-month lag history)
with lag-1/2/3, a rolling-3 mean, cyclical month encoding, and static course
attributes, evaluated by **expanding-window forward chaining** (train on months
< m, forecast month m, for m = 6…12):

| Model | Forward-chaining RMSE | MAE | R² |
|---|---|---|---|
| **RandomForest (winner)** | **3.76** | 2.94 | −0.065 |
| Ridge | 3.86 | 3.01 | −0.127 |
| Naive (last month) | 5.15 | — | — |

The model **beats naive persistence** (RMSE 3.76 vs 5.15), so lags and
seasonality carry a little signal, but the slightly negative R² shows
per-course-month demand is close to stationary noise around its mean
(`reports/figures/monthly_forecast.png`).

## 7. Drivers (plain English)

- **Price is the revenue engine.** Setting a (higher) price is the single
  dominant determinant of course revenue; nothing else moves the needle
  materially. Free courses contribute enrollment and engagement but $0 revenue.
- **Demand barely responds to attributes.** Within this dataset, price, level,
  duration, category, and instructor metrics do **not** reliably predict how many
  people enroll — demand is roughly the same (~167) for every course.
- **Rating is the only (weak) demand nudge.** Higher `CourseRating` has the
  strongest positive association with enrollment (r = +0.29), and higher price a
  weak negative one (r = −0.16) — directionally sensible but not strong enough to
  forecast individual courses.
- **Instructor experience/rating show no robust course-level effect**, partly
  because every course is taught by many teachers (the many-to-many structure
  averages instructor differences away).

## 8. Limitations

- **Small sample (n = 60 courses).** All course-level conclusions rest on 60
  points; CV variance is non-trivial (revenue CV RMSE std ≈ 1,000). We mitigate
  with nested CV + LOOCV and regularised models, but external validity is limited.
- **Free-course revenue = 0 by construction.** Revenue is zero-inflated and, for
  paid courses, mechanically determined by price (Amount ≡ CoursePrice). The
  "predictability" of revenue is therefore partly an accounting identity, not a
  behavioural discovery.
- **No clean course→teacher link.** Instructor features are transaction-weighted
  aggregates over a many-to-many relationship; they cannot isolate the effect of
  any single instructor on a course.
- **Near-uniform demand likely reflects the data-generating process.** The
  dataset appears synthetic with demand drawn from a narrow range; real platforms
  would show more dispersion, and these models should be re-fit on production data
  before operational use.
- **One year, no strong seasonality.** With a single 2025 cycle, seasonal claims
  are tentative.

## 9. Recommendations

1. **Treat pricing as the primary revenue lever** and forecast revenue as
   `price × expected paid-enrollments`; the price coefficient is reliable enough
   to support pricing scenarios in the dashboard.
2. **Do not promise course-level demand forecasts** from attributes alone on this
   data; instead report demand as a band around the platform mean and invest in
   collecting features that actually vary with demand (marketing spend, traffic
   source, recommendation placement, prior-course completion).
3. **Use free courses as a funnel, not a revenue line** — measure their value via
   downstream paid conversion, which this dataset does not yet capture.
4. **Prioritise launches in higher-priced categories** (e.g., Artificial
   Intelligence) for revenue, while using rating/quality to defend demand.
5. **Re-train on production data and expand n** before relying on any course-level
   demand model operationally; revisit the monthly model once multi-year history
   exists.

## 10. Reproducibility

Pinned `requirements.txt`, `random_state = 42` throughout, configuration
centralised in `src/config.py`, no hardcoded absolute paths. The pipeline runs
top to bottom: `data_loader → eda → feature_engineering → train → evaluate →
monthly_model`, then the Streamlit dashboard. See `README.md` for exact commands.
