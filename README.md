# Predictive Modeling for Course Demand & Revenue Forecasting on EduPro

Forecast **course demand (enrollment count)** and **course revenue** for the
EduPro online-learning platform, and reveal the **drivers** behind them, to guide
what to launch, how to price, and which instructors to onboard.

> **Headline finding (honest):** revenue is highly predictable (cross-validated
> R² ≈ 0.99, driven by price), while course-level demand is **not** predictable
> from the available attributes (best CV R² ≈ 0) because enrollments are nearly
> uniform across courses. Full discussion in
> [`reports/research_paper.md`](reports/research_paper.md).

---

## Project layout

```
.
├── data/
│   ├── EduPro_Online_Platform.xlsx   # source workbook (4 sheets)
│   └── course_master.csv             # generated 60-row master table
├── src/
│   ├── config.py                     # all paths, seeds, schema, model grids
│   ├── data_loader.py                # load + validate schema/rows/nulls/dates
│   ├── eda.py                        # exploratory figures + insights
│   ├── feature_engineering.py        # course-level master table (no leakage)
│   ├── modeling.py                   # candidate estimators + CV scoring helpers
│   ├── train.py                      # CV model selection, saves models + metadata
│   ├── evaluate.py                   # 5-fold + LOOCV metrics, importance, drivers
│   └── monthly_model.py              # bonus: monthly demand forecasting
├── app/
│   └── streamlit_app.py              # interactive dashboard
├── models/                           # saved *.joblib models + *_metadata.json
├── reports/
│   ├── figures/                      # all generated PNG figures
│   ├── research_paper.md             # full technical write-up
│   ├── executive_summary.md          # 1–2 page non-technical summary
│   └── driver_findings.md            # generated driver analysis
├── requirements.txt                  # pinned dependencies
└── README.md
```

## Setup

Requires Python 3.11+ (developed on 3.13). From the project root:

**Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

**macOS / Linux (bash):**
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

> The source workbook must be at `data/EduPro_Online_Platform.xlsx`.

## Run the pipeline (top to bottom)

Each stage is an importable module; run them in order. Every stage prints a short
validation report (shapes, null counts, score summary) and writes its artifacts.

```bash
python -m src.data_loader          # 1. load + validate the workbook
python -m src.eda                  # 2. EDA figures  -> reports/figures/
python -m src.feature_engineering  # 3. build master table -> data/course_master.csv
python -m src.train                # 4. CV model selection -> models/
python -m src.evaluate             # 5. metrics, importance, drivers
python -m src.monthly_model        # 6. bonus monthly demand model
```

Reproducibility: `random_state = 42` everywhere, dependencies pinned, no
hardcoded absolute paths (all paths derive from `src/config.py`).

## Launch the dashboard

```bash
streamlit run app/streamlit_app.py
```

Tabs:
- **Demand Prediction** — enter price, duration, level, instructor experience &
  rating, category, free/paid → predicted enrollment, revenue, and
  revenue/enrollment, with an out-of-range extrapolation warning.
- **Revenue Forecast** — revenue visuals including the monthly trend.
- **Feature Importance Explorer** — drivers per target.
- **Category Comparison** — category-level enrollment vs revenue.

## What each target means

| Target | Grain | Rows | Notes |
|---|---|---|---|
| Enrollment count | per course | 60 | count of transactions |
| Course revenue | per course | 60 | sum of `Amount`; $0 for free courses |
| Category revenue | per category | 12 | **rolled up** from course predictions |
| Monthly demand (bonus) | course × month | 540 | lagged features, forward-chained CV |

## Key design decisions

- **Small n = 60** → 5-fold + Leave-One-Out CV, metrics as mean ± std, explicit
  train-vs-CV gap, regularized/shallow models. Never judged on a single split.
- **Many-to-many teacher link** → instructor features are aggregated over each
  course's transactions (no fake 1:1 join).
- **Enrollment ≠ revenue** → modeled separately; `revenue_per_enrollment`
  engineered but excluded as a predictor (target leakage).

See [`reports/research_paper.md`](reports/research_paper.md) for full methodology,
results, drivers, and limitations, and
[`reports/executive_summary.md`](reports/executive_summary.md) for the
non-technical brief.
