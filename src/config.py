"""Central configuration for the EduPro forecasting project.

All paths, constants, schema expectations and modelling knobs live here so the
rest of the codebase contains no hardcoded paths or magic numbers.
"""

from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #
RANDOM_STATE: int = 42

# --------------------------------------------------------------------------- #
# Paths (resolved relative to the project root, never hardcoded absolutes)
# --------------------------------------------------------------------------- #
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
DATA_DIR: Path = PROJECT_ROOT / "data"
MODELS_DIR: Path = PROJECT_ROOT / "models"
REPORTS_DIR: Path = PROJECT_ROOT / "reports"
FIGURES_DIR: Path = REPORTS_DIR / "figures"

RAW_XLSX: Path = DATA_DIR / "EduPro_Online_Platform.xlsx"

for _d in (DATA_DIR, MODELS_DIR, REPORTS_DIR, FIGURES_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------- #
# Expected schema & row counts (used by data_loader validation)
# --------------------------------------------------------------------------- #
SHEETS: dict[str, dict] = {
    "Users": {
        "rows": 3000,
        "columns": ["UserID", "UserName", "Age", "Gender", "Email"],
    },
    "Teachers": {
        "rows": 60,
        "columns": [
            "TeacherID",
            "TeacherName",
            "Age",
            "Gender",
            "Expertise",
            "YearsOfExperience",
            "TeacherRating",
        ],
    },
    "Courses": {
        "rows": 60,
        "columns": [
            "CourseID",
            "CourseName",
            "CourseCategory",
            "CourseType",
            "CourseLevel",
            "CoursePrice",
            "CourseDuration",
            "CourseRating",
        ],
    },
    "Transactions": {
        "rows": 10000,
        "columns": [
            "TransactionID",
            "UserID",
            "CourseID",
            "TransactionDate",
            "Amount",
            "PaymentMethod",
            "TeacherID",
        ],
    },
}

DATE_COLUMN: str = "TransactionDate"
DATE_MIN: str = "2025-01-01"
DATE_MAX: str = "2025-12-30"

# --------------------------------------------------------------------------- #
# Domain encodings
# --------------------------------------------------------------------------- #
LEVEL_ORDER: list[str] = ["Beginner", "Intermediate", "Advanced"]
LEVEL_TO_ORDINAL: dict[str, int] = {lvl: i for i, lvl in enumerate(LEVEL_ORDER)}

# Mapping from a teacher's Expertise to a course CourseCategory, used to compute
# the expertise<->category match score (an instructor feature).
#
# Verified against the data: the Teachers.Expertise column and the
# Courses.CourseCategory column contain the SAME 12 labels, so the mapping is an
# identity over those 12 categories. The match score therefore reduces to an
# equality test (teacher.Expertise == course.CourseCategory), aggregated per
# course as an enrollment-weighted share. We still keep the mapping explicit so
# that if future data introduced differing label sets it could be remapped here
# in one place (see feature_engineering.py).
CATEGORIES: list[str] = [
    "Artificial Intelligence",
    "Business",
    "Cybersecurity",
    "Data Science",
    "Design",
    "Digital Marketing",
    "Finance",
    "Machine Learning",
    "Marketing",
    "Programming",
    "Project Management",
    "Web Development",
]
EXPERTISE_TO_CATEGORY: dict[str, str] = {c: c for c in CATEGORIES}

# --------------------------------------------------------------------------- #
# Modelling
# --------------------------------------------------------------------------- #
CV_FOLDS: int = 5

# Targets modelled at the course level (60 rows).
TARGETS: dict[str, str] = {
    "enrollment_count": "Enrollment Count per course",
    "course_revenue": "Course Revenue per course",
}

# Tree-model constraints appropriate for n=60 (avoid overfitting).
RF_PARAM_GRID: dict[str, list] = {
    "model__n_estimators": [200, 400],
    "model__max_depth": [2, 3, 4],
    "model__min_samples_leaf": [2, 3, 5],
}
GBR_PARAM_GRID: dict[str, list] = {
    "model__n_estimators": [100, 200],
    "model__max_depth": [1, 2, 3],
    "model__learning_rate": [0.03, 0.05, 0.1],
    "model__min_samples_leaf": [2, 3],
}
ALPHA_GRID: list[float] = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]

# Matplotlib figure defaults
FIG_DPI: int = 110
FIG_SIZE: tuple[int, int] = (9, 6)
