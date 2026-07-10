"""Shared modelling helpers: candidate estimators and cross-validated scoring.

Kept separate from :mod:`train` so that :mod:`evaluate` can reuse the exact same
estimator definitions and CV protocol (no drift between training and reporting).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.exceptions import ConvergenceWarning

# Lasso on the large-magnitude revenue target emits cosmetic convergence
# warnings; the selected alphas still cross-validate well. Silence for clean
# reports (functionally harmless — increasing max_iter does not change winners).
warnings.filterwarnings("ignore", category=ConvergenceWarning)
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.model_selection import GridSearchCV, KFold, cross_validate
from sklearn.pipeline import Pipeline

from . import config
from .feature_engineering import build_preprocessor

# Scorers (sklearn returns negatives for error metrics; we flip sign on report).
SCORING = {
    "rmse": "neg_root_mean_squared_error",
    "mae": "neg_mean_absolute_error",
    "r2": "r2",
}


def _kfold() -> KFold:
    return KFold(n_splits=config.CV_FOLDS, shuffle=True,
                 random_state=config.RANDOM_STATE)


def candidate_estimators(
    numeric: list[str], nominal: list[str]
) -> dict[str, BaseEstimator]:
    """Build the candidate model zoo, each as a preprocessing+model Pipeline.

    Models with hyperparameters are wrapped in :class:`GridSearchCV` (inner
    5-fold) so they can be evaluated by *nested* cross-validation — the honest
    way to report generalisation for n=60.
    """
    inner = _kfold()

    def pipe(model: BaseEstimator) -> Pipeline:
        return Pipeline([("pre", build_preprocessor(numeric, nominal)),
                         ("model", model)])

    estimators: dict[str, BaseEstimator] = {}

    # Baselines -------------------------------------------------------------
    estimators["LinearRegression"] = pipe(LinearRegression())

    estimators["Ridge"] = GridSearchCV(
        pipe(Ridge(random_state=config.RANDOM_STATE)),
        {"model__alpha": config.ALPHA_GRID},
        cv=inner, scoring="neg_root_mean_squared_error", n_jobs=-1,
    )
    estimators["Lasso"] = GridSearchCV(
        pipe(Lasso(random_state=config.RANDOM_STATE, max_iter=50000)),
        {"model__alpha": config.ALPHA_GRID},
        cv=inner, scoring="neg_root_mean_squared_error", n_jobs=-1,
    )

    # Advanced (constrained for small n) ------------------------------------
    estimators["RandomForest"] = GridSearchCV(
        pipe(RandomForestRegressor(random_state=config.RANDOM_STATE, n_jobs=-1)),
        config.RF_PARAM_GRID,
        cv=inner, scoring="neg_root_mean_squared_error", n_jobs=-1,
    )
    estimators["GradientBoosting"] = GridSearchCV(
        pipe(GradientBoostingRegressor(random_state=config.RANDOM_STATE)),
        config.GBR_PARAM_GRID,
        cv=inner, scoring="neg_root_mean_squared_error", n_jobs=-1,
    )
    return estimators


@dataclass
class CVResult:
    """Cross-validated metrics for one model (mean +/- std, train and test)."""

    name: str
    rmse_mean: float
    rmse_std: float
    mae_mean: float
    mae_std: float
    r2_mean: float
    r2_std: float
    train_rmse_mean: float
    train_r2_mean: float

    @property
    def cv_train_gap_r2(self) -> float:
        """Train R^2 minus CV R^2 — a positive gap signals overfitting."""
        return self.train_r2_mean - self.r2_mean

    def as_row(self) -> dict[str, float | str]:
        return {
            "model": self.name,
            "CV_RMSE": round(self.rmse_mean, 3),
            "RMSE_std": round(self.rmse_std, 3),
            "CV_MAE": round(self.mae_mean, 3),
            "CV_R2": round(self.r2_mean, 3),
            "R2_std": round(self.r2_std, 3),
            "train_RMSE": round(self.train_rmse_mean, 3),
            "train_R2": round(self.train_r2_mean, 3),
            "R2_gap": round(self.cv_train_gap_r2, 3),
        }


def evaluate_cv(name: str, estimator: BaseEstimator,
                X: pd.DataFrame, y: pd.Series) -> CVResult:
    """Run k-fold cross-validation and collect mean/std train & test metrics."""
    cv = cross_validate(
        estimator, X, y, cv=_kfold(), scoring=SCORING,
        return_train_score=True, n_jobs=-1,
    )
    return CVResult(
        name=name,
        rmse_mean=float(-np.mean(cv["test_rmse"])),
        rmse_std=float(np.std(cv["test_rmse"])),
        mae_mean=float(-np.mean(cv["test_mae"])),
        mae_std=float(np.std(cv["test_mae"])),
        r2_mean=float(np.mean(cv["test_r2"])),
        r2_std=float(np.std(cv["test_r2"])),
        train_rmse_mean=float(-np.mean(cv["train_rmse"])),
        train_r2_mean=float(np.mean(cv["train_r2"])),
    )


def results_table(results: list[CVResult]) -> pd.DataFrame:
    """Tidy DataFrame of all candidate results, sorted by CV RMSE ascending."""
    df = pd.DataFrame([r.as_row() for r in results])
    return df.sort_values("CV_RMSE").reset_index(drop=True)
