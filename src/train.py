"""Train and select models for each course-level target.

For every target we cross-validate a model zoo (LinearRegression, Ridge, Lasso,
RandomForest, GradientBoosting), pick the winner on **CV RMSE** (never the
training score), refit it on all 60 courses, and persist it with a metadata
JSON. Run with::

    python -m src.train
"""

from __future__ import annotations

import json
from datetime import datetime

import joblib
import pandas as pd

from . import config
from .feature_engineering import get_modeling_frame
from .modeling import candidate_estimators, evaluate_cv, results_table


def train_target(
    master: pd.DataFrame, target: str, numeric: list[str], nominal: list[str]
) -> dict:
    """Cross-validate the model zoo for one target and persist the winner."""
    X = master[numeric + nominal]
    y = master[target]

    print("\n" + "-" * 70)
    print(f"TARGET: {target}  ({config.TARGETS.get(target, '')})")
    print("-" * 70)

    results = [
        evaluate_cv(name, est, X, y)
        for name, est in candidate_estimators(numeric, nominal).items()
    ]
    table = results_table(results)
    print(table.to_string(index=False))

    winner_name = table.iloc[0]["model"]
    print(f"\n  -> Winner by CV RMSE: {winner_name}")

    # Refit the winning estimator on all data (GridSearchCV refits internally).
    winner_est = candidate_estimators(numeric, nominal)[winner_name]
    winner_est.fit(X, y)

    model_path = config.MODELS_DIR / f"{target}_best.joblib"
    joblib.dump(winner_est, model_path)

    # Resolve tuned hyperparameters if the winner was a GridSearchCV.
    best_params = getattr(winner_est, "best_params_", {})

    meta = {
        "target": target,
        "winner": winner_name,
        "best_params": {k: _jsonable(v) for k, v in best_params.items()},
        "cv_folds": config.CV_FOLDS,
        "random_state": config.RANDOM_STATE,
        "numeric_features": numeric,
        "nominal_features": nominal,
        "cv_results": table.to_dict(orient="records"),
        "n_rows": int(len(master)),
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "model_file": model_path.name,
    }
    meta_path = config.MODELS_DIR / f"{target}_metadata.json"
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"  Saved model    -> {model_path.name}")
    print(f"  Saved metadata -> {meta_path.name}")
    return meta


def _jsonable(v):
    """Coerce numpy scalars to native types for JSON."""
    try:
        return v.item()
    except AttributeError:
        return v


def main() -> None:
    print("=" * 70)
    print("EduPro model training (CV-selected)")
    print("=" * 70)
    master, numeric, nominal = get_modeling_frame()
    summaries = []
    for target in config.TARGETS:
        summaries.append(train_target(master, target, numeric, nominal))

    print("\n" + "=" * 70)
    print("Training complete. Winners:")
    for s in summaries:
        print(f"  {s['target']:<18} -> {s['winner']}  "
              f"(CV RMSE {s['cv_results'][0]['CV_RMSE']}, "
              f"CV R2 {s['cv_results'][0]['CV_R2']})")


if __name__ == "__main__":
    main()
