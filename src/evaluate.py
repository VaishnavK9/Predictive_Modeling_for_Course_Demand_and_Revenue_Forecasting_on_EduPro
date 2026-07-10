"""Evaluate the trained models and produce the driver (importance) analysis.

Outputs, per target:
  * 5-fold and Leave-One-Out CV metrics as mean +/- std
  * a train-vs-CV gap table (overfitting check)
  * predicted-vs-actual and residual plots (saved to reports/figures)
  * permutation importance plus model-native importances
    (tree importances / standardized linear coefficients)
  * plain-English driver findings written to reports/driver_findings.md

Run with::

    python -m src.evaluate
"""

from __future__ import annotations

import json

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import LeaveOneOut, cross_val_predict, cross_validate

from . import config
from .feature_engineering import get_modeling_frame
from .modeling import SCORING, _kfold

sns_ok = True
try:
    import seaborn as sns

    sns.set_theme(style="whitegrid")
except Exception:  # pragma: no cover
    sns_ok = False

plt.rcParams["figure.dpi"] = config.FIG_DPI


def _load(target: str):
    model = joblib.load(config.MODELS_DIR / f"{target}_best.joblib")
    meta = json.loads((config.MODELS_DIR / f"{target}_metadata.json").read_text())
    return model, meta


def _feature_names(model, numeric: list[str], nominal: list[str]) -> list[str]:
    """Recover post-preprocessing feature names from the fitted pipeline."""
    pipe = getattr(model, "best_estimator_", model)
    pre = pipe.named_steps["pre"]
    return list(pre.get_feature_names_out())


def cv_metric_block(model, X, y) -> dict[str, str]:
    """Return 5-fold and LOOCV MAE/RMSE/R^2 as 'mean +/- std' strings."""
    out: dict[str, str] = {}

    kf = cross_validate(model, X, y, cv=_kfold(), scoring=SCORING,
                        return_train_score=True, n_jobs=-1)
    out["kfold_RMSE"] = f"{-kf['test_rmse'].mean():.3f} +/- {kf['test_rmse'].std():.3f}"
    out["kfold_MAE"] = f"{-kf['test_mae'].mean():.3f} +/- {kf['test_mae'].std():.3f}"
    out["kfold_R2"] = f"{kf['test_r2'].mean():.3f} +/- {kf['test_r2'].std():.3f}"
    out["train_R2"] = f"{kf['train_r2'].mean():.3f}"
    out["R2_gap(train-cv)"] = f"{kf['train_r2'].mean() - kf['test_r2'].mean():.3f}"

    # LOOCV: R^2 across single-point test folds is undefined, so we aggregate
    # predictions and score once (standard practice for LOO).
    loo = LeaveOneOut()
    y_pred = cross_val_predict(model, X, y, cv=loo, n_jobs=-1)
    out["loo_RMSE"] = f"{np.sqrt(mean_squared_error(y, y_pred)):.3f}"
    out["loo_MAE"] = f"{mean_absolute_error(y, y_pred):.3f}"
    out["loo_R2"] = f"{r2_score(y, y_pred):.3f}"
    return out, y_pred


def diagnostics_plot(target: str, y, y_pred) -> None:
    """Predicted-vs-actual and residual plots for one target."""
    resid = y - y_pred
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].scatter(y, y_pred, color="#3b6ea5", alpha=0.8)
    lo, hi = min(y.min(), y_pred.min()), max(y.max(), y_pred.max())
    axes[0].plot([lo, hi], [lo, hi], "k--", lw=1)
    axes[0].set_xlabel("Actual")
    axes[0].set_ylabel("Predicted (LOOCV)")
    axes[0].set_title(f"Predicted vs actual — {target}")

    axes[1].scatter(y_pred, resid, color="#a53b5b", alpha=0.8)
    axes[1].axhline(0, color="k", ls="--", lw=1)
    axes[1].set_xlabel("Predicted (LOOCV)")
    axes[1].set_ylabel("Residual")
    axes[1].set_title(f"Residuals — {target}")
    fig.tight_layout()
    path = config.FIGURES_DIR / f"eval_{target}.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] {path.name}")


def importance_analysis(target: str, model, X, y, numeric, nominal) -> pd.DataFrame:
    """Permutation importance (+ native importances) for the fitted winner."""
    perm = permutation_importance(
        model, X, y, n_repeats=30, random_state=config.RANDOM_STATE,
        scoring="neg_root_mean_squared_error", n_jobs=-1,
    )
    imp = pd.DataFrame(
        {"feature": numeric + nominal,
         "perm_importance": perm.importances_mean,
         "perm_std": perm.importances_std}
    ).sort_values("perm_importance", ascending=False)

    # Native importances (tree) or standardized coefficients (linear).
    pipe = getattr(model, "best_estimator_", model)
    est = pipe.named_steps["model"]
    feat_out = _feature_names(model, numeric, nominal)
    native = None
    if hasattr(est, "feature_importances_"):
        native = pd.Series(est.feature_importances_, index=feat_out,
                           name="tree_importance")
    elif hasattr(est, "coef_"):
        native = pd.Series(np.abs(est.coef_), index=feat_out,
                           name="abs_std_coef")

    fig, ax = plt.subplots(figsize=(9, 6))
    top = imp.head(12).iloc[::-1]
    ax.barh(top["feature"], top["perm_importance"],
            xerr=top["perm_std"], color="#3b6ea5")
    ax.set_title(f"Permutation importance — {target}")
    ax.set_xlabel("Increase in RMSE when shuffled")
    fig.tight_layout()
    path = config.FIGURES_DIR / f"importance_{target}.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] {path.name}")

    if native is not None:
        print(f"  Native importance ({native.name}), top 8:")
        print(native.sort_values(ascending=False).head(8).round(4).to_string())
    return imp


def _driver_sentences(target: str, imp: pd.DataFrame, master: pd.DataFrame) -> list[str]:
    """Translate the top drivers into plain-English statements."""
    top = imp.head(5)["feature"].tolist()
    lines = [f"**{target}** — most influential features (permutation): "
             + ", ".join(top) + "."]
    # Sign/direction hints from simple correlations with the target.
    num = [c for c in imp["feature"] if c in master.columns
           and pd.api.types.is_numeric_dtype(master[c])]
    corr = master[num + [target]].corr()[target].drop(target)
    for f in top:
        if f in corr.index:
            direction = "higher" if corr[f] > 0 else "lower"
            lines.append(
                f"- `{f}`: r={corr[f]:+.2f} with {target} "
                f"({direction} {f} associates with {direction} {target})."
            )
    return lines


def main() -> None:
    print("=" * 70)
    print("EduPro evaluation & driver analysis")
    print("=" * 70)
    master, numeric, nominal = get_modeling_frame()
    X = master[numeric + nominal]

    driver_md = ["# Driver findings (plain English)\n"]
    for target in config.TARGETS:
        print("\n" + "-" * 70)
        print(f"TARGET: {target}")
        print("-" * 70)
        model, meta = _load(target)
        y = master[target]

        metrics, y_pred = cv_metric_block(model, X, y)
        print(f"  Winner: {meta['winner']}")
        for k, v in metrics.items():
            print(f"    {k:<18} {v}")

        diagnostics_plot(target, y, y_pred)
        imp = importance_analysis(target, model, X, y, numeric, nominal)

        driver_md.append(f"\n## {target} (winner: {meta['winner']})\n")
        driver_md.append(
            f"- 5-fold R2: {metrics['kfold_R2']}; LOOCV R2: {metrics['loo_R2']}; "
            f"train-vs-CV R2 gap: {metrics['R2_gap(train-cv)']}.\n"
        )
        for line in _driver_sentences(target, imp, master):
            driver_md.append(line + "\n")

    out = config.REPORTS_DIR / "driver_findings.md"
    out.write_text("\n".join(driver_md), encoding="utf-8")
    print(f"\nDriver findings written -> {out}")


if __name__ == "__main__":
    main()
