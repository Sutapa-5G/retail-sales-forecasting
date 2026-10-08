"""Model selection (validation only), one-time test evaluation, final refit, saving.

Order of operations (the test set is touched exactly once, at step 4):
  1. Split: pool = everything before the final 30 days; test = last 30 days.
  2. Validate baselines and candidate models on expanding-window folds of the POOL.
  3. Select the final model from validation results only.
  4. Evaluate the selected model (fitted on the pool) on the untouched test set.
  5. Refit the selected configuration on ALL data, save it, forecast 30 days ahead.
"""
import json
import platform
import sys
from dataclasses import dataclass, field

import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor

from src import config
from src import data_preprocessing as dp
from src import evaluate as ev
from src import feature_engineering as fe
from src import forecast as fc

PROTOCOL_RECURSIVE = "30-day recursive"
PROTOCOL_ONE_STEP = "1-day-ahead"
METRIC_COLUMNS = ["mae", "rmse", "smape_trading_days", "wape", "r2"]
FAMILY_LABELS = {"linear": "Linear Regression", "random_forest": "Random Forest", "xgboost": "XGBoost"}


# ----------------------------------------------------------------------------
# Candidate models
# ----------------------------------------------------------------------------
@dataclass
class ModelSpec:
    name: str
    family: str
    params: dict = field(default_factory=dict)


def candidate_specs() -> list:
    """A small, explainable search space (1 + 6 + 8 = 15 configurations)."""
    specs = [ModelSpec("Linear Regression", "linear")]
    for depth in (4, 8, None):
        for leaf in (1, 3):
            specs.append(ModelSpec(
                f"Random Forest (depth={depth}, min_leaf={leaf})", "random_forest",
                {"n_estimators": 300, "max_depth": depth, "min_samples_leaf": leaf}))
    for n_est in (200, 400):
        for depth in (3, 5):
            for lr in (0.03, 0.1):
                specs.append(ModelSpec(
                    f"XGBoost (n={n_est}, depth={depth}, lr={lr})", "xgboost",
                    {"n_estimators": n_est, "max_depth": depth, "learning_rate": lr}))
    return specs


def make_estimator(spec: ModelSpec):
    """Build an UNFITTED estimator. Any scaling/encoding lives inside the pipeline,
    so it is fitted on training rows only."""
    if spec.family == "linear":
        # weekday and month as categories; quarter/week/day-of-year are redundant
        # for a linear model once month is one-hot encoded; is_weekend is implied
        # by the weekday dummies.
        prep = ColumnTransformer([
            ("categories", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
             ["day_of_week", "month"]),
            ("numeric", StandardScaler(),
             ["day_of_month"] + fe.LAG_COLUMNS + fe.ROLLING_COLUMNS),
            ("flags", "passthrough", ["is_uk_holiday", "is_year_end_closure"]),
        ], remainder="drop")
        return Pipeline([("prep", prep), ("model", LinearRegression())])
    if spec.family == "random_forest":
        return RandomForestRegressor(random_state=config.RANDOM_SEED, n_jobs=-1, **spec.params)
    if spec.family == "xgboost":
        return XGBRegressor(subsample=0.8, colsample_bytree=0.8, tree_method="hist",
                            random_state=config.RANDOM_SEED, n_jobs=1, verbosity=0, **spec.params)
    raise ValueError(f"unknown model family: {spec.family}")


def fit_model(spec: ModelSpec, series: pd.Series):
    """Fit on the supplied series only (features are built from it alone)."""
    X, y = fe.make_training_set(series)
    model = make_estimator(spec)
    model.fit(X, y)
    return model


def _one_step_predictions(model, features: pd.DataFrame, dates) -> np.ndarray:
    """1-day-ahead predictions from actual lags, clipped at 0, known closures set to 0."""
    pred = np.clip(np.asarray(model.predict(features.loc[dates, fe.FEATURE_COLUMNS])), 0, None)
    pred[fe.known_closure_mask(pd.DatetimeIndex(dates))] = 0.0
    return pred


# ----------------------------------------------------------------------------
# Validation (pool only)
# ----------------------------------------------------------------------------
def run_validation(pool: pd.Series, folds: list, specs: list, verbose: bool = False):
    """Evaluate baselines and candidate models on every fold.

    Receives ONLY the pool, so the test set cannot influence anything here.
    Returns (fold_metrics, predictions).
    """
    features_pool = fe.build_features(pool)
    metric_rows, prediction_rows = [], []

    def record(name, kind, family, params, fold, protocol, actual, predicted):
        metric_rows.append({"model": name, "kind": kind, "family": family, "params": json.dumps(params),
                            "fold": fold.fold, "protocol": protocol,
                            **ev.compute_metrics(actual, predicted)})

    def keep(name, fold, dates, actual, recursive, one_step):
        prediction_rows.append(pd.DataFrame({
            "model": name, "fold": fold.fold, "date": dates, "actual": np.asarray(actual),
            "recursive": np.asarray(recursive), "one_step": np.asarray(one_step)}))

    for fold in folds:
        history = pool.loc[: fold.train_end]
        actual = pool.loc[fold.val_start: fold.val_end]
        dates, horizon = actual.index, len(actual)

        for name in ev.BASELINE_NAMES:
            rec = ev.baseline_window_forecast(name, history, horizon)
            one = ev.baseline_one_step(name, pool, dates)
            record(ev.BASELINE_LABELS[name], "baseline", "baseline", {}, fold, PROTOCOL_RECURSIVE, actual, rec)
            record(ev.BASELINE_LABELS[name], "baseline", "baseline", {}, fold, PROTOCOL_ONE_STEP, actual, one)
            keep(ev.BASELINE_LABELS[name], fold, dates, actual, rec, one)

        for spec in specs:
            model = fit_model(spec, history)
            rec = fc.recursive_forecast(model, history, horizon)
            one = _one_step_predictions(model, features_pool, dates)
            record(spec.name, "ml", spec.family, spec.params, fold, PROTOCOL_RECURSIVE, actual, rec)
            record(spec.name, "ml", spec.family, spec.params, fold, PROTOCOL_ONE_STEP, actual, one)
            keep(spec.name, fold, dates, actual, rec, one)
        if verbose:
            print(f"  validation fold {fold.fold} done ({fold.val_start.date()} -> {fold.val_end.date()})")

    return pd.DataFrame(metric_rows), pd.concat(prediction_rows, ignore_index=True)


def summarize_validation(fold_metrics: pd.DataFrame) -> pd.DataFrame:
    """Mean of each metric across folds, per model and protocol."""
    keys = ["model", "kind", "family", "params", "protocol"]
    summary = fold_metrics.groupby(keys, sort=False)[METRIC_COLUMNS].mean().reset_index()
    summary["n_folds"] = fold_metrics.groupby(keys, sort=False)["fold"].nunique().to_numpy()
    return summary


def select_final_model(summary: pd.DataFrame, specs: list):
    """Pick the final model from VALIDATION results only.

    Rule: lowest mean MAE under the 30-day recursive protocol (the protocol used
    in deployment); RMSE breaks ties. Also returns the best config per family.
    """
    rec = summary[(summary["kind"] == "ml") & (summary["protocol"] == PROTOCOL_RECURSIVE)]
    rec = rec.sort_values(["mae", "rmse"]).reset_index(drop=True)
    best_per_family = rec.groupby("family", sort=False).head(1)
    winner_name = rec.iloc[0]["model"]
    spec = next(s for s in specs if s.name == winner_name)
    return spec, best_per_family


# ----------------------------------------------------------------------------
# One-time test evaluation
# ----------------------------------------------------------------------------
def evaluate_on_test(spec: ModelSpec, pool: pd.Series, test: pd.Series, quantiles: dict) -> dict:
    """Fit on the pool and score on the test set. Called exactly once per run.

    The recursive forecast starts at the end of the pool and uses no test actual.
    The 1-day-ahead score uses actuals of the days BEFORE each test day, which is
    exactly what is known when forecasting one day ahead.
    """
    model = fit_model(spec, pool)
    horizon = len(test)
    rec = fc.recursive_forecast(model, pool, horizon)
    bands = fc.add_intervals(rec, quantiles)

    full = pd.concat([pool, test])
    one = pd.Series(_one_step_predictions(model, fe.build_features(full), test.index), index=test.index)

    result = {
        "model": model,
        "metrics": {PROTOCOL_RECURSIVE: ev.compute_metrics(test, rec),
                    PROTOCOL_ONE_STEP: ev.compute_metrics(test, one)},
        "predictions": pd.DataFrame({"actual": test, "forecast": bands["forecast"], "lower": bands["lower"],
                                     "upper": bands["upper"], "one_step_forecast": one}),
        "interval_coverage": float(((test >= bands["lower"]) & (test <= bands["upper"])).mean()),
        "baselines": {},
    }
    for name in ev.BASELINE_NAMES:
        rec_b = ev.baseline_window_forecast(name, pool, horizon)
        one_b = ev.baseline_one_step(name, full, test.index)
        result["baselines"][name] = {PROTOCOL_RECURSIVE: ev.compute_metrics(test, rec_b),
                                     PROTOCOL_ONE_STEP: ev.compute_metrics(test, one_b)}
        result["predictions"][f"baseline_{name}"] = rec_b
    return result


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def verify_frozen_split(series: pd.Series, pool: pd.Series, test: pd.Series, folds: list) -> None:
    """Refuse to run if the data no longer matches the split frozen in config."""
    frozen = config.FROZEN_SPLIT
    observed = {
        "series_start": str(series.index[0].date()), "series_end": str(series.index[-1].date()),
        "pool_end": str(pool.index[-1].date()),
        "test_start": str(test.index[0].date()), "test_end": str(test.index[-1].date()),
    }
    for key, value in observed.items():
        if frozen[key] != value:
            raise ValueError(f"Frozen split mismatch for {key}: config={frozen[key]}, data={value}")
    got = [{"fold": f.fold, "train_end": str(f.train_end.date()), "val_start": str(f.val_start.date()),
            "val_end": str(f.val_end.date())} for f in folds]
    if got != frozen["validation_folds"]:
        raise ValueError("Frozen validation folds in config do not match the computed folds")


def _clean(obj):
    """Make nested structures JSON-safe (numpy -> python, NaN -> None)."""
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        return None if np.isnan(obj) else float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, (pd.Timestamp,)):
        return str(obj.date())
    return obj


def _comparison_rows(summary: pd.DataFrame, best_per_family: pd.DataFrame, winner: str,
                     test_result: dict) -> pd.DataFrame:
    """Rows for reports/model_comparison.csv."""
    rows = []
    shown = set(best_per_family["model"]) | set(ev.BASELINE_LABELS.values())
    for _, r in summary[summary["model"].isin(shown)].iterrows():
        rows.append({"model": r["model"], "kind": r["kind"], "protocol": r["protocol"],
                     "split": f"validation (mean of {int(r['n_folds'])} folds)",
                     "selected_final_model": r["model"] == winner, "hyperparameters": r["params"],
                     **{m: r[m] for m in METRIC_COLUMNS}})
    for protocol in (PROTOCOL_RECURSIVE, PROTOCOL_ONE_STEP):
        rows.append({"model": winner, "kind": "ml", "protocol": protocol, "split": "test (final 30 days)",
                     "selected_final_model": True, "hyperparameters": "",
                     **test_result["metrics"][protocol]})
        for name in ev.BASELINE_NAMES:
            rows.append({"model": ev.BASELINE_LABELS[name], "kind": "baseline", "protocol": protocol,
                         "split": "test (final 30 days)", "selected_final_model": False, "hyperparameters": "",
                         **test_result["baselines"][name][protocol]})
    out = pd.DataFrame(rows)
    out["split_order"] = out["split"].str.startswith("test").astype(int)
    out["kind_order"] = (out["kind"] == "ml").astype(int)
    out = out.sort_values(["split_order", "protocol", "kind_order", "mae"], ascending=[True, False, True, True])
    return out.drop(columns=["split_order", "kind_order"]).reset_index(drop=True)


# ----------------------------------------------------------------------------
# Main entry point
# ----------------------------------------------------------------------------
def run_training(series: pd.Series = None, save: bool = True, verbose: bool = True, specs: list = None,
                 verify_split: bool = True, n_folds: int = config.N_CV_FOLDS,
                 window: int = config.CV_WINDOW_DAYS) -> dict:
    """Run the full, leakage-safe training workflow described in the module docstring.

    ``specs``, ``n_folds``, ``window`` and ``verify_split`` exist so the tests can run a
    small version on synthetic data; the real run uses the defaults.
    """
    series = dp.load_revenue_series() if series is None else series
    pool, test = ev.chronological_split(series, config.TEST_DAYS)
    folds = ev.expanding_window_folds(pool.index, n_folds=n_folds, window=window)
    if verify_split:
        verify_frozen_split(series, pool, test, folds)
    specs = candidate_specs() if specs is None else specs
    if verbose:
        print(f"pool: {pool.index[0].date()} -> {pool.index[-1].date()} ({len(pool)} days) | "
              f"test: {test.index[0].date()} -> {test.index[-1].date()} ({len(test)} days)")
        print(f"validating {len(specs)} configurations + {len(ev.BASELINE_NAMES)} baselines on {len(folds)} folds")

    # 2-3. validation and selection (pool only)
    fold_metrics, val_predictions = run_validation(pool, folds, specs, verbose)
    summary = summarize_validation(fold_metrics)
    spec, best_per_family = select_final_model(summary, specs)
    if verbose:
        print("selected on validation:", spec.name)

    # residual quantiles for intervals come from the selected model's validation forecasts
    sel = val_predictions[val_predictions["model"] == spec.name]
    quantiles = fc.fit_residual_quantiles(sel["actual"], sel["recursive"], pd.DatetimeIndex(sel["date"]))

    # 4. the single test evaluation
    test_result = evaluate_on_test(spec, pool, test, quantiles)

    # 5. refit on all data, forecast the future
    final_model = fit_model(spec, series)
    future = fc.recursive_forecast(final_model, series, config.FORECAST_HORIZON)
    future_frame = fc.add_intervals(future, quantiles)
    future_frame.index.name = config.DATE_COLUMN

    comparison = _comparison_rows(summary, best_per_family, spec.name, test_result)
    X_full, _ = fe.make_training_set(series)
    selected_val = summary[(summary["model"] == spec.name)].set_index("protocol")[METRIC_COLUMNS]
    metadata = {
        "model_name": spec.name,
        "model_family": FAMILY_LABELS[spec.family],
        "hyperparameters": spec.params,
        "target": f"{config.TARGET_COLUMN} (daily gross revenue, GBP)",
        "feature_columns": fe.FEATURE_COLUMNS,
        "forecast_horizon_days": config.FORECAST_HORIZON,
        "forecast_strategy": "recursive (each day's forecast feeds the next day's lag/rolling features)",
        "selection_rule": "lowest mean MAE over validation folds, 30-day recursive protocol; test set not used",
        "split": config.FROZEN_SPLIT,
        "saved_model_training_period": {"series_start": str(series.index[0].date()),
                                        "series_end": str(series.index[-1].date()),
                                        "first_training_row": str(X_full.index[0].date()),
                                        "n_training_rows": int(len(X_full))},
        "validation_metrics_mean_of_folds": {p: selected_val.loc[p].to_dict() for p in selected_val.index},
        "test_metrics_pool_trained_model": test_result["metrics"],
        "test_note": "Test metrics belong to the same configuration fitted on data up to "
                     f"{pool.index[-1].date()} only. The saved model is refit on all data through "
                     f"{series.index[-1].date()} with the same hyperparameters.",
        "prediction_interval": {"method": "empirical residual quantiles from out-of-sample validation forecasts",
                                "nominal_coverage": config.INTERVAL_COVERAGE,
                                "residual_quantiles": quantiles,
                                "observed_test_coverage": test_result["interval_coverage"]},
        "random_seed": config.RANDOM_SEED,
        "versions": {"python": platform.python_version(), "pandas": pd.__version__, "numpy": np.__version__,
                     "scikit-learn": sklearn.__version__, "xgboost": xgboost.__version__,
                     "joblib": joblib.__version__},
    }

    artifacts = {"spec": spec, "summary": summary, "fold_metrics": fold_metrics,
                 "val_predictions": val_predictions, "test_result": test_result,
                 "comparison": comparison, "metadata": _clean(metadata), "forecast": future_frame,
                 "final_model": final_model, "folds": folds, "pool": pool, "test": test}

    if save:
        config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
        config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(final_model, config.MODEL_FILE, compress=3)
        config.MODEL_METADATA_FILE.write_text(json.dumps(artifacts["metadata"], indent=2))
        comparison.round(4).to_csv(config.MODEL_COMPARISON_FILE, index=False)
        summary.round(4).to_csv(config.HYPERPARAMETER_SEARCH_FILE, index=False)
        fold_metrics.round(4).to_csv(config.VALIDATION_FOLD_METRICS_FILE, index=False)
        future_frame.round(2).to_csv(config.FORECAST_FILE)
        test_result["predictions"].round(2).to_csv(config.TEST_PREDICTIONS_FILE, index_label=config.DATE_COLUMN)
        sel.round({"actual": 2, "recursive": 2, "one_step": 2}).to_csv(
            config.VALIDATION_PREDICTIONS_FILE, index=False)

        # reproducibility check: the reloaded artifact must give identical forecasts
        reloaded, meta = fc.load_artifacts()
        again = fc.forecast_with_artifacts(reloaded, meta, series, config.FORECAST_HORIZON)
        if not np.allclose(again["forecast"].to_numpy(), future_frame["forecast"].to_numpy()):
            raise RuntimeError("Reloaded model does not reproduce the forecast")
        if verbose:
            print("artifacts saved; reloaded model reproduces the 30-day forecast exactly")
    return artifacts


def main() -> None:
    result = run_training()
    print(result["comparison"].round(2).to_string(index=False))


if __name__ == "__main__":
    sys.exit(main())
